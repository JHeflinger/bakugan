#include "loss.hpp"
#include "dino.hpp"
#include "util/box.hpp"
#include "data/field.hpp"
#include <raymath.h>
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>

namespace Bakugan {
    namespace {
        // f(0) .. f(n - 1) on every core.
        void ParallelFor(int n, const std::function<void(int)>& f) {
            int workers = std::min(n, std::max(1, (int)std::thread::hardware_concurrency()));
            std::atomic<int> next{ 0 };
            auto work = [&] {
                for (int i = next++; i < n; i = next++) f(i);
            };
            std::vector<std::thread> threads;
            for (int k = 1; k < workers; k++) threads.emplace_back(work);
            work();
            for (std::thread& t : threads) t.join();
        }

        // Fills a triangle into a supersampled coverage grid (pixel coordinates times ss).
        void FillTriangle(Vector2 a, Vector2 b, Vector2 c, int n, std::vector<unsigned char>& hit) {
            int x0 = std::max(0, (int)floorf(fminf(a.x, fminf(b.x, c.x)))), x1 = std::min(n - 1, (int)ceilf(fmaxf(a.x, fmaxf(b.x, c.x))));
            int y0 = std::max(0, (int)floorf(fminf(a.y, fminf(b.y, c.y)))), y1 = std::min(n - 1, (int)ceilf(fmaxf(a.y, fmaxf(b.y, c.y))));
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++) {
                    float px = x + 0.5f, py = y + 0.5f;
                    float e0 = (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x);
                    float e1 = (c.x - b.x) * (py - b.y) - (c.y - b.y) * (px - b.x);
                    float e2 = (a.x - c.x) * (py - c.y) - (a.y - c.y) * (px - c.x);
                    if ((e0 >= 0 && e1 >= 0 && e2 >= 0) || (e0 <= 0 && e1 <= 0 && e2 <= 0)) hit[(size_t)y * n + x] = 1;
                }
        }
    }

    // ---- setting up: the frame and views, the mesh, the sphere ----------------------------------------------------------

    Loss::Loss(const TriMesh& mesh, const AutoParams& p) : m_P(p) {
        Vector3 lo = { FLT_MAX, FLT_MAX, FLT_MAX }, hi = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
        for (unsigned i : mesh.indices) { lo = Vector3Min(lo, mesh.positions[i]); hi = Vector3Max(hi, mesh.positions[i]); }
        m_L = fmaxf(Vector3Distance(lo, hi), 1e-3f);
        m_Centre = Vector3Scale(Vector3Add(lo, hi), 0.5f);
        m_Half = 0.7f * m_L;                     // the mesh, and room to move and grow it a little, stays in the frame
        m_Res = std::clamp(p.viewResolution, 16, 256);
        m_Tau = 2.0f * m_Half / m_Res;           // one pixel: silhouettes fade over two pixels, for smoother gradients

        // views spread over the upper half of a sphere (from the other half they would be mirror images)
        int n = std::clamp(p.views, 1, 24);
        for (int i = 0; i < n; i++) {
            float y = 1.0f - (i + 0.5f) / n, r = sqrtf(fmaxf(0.0f, 1.0f - y * y)), phi = i * 2.39996323f + 0.6f;
            View w;
            w.dir = Vector3Negate({ r * cosf(phi), y, r * sinf(phi) });
            Vector3 up = fabsf(w.dir.y) < 0.99f ? Vector3{ 0, 1, 0 } : Vector3{ 0, 0, -1 };
            w.u = Vector3Normalize(Vector3CrossProduct(w.dir, up));
            w.v = Vector3CrossProduct(w.u, w.dir);
            m_Views.push_back(w);
        }

        // the mesh's centre of volume and volume (signed tetrahedra from the box's centre): the sphere's
        double volume = 0.0;
        Vector3 moment = { 0, 0, 0 };
        for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
            Vector3 a = Vector3Subtract(mesh.positions[mesh.indices[t]], m_Centre), b = Vector3Subtract(mesh.positions[mesh.indices[t + 1]], m_Centre),
                    c = Vector3Subtract(mesh.positions[mesh.indices[t + 2]], m_Centre);
            double v = Vector3DotProduct(a, Vector3CrossProduct(b, c)) / 6.0;
            volume += v;
            moment = Vector3Add(moment, Vector3Scale(Vector3Add(a, Vector3Add(b, c)), (float)(v / 4.0)));
        }
        bool closed = fabs(volume) > 1e-9 * m_L * m_L * m_L;
        m_MeshCentre = closed ? Vector3Add(m_Centre, Vector3Scale(moment, (float)(1.0 / volume))) : m_Centre;
        m_SphereCentre = m_MeshCentre;
        m_SphereRadius = p.sphereDiameter > 0.0f ? 0.5f * p.sphereDiameter : (closed ? (float)cbrt(3.0 * fabs(volume) / (4.0 * PI)) : 0.25f * m_L);
        m_MeshReach = 0.0f;
        for (unsigned i : mesh.indices) m_MeshReach = fmaxf(m_MeshReach, Vector3Distance(mesh.positions[i], m_MeshCentre));

        // the mesh's silhouette in each view, finely (4 samples a pixel across), where it is: placements resample it
        m_Fine = m_Res * 4;
        m_MeshFine.assign(n, std::vector<float>((size_t)m_Fine * m_Fine, 0.0f));
        ParallelFor(n, [&](int v) {
            std::vector<unsigned char> hit((size_t)m_Fine * m_Fine, 0);
            std::vector<Vector2> pts(mesh.positions.size());
            const View& w = m_Views[v];
            for (size_t i = 0; i < pts.size(); i++) {
                Vector3 q = Vector3Subtract(mesh.positions[i], m_Centre);
                pts[i] = { (Vector3DotProduct(q, w.u) / m_Half + 1.0f) * 0.5f * m_Fine, (1.0f - Vector3DotProduct(q, w.v) / m_Half) * 0.5f * m_Fine };
            }
            for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) FillTriangle(pts[mesh.indices[t]], pts[mesh.indices[t + 1]], pts[mesh.indices[t + 2]], m_Fine, hit);
            for (size_t k = 0; k < hit.size(); k++) m_MeshFine[v][k] = hit[k];
        });

        // the collisions' sample points
        std::mt19937 rng(p.seed * 31u + 7u);
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);
        for (int i = 0; i < 128; i++) m_PairUnit.push_back({ u01(rng), u01(rng), u01(rng) });
        for (int i = 0; i < 1024; i++) {                 // a Fibonacci sphere
            float y = 1.0f - (i + 0.5f) / 512.0f, r = sqrtf(fmaxf(0.0f, 1.0f - y * y)), phi = i * 2.39996323f;
            m_SurfaceDirs.push_back({ r * cosf(phi), y, r * sinf(phi) });
        }
        if (p.semanticWeight > 0.0f) SetUpSemantic(mesh);
        m_Target = TargetSilhouettes(m_Placed);
        if (m_Dino) m_SemTarget = SemanticTargets(m_Placed);
    }

    Loss::~Loss() = default;

    std::vector<Vector3> Loss::ViewDirections() const {
        std::vector<Vector3> out;
        for (const View& w : m_Views) out.push_back(w.dir);
        return out;
    }

    // ---- the placed mesh --------------------------------------------------------------------------------------------------

    Vector2 Loss::Unplace(const View& w, float sx, float sy, const Placement& t) const {
        Vector3 at = Vector3Add(m_Centre, Vector3Add(Vector3Scale(w.u, sx), Vector3Scale(w.v, sy)));
        Vector3 q = Vector3Add(m_MeshCentre, Vector3Scale(Vector3Subtract(Vector3Subtract(at, m_MeshCentre), t.offset), 1.0f / fmaxf(t.scale, 1e-6f)));
        Vector3 r = Vector3Subtract(q, m_Centre);
        return { Vector3DotProduct(r, w.u), Vector3DotProduct(r, w.v) };
    }

    namespace {
        // A fine map (n x n) at a point given in its pixels (centres at +0.5), bilinear, nothing outside it.
        float Bilinear(const std::vector<float>& m, int n, float fx, float fy) {
            fx -= 0.5f;
            fy -= 0.5f;
            int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
            float ax = fx - x0, ay = fy - y0;
            auto at = [&](int x, int y) { return x < 0 || y < 0 || x >= n || y >= n ? 0.0f : m[(size_t)y * n + x]; };
            return (1 - ay) * ((1 - ax) * at(x0, y0) + ax * at(x0 + 1, y0)) + ay * ((1 - ax) * at(x0, y0 + 1) + ax * at(x0 + 1, y0 + 1));
        }
    }

    std::vector<std::vector<float>> Loss::TargetSilhouettes(const Placement& t) const {
        const int nv = (int)m_Views.size();
        std::vector<std::vector<float>> out(nv, std::vector<float>((size_t)m_Res * m_Res, 0.0f));
        for (int v = 0; v < nv; v++)
            for (int y = 0; y < m_Res; y++)
                for (int x = 0; x < m_Res; x++) {
                    float sum = 0.0f;
                    for (int k = 0; k < 4; k++) {                // 2 x 2 samples a pixel
                        float sx = ((x + 0.25f + 0.5f * (k & 1)) / m_Res * 2.0f - 1.0f) * m_Half, sy = (1.0f - (y + 0.25f + 0.5f * (k >> 1)) / m_Res * 2.0f) * m_Half;
                        Vector2 b = Unplace(m_Views[v], sx, sy, t);
                        sum += Bilinear(m_MeshFine[v], m_Fine, (b.x / m_Half + 1.0f) * 0.5f * m_Fine, (1.0f - b.y / m_Half) * 0.5f * m_Fine);
                    }
                    out[v][(size_t)y * m_Res + x] = 0.25f * sum;
                }
        return out;
    }

    Placement Loss::KeepTargetInFrame(Placement t) const {
        t.scale = std::clamp(t.scale, 0.3f, 3.3f);
        float room = 0.98f * m_Half;                     // the mesh's reach around its centre, inside the frame
        t.scale = fminf(t.scale, room / fmaxf(m_MeshReach, 1e-6f));
        Vector3 at = Vector3Add(m_MeshCentre, t.offset), off = Vector3Subtract(at, m_Centre);
        float l = Vector3Length(off), free = fmaxf(0.0f, room - t.scale * m_MeshReach);
        if (l > free) t.offset = Vector3Subtract(Vector3Add(m_Centre, Vector3Scale(off, free / l)), m_MeshCentre);
        return t;
    }

    // ---- drawing the design ---------------------------------------------------------------------------------------------

    float Loss::Trace(const BodyField& f, const BoundingBox& box, const Rigid& xf, const View& w, float sx, float sy, float tau,
                      Vector3* at, Vector3* hit) const {
        const float far = 4.0f * m_Half + Vector3Distance(box.min, box.max);
        Vector3 o = Vector3Add(Vector3Subtract(m_Centre, Vector3Scale(w.dir, far)), Vector3Add(Vector3Scale(w.u, sx), Vector3Scale(w.v, sy)));
        // the ray in the body's own frame
        Quaternion inv = { -xf.rotation.x, -xf.rotation.y, -xf.rotation.z, xf.rotation.w };
        Vector3 lo = xf.ApplyInverse(o), dir = Vector3RotateByQuaternion(w.dir, inv);
        // its stretch inside the box (slabs)
        float tn = 0.0f, tf = 2.0f * far;
        for (int a = 0; a < 3; a++) {
            float oa = (&lo.x)[a], da = (&dir.x)[a], l = (&box.min.x)[a] - tau, h = (&box.max.x)[a] + tau;
            if (fabsf(da) < 1e-9f) {
                if (oa < l || oa > h) return FLT_MAX;
                continue;
            }
            float t0 = (l - oa) / da, t1 = (h - oa) / da;
            if (t0 > t1) std::swap(t0, t1);
            tn = fmaxf(tn, t0);
            tf = fminf(tf, t1);
            if (tn > tf) return FLT_MAX;
        }
        float t = tn, dmin = FLT_MAX, tmin = tn, outside = -1.0f, inside = -1.0f;   // outside: the last t outside
        for (int k = 0; k < 64 && t <= tf; k++) {
            float d = f.Distance(Vector3Add(lo, Vector3Scale(dir, t)));
            if (d < dmin) { dmin = d; tmin = t; }
            if (inside < 0.0f) (d >= 0.0f ? outside : inside) = t;
            if (d < -tau) break;                       // well inside: fully covered
            t += d > tau ? 0.9f * d : 0.5f * tau;    // small steps near the surface
        }
        if (fabsf(dmin) < tau) {
            // on the soft edge: the steps only bracket the nearest point, so close in on it (golden section), or the
            // edge would shift with where the steps happen to land, i.e. with how the body's field is put together
            const float g = 0.618034f, h = 0.5f * tau;
            float a = tmin - h, b = tmin + h;
            for (int k = 0; k < 8; k++) {
                float c = b - g * (b - a), e = a + g * (b - a);
                float dc = f.Distance(Vector3Add(lo, Vector3Scale(dir, c))), de = f.Distance(Vector3Add(lo, Vector3Scale(dir, e)));
                if (dc < dmin) { dmin = dc; tmin = c; }
                if (de < dmin) { dmin = de; tmin = e; }
                if (dc < de) b = e; else a = c;
            }
        }
        *at = Vector3Add(o, Vector3Scale(w.dir, tmin));
        if (hit && dmin < 0.0f && inside >= 0.0f) {
            // where it enters: between the last step outside and the first inside (bisection)
            float a = outside >= 0.0f ? outside : inside - 0.5f * tau, c = inside;
            for (int k = 0; k < 10; k++) {
                float m = 0.5f * (a + c);
                (f.Distance(Vector3Add(lo, Vector3Scale(dir, m))) < 0.0f ? c : a) = m;
            }
            *hit = Vector3Add(o, Vector3Scale(w.dir, c));
        }
        return dmin;
    }

    // ---- where the poses between are checked ---------------------------------------------------------------------------

    std::vector<float> CheckTimes(const Design& d, int n, float bias) {
        if (n <= 0) return {};
        // the hinges that move, and the timeline cut where any of them starts or stops
        std::vector<std::pair<float, float>> windows;
        std::vector<float> cuts = { 0.0f, 1.0f };
        for (const Hinge& h : d.hinges) {
            if (fabsf(h.end - h.start) < 1e-4f) continue;
            float a = std::clamp(h.t0, 0.0f, 1.0f), b = std::clamp(h.t1, 0.0f, 1.0f);
            if (b <= a) continue;
            windows.push_back({ a, b });
            cuts.push_back(a);
            cuts.push_back(b);
        }
        std::sort(cuts.begin(), cuts.end());
        cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
        // how many move in each stretch; the stretches' lengths where at least one does, and weighted by how many
        std::vector<int> moving(cuts.size() - 1, 0);
        double some = 0.0, weighted = 0.0;
        for (size_t k = 0; k + 1 < cuts.size(); k++) {
            float mid = 0.5f * (cuts[k] + cuts[k + 1]), len = cuts[k + 1] - cuts[k];
            for (auto [a, b] : windows) moving[k] += a <= mid && mid < b;
            if (moving[k] > 0) some += len;
            weighted += (double)len * moving[k];
        }
        if (some <= 0.0) return {};
        // each stretch's share of the checks: (1 - bias) evenly over the moving time, bias by how many move
        bias = std::clamp(bias, 0.0f, 1.0f);
        std::vector<double> mass(moving.size());
        for (size_t k = 0; k < moving.size(); k++) {
            double len = cuts[k + 1] - cuts[k];
            mass[k] = moving[k] > 0 ? (1.0 - bias) * len / some + bias * len * moving[k] / weighted : 0.0;
        }
        // the checks at evenly spaced quantiles of that distribution
        std::vector<float> out;
        size_t k = 0;
        double below = 0.0;                   // mass before stretch k
        for (int i = 0; i < n; i++) {
            double q = (i + 0.5) / n;
            while (k + 1 < mass.size() && below + mass[k] < q) below += mass[k++];
            double f = mass[k] > 0.0 ? std::clamp((q - below) / mass[k], 0.0, 1.0) : 0.5;
            out.push_back(cuts[k] + (float)f * (cuts[k + 1] - cuts[k]));
        }
        return out;
    }

    // ---- the loss -------------------------------------------------------------------------------------------------------

    AutoScore Loss::Evaluate(const Carving& c, const Placement& target, const std::vector<Param>& ps, std::vector<float>* grad) {
        if (grad) grad->assign(ps.size(), 0.0f);
        std::vector<int> bodyOf;
        Design d = c.Build(&bodyOf);
        Bodies b;
        for (size_t k = 0; k < c.pieces.size(); k++) {
            b.fields.push_back(BodyField::Body(d, bodyOf[k]));
            b.box.push_back(Grow(c.PieceBounds(c.pieces[k]), 0.002f * m_L));
            b.group.push_back(d.GroupIndex(bodyOf[k]));
        }
        Slopes slopes(b.fields.size());
        Shades shades(b.fields.size());
        AutoScore s = Score(d, b, target, &slopes, &shades);
        s.surface = SurfaceTerm(c, &s.surfaceCarved);
        s.total += m_P.surfaceWeight * s.surface;
        if (!grad || ps.empty() || !s.valid) return s;
        Gradient(c, target, bodyOf, b, slopes, shades, ps, *grad);

        // the placement: the targets drawn again with it nudged two units either way, against the design as drawn
        const float silhouette = std::max(0.0f, m_P.silhouetteWeight);
        for (size_t j = 0; j < ps.size(); j++) {
            if (ps[j].kind != Param::Target) continue;
            double loss[2] = { 0.0, 0.0 };
            float value[2];
            for (int side = 0; side < 2; side++) {
                Carving cc = c;
                Placement t = target;
                Set(cc, t, ps[j], Get(c, target, ps[j]) + (side == 0 ? 2.0f : -2.0f) * ps[j].unit, m_L);   // (2 units: the silhouettes wiggle finer)
                value[side] = Get(cc, t, ps[j]);
                if (silhouette > 0.0f) loss[side] += silhouette * FitLoss(TargetSilhouettes(t));
                if (m_Dino && m_SemGradTarget.size() == m_SemDrawn.size()) {
                    std::vector<std::vector<float>> pictures = SemanticTargets(t);
                    const size_t px = (size_t)m_SemRes * m_SemRes;
                    for (size_t v = 0; v < pictures.size(); v++)
                        for (size_t k = 0; k < px; k++) loss[side] += m_SemGradTarget[v * px + k] * pictures[v][k];
                }
            }
            float step = value[0] - value[1];
            if (step > 0.0f) (*grad)[j] = (float)((loss[0] - loss[1]) / step) * ps[j].unit;
        }
        return s;
    }

    AutoScore Loss::Evaluate(const Design& d, const Placement& target) {
        Bodies b;
        for (int g : d.Bodies()) {
            BodyField f = BodyField::Body(d, d.groups[g].id);
            if (f.Empty()) continue;
            b.box.push_back(Grow(d.BodyBounds(d.groups[g].id), 0.002f * m_L));
            b.fields.push_back(std::move(f));
            b.group.push_back(g);
        }
        return Score(d, b, target, nullptr, nullptr);
    }

    AutoScore Loss::Score(const Design& d, Bodies& b, const Placement& target, Slopes* slopesOut, Shades* shadesOut) {
        AutoScore s;
        s.target = target;
        s.target.centre = m_MeshCentre;
        s.centre = m_SphereCentre;
        s.radius = m_SphereRadius;
        if (target.offset.x != m_Placed.offset.x || target.offset.y != m_Placed.offset.y || target.offset.z != m_Placed.offset.z ||
            target.scale != m_Placed.scale) {
            m_Placed = target;
            m_Target = TargetSilhouettes(target);
            if (m_Dino) m_SemTarget = SemanticTargets(target);
        }
        if (b.fields.empty()) return s;
        b.fieldOf.assign(d.groups.size(), -1);
        for (size_t i = 0; i < b.group.size(); i++) b.fieldOf[b.group[i]] = (int)i;
        s.checkTimes = m_P.collisionWeight > 0.0f ? CheckTimes(d, m_P.motionChecks, m_P.motionCheckBias) : std::vector<float>{};
        b.times = { 0.0f };
        b.times.insert(b.times.end(), s.checkTimes.begin(), s.checkTimes.end());
        for (float t : b.times) b.xf.push_back(d.Transforms(t, true));

        Slopes slopes(b.fields.size());
        Shades shades(b.fields.size());
        Silhouettes(b, s, slopes);
        if (m_Dino) Semantics(b, s, slopes, shades);
        if (m_P.collisionWeight > 0.0f) Collisions(b, s, slopes);
        Hinges(d, b, s);
        s.total = std::max(0.0f, m_P.silhouetteWeight) * (s.fit + s.coveragePenalty) + (m_Dino ? m_P.semanticWeight * s.semantic : 0.0f) +
                  m_P.collisionWeight * s.collision + m_P.hingeWeight * s.hinges;
        s.valid = true;
        if (slopesOut) *slopesOut = std::move(slopes);
        if (shadesOut) *shadesOut = std::move(shades);
        return s;
    }

    // ---- silhouettes ------------------------------------------------------------------------------------------------------

    // The open pose drawn (each pixel keeps the piece covering it most, and where its ray came closest to it),
    // compared with the placed mesh, and the slopes of the soft edge pixels:
    //   d loss / d D: -1/sum(T) where the design covers less than the target, +1/sum(T) where it covers more, plus the
    //   coverage penalty's share; and d D / d distance = -1 / (2 tau) on the edge (0 < D < 1).
    void Loss::Silhouettes(const Bodies& b, AutoScore& s, Slopes& slopes) {
        const int nv = (int)m_Views.size(), px = m_Res * m_Res;
        std::vector<int> who((size_t)nv * px, -1);
        std::vector<Vector3> at((size_t)nv * px, { 0, 0, 0 });
        m_Cover.assign((size_t)nv * px, 0.0f);
        ParallelFor(nv * m_Res, [&](int task) {               // every row of every view
            int v = task / m_Res, y = task % m_Res;
            for (int x = 0; x < m_Res; x++) {
                size_t k = (size_t)v * px + (size_t)y * m_Res + x;
                for (size_t i = 0; i < b.fields.size(); i++) {
                    Vector3 p;
                    float dmin = Trace(b.fields[i], b.box[i], b.xf[0][b.group[i]], m_Views[v], FrameX(x, m_Res), FrameY(y, m_Res), m_Tau, &p);
                    float c = std::clamp(0.5f - dmin / (2.0f * m_Tau), 0.0f, 1.0f);
                    if (c > m_Cover[k]) {
                        m_Cover[k] = c;
                        who[k] = (int)i;
                        at[k] = p;
                    }
                }
            }
        });

        const float minCover = std::clamp(m_P.minViewCoverage, 0.0f, 1.0f), silhouette = std::max(0.0f, m_P.silhouetteWeight);
        const float weight = 1.0f / nv;
        // where the design shows outside the mesh: a spread of such pixels, as points of their pieces, closed
        m_Extra.clear();
        {
            std::vector<size_t> extra;
            for (int v = 0; v < nv; v++)
                for (int k = 0; k < px; k++) {
                    size_t kk = (size_t)v * px + k;
                    if (who[kk] >= 0 && m_Cover[kk] - m_Target[v][k] > 0.5f) extra.push_back(kk);
                }
            size_t stride = std::max<size_t>(1, extra.size() / 256);
            for (size_t e = 0; e < extra.size(); e += stride) {
                size_t kk = extra[e];
                int i = who[kk];
                const Rigid& xf = b.xf[0][b.group[i]];
                Quaternion inv = { -xf.rotation.x, -xf.rotation.y, -xf.rotation.z, xf.rotation.w };
                m_Extra.push_back({ i, xf.ApplyInverse(at[kk]), Vector3RotateByQuaternion(m_Views[kk / px].dir, inv) });
            }
        }
        s.worstView = 1.0f;
        double lossSum = 0.0, coverSum = 0.0, extraSum = 0.0;
        for (int v = 0; v < nv; v++) {
            const float* D = m_Cover.data() + (size_t)v * px;
            const std::vector<float>& T = m_Target[v];
            double tsum = 0.0, hit = 0.0, spill = 0.0;
            for (int k = 0; k < px; k++) {
                tsum += T[k];
                hit += fminf(T[k], D[k]);
                spill += fmaxf(0.0f, D[k] - T[k]);
            }
            if (tsum <= 0.0) continue;
            float cover = (float)(hit / tsum), extra = (float)(spill / tsum);
            lossSum += (1.0f - cover) + extra;
            coverSum += cover;
            extraSum += extra;
            bool shortOf = cover < minCover;
            s.worstView = fminf(s.worstView, cover);
            s.coveragePenalty += 10.0f * fmaxf(0.0f, minCover - cover);
            if (silhouette <= 0.0f) continue;
            for (int k = 0; k < px; k++) {
                size_t kk = (size_t)v * px + k;
                float c = D[k], t = T[k];
                int i = who[kk];
                if (i < 0 || c <= 0.0f || c >= 1.0f || c == t) continue;     // not on an edge, or no slope
                float dD = c < t ? (float)((-weight - (shortOf ? 10.0f : 0.0f)) / tsum) : (float)(weight / tsum);
                slopes[i].push_back({ 0, at[kk], silhouette * dD * (-1.0f / (2.0f * m_Tau)) });
            }
        }
        s.fit = (float)(lossSum / nv);
        s.viewCover = (float)(coverSum / nv);
        s.viewExtra = (float)(extraSum / nv);
    }

    // The silhouettes' part of the loss (fit + coverage penalty) for the last drawing, against other targets.
    float Loss::FitLoss(const std::vector<std::vector<float>>& target) const {
        const int nv = (int)m_Views.size(), px = m_Res * m_Res;
        const float minCover = std::clamp(m_P.minViewCoverage, 0.0f, 1.0f);
        if (m_Cover.size() != (size_t)nv * px) return 0.0f;
        double loss = 0.0;
        for (int v = 0; v < nv; v++) {
            const float* D = m_Cover.data() + (size_t)v * px;
            double tsum = 0.0, hit = 0.0, spill = 0.0;
            for (int k = 0; k < px; k++) {
                float t = target[v][k];
                tsum += t;
                hit += fminf(t, D[k]);
                spill += fmaxf(0.0f, D[k] - t);
            }
            if (tsum <= 0.0) continue;
            double cover = hit / tsum;
            loss += ((1.0 - cover) + spill / tsum) / nv + 10.0 * fmax(0.0, minCover - cover);
        }
        return (float)loss;
    }

    // ---- collisions -------------------------------------------------------------------------------------------------------

    // In every pose (open, and the checks between), every two pieces whose boxes meet are sampled where they meet (the
    // same unit points each time). A point inside both by dd (the larger, so nearer zero, of their two distances, both
    // negative) adds dd's depth times its share of the box's volume; its slope goes to the piece whose distance that is.
    void Loss::Collisions(const Bodies& b, AutoScore& s, Slopes& slopes) {
        const double sphereVolume = 4.0 / 3.0 * PI * pow(m_SphereRadius, 3.0), norm = sphereVolume * 0.02 * m_L;
        const int poses = (int)b.times.size(), between = poses - 1;
        struct Pair {
            int pose, a, c;
            BoundingBox meet;
            double depth = 0.0, volume = 0.0;
            std::vector<std::pair<int, Slope>> slopes;
        };
        std::vector<Pair> pairs;
        for (int pose = 0; pose < poses; pose++) {
            std::vector<BoundingBox> world(b.fields.size());
            for (size_t i = 0; i < b.fields.size(); i++) world[i] = TransformBox(b.box[i], b.xf[pose][b.group[i]]);
            for (int i = 0; i < (int)b.fields.size(); i++)
                for (int j = i + 1; j < (int)b.fields.size(); j++)
                    if (Overlap(world[i], world[j]))
                        pairs.push_back({ pose, i, j, { Vector3Max(world[i].min, world[j].min), Vector3Min(world[i].max, world[j].max) } });
        }
        ParallelFor((int)pairs.size(), [&](int k) {
            Pair& pr = pairs[k];
            // the pose's share of the collision term: the open pose counts fully, the poses between share one
            const float poseWeight = pr.pose == 0 ? 1.0f : 1.0f / between;
            const Rigid& xa = b.xf[pr.pose][b.group[pr.a]];
            const Rigid& xc = b.xf[pr.pose][b.group[pr.c]];
            Vector3 ext = Vector3Subtract(pr.meet.max, pr.meet.min);
            const double dv = (double)ext.x * ext.y * ext.z / m_PairUnit.size();
            for (Vector3 u : m_PairUnit) {
                Vector3 p = { pr.meet.min.x + u.x * ext.x, pr.meet.min.y + u.y * ext.y, pr.meet.min.z + u.z * ext.z };
                Vector3 qa = xa.ApplyInverse(p), qc = xc.ApplyInverse(p);
                if (!InBox(qa, b.box[pr.a]) || !InBox(qc, b.box[pr.c])) continue;      // outside a body's box: outside it
                float da = b.fields[pr.a].Distance(qa);
                if (da >= 0.0f) continue;
                float dc = b.fields[pr.c].Distance(qc);
                if (dc >= 0.0f) continue;
                float dd = fmaxf(da, dc);
                pr.depth += -dd * dv;
                pr.volume += dv;
                // d loss / d dd = -weight · poseWeight · dv / norm
                pr.slopes.push_back({ da > dc ? pr.a : pr.c, { pr.pose, p, (float)(-m_P.collisionWeight * poseWeight * dv / norm) } });
            }
        });
        std::vector<double> depth(poses, 0.0), volume(poses, 0.0);
        for (const Pair& pr : pairs) {
            depth[pr.pose] += pr.depth;
            volume[pr.pose] += pr.volume;
            for (const auto& [field, slope] : pr.slopes) slopes[field].push_back(slope);
        }
        double betweenSum = 0.0;
        for (int pose = 1; pose < poses; pose++) {
            betweenSum += depth[pose] / norm;
            float share = (float)(volume[pose] / sphereVolume);
            if (share > s.clashMotion) {
                s.clashMotion = share;
                s.clashMotionT = b.times[pose];
            }
        }
        s.collision = (float)(depth[0] / norm + (between > 0 ? betweenSum / between : 0.0));
        s.clashOpen = (float)(volume[0] / sphereVolume);
    }

    // ---- hinges ----------------------------------------------------------------------------------------------------------

    // Each pivot against the two pieces it joins, both in the design pose (closed: the frame pivots are given in).
    void Loss::Hinges(const Design& d, const Bodies& b, AutoScore& s) const {
        for (const Hinge& h : d.hinges) {
            int pg = d.GroupIndex(h.parent), cg = d.GroupIndex(h.child);
            int pi = pg >= 0 ? b.fieldOf[pg] : -1, ci = cg >= 0 ? b.fieldOf[cg] : -1;
            if (pi < 0 || ci < 0) continue;
            s.hinges += (PivotGap(b.fields[pi], h.pivot) + PivotGap(b.fields[ci], h.pivot)) / m_L;
            s.hingeGap = fmaxf(s.hingeGap, fmaxf(b.fields[pi].Distance(h.pivot), b.fields[ci].Distance(h.pivot)));
        }
    }

    // ---- the surface --------------------------------------------------------------------------------------------------------

    // Points just under the sphere's surface, each against the hollows (the union of them): how deep inside one it is,
    // d, counts d / (d + 1% of the mesh's size): nothing until a hollow reaches it, nearly 1 once well inside.
    float Loss::SurfaceTerm(const Carving& c, float* carved) const {
        const float scale = 0.01f * m_L, under = c.radius * 0.997f;
        double sum = 0.0;
        int in = 0;
        bool any = false;
        for (const Cutter& k : c.cutters) any |= k.hollow;
        if (any)
            for (Vector3 dir : m_SurfaceDirs) {
                Vector3 p = Vector3Add(c.centre, Vector3Scale(dir, under));
                float d = FLT_MAX;
                for (const Cutter& k : c.cutters)
                    if (k.hollow) d = fminf(d, k.sf.Distance(p));
                if (d >= 0.0f) continue;
                in++;
                sum += -d / (-d + scale);
            }
        if (carved) *carved = (float)in / m_SurfaceDirs.size();
        return (float)(sum / m_SurfaceDirs.size());
    }

    // ---- the gradient -------------------------------------------------------------------------------------------------------

    // d loss / d parameter = sum over the slopes of (slope · d distance / d parameter). To first order a distance only
    // changes through the piece's field and pose at its fixed world point (for a silhouette's ray, where it comes
    // closest: the envelope theorem), so d distance / d parameter comes from central differences there: the parameter
    // nudged either way, the carving built again, the fields of the pieces a cutter shapes (a cutter's parameter) or
    // the poses (a joint's) rebuilt, the distance measured again. The hinge term is measured again directly, for the
    // joints the parameter touches. The poses' times stay as they were (the checks are placed by the design, not part
    // of what is moved).
    void Loss::Gradient(const Carving& c, const Placement& target, const std::vector<int>& bodyOf, const Bodies& b, const Slopes& slopes,
                        const Shades& shades, const std::vector<Param>& ps, std::vector<float>& grad) {
        ParallelFor((int)ps.size(), [&](int j) {
            const Param& p = ps[j];
            if (p.kind == Param::Target) return;            // the placement's: see Evaluate
            bool ok = false;
            float base = Get(c, target, p, &ok);
            if (!ok) return;
            double sum[2] = { 0.0, 0.0 };
            float value[2] = { 0.0f, 0.0f };
            for (int side = 0; side < 2; side++) {
                Carving e = c;
                Placement t = target;
                Set(e, t, p, base + (side == 0 ? 0.1f : -0.1f) * p.unit, m_L);
                value[side] = Get(e, t, p);
                Tidy(e);
                Design de = e.Build();
                std::vector<std::vector<Rigid>> xf;
                for (float time : b.times) xf.push_back(de.Transforms(time, true));
                // a cutter's parameter changes the fields of the pieces it shapes
                std::vector<int> changedOf(b.fields.size(), -1);
                std::vector<BodyField> changed;
                if (p.kind == Param::Cutter)
                    for (int k : e.PiecesUsing(p.id)) {
                        changedOf[k] = (int)changed.size();
                        changed.push_back(BodyField::Body(de, bodyOf[k]));
                    }
                auto field = [&](size_t i) -> const BodyField& { return changedOf[i] >= 0 ? changed[changedOf[i]] : b.fields[i]; };
                for (size_t i = 0; i < b.fields.size(); i++) {
                    for (const Slope& sl : slopes[i]) {
                        const Rigid& before = b.xf[sl.pose][b.group[i]];
                        const Rigid& after = xf[sl.pose][b.group[i]];
                        bool poseMoves = memcmp(&before, &after, sizeof(Rigid)) != 0;
                        if (changedOf[i] < 0 && !poseMoves) continue;   // this parameter doesn't touch it
                        sum[side] += sl.weight * field(i).Distance(after.ApplyInverse(sl.at));
                    }
                    // the semantic pixels' shades: the surface point slid along its ray, and shaded again
                    for (const Shade& sh : shades[i]) {
                        const Rigid& before = b.xf[sh.pose][b.group[i]];
                        const Rigid& after = xf[sh.pose][b.group[i]];
                        bool poseMoves = memcmp(&before, &after, sizeof(Rigid)) != 0;
                        if (changedOf[i] < 0 && !poseMoves) continue;
                        sum[side] += sh.weight * ShadeAt(field(i), after, sh);
                    }
                }
                // the surface term, for a hollow's parameter
                if (p.kind == Param::Cutter && m_P.surfaceWeight > 0.0f) {
                    int k = e.CutterIndex(p.id);
                    if (k >= 0 && e.cutters[k].hollow) sum[side] += m_P.surfaceWeight * SurfaceTerm(e);
                }
                // the hinge term, for the joints this parameter touches: a pivot it moves, or a piece whose field it changes
                if (m_P.hingeWeight > 0.0f)
                    for (const Hinge& h : de.hinges) {
                        int pg = de.GroupIndex(h.parent), cg = de.GroupIndex(h.child);
                        int pi = pg >= 0 ? b.fieldOf[pg] : -1, ci = cg >= 0 ? b.fieldOf[cg] : -1;
                        if (pi < 0 || ci < 0) continue;
                        bool pivotMoves = p.kind == Param::Joint && h.name == "Joint " + std::to_string(p.id) && p.field < 3;
                        if (!pivotMoves && changedOf[pi] < 0 && changedOf[ci] < 0) continue;
                        sum[side] += m_P.hingeWeight * (PivotGap(field(pi), h.pivot) + PivotGap(field(ci), h.pivot)) / m_L;
                    }
            }
            float step = value[0] - value[1];
            if (step > 0.0f) grad[j] = (float)((sum[0] - sum[1]) / step) * p.unit;    // per unit of the parameter
        });
    }

    // ---- the picture for the UI ---------------------------------------------------------------------------------------

    // Dark where both cover a pixel, red where only the placed mesh does (missing), blue where only the design does (extra).
    void Loss::Picture(std::vector<unsigned char>* rgba, int* w, int* h) const {
        const int nv = (int)m_Views.size(), px = m_Res * m_Res, gap = 3;
        if (m_Cover.size() != (size_t)nv * px) { *w = *h = 0; rgba->clear(); return; }
        *w = nv * m_Res + (nv + 1) * gap;
        *h = m_Res + 2 * gap;
        rgba->assign((size_t)*w * *h * 4, 255);
        for (size_t k = 0; k < rgba->size(); k += 4) { (*rgba)[k] = 222; (*rgba)[k + 1] = 225; (*rgba)[k + 2] = 232; }
        const float both[3] = { 70, 74, 86 }, miss[3] = { 235, 110, 95 }, extra[3] = { 90, 140, 235 };
        for (int v = 0; v < nv; v++)
            for (int k = 0; k < px; k++) {
                int x = k % m_Res, y = k / m_Res;
                float t = m_Target[v][k], dd = m_Cover[(size_t)v * px + k];
                float ov = fminf(t, dd), ms = t - ov, ex = dd - ov, rest = 1.0f - fmaxf(t, dd);
                unsigned char* out = &(*rgba)[((size_t)(gap + y) * *w + gap + v * (m_Res + gap) + x) * 4];
                for (int c = 0; c < 3; c++) out[c] = (unsigned char)std::clamp(lroundf(ov * both[c] + ms * miss[c] + ex * extra[c] + rest * 255.0f), 0L, 255L);
            }
    }
}
