#include "loss.hpp"
#include "dino.hpp"
#include "util/box.hpp"
#include "data/clash.hpp"
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
        m_Hinge = PlanHinge(m_SphereRadius, p.hinges);
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
        Vector3 o = RayStart(box, w, sx, sy);
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

    // ---- the loss -------------------------------------------------------------------------------------------------------

    AutoScore Loss::Evaluate(const Carving& c, const Placement& target, const std::vector<Param>& ps, std::vector<float>* grad, float phase) {
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
        AutoScore s = Score(d, b, target, &slopes, &shades, phase);
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

    AutoScore Loss::Evaluate(const Design& d, const Placement& target, float phase) {
        Bodies b;
        for (int g : d.Bodies()) {
            BodyField f = BodyField::Body(d, d.groups[g].id);
            if (f.Empty()) continue;
            b.box.push_back(Grow(d.BodyBounds(d.groups[g].id), 0.002f * m_L));
            b.fields.push_back(std::move(f));
            b.group.push_back(g);
        }
        return Score(d, b, target, nullptr, nullptr, phase);
    }

    AutoScore Loss::Score(const Design& d, Bodies& b, const Placement& target, Slopes* slopesOut, Shades* shadesOut, float phase) {
        AutoScore s;
        s.target = target;
        s.target.centre = m_MeshCentre;
        s.centre = m_SphereCentre;
        s.radius = m_SphereRadius;
        s.meshSize = m_L;
        if (target.offset.x != m_Placed.offset.x || target.offset.y != m_Placed.offset.y || target.offset.z != m_Placed.offset.z ||
            target.scale != m_Placed.scale) {
            m_Placed = target;
            m_Target = TargetSilhouettes(target);
            if (m_Dino) m_SemTarget = SemanticTargets(target);
        }
        if (b.fields.empty()) return s;
        b.fieldOf.assign(d.groups.size(), -1);
        for (size_t i = 0; i < b.group.size(); i++) b.fieldOf[b.group[i]] = (int)i;
        b.times = { 0.0f };
        b.hinge = { -1 };
        if (m_P.collisionWeight > 0.0f)
            for (const MotionCheck& c : HingeChecks(d, m_P.hingeChecks, phase)) {
                b.times.push_back(c.t);
                b.hinge.push_back(c.hinge);
                s.checkTimes.push_back(c.t);
            }
        if (m_P.dependencies) {                          // closed: where holding is measured
            b.times.push_back(1.0f);
            b.hinge.push_back(-2);
        }
        for (float t : b.times) b.xf.push_back(d.Transforms(t, true));

        Slopes slopes(b.fields.size());
        Shades shades(b.fields.size());
        Silhouettes(b, s, slopes);
        if (m_Dino) Semantics(b, s, slopes, shades);
        if (m_P.collisionWeight > 0.0f) Collisions(d, b, s, slopes, phase);
        Hinges(d, b, s);
        if (m_P.dependencies) Dependencies(d, b, s, slopes);
        s.total = std::max(0.0f, m_P.silhouetteWeight) * (s.fit + s.coveragePenalty) + (m_Dino ? m_P.semanticWeight * s.semantic : 0.0f) +
                  CollisionWeight() * s.collision + HingeWeight() * s.hinges + (m_P.dependencies ? m_P.dependencyWeight * s.dependency : 0.0f);
        s.collisionScale = m_CollisionScale;
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
    Vector3 Loss::RayStart(const BoundingBox& box, const View& w, float sx, float sy) const {
        const float far = 4.0f * m_Half + Vector3Distance(box.min, box.max);
        return Vector3Add(Vector3Subtract(m_Centre, Vector3Scale(w.dir, far)), Vector3Add(Vector3Scale(w.u, sx), Vector3Scale(w.v, sy)));
    }

    void Loss::Draw(const Bodies& b, std::vector<int>& who, std::vector<Vector3>& at, bool* traced) {
        const int nv = (int)m_Views.size(), px = m_Res * m_Res;
        who.assign((size_t)nv * px, -1);
        at.assign((size_t)nv * px, { 0, 0, 0 });
        m_Cover.assign((size_t)nv * px, 0.0f);
        *traced = m_P.gpu && DrawTraced(b, who, at);
        if (*traced) return;
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
    }

    // The same drawing by the tracer: every piece's field as GLSL, marched there the way Trace marches it here.
    bool Loss::DrawTraced(const Bodies& b, std::vector<int>& who, std::vector<Vector3>& at) {
        const int nv = (int)m_Views.size(), px = m_Res * m_Res;
        TraceRequest rq;
        rq.centre = m_Centre;
        rq.half = m_Half;
        rq.res = m_Res;
        rq.tau = m_Tau;
        for (const View& w : m_Views) rq.views.push_back({ w.dir, w.u, w.v });
        for (size_t i = 0; i < b.fields.size(); i++) {
            TracePiece piece;
            piece.code = b.fields[i].Glsl(&piece.params);
            piece.box = b.box[i];
            piece.xf = b.xf[0][b.group[i]];
            rq.pieces.push_back(std::move(piece));
        }
        TraceResult r;
        if (!m_P.gpu->Trace(rq, &r)) return false;
        if (r.piece.size() != (size_t)nv * px || r.nearest.size() != r.piece.size() || r.along.size() != r.piece.size()) return false;
        for (int v = 0; v < nv; v++)
            for (int k = 0; k < px; k++) {
                size_t kk = (size_t)v * px + k;
                int i = r.piece[kk];
                if (i < 0 || i >= (int)b.fields.size()) continue;
                float c = std::clamp(0.5f - r.nearest[kk] / (2.0f * m_Tau), 0.0f, 1.0f);
                if (c <= 0.0f) continue;
                m_Cover[kk] = c;
                who[kk] = i;
                at[kk] = Vector3Add(RayStart(b.box[i], m_Views[v], FrameX(k % m_Res, m_Res), FrameY(k / m_Res, m_Res)),
                                    Vector3Scale(m_Views[v].dir, r.along[kk]));
            }
        return true;
    }

    void Loss::Silhouettes(const Bodies& b, AutoScore& s, Slopes& slopes) {
        const int nv = (int)m_Views.size(), px = m_Res * m_Res;
        std::vector<int> who;
        std::vector<Vector3> at;
        Draw(b, who, at, &s.traced);

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

    namespace {
        float Frac(float x) { return x - floorf(x); }
        bool Turns(const Hinge& h) { return fabsf(h.end - h.start) > 1e-4f && h.t1 - h.t0 > 1e-5f; }
    }

    // A point inside two pieces by `depth` costs its depth, and up to 2% of the mesh's size more, reached at a quarter
    // of that depth: a shallow overlap already costs five times its depth (so it isn't next to free, as depth alone
    // would have it: its integral over a thin overlap goes with the square of the thickness), a deep one its volume too.
    float Loss::ClashCost(float depth, float* slope) const {
        const float d0 = 0.02f * m_L, w = 0.25f * d0;
        if (slope) *slope = 1.0f + (depth < w ? d0 / w : 0.0f);
        return depth + d0 * fminf(depth / w, 1.0f);
    }

    // In the open pose, every two pieces; in a pose of a hinge's turn, every two that move against each other there
    // (the others stand as they did at the end of an earlier turn, or closed, side by side). Each pair is sampled on a
    // grid where the two can meet (SampleOverlap: cells their surfaces pass through are sampled finer), shifted with
    // the phase. A sample's slope goes to the piece whose surface is the nearer one (whose distance is its depth).
    void Loss::Collisions(const Design& d, const Bodies& b, AutoScore& s, Slopes& slopes, float phase) {
        const double sphereVolume = 4.0 / 3.0 * PI * pow(m_SphereRadius, 3.0), norm = sphereVolume * 0.02 * m_L;
        const int poses = (int)b.times.size(), perHinge = std::max(1, m_P.hingeChecks);
        const float cell = m_SphereRadius / 5.0f;
        const Vector3 jitter = { Frac(0.5f + (phase - 0.5f) * 7.0f), Frac(0.5f + (phase - 0.5f) * 11.0f), Frac(0.5f + (phase - 0.5f) * 13.0f) };
        const std::vector<std::vector<char>> under = HingeSubtrees(d);
        struct Pair {
            int pose, a, c;
            double cost = 0.0, volume = 0.0;
            std::vector<std::pair<int, Slope>> slopes;
        };
        std::vector<Pair> pairs;
        for (int pose = 0; pose < poses; pose++) {
            if (b.hinge[pose] == -2) continue;
            std::vector<BoundingBox> world(b.fields.size());
            for (size_t i = 0; i < b.fields.size(); i++) world[i] = TransformBox(b.box[i], b.xf[pose][b.group[i]]);
            for (int i = 0; i < (int)b.fields.size(); i++)
                for (int j = i + 1; j < (int)b.fields.size(); j++) {
                    if (b.hinge[pose] >= 0 && !MovingApart(d, under, b.group[i], b.group[j], b.times[pose])) continue;
                    if (Overlap(world[i], world[j])) pairs.push_back({ pose, i, j });
                }
        }
        ParallelFor((int)pairs.size(), [&](int k) {
            Pair& pr = pairs[k];
            // the pose's share of the collision term: the open pose counts fully, a hinge's poses share one
            const float poseWeight = pr.pose == 0 ? 1.0f : 1.0f / perHinge;
            Clash clash;
            SampleOverlap({ &b.fields[pr.a], b.box[pr.a], b.xf[pr.pose][b.group[pr.a]] }, { &b.fields[pr.c], b.box[pr.c], b.xf[pr.pose][b.group[pr.c]] },
                          cell, jitter, &clash, true);
            // next to a hinge that will be built under the surface, its two pieces are in each other's way as they turn,
            // whatever their shapes: AddHinges cuts that room (HingeSizes::relief), so it isn't a collision here
            const Hinge* joint = nullptr;
            if (m_P.buildHinges && m_Hinge.relief > 0.0f)
                for (const Hinge& h : d.hinges) {
                    const int ia = d.groups[b.group[pr.a]].id, ic = d.groups[b.group[pr.c]].id;
                    if ((h.parent == ia && h.child == ic) || (h.child == ia && h.parent == ic)) joint = &h;
                }
            pr.volume = joint ? 0.0 : clash.volume;
            for (const OverlapSample& sm : clash.samples) {
                if (joint) {
                    const Vector3 a = Vector3Normalize(joint->axis), q = Vector3Subtract(b.xf[pr.pose][b.group[pr.a]].ApplyInverse(sm.at), joint->pivot);
                    if (Vector3Length(Vector3Subtract(q, Vector3Scale(a, Vector3DotProduct(q, a)))) < m_Hinge.relief) continue;
                    pr.volume += sm.volume;
                }
                float slope;
                pr.cost += (double)ClashCost(sm.depth, &slope) * sm.volume;
                // d loss / d distance = -weight · poseWeight · slope · volume / norm (the distance is minus the depth)
                pr.slopes.push_back({ sm.first ? pr.a : pr.c, { pr.pose, sm.at, (float)(-CollisionWeight() * poseWeight * slope * sm.volume / norm) } });
            }
        });
        std::vector<double> cost(poses, 0.0), volume(poses, 0.0);
        for (const Pair& pr : pairs) {
            cost[pr.pose] += pr.cost;
            volume[pr.pose] += pr.volume;
            for (const auto& [field, slope] : pr.slopes) slopes[field].push_back(slope);
        }
        double turns = 0.0;
        for (int pose = 1; pose < poses; pose++) {
            if (b.hinge[pose] < 0) continue;
            turns += cost[pose] / norm / perHinge;
            float share = (float)(volume[pose] / sphereVolume);
            if (share > s.clashMotion) {
                s.clashMotion = share;
                s.clashMotionT = b.times[pose];
            }
        }
        s.collision = (float)(cost[0] / norm + turns);
        s.clashOpen = (float)(volume[0] / sphereVolume);
    }

    // ---- dependencies ----------------------------------------------------------------------------------------------------

    float Loss::SurfaceSpacing() const { return sqrtf(4.0f * PI / m_SurfaceDirs.size()) * m_SphereRadius; }

    std::vector<int> Loss::SurfaceOwners(const Bodies& b) const {
        std::vector<int> owner(m_SurfaceDirs.size(), -1);
        ParallelFor((int)owner.size(), [&](int k) {
            Vector3 p = Vector3Add(m_SphereCentre, Vector3Scale(m_SurfaceDirs[k], 0.985f * m_SphereRadius));
            for (size_t i = 0; i < b.fields.size(); i++)
                if (InBox(p, b.box[i]) && b.fields[i].Distance(p) < 0.0f) {
                    owner[k] = (int)i;
                    break;
                }
        });
        return owner;
    }

    // A piece can hold hinge h shut where it lies over a piece the hinge turns: it must not turn with the hinge itself,
    // and a hinge that opens earlier must lie between it and the hinge's parent (so that it has moved off by the
    // hinge's turn; the parent itself never does). Held points are the turned pieces' surface where it would lift off
    // when the hinge opens, away from the axis (at least 0.4 of the farthest's lever: a hold near the axis holds little).
    float Loss::HoldGap(const Design& d, const Bodies& b, const std::vector<std::vector<char>>& under, const std::vector<int>& owner, int h,
                        const std::vector<int>& earlier, int* from, int* to) const {
        const Hinge& hg = d.hinges[h];
        const int pg = d.GroupIndex(hg.parent);
        if (pg < 0) return FLT_MAX;
        std::vector<char> turned(b.fields.size(), 0), holder(b.fields.size(), 0);
        for (size_t i = 0; i < b.fields.size(); i++) {
            int g = b.group[i];
            turned[i] = under[h][g];
            if (turned[i]) continue;
            for (int e : earlier) holder[i] |= under[e][g] != under[e][pg];
        }
        const Vector3 axis = Vector3Length(hg.axis) > 1e-6f ? Vector3Normalize(hg.axis) : Vector3{ 1, 0, 0 };
        const float turn = hg.start >= hg.end ? 1.0f : -1.0f;      // opening turns from the closed angle to the open one
        std::vector<int> held, holders;
        std::vector<float> lever;
        float farthest = 0.0f;
        for (size_t k = 0; k < owner.size(); k++) {
            if (owner[k] < 0) continue;
            if (holder[owner[k]]) holders.push_back((int)k);
            if (!turned[owner[k]]) continue;
            Vector3 p = Vector3Add(m_SphereCentre, Vector3Scale(m_SurfaceDirs[k], m_SphereRadius));
            Vector3 r = Vector3Subtract(p, hg.pivot);
            Vector3 v = Vector3Scale(Vector3CrossProduct(axis, r), turn);
            float l = Vector3Length(v);                       // (its speed is its lever)
            if (Vector3DotProduct(v, m_SurfaceDirs[k]) < 0.2f * l) continue;   // it doesn't lift off there
            held.push_back((int)k);
            lever.push_back(l);
            farthest = fmaxf(farthest, l);
        }
        float best = FLT_MAX;
        for (size_t i = 0; i < held.size(); i++) {
            if (lever[i] < 0.4f * farthest) continue;
            Vector3 a = m_SurfaceDirs[held[i]];
            for (int k : holders) {
                float dist = Vector3Distance(a, m_SurfaceDirs[k]);
                if (dist < best) {
                    best = dist;
                    if (from) *from = held[i];
                    if (to) *to = k;
                }
            }
        }
        if (best == FLT_MAX) return FLT_MAX;
        // neighbours on the surface are a spacing apart: touching pieces are no gap
        return fmaxf(0.0f, best * m_SphereRadius - 1.5f * SurfaceSpacing());
    }

    // Per hinge after the first to open: how far the nearest holder is, over the sphere's radius (1 at most, and when
    // there is none). Its gradient is a stand-in: along the surface between the two nearest points, whatever lies
    // outside both pieces pulls the nearer of them toward it.
    void Loss::Dependencies(const Design& d, const Bodies& b, AutoScore& s, Slopes& slopes) {
        std::vector<int> order;
        for (int h : OpeningOrder(d))
            if (Turns(d.hinges[h])) order.push_back(h);
        if (order.size() < 2) return;
        s.dependents = (int)order.size() - 1;
        int closed = -1;
        for (size_t k = 0; k < b.hinge.size(); k++)
            if (b.hinge[k] == -2) closed = (int)k;
        const std::vector<std::vector<char>> under = HingeSubtrees(d);
        const std::vector<int> owner = SurfaceOwners(b);
        const float R = m_SphereRadius, reach = m_P.ledges.reach > 0.0f ? m_P.ledges.reach : 0.35f * R;
        std::vector<int> earlier;
        for (size_t r = 0; r < order.size(); r++) {
            if (r > 0) {
                int from = -1, to = -1;
                float gap = HoldGap(d, b, under, owner, order[r], earlier, &from, &to);
                s.dependency += (gap == FLT_MAX ? 1.0f : fminf(1.0f, gap / R)) / s.dependents;
                if (gap <= reach) s.holdable++;
                if (gap != FLT_MAX && gap > 0.0f && closed >= 0) {
                    const int steps = 6;
                    const float weight = m_P.dependencyWeight / (steps * R * s.dependents);
                    for (int k = 0; k < steps; k++) {
                        Vector3 dir = Vector3Normalize(Vector3Lerp(m_SurfaceDirs[from], m_SurfaceDirs[to], (k + 0.5f) / steps));
                        Vector3 p = Vector3Add(m_SphereCentre, Vector3Scale(dir, 0.985f * R));
                        float dt = b.fields[owner[from]].Distance(p), dh = b.fields[owner[to]].Distance(p);
                        if (dt <= 0.0f || dh <= 0.0f) continue;
                        slopes[dt < dh ? owner[from] : owner[to]].push_back({ closed, p, weight });
                    }
                }
            }
            earlier.push_back(order[r]);
        }
    }

    // ---- the opening order -----------------------------------------------------------------------------------------------

    std::vector<int> Loss::BestOrder(const Carving& c) const {
        std::vector<int> current = c.Order(), moving, still;
        for (int id : current) (fabsf(c.joints[c.JointIndex(id)].open) > 1e-3f ? moving : still).push_back(id);
        if (moving.size() < 2 || moving.size() > 24) return current;
        std::vector<int> bodyOf;
        Design d = c.Build(&bodyOf);
        Bodies b;
        for (size_t k = 0; k < c.pieces.size(); k++) {
            b.fields.push_back(BodyField::Body(d, bodyOf[k]));
            b.box.push_back(Grow(c.PieceBounds(c.pieces[k]), 0.002f * m_L));
            b.group.push_back(d.GroupIndex(bodyOf[k]));
        }
        auto hingeOf = [&](int joint) {
            std::string name = "Joint " + std::to_string(joint);
            for (size_t h = 0; h < d.hinges.size(); h++)
                if (d.hinges[h].name == name) return (int)h;
            return -1;
        };
        const std::vector<std::vector<char>> under = HingeSubtrees(d);
        const std::vector<int> owner = m_P.dependencies ? SurfaceOwners(b) : std::vector<int>{};
        const double norm = 4.0 / 3.0 * PI * pow(m_SphereRadius, 3.0) * 0.02 * m_L;
        const float cell = m_SphereRadius / 4.0f, d0 = 0.02f * m_L;
        const int checks = 3;
        std::vector<int> opened, openedHinges;
        std::vector<float> angles(d.hinges.size());
        for (size_t h = 0; h < d.hinges.size(); h++) angles[h] = d.hinges[h].end;
        while (!moving.empty()) {
            std::vector<double> cost(moving.size(), 0.0);
            ParallelFor((int)moving.size(), [&](int k) {
                int h = hingeOf(moving[k]);
                if (h < 0) return;
                const Hinge& hg = d.hinges[h];
                double sum = 0.0;
                for (int q = 0; q < checks; q++) {
                    std::vector<float> at = angles;
                    at[h] = hg.end + (hg.start - hg.end) * (q + 0.5f) / checks;
                    std::vector<Rigid> xf = d.TransformsAt(at);
                    Clash clash;
                    for (size_t i = 0; i < b.fields.size(); i++)
                        for (size_t j = i + 1; j < b.fields.size(); j++)
                            if (under[h][b.group[i]] != under[h][b.group[j]])
                                SampleOverlap({ &b.fields[i], b.box[i], xf[b.group[i]] }, { &b.fields[j], b.box[j], xf[b.group[j]] }, cell,
                                              { 0.5f, 0.5f, 0.5f }, &clash, false);
                    sum += clash.depth + d0 * clash.volume;
                }
                cost[k] = sum / norm / checks;
                if (m_P.dependencies && !opened.empty()) {
                    float gap = HoldGap(d, b, under, owner, h, openedHinges);
                    cost[k] += 0.02 * (gap == FLT_MAX ? 1.0 : std::min(1.0, (double)gap / m_SphereRadius));
                }
            });
            size_t pick = 0;                                   // (ties: as they stand)
            for (size_t k = 1; k < moving.size(); k++)
                if (cost[k] < cost[pick] - 1e-6) pick = k;
            int h = hingeOf(moving[pick]);
            if (h >= 0) {
                angles[h] = d.hinges[h].start;
                openedHinges.push_back(h);
            }
            opened.push_back(moving[pick]);
            moving.erase(moving.begin() + pick);
        }
        opened.insert(opened.end(), still.begin(), still.end());
        return opened;
    }

    // ---- hinges ----------------------------------------------------------------------------------------------------------

    // Each pivot against the two pieces it joins, both in the design pose (closed: the frame pivots are given in).
    void Loss::Hinges(const Design& d, const Bodies& b, AutoScore& s) const {
        for (const Hinge& h : d.hinges) {
            int pg = d.GroupIndex(h.parent), cg = d.GroupIndex(h.child);
            int pi = pg >= 0 ? b.fieldOf[pg] : -1, ci = cg >= 0 ? b.fieldOf[cg] : -1;
            if (pi < 0 || ci < 0) continue;
            float farthest = 0.0f;
            s.hinges += HingeCost(h, b.fields[pi], b.fields[ci], &farthest);
            s.hingeGap = fmaxf(s.hingeGap, farthest);
        }
    }

    float Loss::HingeCost(const Hinge& h, const BodyField& parent, const BodyField& child, float* farthest) const {
        const Vector3 a = Vector3Normalize(h.axis);
        const float half = 0.5f * m_Hinge.length, slack = HingeSlack();
        float sum = 0.0f;
        for (float w : { -half, 0.0f, half }) {
            Vector3 q = Vector3Add(h.pivot, Vector3Scale(a, w));
            float dp = parent.Distance(q), dc = child.Distance(q);
            sum += (fmaxf(0.0f, dp - slack) + fmaxf(0.0f, dc - slack)) / 3.0f;
            if (farthest) *farthest = fmaxf(*farthest, fmaxf(dp, dc));
        }
        if (m_P.buildHinges && m_P.hinges.flush) sum += fmaxf(0.0f, Vector3Distance(h.pivot, m_SphereCentre) - m_Hinge.pivotRadius);
        return sum / m_L;
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
                        bool pivotMoves = p.kind == Param::Joint && h.name == "Joint " + std::to_string(p.id) && p.field < 6;   // pivot or axis
                        if (!pivotMoves && changedOf[pi] < 0 && changedOf[ci] < 0) continue;
                        sum[side] += HingeWeight() * HingeCost(h, field(pi), field(ci));
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
