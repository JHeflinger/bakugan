#include "loss.hpp"
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

    // ---- setting up: views, the mesh's silhouettes, the sphere --------------------------------------------------------

    Loss::Loss(const TriMesh& mesh, const AutoParams& p) : m_P(p) {
        Vector3 lo = { FLT_MAX, FLT_MAX, FLT_MAX }, hi = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
        for (unsigned i : mesh.indices) { lo = Vector3Min(lo, mesh.positions[i]); hi = Vector3Max(hi, mesh.positions[i]); }
        m_L = fmaxf(Vector3Distance(lo, hi), 1e-3f);
        m_Centre = Vector3Scale(Vector3Add(lo, hi), 0.5f);
        m_Half = 0.6f * m_L;                     // the mesh, and anything near it, stays in the frame
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

        // the mesh's silhouette in each view: triangles filled at 4x4 samples a pixel
        const int ss = 4, big = m_Res * ss;
        m_Mesh.assign(n, std::vector<float>((size_t)m_Res * m_Res, 0.0f));
        ParallelFor(n, [&](int v) {
            std::vector<unsigned char> hit((size_t)big * big, 0);
            std::vector<Vector2> pts(mesh.positions.size());
            for (size_t i = 0; i < pts.size(); i++) pts[i] = Vector2Scale(ToPixel(m_Views[v], mesh.positions[i]), (float)ss);
            for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) FillTriangle(pts[mesh.indices[t]], pts[mesh.indices[t + 1]], pts[mesh.indices[t + 2]], big, hit);
            for (int y = 0; y < big; y++)
                for (int x = 0; x < big; x++)
                    if (hit[(size_t)y * big + x]) m_Mesh[v][(size_t)(y / ss) * m_Res + x / ss] += 1.0f / (ss * ss);
        });

        // the sphere: at the mesh's centre of volume, with its volume (signed tetrahedra from the box's centre)
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
        m_SphereCentre = closed ? Vector3Add(m_Centre, Vector3Scale(moment, (float)(1.0 / volume))) : m_Centre;
        m_SphereRadius = p.sphereDiameter > 0.0f ? 0.5f * p.sphereDiameter : (closed ? (float)cbrt(3.0 * fabs(volume) / (4.0 * PI)) : 0.25f * m_L);

        // points on the surface (area weighted), where the grammar puts new shapes; and the collisions' sample points
        std::mt19937 rng(p.seed * 31u + 7u);
        std::vector<double> cdf;
        double area = 0.0;
        for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
            Vector3 a = mesh.positions[mesh.indices[t]], b = mesh.positions[mesh.indices[t + 1]], c = mesh.positions[mesh.indices[t + 2]];
            area += 0.5 * Vector3Length(Vector3CrossProduct(Vector3Subtract(b, a), Vector3Subtract(c, a)));
            cdf.push_back(area);
        }
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);
        for (int i = 0; i < 2000 && area > 0.0; i++) {
            size_t t = std::min((size_t)(std::lower_bound(cdf.begin(), cdf.end(), u01(rng) * area) - cdf.begin()), cdf.size() - 1);
            float u = u01(rng), v = u01(rng);
            if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
            Vector3 a = mesh.positions[mesh.indices[t * 3]], b = mesh.positions[mesh.indices[t * 3 + 1]], c = mesh.positions[mesh.indices[t * 3 + 2]];
            m_Surface.push_back(Vector3Add(a, Vector3Add(Vector3Scale(Vector3Subtract(b, a), u), Vector3Scale(Vector3Subtract(c, a), v))));
        }
        for (int i = 0; i < 128; i++) m_PairUnit.push_back({ u01(rng), u01(rng), u01(rng) });
    }

    Vector2 Loss::ToPixel(const View& w, Vector3 p) const {
        Vector3 q = Vector3Subtract(p, m_Centre);
        return { (Vector3DotProduct(q, w.u) / m_Half + 1.0f) * 0.5f * m_Res, (1.0f - Vector3DotProduct(q, w.v) / m_Half) * 0.5f * m_Res };
    }

    float Loss::Disc(const View& w, int x, int y) const {
        Vector2 c = ToPixel(w, m_SphereCentre);
        float dx = x + 0.5f - c.x, dy = y + 0.5f - c.y;
        return std::clamp(m_SphereRadius / (2.0f * m_Half / m_Res) - sqrtf(dx * dx + dy * dy) + 0.5f, 0.0f, 1.0f);
    }

    // ---- drawing the design ---------------------------------------------------------------------------------------------

    // One pixel's ray against one body (moved by xf), sphere traced through its field inside its box. Returns the
    // nearest the ray gets to the body (negative: how deep into it), and where (world).
    float Loss::Trace(const BodyField& f, const BoundingBox& box, const Rigid& xf, const View& w, int x, int y, Vector3* at) const {
        const float far = 4.0f * m_Half + Vector3Distance(box.min, box.max);
        float sx = ((x + 0.5f) / m_Res * 2.0f - 1.0f) * m_Half, sy = (1.0f - (y + 0.5f) / m_Res * 2.0f) * m_Half;
        Vector3 o = Vector3Add(Vector3Subtract(m_Centre, Vector3Scale(w.dir, far)), Vector3Add(Vector3Scale(w.u, sx), Vector3Scale(w.v, sy)));
        // the ray in the body's own frame
        Quaternion inv = { -xf.rotation.x, -xf.rotation.y, -xf.rotation.z, xf.rotation.w };
        Vector3 lo = xf.ApplyInverse(o), dir = Vector3RotateByQuaternion(w.dir, inv);
        // its stretch inside the box (slabs)
        float tn = 0.0f, tf = 2.0f * far;
        for (int a = 0; a < 3; a++) {
            float oa = (&lo.x)[a], da = (&dir.x)[a], l = (&box.min.x)[a] - m_Tau, h = (&box.max.x)[a] + m_Tau;
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
        float t = tn, dmin = FLT_MAX, tmin = tn;
        for (int k = 0; k < 64 && t <= tf; k++) {
            float d = f.Distance(Vector3Add(lo, Vector3Scale(dir, t)));
            if (d < dmin) { dmin = d; tmin = t; }
            if (d < -m_Tau) break;                       // well inside: fully covered
            t += d > m_Tau ? 0.9f * d : 0.5f * m_Tau;    // small steps near the surface
        }
        *at = Vector3Add(o, Vector3Scale(w.dir, tmin));
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

    AutoScore Loss::Evaluate(const Design& d, const std::vector<Param>& ps, std::vector<float>* grad) {
        AutoScore s;
        s.centre = m_SphereCentre;
        s.radius = m_SphereRadius;
        if (grad) grad->assign(ps.size(), 0.0f);

        // the bodies, in every pose
        Bodies b;
        for (int g : d.Bodies()) {
            BodyField f = BodyField::Body(d, d.groups[g].id);
            if (f.Empty()) continue;
            b.box.push_back(Grow(d.BodyBounds(d.groups[g].id), 0.002f * m_L));
            b.fields.push_back(std::move(f));
            b.group.push_back(g);
        }
        if (b.fields.empty()) return s;
        s.checkTimes = m_P.collisionWeight > 0.0f ? CheckTimes(d, m_P.motionChecks, m_P.motionCheckBias) : std::vector<float>{};
        b.times = { 0.0f, 1.0f };
        b.times.insert(b.times.end(), s.checkTimes.begin(), s.checkTimes.end());
        for (float t : b.times) b.xf.push_back(d.Transforms(t, true));

        Slopes slopes(b.fields.size());
        Silhouettes(b, s, slopes);
        if (m_P.collisionWeight > 0.0f) Collisions(b, s, slopes);
        s.total = (1.0f - std::clamp(m_P.sphereBias, 0.0f, 1.0f)) * s.fit + std::clamp(m_P.sphereBias, 0.0f, 1.0f) * s.sphere +
                  s.coveragePenalty + m_P.collisionWeight * s.collision;
        s.valid = true;
        if (grad && !ps.empty()) Gradient(d, b, slopes, ps, *grad);
        return s;
    }

    // ---- silhouettes ------------------------------------------------------------------------------------------------------

    // The open and closed poses drawn (each pixel keeps the body covering it most, and where its ray came closest to
    // it), compared with their targets, and the slopes of the soft edge pixels:
    //   d loss / d D: -1/sum(T) where the design covers less than the target, +1/sum(T) where it covers more, plus the
    //   coverage penalty's share; and d D / d distance = -1 / (2 tau) on the edge (0 < D < 1).
    void Loss::Silhouettes(const Bodies& b, AutoScore& s, Slopes& slopes) {
        const int nv = (int)m_Views.size(), px = m_Res * m_Res;
        std::vector<int> who[2];
        std::vector<Vector3> at[2];
        for (int pose = 0; pose < 2; pose++) {
            m_Cover[pose].assign((size_t)nv * px, 0.0f);
            who[pose].assign((size_t)nv * px, -1);
            at[pose].assign((size_t)nv * px, { 0, 0, 0 });
        }
        ParallelFor(2 * nv * m_Res, [&](int task) {            // every row of every view of both poses
            int pose = task / (nv * m_Res), v = (task / m_Res) % nv, y = task % m_Res;
            for (int x = 0; x < m_Res; x++) {
                size_t k = (size_t)v * px + (size_t)y * m_Res + x;
                for (size_t i = 0; i < b.fields.size(); i++) {
                    Vector3 p;
                    float dmin = Trace(b.fields[i], b.box[i], b.xf[pose][b.group[i]], m_Views[v], x, y, &p);
                    float c = std::clamp(0.5f - dmin / (2.0f * m_Tau), 0.0f, 1.0f);
                    if (c > m_Cover[pose][k]) {
                        m_Cover[pose][k] = c;
                        who[pose][k] = (int)i;
                        at[pose][k] = p;
                    }
                }
            }
        });

        const float bias = std::clamp(m_P.sphereBias, 0.0f, 1.0f), minCover = std::clamp(m_P.minViewCoverage, 0.0f, 1.0f);
        s.worstView = 1.0f;
        for (int pose = 0; pose < 2; pose++) {
            const float weight = (pose == 0 ? 1.0f - bias : bias) / nv;
            const bool counts = pose == 0 ? bias < 1.0f : bias > 0.0f;       // its views' minimum coverage applies
            double lossSum = 0.0, coverSum = 0.0, extraSum = 0.0;
            for (int v = 0; v < nv; v++) {
                const float* D = m_Cover[pose].data() + (size_t)v * px;
                auto target = [&](int k) { return pose == 0 ? m_Mesh[v][k] : Disc(m_Views[v], k % m_Res, k / m_Res); };
                double tsum = 0.0, hit = 0.0, spill = 0.0;
                for (int k = 0; k < px; k++) {
                    float t = target(k);
                    tsum += t;
                    hit += fminf(t, D[k]);
                    spill += fmaxf(0.0f, D[k] - t);
                }
                if (tsum <= 0.0) continue;
                float cover = (float)(hit / tsum), extra = (float)(spill / tsum);
                lossSum += (1.0f - cover) + extra;
                coverSum += cover;
                extraSum += extra;
                bool shortOf = counts && cover < minCover;
                if (counts) {
                    s.worstView = fminf(s.worstView, cover);
                    s.coveragePenalty += 10.0f * fmaxf(0.0f, minCover - cover);
                }
                for (int k = 0; k < px; k++) {
                    size_t kk = (size_t)v * px + k;
                    float c = D[k], t = target(k);
                    int i = who[pose][kk];
                    if (i < 0 || c <= 0.0f || c >= 1.0f || c == t) continue;     // not on an edge, or no slope
                    float dD = c < t ? (float)((-weight - (shortOf ? 10.0f : 0.0f)) / tsum) : (float)(weight / tsum);
                    slopes[i].push_back({ pose, at[pose][kk], dD * (-1.0f / (2.0f * m_Tau)) });
                }
            }
            (pose == 0 ? s.fit : s.sphere) = (float)(lossSum / nv);
            (pose == 0 ? s.viewCover : s.sphereViewCover) = (float)(coverSum / nv);
            (pose == 0 ? s.viewExtra : s.sphereViewExtra) = (float)(extraSum / nv);
        }
    }

    // ---- collisions -------------------------------------------------------------------------------------------------------

    // In every pose, every two bodies whose boxes meet are sampled where they meet (the same unit points each time). A
    // point inside both by dd (the larger, so nearer zero, of their two distances, both negative) adds dd's depth
    // times its share of the box's volume; its slope goes to the body whose distance that is.
    void Loss::Collisions(const Bodies& b, AutoScore& s, Slopes& slopes) const {
        const double sphereVolume = 4.0 / 3.0 * PI * pow(m_SphereRadius, 3.0), norm = sphereVolume * 0.02 * m_L;
        const int poses = (int)b.times.size(), between = poses - 2;
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
            // the pose's share of the collision term: the ends count fully, the poses between share one
            const float poseWeight = pr.pose < 2 ? 1.0f : 1.0f / between;
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
        for (int pose = 2; pose < poses; pose++) {
            betweenSum += depth[pose] / norm;
            float share = (float)(volume[pose] / sphereVolume);
            if (share > s.clashMotion) {
                s.clashMotion = share;
                s.clashMotionT = b.times[pose];
            }
        }
        s.collision = (float)(depth[0] / norm + depth[1] / norm + (between > 0 ? betweenSum / between : 0.0));
        s.clashOpen = (float)(volume[0] / sphereVolume);
        s.clashClosed = (float)(volume[1] / sphereVolume);
    }

    // ---- the gradient -------------------------------------------------------------------------------------------------------

    // d loss / d parameter = sum over the slopes of (slope · d distance / d parameter). To first order a distance only
    // changes through the body's field and pose at its fixed world point (for a silhouette's ray, where it comes
    // closest: the envelope theorem), so d distance / d parameter comes from central differences there: the parameter
    // nudged either way, the body's field (a shape's parameter) or the poses (a hinge's) rebuilt, the distance measured
    // again. The poses' times stay as they were (the checks are placed by the design, not part of what is moved).
    void Loss::Gradient(const Design& d, const Bodies& b, const Slopes& slopes, const std::vector<Param>& ps, std::vector<float>& grad) const {
        std::vector<int> fieldOf(d.groups.size(), -1);               // group index -> field
        for (size_t i = 0; i < b.group.size(); i++) fieldOf[b.group[i]] = (int)i;
        ParallelFor((int)ps.size(), [&](int j) {
            const Param& p = ps[j];
            double sum[2] = { 0.0, 0.0 };
            float value[2] = { 0.0f, 0.0f };
            for (int side = 0; side < 2; side++) {
                Design e = d;
                float* ref = Ref(e, p);
                if (!ref) return;
                SetParam(e, p, *ref + (side == 0 ? 0.1f : -0.1f) * p.unit, m_L);
                value[side] = *Ref(e, p);
                Tidy(e);
                std::vector<std::vector<Rigid>> xf;
                for (float t : b.times) xf.push_back(e.Transforms(t, true));
                int own = -1;                                           // a shape's parameter changes its body's field
                BodyField changed;
                if (p.kind == Param::Shape) {
                    int body = e.GroupIndex(e.ShapeBody(e.ShapeIndex(p.id)));
                    own = body >= 0 ? fieldOf[body] : -1;
                    if (own >= 0) changed = BodyField::Body(e, e.groups[body].id);
                }
                for (size_t i = 0; i < b.fields.size(); i++)
                    for (const Slope& sl : slopes[i]) {
                        const Rigid& before = b.xf[sl.pose][b.group[i]];
                        const Rigid& after = xf[sl.pose][b.group[i]];
                        bool fieldMoves = (int)i == own, poseMoves = memcmp(&before, &after, sizeof(Rigid)) != 0;
                        if (!fieldMoves && !poseMoves) continue;        // this parameter doesn't touch it
                        sum[side] += sl.weight * (fieldMoves ? changed : b.fields[i]).Distance(after.ApplyInverse(sl.at));
                    }
            }
            float step = value[0] - value[1];
            if (step > 0.0f) grad[j] = (float)((sum[0] - sum[1]) / step) * p.unit;    // per unit of the parameter
        });
    }

    // ---- the picture for the UI ---------------------------------------------------------------------------------------

    // Dark where both cover a pixel, red where only the target does (missing), blue where only the design does (extra).
    void Loss::Picture(std::vector<unsigned char>* rgba, int* w, int* h) const {
        const int nv = (int)m_Views.size(), px = m_Res * m_Res, gap = 3;
        if (m_Cover[0].size() != (size_t)nv * px) { *w = *h = 0; rgba->clear(); return; }
        *w = nv * m_Res + (nv + 1) * gap;
        *h = 2 * m_Res + 3 * gap;
        rgba->assign((size_t)*w * *h * 4, 255);
        for (size_t k = 0; k < rgba->size(); k += 4) { (*rgba)[k] = 222; (*rgba)[k + 1] = 225; (*rgba)[k + 2] = 232; }
        const float both[3] = { 70, 74, 86 }, miss[3] = { 235, 110, 95 }, extra[3] = { 90, 140, 235 };
        for (int pose = 0; pose < 2; pose++)
            for (int v = 0; v < nv; v++)
                for (int k = 0; k < px; k++) {
                    int x = k % m_Res, y = k / m_Res;
                    float t = pose == 0 ? m_Mesh[v][k] : Disc(m_Views[v], x, y), dd = m_Cover[pose][(size_t)v * px + k];
                    float ov = fminf(t, dd), ms = t - ov, ex = dd - ov, rest = 1.0f - fmaxf(t, dd);
                    unsigned char* out = &(*rgba)[((size_t)(gap + pose * (m_Res + gap) + y) * *w + gap + v * (m_Res + gap) + x) * 4];
                    for (int c = 0; c < 3; c++) out[c] = (unsigned char)std::clamp(lroundf(ov * both[c] + ms * miss[c] + ex * extra[c] + rest * 255.0f), 0L, 255L);
                }
    }
}
