#include "silhouette.hpp"
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

    Silhouettes::Silhouettes(const TriMesh& mesh, const AutoParams& p) : m_P(p) {
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

        // points on the surface (area weighted), where the grammar puts new shapes
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
    }

    Vector2 Silhouettes::ToPixel(const View& w, Vector3 p) const {
        Vector3 q = Vector3Subtract(p, m_Centre);
        return { (Vector3DotProduct(q, w.u) / m_Half + 1.0f) * 0.5f * m_Res, (1.0f - Vector3DotProduct(q, w.v) / m_Half) * 0.5f * m_Res };
    }

    float Silhouettes::Disc(const View& w, int x, int y) const {
        Vector2 c = ToPixel(w, m_SphereCentre);
        float dx = x + 0.5f - c.x, dy = y + 0.5f - c.y;
        return std::clamp(m_SphereRadius / (2.0f * m_Half / m_Res) - sqrtf(dx * dx + dy * dy) + 0.5f, 0.0f, 1.0f);
    }

    // ---- drawing the design ---------------------------------------------------------------------------------------------

    // One pixel's ray against one body (moved by xf), sphere traced through its field inside its box. Returns the
    // nearest the ray gets to the body (negative: how deep into it), and where (world).
    float Silhouettes::Trace(const BodyField& f, const BoundingBox& box, const Rigid& xf, const View& w, int x, int y, Vector3* at) const {
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

    void Silhouettes::Draw(const Design& d, Render& r) const {
        r = Render{};
        for (int g : d.Bodies()) {
            BodyField f = BodyField::Body(d, d.groups[g].id);
            if (f.Empty()) continue;
            BoundingBox b = d.BodyBounds(d.groups[g].id);
            r.box.push_back({ Vector3SubtractValue(b.min, 0.002f * m_L), Vector3AddValue(b.max, 0.002f * m_L) });
            r.fields.push_back(std::move(f));
            r.group.push_back(g);
        }
        r.xf[0] = d.Transforms(0.0f, false);
        r.xf[1] = d.Transforms(1.0f, false);
        const int nv = (int)m_Views.size(), px = m_Res * m_Res;
        for (int pose = 0; pose < 2; pose++) {
            r.cover[pose].assign((size_t)nv * px, 0.0f);
            r.who[pose].assign((size_t)nv * px, -1);
            r.at[pose].assign((size_t)nv * px, { 0, 0, 0 });
        }
        // every row of every view of both poses, in parallel; each pixel keeps the body covering it most
        ParallelFor(2 * nv * m_Res, [&](int task) {
            int pose = task / (nv * m_Res), v = (task / m_Res) % nv, y = task % m_Res;
            for (int x = 0; x < m_Res; x++) {
                size_t k = (size_t)v * px + (size_t)y * m_Res + x;
                for (size_t i = 0; i < r.fields.size(); i++) {
                    Vector3 at;
                    float dmin = Trace(r.fields[i], r.box[i], r.xf[pose][r.group[i]], m_Views[v], x, y, &at);
                    float c = std::clamp(0.5f - dmin / (2.0f * m_Tau), 0.0f, 1.0f);
                    if (c > r.cover[pose][k]) {
                        r.cover[pose][k] = c;
                        r.who[pose][k] = (int)i;
                        r.at[pose][k] = at;
                    }
                }
            }
        });
    }

    // ---- the loss and its gradient ----------------------------------------------------------------------------------------

    // The gradient follows the loss back in three steps:
    //   1. d loss / d D per pixel: -1/sum(T) where the design covers less than the target (more would cover more),
    //      +1/sum(T) where it covers more (less would be less extra), plus the coverage penalty's share
    //   2. d D / d dmin = -1 / (2 tau) on the soft edge of a silhouette (0 < D < 1), 0 elsewhere: dmin is the nearest
    //      the pixel's ray gets to the body covering it most
    //   3. d dmin / d parameter: the ray's nearest point moves, but to first order the distance there only changes
    //      through the field (the envelope theorem), so it is the change of that body's distance at that fixed world
    //      point, here by central differences: the parameter nudged either way, the body's field (a shape's
    //      parameter) or pose (a hinge's) rebuilt, and the distance there measured again.
    AutoScore Silhouettes::Loss(const Design& d, const std::vector<Param>& ps, std::vector<float>* grad) {
        Draw(d, m_Last);
        const Render& r = m_Last;
        AutoScore s;
        s.centre = m_SphereCentre;
        s.radius = m_SphereRadius;
        if (grad) grad->assign(ps.size(), 0.0f);
        if (r.fields.empty()) return s;

        const int nv = (int)m_Views.size(), px = m_Res * m_Res;
        const float bias = std::clamp(m_P.sphereBias, 0.0f, 1.0f), minCover = std::clamp(m_P.minViewCoverage, 0.0f, 1.0f);
        std::vector<float> dD[2] = { std::vector<float>((size_t)nv * px, 0.0f), std::vector<float>((size_t)nv * px, 0.0f) };
        s.worstView = 1.0f;
        for (int pose = 0; pose < 2; pose++) {
            const float weight = (pose == 0 ? 1.0f - bias : bias) / nv;
            const bool counts = pose == 0 ? bias < 1.0f : bias > 0.0f;       // its views' minimum coverage applies
            double lossSum = 0.0, coverSum = 0.0, extraSum = 0.0;
            for (int v = 0; v < nv; v++) {
                const float* D = r.cover[pose].data() + (size_t)v * px;
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
                bool short_ = counts && cover < minCover;
                if (counts) {
                    s.worstView = fminf(s.worstView, cover);
                    s.coveragePenalty += 10.0f * fmaxf(0.0f, minCover - cover);
                }
                float* g = dD[pose].data() + (size_t)v * px;
                for (int k = 0; k < px; k++) {
                    float t = target(k);
                    if (D[k] < t) g[k] = (float)((-weight - (short_ ? 10.0f : 0.0f)) / tsum);
                    else if (D[k] > t) g[k] = (float)(weight / tsum);
                }
            }
            if (pose == 0) {
                s.fit = (float)(lossSum / nv);
                s.viewCover = (float)(coverSum / nv);
                s.viewExtra = (float)(extraSum / nv);
            } else {
                s.sphere = (float)(lossSum / nv);
                s.sphereViewCover = (float)(coverSum / nv);
                s.sphereViewExtra = (float)(extraSum / nv);
            }
        }
        s.total = (1.0f - bias) * s.fit + bias * s.sphere + s.coveragePenalty;
        s.valid = true;
        if (!grad || ps.empty()) return s;

        // the pixels the parameters move: on the soft edge of a silhouette, with a slope in the loss
        struct Edge {
            int pose, field;
            Vector3 at;
            float weight;                    // d loss / d dmin
        };
        std::vector<std::vector<Edge>> edges(r.fields.size());    // by field
        for (int pose = 0; pose < 2; pose++)
            for (size_t k = 0; k < r.cover[pose].size(); k++) {
                float c = r.cover[pose][k];
                int i = r.who[pose][k];
                if (i < 0 || c <= 0.0f || c >= 1.0f || dD[pose][k] == 0.0f) continue;
                edges[i].push_back({ pose, i, r.at[pose][k], dD[pose][k] * (-1.0f / (2.0f * m_Tau)) });
            }
        std::vector<int> fieldOf(d.groups.size(), -1);           // group index -> field
        for (size_t i = 0; i < r.group.size(); i++) fieldOf[r.group[i]] = (int)i;

        ParallelFor((int)ps.size(), [&](int j) {
            const Param& p = ps[j];
            // the loss's slope along this parameter, from the edge pixels' distances with it nudged up and down
            double sum[2] = { 0.0, 0.0 };
            float value[2] = { 0.0f, 0.0f };
            for (int side = 0; side < 2; side++) {
                Design e = d;
                float* ref = Ref(e, p);
                if (!ref) return;
                SetParam(e, p, *ref + (side == 0 ? 0.1f : -0.1f) * p.unit, m_L);
                value[side] = *Ref(e, p);
                Tidy(e);
                std::vector<Rigid> xf[2] = { e.Transforms(0.0f, false), e.Transforms(1.0f, false) };
                int own = -1;                                     // a shape's parameter changes its body's field
                BodyField changed;
                if (p.kind == Param::Shape) {
                    int body = e.GroupIndex(e.ShapeBody(e.ShapeIndex(p.id)));
                    own = body >= 0 ? fieldOf[body] : -1;
                    if (own >= 0) changed = BodyField::Body(e, e.groups[body].id);
                }
                for (size_t i = 0; i < r.fields.size(); i++) {
                    bool fieldMoves = (int)i == own;
                    for (const Edge& g : edges[i]) {
                        const Rigid& before = r.xf[g.pose][r.group[i]];
                        const Rigid& after = xf[g.pose][r.group[i]];
                        bool poseMoves = memcmp(&before, &after, sizeof(Rigid)) != 0;
                        if (!fieldMoves && !poseMoves) continue;          // this parameter doesn't touch it
                        const BodyField& f = fieldMoves ? changed : r.fields[i];
                        sum[side] += g.weight * f.Distance(after.ApplyInverse(g.at));
                    }
                }
            }
            float step = value[0] - value[1];
            if (step > 0.0f) (*grad)[j] = (float)((sum[0] - sum[1]) / step) * p.unit;   // per unit of the parameter
        });
        return s;
    }

    // ---- the picture for the UI ---------------------------------------------------------------------------------------

    // Dark where both cover a pixel, red where only the target does (missing), blue where only the design does (extra).
    void Silhouettes::Picture(const AutoScore&, std::vector<unsigned char>* rgba, int* w, int* h) const {
        const int nv = (int)m_Views.size(), px = m_Res * m_Res, gap = 3;
        if (m_Last.cover[0].size() != (size_t)nv * px) { *w = *h = 0; rgba->clear(); return; }
        *w = nv * m_Res + (nv + 1) * gap;
        *h = 2 * m_Res + 3 * gap;
        rgba->assign((size_t)*w * *h * 4, 255);
        for (size_t k = 0; k < rgba->size(); k += 4) { (*rgba)[k] = 222; (*rgba)[k + 1] = 225; (*rgba)[k + 2] = 232; }
        const float both[3] = { 70, 74, 86 }, miss[3] = { 235, 110, 95 }, extra[3] = { 90, 140, 235 };
        for (int pose = 0; pose < 2; pose++)
            for (int v = 0; v < nv; v++)
                for (int k = 0; k < px; k++) {
                    int x = k % m_Res, y = k / m_Res;
                    float t = pose == 0 ? m_Mesh[v][k] : Disc(m_Views[v], x, y), dd = m_Last.cover[pose][(size_t)v * px + k];
                    float ov = fminf(t, dd), ms = t - ov, ex = dd - ov, rest = 1.0f - fmaxf(t, dd);
                    unsigned char* out = &(*rgba)[((size_t)(gap + pose * (m_Res + gap) + y) * *w + gap + v * (m_Res + gap) + x) * 4];
                    for (int c = 0; c < 3; c++) out[c] = (unsigned char)std::clamp(lroundf(ov * both[c] + ms * miss[c] + ex * extra[c] + rest * 255.0f), 0L, 255L);
                }
    }
}
