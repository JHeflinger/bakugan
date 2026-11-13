// The 3D part of the loss: the open design against the placed mesh in 3D, not as outlines. Two terms:
//
// Chamfer (chamferWeight). The two surfaces should lie on each other, both ways round:
//   mesh -> design   points on the mesh's surface: each costs how far the design's surface is from it, |D|
//   design -> mesh   points on the design's surface: each costs how far the mesh's surface is from it
// (both rounded off near 0, so their slopes don't flip at every step). The first sees what the design misses, the
// second what it has too much of.
//
// SDF (sdfWeight). Points in and around the mesh, whose signed distance T to it is known (negative inside): the
// design's signed distance D should match it. Within delta of the mesh's surface they should agree, |D - T|; deeper
// inside the mesh the design only has to be at least delta inside too, max(0, D + delta); farther out, at least delta
// outside, max(0, delta - D). So it tells inside from outside (which the Chamfer distance alone can't: a hollow shell
// and a solid scores the same) and says which way to move wherever the design is wrong.
//
// Each part is a mean over its points, over 0.1 of the mesh's size.
//
// The mesh's points are fixed in its own frame and moved with the placement (an offset, and a scale about its centre
// of volume, which scales the signed distances with it). The design's surface points are found again every
// evaluation: fixed points in each piece's box, pulled onto its surface (Newton steps along its field's gradient),
// kept where no other piece covers them.
//
// The gradient (loss.cpp) needs nothing new: every term depends on the carving only through the pieces' distances at
// world points, as the silhouettes do.
//   - A mesh point's |D| (and an SDF point's) depends on the nearest piece's distance there.
//   - A design point sits on its piece's surface. A change that raises the piece's distance there by e moves the
//     surface back by e along its normal n, so the point's cost changes by -e (grad u . n), u the distance to the mesh.
// The placement's slope is measured directly, the mesh's points moved with it nudged either way (in Evaluate).
//
// The mesh's signed distance at the SDF points is measured once: the distance to the nearest triangle (a box tree over
// them), negative where the mesh's winding number is over 1/2, which tolerates small holes. The design -> mesh term
// asks the box tree for the nearest point every evaluation, which needs no sign.
#include "loss.hpp"
#include "util/box.hpp"
#include <raymath.h>
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <functional>
#include <numeric>
#include <random>
#include <thread>

namespace Bakugan {
    namespace {
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

        // The point of triangle abc nearest p (Ericson, Real-Time Collision Detection, 5.1.5).
        Vector3 ClosestOnTriangle(Vector3 p, Vector3 a, Vector3 b, Vector3 c) {
            Vector3 ab = Vector3Subtract(b, a), ac = Vector3Subtract(c, a), ap = Vector3Subtract(p, a);
            float d1 = Vector3DotProduct(ab, ap), d2 = Vector3DotProduct(ac, ap);
            if (d1 <= 0.0f && d2 <= 0.0f) return a;
            Vector3 bp = Vector3Subtract(p, b);
            float d3 = Vector3DotProduct(ab, bp), d4 = Vector3DotProduct(ac, bp);
            if (d3 >= 0.0f && d4 <= d3) return b;
            float vc = d1 * d4 - d3 * d2;
            if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) return Vector3Add(a, Vector3Scale(ab, d1 / (d1 - d3)));
            Vector3 cp = Vector3Subtract(p, c);
            float d5 = Vector3DotProduct(ab, cp), d6 = Vector3DotProduct(ac, cp);
            if (d6 >= 0.0f && d5 <= d6) return c;
            float vb = d5 * d2 - d1 * d6;
            if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) return Vector3Add(a, Vector3Scale(ac, d2 / (d2 - d6)));
            float va = d3 * d6 - d5 * d4;
            if (va <= 0.0f && d4 - d3 >= 0.0f && d5 - d6 >= 0.0f)
                return Vector3Add(b, Vector3Scale(Vector3Subtract(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
            float den = 1.0f / (va + vb + vc), v = vb * den, w = vc * den;
            return Vector3Add(a, Vector3Add(Vector3Scale(ab, v), Vector3Scale(ac, w)));
        }

        // The mesh's winding number at p: about 1 inside a closed mesh, 0 outside (-1 inside if it's turned inside out).
        // Each triangle adds the solid angle it covers seen from p (Van Oosterom & Strackee), over 4 pi.
        double Winding(const TriMesh& m, Vector3 p) {
            double sum = 0.0;
            for (size_t t = 0; t + 2 < m.indices.size(); t += 3) {
                Vector3 a = Vector3Subtract(m.positions[m.indices[t]], p), b = Vector3Subtract(m.positions[m.indices[t + 1]], p),
                        c = Vector3Subtract(m.positions[m.indices[t + 2]], p);
                double la = Vector3Length(a), lb = Vector3Length(b), lc = Vector3Length(c);
                double num = Vector3DotProduct(a, Vector3CrossProduct(b, c));
                double den = la * lb * lc + Vector3DotProduct(a, b) * lc + Vector3DotProduct(b, c) * la + Vector3DotProduct(c, a) * lb;
                sum += 2.0 * atan2(num, den);
            }
            return sum / (4.0 * PI);
        }

        // |x|, rounded off within eps of 0, and its slope.
        float SmoothAbs(float x, float eps, float* slope) {
            float r = sqrtf(x * x + eps * eps);
            *slope = x / r;
            return r - eps;
        }
    }

    // A box tree over a mesh's triangles (its own copy), for the nearest point of the mesh to a point.
    class TriangleTree {
    public:
        explicit TriangleTree(const TriMesh& m) : m_P(m.positions), m_I(m.indices) {
            int n = (int)(m_I.size() / 3);
            m_Tris.resize(n);
            std::iota(m_Tris.begin(), m_Tris.end(), 0);
            m_Centroid.resize(n);
            for (int t = 0; t < n; t++) m_Centroid[t] = Vector3Scale(Vector3Add(V(t, 0), Vector3Add(V(t, 1), V(t, 2))), 1.0f / 3.0f);
            m_Nodes.reserve(2 * (size_t)n / Leaf + 2);
            if (n > 0) Build(0, n);
        }

        // The distance from p to the mesh, and (nearest) the point of it there.
        float Distance(Vector3 p, Vector3* nearest = nullptr) const {
            if (m_Nodes.empty()) return FLT_MAX;
            float best = FLT_MAX;                    // squared
            Vector3 at = p;
            int stack[128], top = 0;
            stack[top++] = 0;
            while (top > 0) {
                const Node& nd = m_Nodes[stack[--top]];
                if (BoxDistance2(p, nd.box) >= best) continue;
                if (nd.leaf) {
                    for (int k = nd.a; k < nd.b; k++) {
                        int t = m_Tris[k];
                        Vector3 q = ClosestOnTriangle(p, V(t, 0), V(t, 1), V(t, 2));
                        float d = Vector3DistanceSqr(p, q);
                        if (d < best) { best = d; at = q; }
                    }
                    continue;
                }
                if (top + 2 > 128) continue;
                float da = BoxDistance2(p, m_Nodes[nd.a].box), db = BoxDistance2(p, m_Nodes[nd.b].box);
                if (da < db) { stack[top++] = nd.b; stack[top++] = nd.a; }   // the nearer child is searched first
                else { stack[top++] = nd.a; stack[top++] = nd.b; }
            }
            if (nearest) *nearest = at;
            return sqrtf(best);
        }

    private:
        static constexpr int Leaf = 8;
        struct Node {
            BoundingBox box;
            bool leaf;
            int a, b;                                // leaf: triangles a..b; otherwise the children's nodes
        };
        Vector3 V(int t, int k) const { return m_P[m_I[(size_t)t * 3 + k]]; }
        static float BoxDistance2(Vector3 p, const BoundingBox& b) {
            Vector3 q = Vector3Subtract(Vector3Max(b.min, Vector3Min(p, b.max)), p);
            return Vector3DotProduct(q, q);
        }
        int Build(int first, int last) {
            BoundingBox box = { { FLT_MAX, FLT_MAX, FLT_MAX }, { -FLT_MAX, -FLT_MAX, -FLT_MAX } };
            Vector3 clo = box.min, chi = box.max;
            for (int k = first; k < last; k++) {
                int t = m_Tris[k];
                for (int j = 0; j < 3; j++) { box.min = Vector3Min(box.min, V(t, j)); box.max = Vector3Max(box.max, V(t, j)); }
                clo = Vector3Min(clo, m_Centroid[t]);
                chi = Vector3Max(chi, m_Centroid[t]);
            }
            int idx = (int)m_Nodes.size();
            m_Nodes.push_back({ box, true, first, last });
            if (last - first <= Leaf) return idx;
            Vector3 ext = Vector3Subtract(chi, clo);
            int axis = ext.x >= ext.y && ext.x >= ext.z ? 0 : (ext.y >= ext.z ? 1 : 2);
            int mid = (first + last) / 2;
            std::nth_element(m_Tris.begin() + first, m_Tris.begin() + mid, m_Tris.begin() + last,
                             [&](int x, int y) { return (&m_Centroid[x].x)[axis] < (&m_Centroid[y].x)[axis]; });
            int l = Build(first, mid), r = Build(mid, last);
            m_Nodes[idx].leaf = false;
            m_Nodes[idx].a = l;
            m_Nodes[idx].b = r;
            return idx;
        }

        std::vector<Vector3> m_P;
        std::vector<unsigned> m_I;
        std::vector<int> m_Tris;
        std::vector<Vector3> m_Centroid;
        std::vector<Node> m_Nodes;
    };

    // ---- setting up: the mesh's points, and its signed distance at them -----------------------------------------------------

    void Loss::SetUpShape(const TriMesh& mesh) {
        const int n = std::clamp(m_P.shapeSamples, 300, 50000), ns = n / 3, nv = n - ns;
        std::mt19937 rng(m_P.seed * 131u + 3u);
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);
        std::normal_distribution<float> gauss(0.0f, 1.0f);

        std::vector<double> cdf;
        double area = 0.0;
        Vector3 lo = { FLT_MAX, FLT_MAX, FLT_MAX }, hi = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
        for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
            Vector3 a = mesh.positions[mesh.indices[t]], b = mesh.positions[mesh.indices[t + 1]], c = mesh.positions[mesh.indices[t + 2]];
            area += 0.5 * Vector3Length(Vector3CrossProduct(Vector3Subtract(b, a), Vector3Subtract(c, a)));
            cdf.push_back(area);
            lo = Vector3Min(lo, Vector3Min(a, Vector3Min(b, c)));
            hi = Vector3Max(hi, Vector3Max(a, Vector3Max(b, c)));
        }
        if (cdf.empty() || area <= 0.0) return;
        auto onMesh = [&]() {                     // area weighted
            size_t t = std::min((size_t)(std::lower_bound(cdf.begin(), cdf.end(), u01(rng) * area) - cdf.begin()), cdf.size() - 1);
            float u = u01(rng), v = u01(rng);
            if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
            Vector3 a = mesh.positions[mesh.indices[t * 3]], b = mesh.positions[mesh.indices[t * 3 + 1]], c = mesh.positions[mesh.indices[t * 3 + 2]];
            return Vector3Add(a, Vector3Add(Vector3Scale(Vector3Subtract(b, a), u), Vector3Scale(Vector3Subtract(c, a), v)));
        };

        // points on the mesh; SDF points, half just off its surface (so thin parts get their share), half anywhere in a
        // box around it with room to spare (so parts of the design sticking out are seen)
        m_ShapeSurface.clear();
        m_ShapeVolume.clear();
        for (int i = 0; i < ns; i++) m_ShapeSurface.push_back(onMesh());
        const float near = 0.03f * m_L;
        BoundingBox box = Grow({ lo, hi }, 0.2f * m_L);
        for (int i = 0; i < nv; i++) {
            if (i % 2 == 0) m_ShapeVolume.push_back(Vector3Add(onMesh(), { near * gauss(rng), near * gauss(rng), near * gauss(rng) }));
            else m_ShapeVolume.push_back({ box.min.x + u01(rng) * (box.max.x - box.min.x), box.min.y + u01(rng) * (box.max.y - box.min.y),
                                           box.min.z + u01(rng) * (box.max.z - box.min.z) });
        }
        m_Tree = std::make_shared<TriangleTree>(mesh);
        m_ShapeTarget.assign(nv, 0.0f);
        ParallelFor(nv, [&](int i) {
            float d = m_Tree->Distance(m_ShapeVolume[i]);
            m_ShapeTarget[i] = fabs(Winding(mesh, m_ShapeVolume[i])) > 0.5 ? -d : d;
        });
        // fixed points in a unit box, to pull onto each piece's surface
        m_DesignUnit.clear();
        for (int i = 0; i < 256; i++) m_DesignUnit.push_back({ u01(rng), u01(rng), u01(rng) });
    }

    // ---- the design's distance, and its surface -----------------------------------------------------------------------------

    float Loss::DesignDistance(const Bodies& b, int pose, Vector3 p, int* who, int skip) const {
        float best = FLT_MAX;
        if (who) *who = -1;
        for (size_t i = 0; i < b.fields.size(); i++) {
            if ((int)i == skip) continue;
            Vector3 q = b.xf[pose][b.group[i]].ApplyInverse(p);
            // outside the piece's box by g: it's no nearer than g, so it can't beat a nearer piece
            Vector3 g = Vector3Subtract(Vector3Max(b.box[i].min, Vector3Min(q, b.box[i].max)), q);
            if (Vector3Length(g) >= best) continue;
            float d = b.fields[i].Distance(q);
            if (d < best) {
                best = d;
                if (who) *who = (int)i;
            }
        }
        return best;
    }

    void Loss::SampleDesignSurface(const Bodies& b) {
        const int per = (int)m_DesignUnit.size(), nf = (int)b.fields.size();
        const float h = 0.001f * m_L, tol = 0.002f * m_L;
        std::vector<DesignPoint> found((size_t)per * nf);
        std::vector<char> ok((size_t)per * nf, 0);
        ParallelFor(nf * per, [&](int task) {
            int i = task / per, k = task % per;
            const BodyField& f = b.fields[i];
            const BoundingBox& box = b.box[i];
            Vector3 u = m_DesignUnit[k];
            Vector3 p = { box.min.x + u.x * (box.max.x - box.min.x), box.min.y + u.y * (box.max.y - box.min.y), box.min.z + u.z * (box.max.z - box.min.z) };
            Vector3 g = { 0, 0, 0 };
            for (int it = 0; it < 6; it++) {        // Newton steps onto the surface
                float d = f.Distance(p);
                g = f.Gradient(p, h);
                float gg = Vector3DotProduct(g, g);
                if (gg < 1e-8f) return;
                p = Vector3Subtract(p, Vector3Scale(g, d / gg));
                if (fabsf(d) < 0.1f * tol) break;
            }
            if (fabsf(f.Distance(p)) > tol || !InBox(p, box)) return;
            g = f.Gradient(p, h);
            float gl = Vector3Length(g);
            if (gl < 1e-4f) return;
            const Rigid& xf = b.xf[0][b.group[i]];
            Vector3 at = xf.Apply(p);
            if (DesignDistance(b, 0, at, nullptr, i) < -tol) return;     // inside another piece: not the design's surface
            found[task] = { i, at, Vector3RotateByQuaternion(Vector3Scale(g, 1.0f / gl), xf.rotation) };
            ok[task] = 1;
        });
        m_DesignSurface.clear();
        for (size_t k = 0; k < found.size(); k++)
            if (ok[k]) m_DesignSurface.push_back(found[k]);
    }

    // ---- the loss -------------------------------------------------------------------------------------------------------

    float Loss::Shape(const Bodies& b, const Placement& t, AutoScore* s, Slopes* slopes) const {
        if (m_ShapeSurface.empty() || b.fields.empty()) return 0.0f;
        const float chamferW = std::max(0.0f, m_P.chamferWeight), sdfW = std::max(0.0f, m_P.sdfWeight);
        const float scale = 0.1f * m_L, eps = 0.005f * m_L, delta = 0.05f * m_L * t.scale;
        const int ns = (int)m_ShapeSurface.size(), nv = (int)m_ShapeVolume.size(), nd = (int)m_DesignSurface.size();
        auto place = [&](Vector3 q) { return Vector3Add(Vector3Add(m_MeshCentre, t.offset), Vector3Scale(Vector3Subtract(q, m_MeshCentre), t.scale)); };
        auto unplace = [&](Vector3 p) {
            return Vector3Add(m_MeshCentre, Vector3Scale(Vector3Subtract(Vector3Subtract(p, m_MeshCentre), t.offset), 1.0f / fmaxf(t.scale, 1e-6f)));
        };

        // the design's distance at the mesh's points (its surface's, then the SDF's), placed
        std::vector<Vector3> pts(ns + nv);
        for (int k = 0; k < ns; k++) pts[k] = place(m_ShapeSurface[k]);
        for (int k = 0; k < nv; k++) pts[ns + k] = place(m_ShapeVolume[k]);
        std::vector<float> D(ns + nv);
        std::vector<int> who(ns + nv);
        ParallelFor((ns + nv + 63) / 64, [&](int chunk) {
            for (int k = chunk * 64; k < std::min(ns + nv, chunk * 64 + 64); k++) D[k] = DesignDistance(b, 0, pts[k], &who[k]);
        });
        // the mesh's distance at the design's surface points, and which way it grows
        std::vector<float> U(nd);
        std::vector<Vector3> away(nd);
        ParallelFor((nd + 63) / 64, [&](int chunk) {
            for (int k = chunk * 64; k < std::min(nd, chunk * 64 + 64); k++) {
                Vector3 q = unplace(m_DesignSurface[k].at), near;
                float u = m_Tree->Distance(q, &near);
                U[k] = u * t.scale;
                Vector3 dir = Vector3Subtract(q, near);
                float l = Vector3Length(dir);
                away[k] = l > 1e-9f ? Vector3Scale(dir, 1.0f / l) : Vector3{ 0, 0, 0 };    // (the same way placed: the scale is uniform)
            }
        });

        double toDesign = 0.0, toMesh = 0.0, sdf = 0.0, meshGap = 0.0, designGap = 0.0;
        std::vector<float> slope(ns + nv, 0.0f);
        for (int k = 0; k < ns; k++) {          // mesh -> design
            toDesign += SmoothAbs(D[k], eps, &slope[k]);
            meshGap += fabsf(D[k]);
            slope[k] *= chamferW / (ns * scale);
        }
        for (int k = 0; k < nv; k++) {          // SDF
            float d = D[ns + k], tt = t.scale * m_ShapeTarget[k], sl = 0.0f, cost = 0.0f;
            if (tt <= -delta) { if (d > -delta) { cost = d + delta; sl = 1.0f; } }         // well inside: be inside too
            else if (tt >= delta) { if (d < delta) { cost = delta - d; sl = -1.0f; } }      // well outside: be outside too
            else cost = SmoothAbs(d - tt, eps, &sl);                                         // near the surface: agree
            sdf += cost;
            slope[ns + k] = sl * sdfW / (nv * scale);
        }
        std::vector<float> slopeD(nd, 0.0f);
        for (int k = 0; k < nd; k++) {          // design -> mesh
            float sl;
            toMesh += SmoothAbs(U[k], eps, &sl);
            designGap += U[k];
            // the piece's distance up by e: its surface back by e along n, so u changes by -e (away . n)
            slopeD[k] = -sl * Vector3DotProduct(away[k], m_DesignSurface[k].normal) * chamferW / (nd * scale);
        }

        float chamfer = (float)(toDesign / (ns * scale) + (nd > 0 ? toMesh / (nd * scale) : 0.0));
        float sdfLoss = (float)(sdf / (nv * scale));
        if (s) {
            s->chamfer = chamfer;
            s->sdf = sdfLoss;
            s->meshGap = (float)(meshGap / ns);
            s->designGap = nd > 0 ? (float)(designGap / nd) : 0.0f;
        }
        if (slopes) {
            for (int k = 0; k < ns + nv; k++)
                if (slope[k] != 0.0f && who[k] >= 0) (*slopes)[who[k]].push_back({ 0, pts[k], slope[k] });
            for (int k = 0; k < nd; k++)
                if (slopeD[k] != 0.0f) (*slopes)[m_DesignSurface[k].field].push_back({ 0, m_DesignSurface[k].at, slopeD[k] });
        }
        return chamferW * chamfer + sdfW * sdfLoss;
    }
}
