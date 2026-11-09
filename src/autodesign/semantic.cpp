// The semantic part of the loss (see loss.hpp): the design drawn from each view, compared with the mesh's and the
// sphere's pictures by DINOv3, and the gradient that comes back through the pixels.
//
// A pixel's value: each body its ray reaches (within a pixel's softness) is a layer, with a coverage c (from the ray's
// nearest distance, as for silhouettes) and a shade s (of the surface where the ray enters it, or on an edge, where it
// comes closest). Layers composite front to back over the background:  I = sum_k T_k c_k s_k + T_end bg,  where
// T_k = prod_{j<k} (1 - c_j). DINOv3 gives d loss / d I, and then
//   d I / d c_k = T_k (s_k - B_k)    (B_k: what is composited behind layer k)     -> a Slope at the edge, like silhouettes
//   d I / d s_k = T_k c_k                                                         -> a Shade, measured again per parameter
// Shades are many (every covered pixel), and each costs a few field evaluations per parameter, so the gradient takes a
// fixed number of them, drawn by how much they matter (|slope|) and weighted to stay unbiased. Edges are all kept.
#include "loss.hpp"
#include "dino.hpp"
#include <raymath.h>
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <functional>
#include <random>
#include <thread>
#include <unordered_map>

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

        Quaternion Conjugate(Quaternion q) { return { -q.x, -q.y, -q.z, q.w }; }

        // A pixel's layers: the bodies its ray reaches, nearest first (at most this many are kept).
        constexpr int MaxLayers = 4;
        struct Layer {
            int field;
            float c, s, depth;
            float along;                         // d distance / d t at q (< 0: the ray enters there); 0: an edge
            Vector3 at;                          // where the ray comes closest (the edge's point)
            Vector3 q;                           // the point shaded
        };
    }

    // ---- setting up ---------------------------------------------------------------------------------------------------

    void Loss::SetUpSemantic(const TriMesh& mesh) {
        m_Dino = std::make_unique<Dino>(m_P.onnxRuntime, m_P.semanticModel, m_P.semanticGpu);
        if (!m_Dino->Ok()) {
            m_Error = "Semantic loss off: " + m_Dino->Error();
            m_Dino.reset();
            return;
        }
        m_SemRes = m_Dino->Size();
        m_SemTau = 2.0f * m_Half / m_SemRes;
        const int S = m_SemRes, nv = (int)m_Views.size();

        // smooth normals at the vertices (welded by position: a mesh is often split along its seams), so the mesh
        // shades as smoothly as the bodies' fields do
        struct Key {
            float x, y, z;
            bool operator==(const Key& o) const { return x == o.x && y == o.y && z == o.z; }
        };
        struct Hash {
            size_t operator()(const Key& k) const {
                return std::hash<float>()(k.x) * 73856093u ^ std::hash<float>()(k.y) * 19349663u ^ std::hash<float>()(k.z) * 83492791u;
            }
        };
        std::unordered_map<Key, int, Hash> weld;
        std::vector<int> vertexOf(mesh.positions.size());
        for (size_t i = 0; i < mesh.positions.size(); i++) {
            Key k = { mesh.positions[i].x, mesh.positions[i].y, mesh.positions[i].z };
            vertexOf[i] = weld.emplace(k, (int)weld.size()).first->second;
        }
        std::vector<Vector3> normal(weld.size(), { 0, 0, 0 });
        for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
            Vector3 a = mesh.positions[mesh.indices[t]], b = mesh.positions[mesh.indices[t + 1]], c = mesh.positions[mesh.indices[t + 2]];
            Vector3 n = Vector3CrossProduct(Vector3Subtract(b, a), Vector3Subtract(c, a));     // area weighted
            for (int k = 0; k < 3; k++) normal[vertexOf[mesh.indices[t + k]]] = Vector3Add(normal[vertexOf[mesh.indices[t + k]]], n);
        }

        // each view: triangles depth buffered at 3x3 samples a pixel, each sample shaded, averaged
        const int ss = 3, big = S * ss;
        m_MeshLook.assign(nv, std::vector<float>((size_t)S * S, 0.0f));
        ParallelFor(nv, [&](int v) {
            const View& w = m_Views[v];
            std::vector<float> depth((size_t)big * big, FLT_MAX);
            std::vector<Vector3> norm((size_t)big * big, { 0, 0, 0 });
            auto project = [&](Vector3 p) {
                Vector3 q = Vector3Subtract(p, m_Centre);
                return Vector3{ (Vector3DotProduct(q, w.u) / m_Half + 1.0f) * 0.5f * big, (1.0f - Vector3DotProduct(q, w.v) / m_Half) * 0.5f * big,
                                Vector3DotProduct(q, w.dir) };
            };
            for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
                Vector3 P[3], N[3];
                for (int k = 0; k < 3; k++) {
                    P[k] = project(mesh.positions[mesh.indices[t + k]]);
                    N[k] = normal[vertexOf[mesh.indices[t + k]]];
                }
                float area = (P[1].x - P[0].x) * (P[2].y - P[0].y) - (P[1].y - P[0].y) * (P[2].x - P[0].x);
                if (fabsf(area) < 1e-12f) continue;
                int x0 = std::max(0, (int)floorf(fminf(P[0].x, fminf(P[1].x, P[2].x)))), x1 = std::min(big - 1, (int)ceilf(fmaxf(P[0].x, fmaxf(P[1].x, P[2].x))));
                int y0 = std::max(0, (int)floorf(fminf(P[0].y, fminf(P[1].y, P[2].y)))), y1 = std::min(big - 1, (int)ceilf(fmaxf(P[0].y, fmaxf(P[1].y, P[2].y))));
                for (int y = y0; y <= y1; y++)
                    for (int x = x0; x <= x1; x++) {
                        float px = x + 0.5f, py = y + 0.5f;
                        float b0 = ((P[1].x - px) * (P[2].y - py) - (P[1].y - py) * (P[2].x - px)) / area;
                        float b1 = ((P[2].x - px) * (P[0].y - py) - (P[2].y - py) * (P[0].x - px)) / area;
                        float b2 = 1.0f - b0 - b1;
                        if (b0 < 0.0f || b1 < 0.0f || b2 < 0.0f) continue;
                        float z = b0 * P[0].z + b1 * P[1].z + b2 * P[2].z;
                        size_t k = (size_t)y * big + x;
                        if (z >= depth[k]) continue;
                        depth[k] = z;
                        norm[k] = Vector3Add(Vector3Scale(N[0], b0), Vector3Add(Vector3Scale(N[1], b1), Vector3Scale(N[2], b2)));
                    }
            }
            const float bg = Background();
            for (int y = 0; y < big; y++)
                for (int x = 0; x < big; x++) {
                    size_t k = (size_t)y * big + x;
                    float value = bg;
                    if (depth[k] < FLT_MAX) {
                        float sx = ((x + 0.5f) / big * 2.0f - 1.0f) * m_Half, sy = (1.0f - (y + 0.5f) / big * 2.0f) * m_Half;
                        Vector3 p = Vector3Add(m_Centre, Vector3Add(Vector3Add(Vector3Scale(w.u, sx), Vector3Scale(w.v, sy)), Vector3Scale(w.dir, depth[k])));
                        Vector3 n = Vector3Normalize(norm[k]);
                        if (Vector3DotProduct(n, w.dir) > 0.0f) n = Vector3Negate(n);    // the side facing the view
                        value = Look(w, p, n);
                    }
                    m_MeshLook[v][(size_t)(y / ss) * S + x / ss] += value / (ss * ss);
                }
        });
    }

    std::string Loss::SemanticDevice() const { return m_Dino ? m_Dino->Device() : ""; }

    // ---- looks ------------------------------------------------------------------------------------------------------------

    float Loss::Background() const { return m_P.semanticLook == SemanticLook::Shaded ? 0.05f : 0.0f; }

    float Loss::Look(const View& w, Vector3 p, Vector3 n) const {
        switch (m_P.semanticLook) {
        case SemanticLook::Silhouette:
            return 1.0f;
        case SemanticLook::Depth:                         // the frame's near side bright, its far side dim
            return std::clamp(0.5f - Vector3DotProduct(Vector3Subtract(p, m_Centre), w.dir) / (2.4f * m_Half), 0.05f, 1.0f);
        case SemanticLook::Shaded:
        default: {
            // a key light from the viewer's upper left and a dimmer fill from the right, on a light grey material
            static const float key[3] = { -0.6f, 0.8f, 1.0f }, fill[3] = { 0.9f, -0.2f, 1.0f };
            auto toward = [&](const float* l) {
                return Vector3Normalize(Vector3Add(Vector3Add(Vector3Scale(w.u, l[0]), Vector3Scale(w.v, l[1])), Vector3Scale(w.dir, -l[2])));
            };
            float k = fmaxf(0.0f, Vector3DotProduct(n, toward(key))), f = fmaxf(0.0f, Vector3DotProduct(n, toward(fill)));
            return std::clamp(0.8f * (0.12f + 0.75f * k + 0.3f * f), 0.0f, 1.0f);
        }
        }
    }

    // Measured across a pixel, which blurs creases (where unions and cuts meet) over about that much: a crease's shading
    // then moves smoothly as it does, instead of jumping from pixel to pixel, which no gradient would follow.
    Vector3 Loss::Normal(const BodyField& f, Vector3 p) const {
        const float h = m_SemTau;
        Vector3 g = { f.Distance({ p.x + h, p.y, p.z }) - f.Distance({ p.x - h, p.y, p.z }),
                      f.Distance({ p.x, p.y + h, p.z }) - f.Distance({ p.x, p.y - h, p.z }),
                      f.Distance({ p.x, p.y, p.z + h }) - f.Distance({ p.x, p.y, p.z - h }) };
        float l = Vector3Length(g);
        return l > 1e-12f ? Vector3Scale(g, 1.0f / l) : Vector3{ 0, 0, 1 };
    }

    // A shaded point again, with the body's field or pose changed: slid along its ray back onto the surface (one Newton
    // step from where it was: the distance there over the distance's slope along the ray), then shaded.
    float Loss::ShadeAt(const BodyField& f, const Rigid& xf, const Shade& sh) const {
        const View& w = m_Views[sh.view];
        Vector3 p = xf.ApplyInverse(sh.at);
        Vector3 dir = Vector3RotateByQuaternion(w.dir, Conjugate(xf.rotation));
        if (sh.along < -1e-3f) {
            float t = std::clamp(-f.Distance(p) / sh.along, -2.0f * m_SemTau, 2.0f * m_SemTau);
            p = Vector3Add(p, Vector3Scale(dir, t));
        }
        Vector3 n = m_P.semanticLook == SemanticLook::Shaded ? Vector3RotateByQuaternion(Normal(f, p), xf.rotation) : Vector3{ 0, 0, 1 };
        return Look(w, xf.Apply(p), n);
    }

    // The sphere's picture, the way the bodies are drawn: the soft edge from how near each ray comes to it, the shade of
    // where the ray enters it (or comes closest, past its edge).
    void Loss::SphereLook(const View& w, Vector3 centre, float* out) const {
        const int S = m_SemRes;
        const float R = m_SphereRadius, bg = Background();
        Vector3 c = Vector3Subtract(centre, m_Centre);
        float cx = Vector3DotProduct(c, w.u), cy = Vector3DotProduct(c, w.v);
        for (int y = 0; y < S; y++)
            for (int x = 0; x < S; x++) {
                float dx = FrameX(x, S) - cx, dy = FrameY(y, S) - cy, r = sqrtf(dx * dx + dy * dy);
                float cover = std::clamp(0.5f - (r - R) / (2.0f * m_SemTau), 0.0f, 1.0f);
                float value = bg;
                if (cover > 0.0f) {
                    Vector3 side = Vector3Add(Vector3Scale(w.u, dx), Vector3Scale(w.v, dy)), q, n;
                    if (r < R) {
                        n = Vector3Add(Vector3Scale(side, 1.0f / R), Vector3Scale(w.dir, -sqrtf(fmaxf(0.0f, R * R - r * r)) / R));
                        q = Vector3Add(centre, Vector3Scale(n, R));
                    } else {
                        n = Vector3Scale(side, 1.0f / fmaxf(r, 1e-9f));
                        q = Vector3Add(centre, Vector3Scale(n, R));
                    }
                    value = cover * Look(w, q, n) + (1.0f - cover) * bg;
                }
                out[(size_t)y * S + x] = value;
            }
    }

    // ---- the loss ---------------------------------------------------------------------------------------------------------

    void Loss::Semantics(const Bodies& b, AutoScore& s, Slopes& slopes, Shades& shades) {
        const int S = m_SemRes, nv = (int)m_Views.size(), px = S * S;
        const float bias = std::clamp(m_P.sphereBias, 0.0f, 1.0f), bg = Background(), tau = m_SemTau;
        const float poseWeight[2] = { m_P.semanticWeight * (1.0f - bias) / nv, m_P.semanticWeight * bias / nv };
        m_SemSphereGrad = { 0, 0, 0 };
        std::vector<Layer> layers;
        std::vector<unsigned char> count;
        struct Candidate {
            int field;
            Shade shade;
        };
        std::vector<Candidate> candidates;
        std::vector<float> gRender(px), gTarget(px), plus(px), minus(px);
        s.worstSemantic = 0.0f;

        for (int pose = 0; pose < 2; pose++) {
            m_SemDrawn[pose].clear();
            m_SemTarget[pose].clear();
            if (poseWeight[pose] <= 0.0f) continue;              // this pose doesn't count: not drawn
            m_SemDrawn[pose].assign((size_t)nv * px, bg);
            m_SemTarget[pose].assign((size_t)nv * px, bg);
            layers.assign((size_t)nv * px * MaxLayers, Layer{});
            count.assign((size_t)nv * px, 0);

            // drawing: every row of every view
            ParallelFor(nv * S, [&](int task) {
                int v = task / S, y = task % S;
                const View& w = m_Views[v];
                for (int x = 0; x < S; x++) {
                    size_t k = (size_t)v * px + (size_t)y * S + x;
                    Layer* L = &layers[k * MaxLayers];
                    int n = 0;
                    for (size_t i = 0; i < b.fields.size(); i++) {
                        const Rigid& xf = b.xf[pose][b.group[i]];
                        Vector3 at, hit = { NAN, NAN, NAN };
                        float dmin = Trace(b.fields[i], b.box[i], xf, w, FrameX(x, S), FrameY(y, S), tau, &at, &hit);
                        float c = std::clamp(0.5f - dmin / (2.0f * tau), 0.0f, 1.0f);
                        if (c <= 0.0f) continue;
                        bool enters = !std::isnan(hit.x);
                        Layer l;
                        l.field = (int)i;
                        l.c = c;
                        l.at = at;
                        l.q = enters ? hit : at;
                        Vector3 ql = xf.ApplyInverse(l.q), nl = Normal(b.fields[i], ql), nw = Vector3RotateByQuaternion(nl, xf.rotation);
                        if (enters) {       // the distance's slope along the ray, measured (fields needn't be exact distances)
                            Vector3 dl = Vector3Scale(Vector3RotateByQuaternion(w.dir, Conjugate(xf.rotation)), 0.25f * tau);
                            l.along = fminf((b.fields[i].Distance(Vector3Add(ql, dl)) - b.fields[i].Distance(Vector3Subtract(ql, dl))) / (0.5f * tau), -1e-3f);
                        } else
                            l.along = 0.0f;
                        l.s = Look(w, l.q, nw);
                        l.depth = Vector3DotProduct(Vector3Subtract(l.q, m_Centre), w.dir);
                        // nearest first; past MaxLayers the farthest is dropped
                        if (n < MaxLayers) L[n++] = l;
                        else if (l.depth < L[n - 1].depth) L[n - 1] = l;
                        else continue;
                        for (int j = n - 1; j > 0 && L[j].depth < L[j - 1].depth; j--) std::swap(L[j], L[j - 1]);
                    }
                    count[k] = (unsigned char)n;
                    float T = 1.0f, value = 0.0f;
                    for (int j = 0; j < n; j++) {
                        value += T * L[j].c * L[j].s;
                        T *= 1.0f - L[j].c;
                    }
                    m_SemDrawn[pose][k] = value + T * bg;
                }
            });

            // each view against its target, by DINOv3; its slopes back through the pixels
            double sum = 0.0;
            for (int v = 0; v < nv; v++) {
                float* target = m_SemTarget[pose].data() + (size_t)v * px;
                if (pose == 0) std::copy(m_MeshLook[v].begin(), m_MeshLook[v].end(), target);
                else SphereLook(m_Views[v], m_SphereCentre, target);
                float loss = 0.0f;
                if (!m_Dino->Run(m_SemDrawn[pose].data() + (size_t)v * px, target, std::clamp(m_P.patchWeight, 0.0f, 1.0f), &loss,
                                 gRender.data(), pose == 1 ? gTarget.data() : nullptr)) {
                    m_Error = m_Dino->Error();
                    continue;
                }
                sum += loss;
                s.worstSemantic = fmaxf(s.worstSemantic, loss);

                for (int p = 0; p < px; p++) {
                    float g = poseWeight[pose] * gRender[p];
                    if (g == 0.0f) continue;
                    size_t k = (size_t)v * px + p;
                    const Layer* L = &layers[k * MaxLayers];
                    int n = count[k];
                    // behind[j]: what shows behind layer j; front[j]: how much of layer j shows
                    float behind[MaxLayers], front[MaxLayers];
                    float B = bg;
                    for (int j = n - 1; j >= 0; j--) {
                        behind[j] = B;
                        B = L[j].c * L[j].s + (1.0f - L[j].c) * B;
                    }
                    float T = 1.0f;
                    for (int j = 0; j < n; j++) {
                        front[j] = T;
                        T *= 1.0f - L[j].c;
                    }
                    for (int j = 0; j < n; j++) {
                        if (L[j].c < 1.0f)          // the soft edge: d c / d distance = -1 / (2 tau)
                            slopes[L[j].field].push_back({ pose, L[j].at, g * front[j] * (L[j].s - behind[j]) * (-1.0f / (2.0f * tau)) });
                        if (m_P.semanticLook != SemanticLook::Silhouette)
                            candidates.push_back({ L[j].field, { pose, v, L[j].q, L[j].along, g * front[j] * L[j].c } });
                    }
                }
                // the sphere: its picture moves with its centre (a quarter pixel either way)
                if (pose == 1) {
                    const float h = 0.25f * tau;
                    for (int a = 0; a < 3; a++) {
                        Vector3 e = { 0, 0, 0 };
                        (&e.x)[a] = h;
                        SphereLook(m_Views[v], Vector3Add(m_SphereCentre, e), plus.data());
                        SphereLook(m_Views[v], Vector3Subtract(m_SphereCentre, e), minus.data());
                        double d = 0.0;
                        for (int p = 0; p < px; p++) d += gTarget[p] * (plus[p] - minus[p]);
                        (&m_SemSphereGrad.x)[a] += poseWeight[pose] * (float)(d / (2.0 * h));
                    }
                }
            }
            (pose == 0 ? s.semanticOpen : s.semanticClosed) = (float)(sum / nv);
        }
        s.semantic = (1.0f - bias) * s.semanticOpen + bias * s.semanticClosed;

        // the shades the gradient measures: all of them if few, else semanticSamples drawn by |slope| (each then
        // stands for total |slope| / samples, with its own sign)
        const size_t want = (size_t)std::max(0, m_P.semanticSamples);
        if (candidates.size() <= want) {
            for (const Candidate& c : candidates) shades[c.field].push_back(c.shade);
        } else if (want > 0) {
            std::vector<double> cdf(candidates.size());
            double total = 0.0;
            for (size_t i = 0; i < candidates.size(); i++) cdf[i] = total += fabs(candidates[i].shade.weight);
            if (total > 0.0) {
                std::mt19937 rng((unsigned)(candidates.size() * 2654435761u) ^ m_P.seed);
                std::uniform_real_distribution<double> u(0.0, total);
                for (size_t k = 0; k < want; k++) {
                    size_t i = std::min((size_t)(std::lower_bound(cdf.begin(), cdf.end(), u(rng)) - cdf.begin()), candidates.size() - 1);
                    Shade sh = candidates[i].shade;
                    sh.weight = (float)(sh.weight >= 0.0f ? total / want : -total / want);
                    shades[candidates[i].field].push_back(sh);
                }
            }
        }
    }

    // ---- the picture for the UI ---------------------------------------------------------------------------------------

    void Loss::SemanticPicture(std::vector<unsigned char>* rgba, int* w, int* h) const {
        const int nv = (int)m_Views.size(), S = m_SemRes, px = S * S, gap = 3;
        const std::vector<float>* rows[4] = { &m_SemTarget[0], &m_SemDrawn[0], &m_SemTarget[1], &m_SemDrawn[1] };
        int shown = 0;
        for (const auto* r : rows) shown += r->size() == (size_t)nv * px;
        if (!m_Dino || shown == 0) { *w = *h = 0; rgba->clear(); return; }
        *w = nv * S + (nv + 1) * gap;
        *h = shown * S + (shown + 1) * gap;
        rgba->assign((size_t)*w * *h * 4, 255);
        for (size_t k = 0; k < rgba->size(); k += 4) { (*rgba)[k] = 222; (*rgba)[k + 1] = 225; (*rgba)[k + 2] = 232; }
        int row = 0;
        for (const auto* r : rows) {
            if (r->size() != (size_t)nv * px) continue;
            for (int v = 0; v < nv; v++)
                for (int k = 0; k < px; k++) {
                    unsigned char g = (unsigned char)std::clamp(lroundf((*r)[(size_t)v * px + k] * 255.0f), 0L, 255L);
                    unsigned char* out = &(*rgba)[((size_t)(gap + row * (S + gap) + k / S) * *w + gap + v * (S + gap) + k % S) * 4];
                    out[0] = out[1] = out[2] = g;
                }
            row++;
        }
    }
}
