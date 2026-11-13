// The transport part of the loss (transportWeight): how far the sphere's material has to travel, opened, to fill the
// placed mesh. Optimal transport (the earth mover's distance) between the two volumes.
//
// Why: the carving can't grow or shrink, only rearrange the sphere's material (less its hollows), so the question is
// where that material goes. Silhouettes, the Chamfer distance and the SDF only see mistakes near the mesh; a piece
// opened to the wrong place, far from where it's needed, gets no pull from them. Transport pairs every bit of material
// with a bit of the mesh it should fill, however far away, and pulls it there.
//
// The material: fixed points spread evenly through the sphere (in its frame). Each belongs to the piece it's inside (a
// point in a hollow: to none), equal mass each, and moves with its piece's open pose: rigidly, so there's nothing to
// sample again. The mesh: fixed points filling it (its winding number over 1/2), equal mass each, moved with the
// placement. Both add up to 1: transport compares where the material is, not how much of it.
//
// The cost is entropic unbalanced optimal transport (Sinkhorn, in the log domain, warm started from the last
// evaluation) with squared distances, over 0.1 of the mesh's size squared:
//   OT = min over plans P of sum P_kj |x_k - y_j|^2 + rho KL(P 1 | a) + rho KL(P^T 1 | b)  (+ a little entropy, eps)
// Its gradient comes from the plan held fixed (the envelope theorem):
//   - each material point:  2 (sum_j P_kj) (x_k - T_k), T_k the mesh points it's sent to (their mean, weighted by the plan);
//     it moves with its piece's pose, so this is what the joints' parameters see (Anchor.grad)
//   - each mesh point: 2 (sum_k P_kj) (y_j - S_j), likewise; the placement's offset and scale move them (exact, in Evaluate)
//   - a cutter moves material from one piece to another along its face. Mass added at x changes OT by the dual
//     potential phi(x) there. Near a piece's surface its share of a point is
//     taken as soft, 1/2 - D / (2 tau), so the slope against its distance there is -phi(x) a / (2 tau), x
//     where the point would be if it were that piece's (Anchor.massSlope).
#include "loss.hpp"
#include "util/box.hpp"
#include <raymath.h>
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <functional>
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

        double WindingAt(const TriMesh& m, Vector3 p) {
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

        // log(sum exp(v)), steady for large and very negative v
        double LogSumExp(const float* v, int n, int stride = 1) {
            float m = -FLT_MAX;
            for (int i = 0; i < n; i++) m = fmaxf(m, v[(size_t)i * stride]);
            if (m == -FLT_MAX) return -FLT_MAX;
            double s = 0.0;
            for (int i = 0; i < n; i++) s += expf(v[(size_t)i * stride] - m);
            return m + log(s);
        }
    }

    // ---- setting up: points through the sphere, points filling the mesh ---------------------------------------------------

    void Loss::SetUpTransport(const TriMesh& mesh) {
        const int n = std::clamp(m_P.transportPoints, 128, 8192);
        std::mt19937 rng(m_P.seed * 977u + 11u);
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);
        // the sphere's: evenly through the unit ball (stratified, then jittered), in its frame
        m_BallUnit.clear();
        while ((int)m_BallUnit.size() < n) {
            Vector3 p = { 2.0f * u01(rng) - 1.0f, 2.0f * u01(rng) - 1.0f, 2.0f * u01(rng) - 1.0f };
            if (Vector3LengthSqr(p) <= 1.0f) m_BallUnit.push_back(p);
        }
        // the mesh's: inside it, where it is (placements move them)
        Vector3 lo = { FLT_MAX, FLT_MAX, FLT_MAX }, hi = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
        for (unsigned i : mesh.indices) { lo = Vector3Min(lo, mesh.positions[i]); hi = Vector3Max(hi, mesh.positions[i]); }
        m_MeshFill.clear();
        for (int round = 0; round < 40 && (int)m_MeshFill.size() < n; round++) {
            std::vector<Vector3> cand(n);
            for (Vector3& c : cand) c = { lo.x + u01(rng) * (hi.x - lo.x), lo.y + u01(rng) * (hi.y - lo.y), lo.z + u01(rng) * (hi.z - lo.z) };
            std::vector<char> in(n, 0);
            ParallelFor(n, [&](int i) { in[i] = fabs(WindingAt(mesh, cand[i])) > 0.5; });
            for (int i = 0; i < n && (int)m_MeshFill.size() < n; i++)
                if (in[i]) m_MeshFill.push_back(cand[i]);
        }
        m_TransportG.assign(m_MeshFill.size(), 0.0f);
        m_TransportWarm = false;
    }

    // ---- the loss -------------------------------------------------------------------------------------------------------

    float Loss::Transport(const Carving& c, const Bodies& b, const Placement& t, AutoScore& s, Anchors* anchors, Vector3* gradOffset,
                          float* gradLogScale) {
        if (gradOffset) *gradOffset = { 0, 0, 0 };
        if (gradLogScale) *gradLogScale = 0.0f;
        const int nb = (int)m_BallUnit.size(), m = (int)m_MeshFill.size(), nf = (int)b.fields.size();
        if (nb == 0 || m == 0 || nf == 0) return 0.0f;
        const float norm = 1.0f / (0.01f * m_L * m_L);            // over (0.1 L)^2
        const float eps = 0.0025f * m_L * m_L;                     // (0.05 L)^2
        const float tau = 0.02f * m_L;                             // how soft a piece's share of a point is, near its surface

        // the material: which piece each sphere point is in (the deepest), and its distance to every piece near it
        std::vector<Vector3> local(nb);
        std::vector<int> owner(nb, -1);
        std::vector<float> dist((size_t)nb * nf);
        ParallelFor((nb + 63) / 64, [&](int chunk) {
            for (int k = chunk * 64; k < std::min(nb, chunk * 64 + 64); k++) {
                local[k] = Vector3Add(c.centre, Vector3Scale(m_BallUnit[k], c.radius));
                float best = 0.0f;
                for (int i = 0; i < nf; i++) {
                    float d = InBox(local[k], Grow(b.box[i], tau)) ? b.fields[i].Distance(local[k]) : FLT_MAX;
                    dist[(size_t)k * nf + i] = d;
                    if (d < best) { best = d; owner[k] = i; }
                }
            }
        });
        std::vector<int> owned;                                    // sphere point index, for each material point
        std::vector<Vector3> X;
        for (int k = 0; k < nb; k++)
            if (owner[k] >= 0) {
                owned.push_back(k);
                X.push_back(b.xf[0][b.group[owner[k]]].Apply(local[k]));
            }
        const int n = (int)X.size();
        if (n == 0) return 0.0f;
        std::vector<Vector3> Y(m);
        const Vector3 base = Vector3Add(m_MeshCentre, t.offset);
        for (int j = 0; j < m; j++) Y[j] = Vector3Add(base, Vector3Scale(Vector3Subtract(m_MeshFill[j], m_MeshCentre), t.scale));

        // the masses, in shares of the sphere's volume: a material point's is its share of the sphere; a mesh point's,
        // its share of the placed mesh's volume, over the sphere's. So a mesh with less room than the sphere's material
        // can't take all of it, and material taken away by hollows leaves some of the mesh unfilled: both cost.
        const double sphereVolume = 4.0 / 3.0 * PI * pow(c.radius, 3.0);
        const double meshShare = m_MeshVolume > 0.0f ? m_MeshVolume * pow(t.scale, 3.0) / sphereVolume : (double)n / nb;
        const float a = 1.0f / nb, bm = (float)(meshShare / m);
        const double A = (double)n * a, B = meshShare;
        const float logA = logf(a), logB = logf(bm);

        // unbalanced Sinkhorn, in the log domain: f (material), g (mesh). Leaving mass unmatched costs rho per unit
        // (relative entropy on both sides); rho is the mesh's size squared, so moving material is always cheaper than
        // dropping it, and only a real difference in amount goes unmatched.
        const float rho = m_L * m_L, kappa = rho / (rho + eps);
        std::vector<float> C((size_t)n * m), f(n, 0.0f), work((size_t)n * m);
        ParallelFor(n, [&](int k) {
            for (int j = 0; j < m; j++) C[(size_t)k * m + j] = Vector3DistanceSqr(X[k], Y[j]);
        });
        std::vector<float>& g = m_TransportG;
        if ((int)g.size() != m) g.assign(m, 0.0f);
        const int iterations = m_TransportWarm ? 40 : 200;
        auto updateF = [&]() {
            ParallelFor(n, [&](int k) {
                float* row = &work[(size_t)k * m];
                for (int j = 0; j < m; j++) row[j] = (g[j] - C[(size_t)k * m + j]) / eps + logB;
                f[k] = (float)(-kappa * eps * LogSumExp(row, m));
            });
        };
        auto updateG = [&]() {
            ParallelFor(m, [&](int j) {
                std::vector<float> col(n);
                for (int k = 0; k < n; k++) col[k] = (f[k] - C[(size_t)k * m + j]) / eps + logA;
                g[j] = (float)(-kappa * eps * LogSumExp(col.data(), n));
            });
        };
        // With rho large, plain unbalanced Sinkhorn settles the balance between the two sides (a constant shared by
        // f and g) very slowly: about eps / rho of it a round. So after each round the best such shift is made
        // exactly (Sejourne et al. 2022, translation invariant Sinkhorn): f + l, g - l, l = rho/2 log(<a, e^(-f/rho)> /
        // <b, e^(-g/rho)>), which maximizes the dual over l.
        auto balance = [&]() {
            double fa = 0.0, gb = 0.0;
            for (int k = 0; k < n; k++) fa += a * exp(-f[k] / rho);
            for (int j = 0; j < m; j++) gb += bm * exp(-g[j] / rho);
            if (fa <= 0.0 || gb <= 0.0) return;
            float l = (float)(0.5 * rho * log(fa / gb));
            for (float& v : f) v += l;
            for (float& v : g) v -= l;
        };
        for (int it = 0; it < iterations; it++) {
            updateF();
            balance();
            updateG();
            balance();
        }
        updateF();
        m_TransportWarm = true;

        // the plan P_kj = a b exp((f_k + g_j - C_kj) / eps): where each material point goes (T), where each mesh point's
        // material comes from (S), how much each sends and gets, and the cost
        std::vector<Vector3> T(n), S(m, { 0, 0, 0 });
        std::vector<double> rowMass(n, 0.0), colMass(m, 0.0), rowCost(n, 0.0);
        ParallelFor(n, [&](int k) {
            Vector3 acc = { 0, 0, 0 };
            double mass = 0.0, cost = 0.0;
            for (int j = 0; j < m; j++) {
                float pk = expf((f[k] + g[j] - C[(size_t)k * m + j]) / eps + logA + logB);
                work[(size_t)k * m + j] = pk;
                acc = Vector3Add(acc, Vector3Scale(Y[j], pk));
                mass += pk;
                cost += pk * C[(size_t)k * m + j];
            }
            T[k] = mass > 0.0 ? Vector3Scale(acc, (float)(1.0 / mass)) : X[k];
            rowMass[k] = mass;
            rowCost[k] = cost;
        });
        double sent = 0.0, moved = 0.0;
        for (int k = 0; k < n; k++) { sent += rowMass[k]; moved += rowCost[k]; }
        for (int k = 0; k < n; k++)
            for (int j = 0; j < m; j++) {
                float p = work[(size_t)k * m + j];
                S[j] = Vector3Add(S[j], Vector3Scale(X[k], p));
                colMass[j] += p;
            }
        // the dual's value: rho <a, 1 - e^(-f/rho)> + rho <b, 1 - e^(-g/rho)> - eps (sum P - A B)
        double dual = 0.0;
        for (int k = 0; k < n; k++) dual += rho * a * (1.0 - exp(-f[k] / rho));
        for (int j = 0; j < m; j++) dual += rho * bm * (1.0 - exp(-g[j] / rho));
        dual -= eps * (sent - A * B);
        s.transport = (float)(dual * norm);
        s.transportMove = sent > 0.0 ? (float)sqrt(fmax(0.0, moved / sent)) : 0.0f;
        s.transportUnfilled = B > 0.0 ? (float)std::clamp(1.0 - sent / B, 0.0, 1.0) : 0.0f;
        const float w = std::max(0.0f, m_P.transportWeight);

        // the placement: each mesh point's slope, 2 (sum_k P_kj) (y_j - S_j); and through the scale, its mass too
        // (b grows with the scale cubed; d dual / d b_j = rho (1 - e^(-g/rho)) - eps (sum_k P_kj / b_j - A))
        if (gradOffset || gradLogScale)
            for (int j = 0; j < m; j++) {
                Vector3 src = colMass[j] > 1e-20 ? Vector3Scale(S[j], (float)(1.0 / colMass[j])) : Y[j];
                Vector3 h = Vector3Scale(Vector3Subtract(Y[j], src), (float)(2.0 * colMass[j]) * w * norm);
                if (gradOffset) *gradOffset = Vector3Add(*gradOffset, h);
                if (gradLogScale) {
                    double psi = rho * (1.0 - exp(-g[j] / rho)) - eps * (colMass[j] / bm - A);
                    *gradLogScale += Vector3DotProduct(h, Vector3Scale(Vector3Subtract(m_MeshFill[j], m_MeshCentre), t.scale)) +
                                     (float)(3.0 * bm * psi) * w * norm;
                }
            }
        m_Grains.clear();
        for (int q = 0; q < n; q++) m_Grains.push_back({ owner[owned[q]], local[owned[q]], X[q], T[q] });
        if (!anchors || w <= 0.0f) return w * s.transport;

        // the material's slopes: moved by its piece's pose ...
        for (int q = 0; q < n; q++) {
            Vector3 grad = Vector3Scale(Vector3Subtract(X[q], T[q]), (float)(2.0 * rowMass[q]) * w * norm);
            anchors->push_back({ owner[owned[q]], local[owned[q]], grad, 0.0f });
        }
        // ... and handed from piece to piece by the cutters (or taken away by hollows): mass a added at x changes the
        // dual by a phi(x), phi(x) = rho (1 - e^(-f(x)/rho)) - eps (e^(-f(x)/rho) - B), f(x) its potential there
        auto phi = [&](Vector3 x) {
            std::vector<float> v(m);
            for (int j = 0; j < m; j++) v[j] = (g[j] - Vector3DistanceSqr(x, Y[j])) / eps + logB;
            double fx = -kappa * eps * LogSumExp(v.data(), m), e = exp(-fx / rho);
            return (float)(rho * (1.0 - e) - eps * (e - B));
        };
        std::vector<std::vector<Anchor>> edge(nb);
        ParallelFor(nb, [&](int k) {
            for (int i = 0; i < nf; i++) {
                float d = dist[(size_t)k * nf + i];
                if (fabsf(d) >= tau) continue;
                Vector3 x = b.xf[0][b.group[i]].Apply(local[k]);
                float slope = -phi(x) * a * w * norm / (2.0f * tau);
                edge[k].push_back({ i, local[k], { 0, 0, 0 }, slope });
            }
        });
        for (auto& e : edge) anchors->insert(anchors->end(), e.begin(), e.end());
        return w * s.transport;
    }
}
