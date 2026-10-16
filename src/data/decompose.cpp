// TBB first: raylib's macros must not leak into its headers.
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include "decompose.hpp"
#include "util/optim.hpp"
#include "util/utils.hpp"
#include <raymath.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace Bakugan {
    namespace {
        using Clock = std::chrono::steady_clock;
        constexpr int NP = SuperFrustum::ParamCount + 3;   // shape params + rotation delta

        // ---- grid helpers -------------------------------------------------------------------------------------

        void Edt1D(const float* f, int n, float* d, int* v, float* z) {
            const float INF = 1e20f;
            int k = 0;
            v[0] = 0;
            z[0] = -INF;
            z[1] = INF;
            for (int q = 1; q < n; q++) {
                float s = ((f[q] + (float)q * q) - (f[v[k]] + (float)v[k] * v[k])) / (2.0f * q - 2.0f * v[k]);
                while (s <= z[k] && k > 0) {
                    k--;
                    s = ((f[q] + (float)q * q) - (f[v[k]] + (float)v[k] * v[k])) / (2.0f * q - 2.0f * v[k]);
                }
                if (s <= z[k]) { v[k] = q; z[k] = -INF; z[k + 1] = INF; continue; }
                k++;
                v[k] = q;
                z[k] = s;
                z[k + 1] = INF;
            }
            k = 0;
            for (int q = 0; q < n; q++) {
                while (z[k + 1] < q) k++;
                float dq = (float)(q - v[k]);
                d[q] = dq * dq + f[v[k]];
            }
        }

        // Euclidean distance (in voxels) from every voxel to the nearest voxel where target != 0.
        std::vector<float> DistanceTo(const std::vector<uint8_t>& target, const int dim[3]) {
            size_t n = target.size();
            std::vector<float> d(n);
            for (size_t i = 0; i < n; i++) d[i] = target[i] ? 0.0f : 1e20f;
            int maxn = std::max(dim[0], std::max(dim[1], dim[2]));
            std::vector<float> f(maxn), out(maxn), z(maxn + 1);
            std::vector<int> v(maxn);
            size_t stride[3] = { 1, (size_t)dim[0], (size_t)dim[0] * dim[1] };
            for (int a = 0; a < 3; a++) {
                int b = (a + 1) % 3, c = (a + 2) % 3;
                for (int jc = 0; jc < dim[c]; jc++)
                    for (int jb = 0; jb < dim[b]; jb++) {
                        size_t base = jb * stride[b] + jc * stride[c];
                        for (int i = 0; i < dim[a]; i++) f[i] = d[base + i * stride[a]];
                        Edt1D(f.data(), dim[a], out.data(), v.data(), z.data());
                        for (int i = 0; i < dim[a]; i++) d[base + i * stride[a]] = out[i];
                    }
            }
            for (float& x : d) x = sqrtf(x);
            return d;
        }

        // 6-connected component of `mask` containing `seed`.
        std::vector<int> Component(const std::vector<uint8_t>& mask, const int dim[3], int seed, std::vector<uint8_t>& visited) {
            std::vector<int> out, stack = { seed };
            visited[seed] = 1;
            const int sx = 1, sy = dim[0], sz = dim[0] * dim[1];
            while (!stack.empty()) {
                int c = stack.back();
                stack.pop_back();
                out.push_back(c);
                int i = c % dim[0], j = (c / dim[0]) % dim[1], k = c / sz;
                const int nb[6] = { i > 0 ? c - sx : -1, i < dim[0] - 1 ? c + sx : -1, j > 0 ? c - sy : -1,
                                    j < dim[1] - 1 ? c + sy : -1, k > 0 ? c - sz : -1, k < dim[2] - 1 ? c + sz : -1 };
                for (int n : nb)
                    if (n >= 0 && mask[n] && !visited[n]) { visited[n] = 1; stack.push_back(n); }
            }
            return out;
        }

        // Symmetric 3x3 eigen decomposition (cyclic Jacobi). Columns of vec are eigenvectors.
        void Eigen3(double a[3][3], double val[3], double vec[3][3]) {
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++) vec[i][j] = i == j ? 1.0 : 0.0;
            for (int sweep = 0; sweep < 32; sweep++) {
                double off = fabs(a[0][1]) + fabs(a[0][2]) + fabs(a[1][2]);
                if (off < 1e-14) break;
                for (int p = 0; p < 2; p++)
                    for (int q = p + 1; q < 3; q++) {
                        if (fabs(a[p][q]) < 1e-18) continue;
                        double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                        double t = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                        double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
                        for (int k = 0; k < 3; k++) {
                            double akp = a[k][p], akq = a[k][q];
                            a[k][p] = c * akp - s * akq;
                            a[k][q] = s * akp + c * akq;
                        }
                        for (int k = 0; k < 3; k++) {
                            double apk = a[p][k], aqk = a[q][k];
                            a[p][k] = c * apk - s * aqk;
                            a[q][k] = s * apk + c * aqk;
                        }
                        for (int k = 0; k < 3; k++) {
                            double vkp = vec[k][p], vkq = vec[k][q];
                            vec[k][p] = c * vkp - s * vkq;
                            vec[k][q] = s * vkp + c * vkq;
                        }
                    }
            }
            for (int i = 0; i < 3; i++) val[i] = a[i][i];
        }

        Quaternion BasisToQuaternion(Vector3 x, Vector3 y, Vector3 z) {
            Matrix m = MatrixIdentity();
            m.m0 = x.x; m.m1 = x.y; m.m2 = x.z;
            m.m4 = y.x; m.m5 = y.y; m.m6 = y.z;
            m.m8 = z.x; m.m9 = z.y; m.m10 = z.z;
            return QuaternionNormalize(QuaternionFromMatrix(m));
        }

        // PCA initialisation of a primitive from a set of voxel centres.
        SuperFrustum SeedPrimitive(const std::vector<Vector3>& pts, float voxel) {
            SuperFrustum sf;
            Vector3 mean = { 0, 0, 0 };
            for (const Vector3& p : pts) mean = Vector3Add(mean, p);
            mean = Vector3Scale(mean, 1.0f / pts.size());
            double cov[3][3] = {};
            for (const Vector3& p : pts) {
                double d[3] = { p.x - mean.x, p.y - mean.y, p.z - mean.z };
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++) cov[i][j] += d[i] * d[j];
            }
            double val[3], vec[3][3];
            Eigen3(cov, val, vec);
            int order[3] = { 0, 1, 2 };
            std::sort(order, order + 3, [&](int a, int b) { return val[a] > val[b]; });
            Vector3 e[3];
            for (int i = 0; i < 3; i++) e[i] = Vector3Normalize({ (float)vec[0][order[i]], (float)vec[1][order[i]], (float)vec[2][order[i]] });
            double l0 = val[order[0]], l1 = val[order[1]], l2 = val[order[2]];
            Vector3 z, x;
            if (l0 > 1.6 * l1) { z = e[0]; x = e[1]; }            // elongated: sweep along the long axis
            else if (l1 > 1.6 * l2) { z = e[2]; x = e[0]; }       // flat disc: sweep across the thin axis
            else { z = e[0]; x = e[1]; }
            Vector3 y = Vector3Normalize(Vector3CrossProduct(z, x));
            x = Vector3CrossProduct(y, z);

            // robust extents from projections
            std::vector<float> proj[3];
            Vector3 axes[3] = { x, y, z };
            for (const Vector3& p : pts) {
                Vector3 d = Vector3Subtract(p, mean);
                for (int a = 0; a < 3; a++) proj[a].push_back(Vector3DotProduct(d, axes[a]));
            }
            float lo[3], hi[3];
            for (int a = 0; a < 3; a++) {
                std::vector<float>& v = proj[a];
                size_t i0 = (size_t)(v.size() * 0.01), i1 = std::min(v.size() - 1, (size_t)(v.size() * 0.99));
                std::nth_element(v.begin(), v.begin() + i0, v.end());
                lo[a] = v[i0];
                std::nth_element(v.begin(), v.begin() + i1, v.end());
                hi[a] = v[i1];
            }
            Vector3 centre = mean;
            for (int a = 0; a < 3; a++) centre = Vector3Add(centre, Vector3Scale(axes[a], 0.5f * (lo[a] + hi[a])));
            sf.size = { hi[0] - lo[0] + voxel, hi[1] - lo[1] + voxel, hi[2] - lo[2] + voxel };

            // taper: compare the cross-section counts of the two ends; the narrow end goes to +z
            float zc = 0.5f * (lo[2] + hi[2]), span = hi[2] - lo[2];
            int bottom = 0, top = 0;
            for (float pz : proj[2]) {
                float t = (pz - zc) / fmaxf(span, 1e-6f);
                if (t < -0.25f) bottom++;
                else if (t > 0.25f) top++;
            }
            if (top > bottom) { z = Vector3Negate(z); y = Vector3Negate(y); std::swap(top, bottom); }
            sf.taper = bottom > 0 ? std::clamp(sqrtf((float)top / bottom), 0.2f, 1.0f) : 1.0f;
            if (sf.taper > 0.9f) sf.taper = 1.0f;

            float fill = (float)pts.size() * voxel * voxel * voxel / fmaxf(sf.size.x * sf.size.y * sf.size.z, 1e-9f);
            sf.rounding = std::clamp((1.0f - fill) / (1.0f - PI / 4.0f), 0.0f, 1.0f);
            if (l0 < 1.6 * l2 && fill < 0.76f) {
                // a blob (ball, egg): start from a dilated round core, which is close to an ellipsoid
                float dil = 0.42f * fminf(sf.size.x, fminf(sf.size.y, sf.size.z));
                sf.size = { sf.size.x - 2 * dil, sf.size.y - 2 * dil, sf.size.z - 2 * dil };
                sf.dilation = dil;
                sf.rounding = 1.0f;
                sf.taper = 1.0f;
            }
            sf.position = centre;
            sf.rotation = BasisToQuaternion(x, y, z);
            sf.Repair(voxel * 1.5f);
            return sf;
        }

        // Seeds one primitive per region, but splits regions a single primitive explains badly (rings, L shapes)
        // along their main axis first, so each piece gets a primitive of its own.
        void SeedRegion(const std::vector<Vector3>& pts, float voxel, size_t minVoxels, int depth, std::vector<SuperFrustum>& out) {
            SuperFrustum sf = SeedPrimitive(pts, voxel);
            if (depth < 3 && pts.size() >= 2 * minVoxels) {
                size_t inside = 0;
                for (const Vector3& p : pts) inside += sf.Distance(p) < 0.0f;
                BoundingBox b = sf.WorldBounds();
                size_t volume = 0;
                for (float z = b.min.z; z <= b.max.z; z += voxel)
                    for (float y = b.min.y; y <= b.max.y; y += voxel)
                        for (float x = b.min.x; x <= b.max.x; x += voxel) volume += sf.Distance({ x, y, z }) < 0.0f;
                float iou = (float)inside / (float)std::max<size_t>(1, pts.size() + volume - inside);
                if (iou < 0.66f) {
                    Vector3 mean = { 0, 0, 0 };
                    for (const Vector3& p : pts) mean = Vector3Add(mean, p);
                    mean = Vector3Scale(mean, 1.0f / pts.size());
                    double cov[3][3] = {};
                    for (const Vector3& p : pts) {
                        double d[3] = { p.x - mean.x, p.y - mean.y, p.z - mean.z };
                        for (int i = 0; i < 3; i++)
                            for (int j = 0; j < 3; j++) cov[i][j] += d[i] * d[j];
                    }
                    double val[3], vec[3][3];
                    Eigen3(cov, val, vec);
                    int top = 0;
                    for (int i = 1; i < 3; i++) if (val[i] > val[top]) top = i;
                    Vector3 axis = { (float)vec[0][top], (float)vec[1][top], (float)vec[2][top] };
                    std::vector<Vector3> a, c;
                    for (const Vector3& p : pts) (Vector3DotProduct(Vector3Subtract(p, mean), axis) < 0 ? a : c).push_back(p);
                    if (a.size() >= minVoxels && c.size() >= minVoxels) {
                        SeedRegion(a, voxel, minVoxels, depth + 1, out);
                        SeedRegion(c, voxel, minVoxels, depth + 1, out);
                        return;
                    }
                }
            }
            out.push_back(sf);
        }

        // ---- fitting -------------------------------------------------------------------------------------------

        struct Samples {
            std::vector<Vector3> pos;
            std::vector<float> target;
            std::vector<float> weight;
            float totalWeight = 0;
        };

        Samples MakeSamples(const ShapeField& f) {
            Samples s;
            uint32_t h = 2166136261u;
            auto rnd = [&]() { h ^= h << 13; h ^= h >> 17; h ^= h << 5; return (h & 0xFFFFFF) / (float)0xFFFFFF; };
            for (int k = 0; k < f.dim[2]; k++)
                for (int j = 0; j < f.dim[1]; j++)
                    for (int i = 0; i < f.dim[0]; i++) {
                        float d = f.sdf[f.Index(i, j, k)];
                        float ad = fabsf(d) / f.voxel;
                        float keep = ad < 2.5f ? 1.0f : (d < 0 ? 0.35f : 0.12f);
                        if (rnd() > keep) continue;
                        s.pos.push_back(f.Position(i, j, k));
                        s.target.push_back(d < 0 ? 1.0f : 0.0f);
                        float w = (ad < 1.5f ? 1.5f : 1.0f) / keep;
                        s.weight.push_back(w);
                        s.totalWeight += w;
                    }
            return s;
        }

        void PackAll(const SuperFrustum& sf, float* p) {
            sf.Pack(p);
            p[11] = p[12] = p[13] = 0.0f;
        }

        void UnpackAll(SuperFrustum& sf, const float* p, float minSize) {
            sf.Unpack(p);
            Vector3 w = { p[11], p[12], p[13] };
            float a = Vector3Length(w);
            if (a > 1e-9f) sf.rotation = QuaternionMultiply(QuaternionFromAxisAngle(Vector3Scale(w, 1.0f / a), a), sf.rotation);
            sf.Repair(minSize);
        }

        class Fitter {
        public:
            Fitter(const Samples& s, float voxel) : m_S(s), m_Voxel(voxel), m_Tau(0.6f * voxel) {}

            float Occ(const SuperFrustum& sf, Vector3 p) const { return Sigmoid(-sf.Distance(p) / m_Tau); }

            // Soft union reconstruction loss of a whole set (used for pruning).
            float Loss(const std::vector<SuperFrustum>& shapes) const {
                size_t n = m_S.pos.size();
                std::vector<float> keep(n, 1.0f), sum(n, 0.0f);
                for (const SuperFrustum& sf : shapes) {
                    float r = sf.BoundingRadius() + 4 * m_Tau;
                    for (size_t s = 0; s < n; s++) {
                        if (Vector3DistanceSqr(m_S.pos[s], sf.position) > r * r) continue;
                        float o = Occ(sf, m_S.pos[s]);
                        keep[s] *= 1.0f - o;
                        sum[s] += o;
                    }
                }
                double L = 0;
                for (size_t s = 0; s < n; s++) {
                    float e = (1.0f - keep[s]) - m_S.target[s];
                    L += m_S.weight[s] * (e * e + 0.25f * fmaxf(sum[s] - 1.0f, 0.0f));
                }
                return (float)(L / m_S.totalWeight);
            }

            void Optimize(std::vector<SuperFrustum>& shapes, int iterations, int only, JobControl& c,
                          float p0, float p1, const std::function<void()>& publish) {
                int P = (int)shapes.size();
                if (P == 0) return;
                std::vector<Adam> adam(P);
                size_t n = m_S.pos.size();
                const float steps[NP] = { 0.25f * m_Voxel, 0.25f * m_Voxel, 0.25f * m_Voxel,
                                          0.25f * m_Voxel, 0.25f * m_Voxel, 0.25f * m_Voxel,
                                          0.02f, 0.02f, 0.02f, 0.1f * m_Voxel, 0.02f, 0.01f, 0.01f, 0.01f };
                const float rates[NP] = { 0.35f * m_Voxel, 0.35f * m_Voxel, 0.35f * m_Voxel,
                                          0.35f * m_Voxel, 0.35f * m_Voxel, 0.35f * m_Voxel,
                                          0.03f, 0.03f, 0.02f, 0.08f * m_Voxel, 0.0f, 0.02f, 0.02f, 0.02f };
                std::vector<float> logKeep(n), sum(n);
                for (int it = 0; it < iterations && !c.Cancelled(); it++) {
                    float decay = 0.2f + 0.8f * 0.5f * (1.0f + cosf(PI * it / std::max(1, iterations)));
                    std::vector<std::vector<int>> local(P);
                    std::vector<std::vector<float>> occ(P);
                    tbb::parallel_for(0, P, [&](int i) {
                        const SuperFrustum& sf = shapes[i];
                        float r = sf.BoundingRadius() + 3.0f * m_Voxel;
                        for (size_t s = 0; s < n; s++)
                            if (Vector3DistanceSqr(m_S.pos[s], sf.position) < r * r) {
                                local[i].push_back((int)s);
                                occ[i].push_back(Occ(sf, m_S.pos[s]));
                            }
                    });
                    std::fill(logKeep.begin(), logKeep.end(), 0.0f);
                    std::fill(sum.begin(), sum.end(), 0.0f);
                    for (int i = 0; i < P; i++)
                        for (size_t k = 0; k < local[i].size(); k++) {
                            logKeep[local[i][k]] += logf(1.0f - occ[i][k] + 1e-6f);
                            sum[local[i][k]] += occ[i][k];
                        }
                    tbb::parallel_for(0, P, [&](int i) {
                        if (only >= 0 && i != only) return;
                        const std::vector<int>& L = local[i];
                        if (L.empty()) return;
                        std::vector<float> others(L.size()), otherSum(L.size());
                        for (size_t k = 0; k < L.size(); k++) {
                            others[k] = expf(logKeep[L[k]] - logf(1.0f - occ[i][k] + 1e-6f));
                            otherSum[k] = sum[L[k]] - occ[i][k];
                        }
                        auto loss = [&](const SuperFrustum& sf) {
                            double acc = 0;
                            for (size_t k = 0; k < L.size(); k++) {
                                int s = L[k];
                                float o = Occ(sf, m_S.pos[s]);
                                float u = 1.0f - others[k] * (1.0f - o);
                                float e = u - m_S.target[s];
                                acc += m_S.weight[s] * (e * e + 0.25f * fmaxf(otherSum[k] + o - 1.0f, 0.0f));
                            }
                            return (float)(acc / m_S.totalWeight);
                        };
                        float p[NP], g[NP], lr[NP];
                        PackAll(shapes[i], p);
                        for (int q = 0; q < NP; q++) {
                            lr[q] = rates[q] * decay;
                            if (rates[q] == 0.0f) { g[q] = 0; continue; }
                            float save = p[q];
                            SuperFrustum a = shapes[i], b = shapes[i];
                            p[q] = save + steps[q]; UnpackAll(a, p, m_Voxel * 1.5f);
                            p[q] = save - steps[q]; UnpackAll(b, p, m_Voxel * 1.5f);
                            p[q] = save;
                            g[q] = (loss(a) - loss(b)) / (2.0f * steps[q]);
                        }
                        adam[i].Step(p, g, lr, NP);
                        UnpackAll(shapes[i], p, m_Voxel * 1.5f);
                    });
                    c.SetProgress(p0 + (p1 - p0) * (it + 1) / (float)iterations);
                    if (it % 8 == 7) publish();
                }
            }
        private:
            const Samples& m_S;
            float m_Voxel;
            float m_Tau;
        };

        // Union occupancy of the shapes at every voxel of the field.
        std::vector<uint8_t> Covered(const std::vector<SuperFrustum>& shapes, const ShapeField& f) {
            std::vector<uint8_t> out(f.sdf.size(), 0);
            for (const SuperFrustum& sf : shapes) {
                BoundingBox b = sf.WorldBounds();
                int lo[3], hi[3];
                for (int a = 0; a < 3; a++) {
                    lo[a] = std::max(0, (int)floorf(((&b.min.x)[a] - (&f.origin.x)[a]) / f.voxel));
                    hi[a] = std::min(f.dim[a] - 1, (int)ceilf(((&b.max.x)[a] - (&f.origin.x)[a]) / f.voxel));
                }
                tbb::parallel_for(lo[2], hi[2] + 1, [&](int k) {
                    for (int j = lo[1]; j <= hi[1]; j++)
                        for (int i = lo[0]; i <= hi[0]; i++)
                            if (sf.Distance(f.Position(i, j, k)) < 0.0f) out[f.Index(i, j, k)] = 1;
                });
            }
            return out;
        }

        Design PreviewDesign(const std::vector<SuperFrustum>& shapes) {
            Design d;
            for (const SuperFrustum& sf : shapes) d.AddPart(sf, -1);
            return d;
        }
    }

    DecomposeResult Decompose(const std::shared_ptr<const TriMesh>& mesh, const DecomposeParams& prm,
                              const std::vector<SuperFrustum>& initial, JobControl& control) {
        DecomposeResult result;
        auto start = Clock::now();
        if (!mesh || mesh->Empty()) { result.message = "Load a model first."; return result; }
        control.SetStage("Voxelising");
        Vector3 size = Vector3Subtract(mesh->box.max, mesh->box.min);
        float voxel = fmaxf(size.x, fmaxf(size.y, size.z)) / (float)std::max(prm.resolution, 12);
        std::string err;
        ShapeField field = BuildShapeField(*mesh, voxel, voxel * 3.0f, &err);
        if (!field.Valid()) { result.message = err; return result; }
        const int* dim = field.dim;
        size_t N = field.sdf.size();
        std::vector<uint8_t> solid(N);
        size_t solidCount = 0;
        for (size_t i = 0; i < N; i++) { solid[i] = field.sdf[i] < 0; solidCount += solid[i]; }
        if (solidCount == 0) { result.message = "The mesh has no inside (is it closed?)"; return result; }
        size_t minVoxels = std::max<size_t>(6, (size_t)(prm.minRegion * solidCount));

        Samples samples = MakeSamples(field);
        Fitter fitter(samples, voxel);
        std::vector<SuperFrustum> shapes = initial;
        auto publish = [&] { control.Publish(PreviewDesign(shapes)); };

        if (prm.refineOnly) {
            control.SetStage("Fitting");
            fitter.Optimize(shapes, prm.iterations, prm.onlyIndex, control, 0.0f, 1.0f, publish);
        } else {
            int rounds = std::max(1, prm.rounds);
            for (int round = 0; round < rounds && !control.Cancelled(); round++) {
                float r0 = (float)round / rounds, r1 = (float)(round + 1) / rounds;
                // residual: solid voxels the current set does not explain
                std::vector<uint8_t> covered = Covered(shapes, field);
                std::vector<uint8_t> residual(N);
                for (size_t i = 0; i < N; i++) residual[i] = solid[i] && !covered[i];
                {   // drop specks
                    std::vector<uint8_t> visited(N, 0);
                    for (size_t i = 0; i < N; i++)
                        if (residual[i] && !visited[i]) {
                            std::vector<int> comp = Component(residual, dim, (int)i, visited);
                            if (comp.size() < minVoxels)
                                for (int c : comp) residual[c] = 0;
                        }
                }
                size_t residualCount = std::accumulate(residual.begin(), residual.end(), (size_t)0);
                if (round > 0 && residualCount < minVoxels) break;

                // morphological shape decomposition: peel the thickest region first
                control.SetStage("Round " + std::to_string(round + 1) + ": seeding");
                int seeds = round == 0 ? prm.seedsPerRound * 2 : prm.seedsPerRound;
                int failures = 0;
                for (int s = 0; s < seeds && (int)shapes.size() < prm.maxParts && failures < 4; s++) {
                    std::vector<uint8_t> outside(N);
                    for (size_t i = 0; i < N; i++) outside[i] = !residual[i];
                    std::vector<float> depth = DistanceTo(outside, dim);
                    size_t best = 0;
                    for (size_t i = 0; i < N; i++)
                        if (depth[i] > depth[best]) best = i;
                    float maxDepth = depth[best];
                    if (maxDepth < 1.75f) {
                        // only thin structures left (plates, wings, thin rods): erosion would wipe them out, so seed
                        // from the largest connected piece of the residual instead
                        // Fringes left around fitted parts wrap the explained solid; real plates only touch it at
                        // their root. Keep components that mostly border empty space.
                        std::vector<uint8_t> visited(N, 0);
                        std::vector<int> biggest;
                        const int sx = 1, sy = dim[0], sz = dim[0] * dim[1];
                        for (size_t i = 0; i < N; i++)
                            if (residual[i] && !visited[i]) {
                                std::vector<int> comp = Component(residual, dim, (int)i, visited);
                                if (comp.size() <= biggest.size() || comp.size() < minVoxels) continue;
                                size_t touching = 0;
                                for (int c : comp) {
                                    const int nb[6] = { c - sx, c + sx, c - sy, c + sy, c - sz, c + sz };
                                    for (int q : nb)
                                        if (q >= 0 && (size_t)q < N && solid[q] && !residual[q]) { touching++; break; }
                                }
                                if (touching < 0.35f * comp.size()) biggest.swap(comp);
                            }
                        if (biggest.size() >= minVoxels) {
                            std::vector<Vector3> region;
                            for (int c : biggest) {
                                int x = c % dim[0], y = (c / dim[0]) % dim[1], z = c / (dim[0] * dim[1]);
                                region.push_back(field.Position(x, y, z));
                                residual[c] = 0;
                            }
                            SeedRegion(region, voxel, minVoxels, 0, shapes);
                            publish();
                            continue;
                        }
                        if (maxDepth < 1.0f) break;   // nothing thin worth a part either
                    }
                    float r = fmaxf(1.0f, maxDepth * 0.55f);
                    std::vector<uint8_t> core(N), visited(N, 0);
                    for (size_t i = 0; i < N; i++) core[i] = residual[i] && depth[i] >= r;
                    std::vector<int> coreComp = Component(core, dim, (int)best, visited);
                    std::vector<uint8_t> coreMask(N, 0);
                    for (int c : coreComp) coreMask[c] = 1;
                    std::vector<float> toCore = DistanceTo(coreMask, dim);
                    std::vector<Vector3> region;
                    for (size_t i = 0; i < N; i++)
                        if (residual[i] && toCore[i] <= r + 0.75f) {
                            int x = (int)(i % dim[0]), y = (int)((i / dim[0]) % dim[1]), z = (int)(i / ((size_t)dim[0] * dim[1]));
                            region.push_back(field.Position(x, y, z));
                            residual[i] = 0;
                        }
                    if (region.size() < minVoxels) { failures++; continue; }
                    SeedRegion(region, voxel, minVoxels, 0, shapes);
                    publish();
                }
                if (control.Cancelled()) break;

                // refine everything together
                control.SetStage("Round " + std::to_string(round + 1) + ": fitting " + std::to_string(shapes.size()) + " primitives");
                fitter.Optimize(shapes, prm.iterations, -1, control, r0, r0 + (r1 - r0) * 0.9f, publish);
                // prune primitives that do not pay for themselves
                control.SetStage("Round " + std::to_string(round + 1) + ": pruning");
                float base = fitter.Loss(shapes) + prm.parsimony * shapes.size();
                std::vector<int> order(shapes.size());
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](int a, int b) {
                    return shapes[a].size.x * shapes[a].size.y * shapes[a].size.z < shapes[b].size.x * shapes[b].size.y * shapes[b].size.z;
                });
                std::vector<char> removed(shapes.size(), 0);
                for (int idx : order) {
                    std::vector<SuperFrustum> trial;
                    for (size_t j = 0; j < shapes.size(); j++)
                        if (!removed[j] && (int)j != idx) trial.push_back(shapes[j]);
                    if (trial.empty()) continue;
                    float cost = fitter.Loss(trial) + prm.parsimony * trial.size();
                    if (cost < base) { removed[idx] = 1; base = cost; }
                }
                std::vector<SuperFrustum> kept;
                for (size_t j = 0; j < shapes.size(); j++)
                    if (!removed[j]) kept.push_back(shapes[j]);
                shapes.swap(kept);
                publish();
                control.SetProgress(r1);
                if ((int)shapes.size() >= prm.maxParts && round > 0) break;
            }
        }

        // biggest first (the first part becomes the default root); a refinement keeps the caller's order
        if (!prm.refineOnly)
            std::stable_sort(shapes.begin(), shapes.end(), [](const SuperFrustum& a, const SuperFrustum& b) {
                return a.size.x * a.size.y * a.size.z > b.size.x * b.size.y * b.size.z;
            });
        std::vector<uint8_t> covered = Covered(shapes, field);
        size_t inter = 0, uni = 0;
        for (size_t i = 0; i < N; i++) { inter += solid[i] && covered[i]; uni += solid[i] || covered[i]; }
        result.iou = uni ? (float)inter / uni : 0.0f;
        result.shapes = shapes;
        result.cancelled = control.Cancelled();
        result.ok = !shapes.empty();
        result.milliseconds = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        char buf[160];
        snprintf(buf, sizeof(buf), "%s%d superfrustums, IoU %.1f%% (%.1f s)", result.cancelled ? "Stopped early: " : "",
                 (int)shapes.size(), result.iou * 100.0f, result.milliseconds / 1000.0);
        result.message = shapes.empty() ? "No primitives were fitted." : buf;
        return result;
    }
}
