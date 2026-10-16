// TBB first: raylib's macros must not leak into its headers.
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include "fitloss.hpp"
#include "util/optim.hpp"
#include <raymath.h>
#include <algorithm>
#include <cmath>

namespace Bakugan {
    void FitModel::Build(const ShapeField& f, int maxSamples) {
        m_Pos.clear();
        m_Target.clear();
        m_Weight.clear();
        m_Field = &f;
        if (!f.Valid()) return;
        m_Voxel = f.voxel;
        m_Tau = 0.6f * f.voxel;
        // keep everything near the surface, thin out the deep inside and the far outside
        size_t total = f.sdf.size();
        size_t band = 0;
        for (float d : f.sdf) band += fabsf(d) < 2.5f * f.voxel;
        float rest = std::clamp((float)(maxSamples - (int)band) / (float)std::max<size_t>(1, total - band), 0.02f, 1.0f);
        uint32_t h = 2166136261u;
        auto rnd = [&]() { h ^= h << 13; h ^= h >> 17; h ^= h << 5; return (h & 0xFFFFFF) / (float)0xFFFFFF; };
        m_TotalWeight = 0.0f;
        for (int k = 0; k < f.dim[2]; k++)
            for (int j = 0; j < f.dim[1]; j++)
                for (int i = 0; i < f.dim[0]; i++) {
                    float d = f.sdf[f.Index(i, j, k)];
                    float ad = fabsf(d) / f.voxel;
                    float keep = ad < 2.5f ? 1.0f : (d < 0 ? rest * 1.5f : rest * 0.6f);
                    keep = std::min(keep, 1.0f);
                    if (rnd() > keep) continue;
                    m_Pos.push_back(f.Position(i, j, k));
                    m_Target.push_back(d < 0 ? 1.0f : 0.0f);
                    float w = (ad < 1.5f ? 1.5f : 1.0f) / keep;
                    m_Weight.push_back(w);
                    m_TotalWeight += w;
                }
        m_TotalWeight = std::max(m_TotalWeight, 1e-6f);
    }

    float FitModel::Occ(const SuperFrustum& sf, Vector3 p) const {
        return Sigmoid(-sf.Distance(p) / m_Tau);
    }

    void FitModel::Prepare(const std::vector<SuperFrustum>& shapes, Cache& c) const {
        int P = (int)shapes.size();
        size_t n = m_Pos.size();
        c.local.assign(P, {});
        c.occ.assign(P, {});
        c.logKeep.assign(n, 0.0f);
        c.sum.assign(n, 0.0f);
        for (int i = 0; i < P; i++) {
            const SuperFrustum& sf = shapes[i];
            float r = sf.BoundingRadius() + 4.0f * m_Voxel;
            for (size_t s = 0; s < n; s++)
                if (Vector3DistanceSqr(m_Pos[s], sf.position) < r * r) {
                    float o = Occ(sf, m_Pos[s]);
                    c.local[i].push_back((int)s);
                    c.occ[i].push_back(o);
                    c.logKeep[s] += logf(1.0f - o + 1e-6f);
                    c.sum[s] += o;
                }
        }
    }

    float FitModel::Loss(const std::vector<SuperFrustum>& shapes) const {
        if (m_Pos.empty()) return 0.0f;
        size_t n = m_Pos.size();
        double L = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n), 0.0,
            [&](const tbb::blocked_range<size_t>& r, double acc) {
                for (size_t s = r.begin(); s < r.end(); s++) {
                    float keep = 1.0f, sum = 0.0f;
                    for (const SuperFrustum& sf : shapes) {
                        float reach = sf.BoundingRadius() + 4.0f * m_Voxel;
                        if (Vector3DistanceSqr(m_Pos[s], sf.position) > reach * reach) continue;
                        float o = Occ(sf, m_Pos[s]);
                        keep *= 1.0f - o;
                        sum += o;
                    }
                    float e = (1.0f - keep) - m_Target[s];
                    acc += m_Weight[s] * (e * e + 0.25f * fmaxf(sum - 1.0f, 0.0f));
                }
                return acc;
            }, std::plus<double>());
        return (float)(L / m_TotalWeight);
    }

    float FitModel::LocalLoss(int i, const SuperFrustum& sf, const Cache& c) const {
        // samples near either the old or the new primitive change; others are unaffected
        float r = sf.BoundingRadius() + 4.0f * m_Voxel;
        double acc = 0;
        auto term = [&](int s, float oldOcc, float newOcc) {
            float others = expf(c.logKeep[s] - logf(1.0f - oldOcc + 1e-6f));
            float otherSum = c.sum[s] - oldOcc;
            float u = 1.0f - others * (1.0f - newOcc);
            float e = u - m_Target[s];
            return m_Weight[s] * (e * e + 0.25f * fmaxf(otherSum + newOcc - 1.0f, 0.0f));
        };
        const std::vector<int>& L = c.local[i];
        for (size_t k = 0; k < L.size(); k++) acc += term(L[k], c.occ[i][k], Occ(sf, m_Pos[L[k]]));
        // newly reached samples (the primitive grew or moved)
        for (size_t s = 0; s < m_Pos.size(); s++) {
            if (Vector3DistanceSqr(m_Pos[s], sf.position) > r * r) continue;
            if (std::binary_search(L.begin(), L.end(), (int)s)) continue;
            acc += term((int)s, 0.0f, Occ(sf, m_Pos[s]));
            float base = (1.0f - expf(c.logKeep[s])) - m_Target[s];
            acc -= m_Weight[s] * (base * base + 0.25f * fmaxf(c.sum[s] - 1.0f, 0.0f));
        }
        return (float)(acc / m_TotalWeight);
    }

    std::vector<Vector3> FitModel::Unexplained(const std::vector<SuperFrustum>& shapes) const {
        std::vector<Vector3> out;
        for (size_t s = 0; s < m_Pos.size(); s++) {
            if (m_Target[s] < 0.5f) continue;
            float keep = 1.0f;
            for (const SuperFrustum& sf : shapes) {
                float reach = sf.BoundingRadius() + 4.0f * m_Voxel;
                if (Vector3DistanceSqr(m_Pos[s], sf.position) > reach * reach) continue;
                keep *= 1.0f - Occ(sf, m_Pos[s]);
            }
            if (1.0f - keep < 0.3f) out.push_back(m_Pos[s]);
        }
        return out;
    }

    float FitModel::IoU(const std::vector<SuperFrustum>& shapes) const {
        if (!m_Field || !m_Field->Valid()) return 0.0f;
        const ShapeField& f = *m_Field;
        struct Count { size_t inter = 0, uni = 0; };
        size_t total = f.sdf.size();
        Count c = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, total), Count{},
            [&](const tbb::blocked_range<size_t>& r, Count acc) {
                for (size_t idx = r.begin(); idx < r.end(); idx++) {
                    int i = (int)(idx % f.dim[0]), j = (int)((idx / f.dim[0]) % f.dim[1]), k = (int)(idx / ((size_t)f.dim[0] * f.dim[1]));
                    bool solid = f.sdf[idx] < 0.0f;
                    Vector3 p = f.Position(i, j, k);
                    bool covered = false;
                    for (const SuperFrustum& sf : shapes) {
                        float reach = sf.BoundingRadius();
                        if (Vector3DistanceSqr(p, sf.position) > reach * reach) continue;
                        if (sf.Distance(p) < 0.0f) { covered = true; break; }
                    }
                    acc.inter += solid && covered;
                    acc.uni += solid || covered;
                }
                return acc;
            }, [](Count a, Count b) { return Count{ a.inter + b.inter, a.uni + b.uni }; });
        return c.uni ? (float)c.inter / c.uni : 0.0f;
    }
}
