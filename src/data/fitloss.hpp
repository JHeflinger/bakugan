#pragma once
#include "data/shape.hpp"
#include "data/superfrustum.hpp"
#include <vector>

namespace Bakugan {
    // How well a set of superfrustums reconstructs the input: the soft-occupancy loss used by the decomposition
    // (probabilistic union of sigmoid occupancies vs. inside/outside, plus an overlap penalty), on a fixed sample set.
    class FitModel {
    public:
        struct Cache {
            std::vector<std::vector<int>> local;     // per part: samples near it
            std::vector<std::vector<float>> occ;     // per part: its occupancy at those samples
            std::vector<float> logKeep, sum;         // per sample: log prod(1 - occ), sum occ
        };

        void Build(const ShapeField& field, int maxSamples);
        bool Valid() const { return !m_Pos.empty(); }
        float Loss(const std::vector<SuperFrustum>& shapes) const;
        // Loss with part `i` replaced by `sf`, everything else as in the cache (only i's neighbourhood is visited).
        void Prepare(const std::vector<SuperFrustum>& shapes, Cache& cache) const;
        float LocalLoss(int i, const SuperFrustum& sf, const Cache& cache) const;
        // Interior samples no primitive explains (occupancy < 0.3).
        std::vector<Vector3> Unexplained(const std::vector<SuperFrustum>& shapes) const;
        float IoU(const std::vector<SuperFrustum>& shapes) const;
        float Voxel() const { return m_Voxel; }
    private:
        float Occ(const SuperFrustum& sf, Vector3 p) const;
        std::vector<Vector3> m_Pos;
        std::vector<float> m_Target;
        std::vector<float> m_Weight;
        float m_TotalWeight = 1.0f;
        float m_Voxel = 0.05f;
        float m_Tau = 0.03f;
        const ShapeField* m_Field = nullptr;
    };
}
