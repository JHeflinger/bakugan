#pragma once
// The score: one number for a design, lower is better, covering everything the search trades off. See score.cpp for
// the terms, in the order they are measured.
#include "silhouettes.hpp"
#include "target.hpp"

namespace Bakugan::Auto {
    // Fixed unit samples (common random numbers: every design sees the same points, so score differences come from
    // the designs, not from sampling noise).
    struct Samples {
        std::vector<Vector3> openUnit;       // in [0, 1]^3, mapped into the open design's box
        std::vector<Vector3> closedUnit;     // in [0, 1]^3, mapped into the closed shape's box
        std::vector<Vector3> sphereUnit;     // in [-1.35, 1.35]^3, around the sphere ...
        std::vector<char> sphereIn;          // ... and whether each is inside the unit sphere
        std::vector<Vector3> shellDirs;      // on the unit sphere (its surface)
        std::vector<Vector3> pairUnit;       // in [0, 1]^3, mapped into where two bodies' boxes meet
        void Make(int quality, unsigned seed);
    };

    class Scorer {
    public:
        Scorer(const AutoParams& p, const Target& t, Pool& pool, Rng& rng) : m_P(p), m_T(t), m_Pool(pool), m_Rng(rng) {}
        void Prepare(JobControl& c);     // the fixed samples, and the mesh's outlines when scoring by silhouettes

        // Scores a design. With diag, also finds where it is wrong (a few sampled spots of each kind) for the grammar.
        AutoScore Evaluate(const Design& d, Diagnostics* diag);
        int Count() const { return m_Count; }            // scorings so far: the search's budget
        const Silhouettes& Outlines() const { return m_Sil; }
        float Tau() const { return m_Tau; }
    private:
        struct Eval;                     // one scoring's poses, samples and findings
        void SampleDistances(Eval& e);
        void MeshFit(Eval& e, AutoScore& sc);
        void OpenMaterial(Eval& e, AutoScore& sc);
        void ClosedShape(Eval& e, AutoScore& sc);
        void MotionClashes(Eval& e, AutoScore& sc);
        void SphereFit(Eval& e, AutoScore& sc);
        void Diagnose(Eval& e, const AutoScore& sc, Diagnostics& diag);
        float Costs(const Design& d, const Probe& pr) const;
        float Occupancy(float d) const { return 1.0f / (1.0f + expf(d / m_Tau)); }
        float Overlap(float d) const { return Occupancy(d + m_Tau); }   // inside by more than about tau
    private:
        const AutoParams& m_P;
        const Target& m_T;
        Pool& m_Pool;
        Rng& m_Rng;
        Samples m_S;
        Silhouettes m_Sil;
        float m_Tau = 1.0f;              // occupancy softness (mm)
        int m_Count = 0;
    };
}
