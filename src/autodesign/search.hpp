#pragma once
// The search: Stochastic Rewrite Descent over the design space, against the score. Search::Run in search.cpp is the
// whole algorithm, step by step; everything it calls lives in its own module (README.md has the full map):
//
//   target       the input mesh, measured once                        target.hpp
//   start        where a run starts                                   start.cpp
//   space        the limits a design must keep                        space.hpp
//   score        how good a design is (and where it is wrong)         score.hpp, silhouettes.hpp
//   parameters   the continuous half: what a descent step moves       params.hpp
//   grammar      the discrete half: the rewrite rules                 grammar.hpp, placement.cpp, edits.hpp
//   progress     what the UI sees while it runs                       progress.hpp
#include "grammar.hpp"
#include "progress.hpp"

namespace Bakugan::Auto {
    class Search {
    public:
        Search(const TriMesh& mesh, const AutoParams& p, JobControl& c, std::shared_ptr<AutoLive> live);
        AutoResult Run(const Design& start);
    private:
        struct State {
            Design design;
            AutoScore score;
            Diagnostics diag;
        };
        // the stages of a run
        State Start(const Design& start);
        void DescentPhase(State& cur);
        int RewriteRound(State& cur);
        AutoResult Result() const;
        // the two kinds of step
        bool DescentStep(State& cur, const std::vector<Param>& ps, float lr);
        void Lookahead(State& cand, const Focus& f, bool patient);
        // starting designs (start.cpp)
        Design Seed();
        Design FromEditor(const Design& start);
        // bookkeeping
        bool Score(State& s, bool diagnose);           // scores it exactly; every scored design may become the best
        float Objective(const AutoScore& s) const { return s.base + m_Ramp * m_P.collisionWeight * s.collision; }
        void Consider(const Design& d, const AutoScore& s);
        bool Budget(int scorings = 1) const { return m_Scorer.Count() + scorings <= m_P.iterations && !m_C.Cancelled(); }
        Progress::Counts Counts() const { return { m_Scorer.Count(), m_Accepted, m_Rounds, m_Lr }; }
    private:
        AutoParams m_P;
        JobControl& m_C;
        Rng m_Rng;
        Pool m_Pool;
        Target m_T;
        DesignSpace m_Space;
        Scorer m_Scorer;
        Parameters m_Params;
        Grammar m_Grammar;
        Progress m_Progress;
        // the run
        float m_Ramp = 0.1f;             // share of the collision weight in force (it grows over the run)
        float m_Lr = 0.5f;               // descent step (parameter units)
        float m_SpsaRms = 0.0f;          // running RMS of SPSA slopes: steps are measured against it
        State m_Best;
        Outlines m_BestOutlines;
        int m_Accepted = 0, m_Rounds = 0;
        struct RuleStat {
            int tried = 0, helped = 0;   // rewrites applied and scored, and those good enough to keep
        };
        RuleStat m_RuleStats[(int)Rule::Count];
    };
}
