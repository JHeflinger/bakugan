#include "search.hpp"
#include "util/utils.hpp"
#include <cstdio>
#include <cstdlib>
#include <algorithm>

namespace Bakugan::Auto {
    Search::Search(const TriMesh& mesh, const AutoParams& p, JobControl& c, std::shared_ptr<AutoLive> live)
        : m_P(p), m_C(c), m_Rng(p.seed), m_Pool(std::max(0, (int)std::thread::hardware_concurrency() - 1)),
          m_Scorer(m_P, m_T, m_Pool, m_Rng), m_Params(m_Space), m_Grammar(m_P, m_T, m_Space, m_Rng),
          m_Progress(std::move(live), c, m_Space, m_Scorer.Outlines()) {
        m_P.iterations = std::max(0, m_P.iterations);
        m_P.rewritesPerRound = std::max(1, m_P.rewritesPerRound);
        m_P.stepsPerRound = std::max(0, m_P.stepsPerRound);
        m_P.stepSize = std::max(1e-3f, m_P.stepSize);
        m_T.mesh = &mesh;
    }

    // Stochastic Rewrite Descent (Kodnongbua, Zhang, Sharp, Schulz, "Design for Descent", SIGGRAPH Asia 2025):
    //
    //   1. measure the mesh, and the fixed samples every design is scored on
    //   2. a starting design: the mesh split into hinged pieces, or the design in the editor
    //   3. until the budget of scorings runs out, rounds of
    //        a. descent: steps on the continuous parameters, the structure fixed
    //        b. rewrites: a batch of grammar rewrites, each tried with a short lookahead; every one that helps is
    //           applied (best first, none conflicting), all at once
    //        c. back to the best design if the search has wandered far from it
    //      while collisions are ramped in over the first 60% of the budget
    //   4. the best design scored on the way is the result
    AutoResult Search::Run(const Design& start) {
        // 1.
        if (!m_T.Build(*m_T.mesh, m_P.quality, m_Rng, m_Pool, m_C)) {
            AutoResult res;
            res.cancelled = m_C.Cancelled();
            res.message = res.cancelled ? "Stopped." : "The mesh has no usable inside (is it closed?).";
            return res;
        }
        m_Space.Init(m_T, m_P);
        m_Scorer.Prepare(m_C);

        // 2.
        State cur = Start(start);
        if (!cur.score.valid) {
            AutoResult res;
            res.message = "Could not score the starting design.";
            return res;
        }

        // 3.
        const int N = m_P.iterations;
        m_Progress.Begin(N, m_Best.design, m_Best.score, m_Scorer.Count());
        m_Lr = m_P.stepSize;
        m_C.SetStage("Searching");
        while (Budget()) {
            // collisions start cheap and reach their full weight at 60% of the run: hinges and folds get to form
            // before overlaps between the bodies have to be carved or moved away
            m_Ramp = std::min(1.0f, 0.1f + 0.9f * ((float)m_Scorer.Count() / N) / 0.6f);
            DescentPhase(cur);                                  // a.
            if (!Budget()) break;
            m_Accepted += RewriteRound(cur);                    // b.
            m_Rounds++;
            m_Lr = fmaxf(m_Lr, m_P.stepSize);                   //    a new structure: full size steps again
            if (cur.score.valid && Objective(cur.score) > Objective(m_Best.score) + 0.02f) cur = m_Best;   // c.
            m_Progress.Tick(m_Best.design, m_Best.score, Counts());
        }

        // 4. (the descent since the last round has not been scored exactly)
        if (!m_C.Cancelled() && N > 0) Score(cur, false);
        m_Progress.End(m_Best.design, m_Best.score, m_BestOutlines, Counts(), m_C.Cancelled());
        if (getenv("BAKUGAN_RULE_STATS"))           // for tuning the grammar: how each rule fared
            for (int k = 0; k < (int)Rule::Count; k++)
                fprintf(stderr, "%-24s tried %4d  helped %3d\n", Grammar::Name((Rule)k), m_RuleStats[k].tried, m_RuleStats[k].helped);
        return Result();
    }

    // The starting design, valid and scored: the best so far.
    Search::State Search::Start(const Design& start) {
        State cur;
        cur.design = m_P.fromCurrent && !start.shapes.empty() ? FromEditor(start) : Seed();
        // a seed with fewer pieces than asked for (a small mesh) sets the floor
        m_Space.minBodies = std::min(m_Space.minBodies, (int)cur.design.Bodies().size());
        if (!m_Space.Sanitize(cur.design))
            for (Hinge& h : cur.design.hinges) h.hardware = false;
        m_Space.Project(cur.design);
        m_Ramp = 0.1f;
        cur.score = m_Scorer.Evaluate(cur.design, &cur.diag);
        if (!cur.score.valid) return cur;
        m_Best = cur;
        m_BestOutlines = m_Scorer.Outlines().Last();
        m_Progress.Best(m_Best.design, m_Best.score, m_BestOutlines, m_Scorer.Count(), true);
        return cur;
    }

    // Descent, the structure fixed: steps on one random element's parameters at a time. A step that doesn't help
    // mostly means that element is already where it should be, not that the step is too big, so the step size follows
    // the success rate instead of runs of failures: it settles where about one step in four helps (x1.25 on success,
    // x0.93 otherwise).
    void Search::DescentPhase(State& cur) {
        const float minLr = 0.05f * m_P.stepSize, maxLr = 2.0f * m_P.stepSize;
        for (int k = 0; k < m_P.stepsPerRound && Budget(2); k++) {
            Focus block = m_Params.RandomBlock(cur.design, m_Rng);
            bool helped = DescentStep(cur, m_Params.Of(cur.design, &block), m_Lr);
            m_Lr = std::clamp(m_Lr * (helped ? 1.25f : 0.93f), minLr, maxLr);
            m_Progress.Tick(m_Best.design, m_Best.score, Counts());
        }
    }

    // One descent step on the parameters `ps`, with an SPSA gradient (simultaneous perturbation stochastic
    // approximation): every parameter is pushed by +1 or -1 unit at random, all at once, and the score there against
    // the score here gives the slope along that direction. The step goes downhill along it by `lr` units at a typical
    // slope (slopes are measured against their running RMS), at most twice that (the paper's gradient clipping).
    //
    // Here the paper and this code differ: there the score is differentiable code, and every step follows its exact
    // gradient over all parameters. A sampled distance field score has no gradient to take, and an SPSA estimate
    // over every parameter at once is too noisy to descend on (it measured worse than standing still). So a step
    // estimates the gradient of one block (randomized block coordinate descent), and is only taken if it lowers the
    // score: whichever of the probe and the step is better replaces the design when it beats it. The current score
    // is known exactly, so one probe gives the slope (a one-sided estimate): two scorings per step.
    bool Search::DescentStep(State& cur, const std::vector<Param>& ps, float lr) {
        if (ps.empty() || !cur.score.valid) return false;
        std::vector<float> delta(ps.size());
        for (float& x : delta) x = Chance(m_Rng, 0.5f) ? 1.0f : -1.0f;
        State probe;
        probe.design = cur.design;
        m_Params.Step(probe.design, ps, delta);
        if (!m_Space.Sanitize(probe.design) || !Score(probe, false)) return false;
        const float here = Objective(cur.score);
        float slope = Objective(probe.score) - here;
        m_SpsaRms = m_SpsaRms > 0.0f ? sqrtf(0.9f * m_SpsaRms * m_SpsaRms + 0.1f * slope * slope) : fabsf(slope);
        State* best = &probe;
        State next;
        if (m_SpsaRms > 1e-12f) {
            float g = std::clamp(slope / m_SpsaRms, -2.0f, 2.0f);
            std::vector<float> step(ps.size());
            for (size_t i = 0; i < ps.size(); i++) step[i] = -lr * g * delta[i];
            next.design = cur.design;
            m_Params.Step(next.design, ps, step);
            if (m_Space.Sanitize(next.design) && Score(next, false) && Objective(next.score) < Objective(probe.score)) best = &next;
        }
        if (Objective(best->score) >= here) return false;
        cur.design = std::move(best->design);
        cur.score = best->score;
        return true;
    }

    // The paper's LocalOptimize, before a rewrite is judged: descent steps on just the parameters the rewrite created
    // or changed, so a good idea that landed slightly off isn't judged by its first guess. The paper takes one exact
    // gradient step; the estimated steps here are weaker, so up to `lookaheadSteps` are taken, stopping at the first
    // that doesn't help.
    //
    // A patient lookahead (rules whose change is nearly neutral by itself, like splitting a shape) takes twice the
    // steps and carries on past steps that don't help, each on one part (position, size, rotation or profile) of one
    // of the focused shapes, the way the descent does: what such a change buys only shows once its shapes have moved.
    void Search::Lookahead(State& cand, const Focus& f, bool patient) {
        if (!patient) {
            std::vector<Param> ps = m_Params.Of(cand.design, &f);
            if (ps.empty()) return;
            for (int k = 0; k < m_P.lookaheadSteps && Budget(2); k++)
                if (!DescentStep(cand, ps, m_P.stepSize)) break;
            return;
        }
        const Part parts[] = { Part::Position, Part::Size, Part::Rotation, Part::Profile };
        for (int k = 0; k < 2 * m_P.lookaheadSteps && Budget(2); k++) {
            Focus one;
            one.shape = f.shape2 >= 0 && Chance(m_Rng, 0.5f) ? f.shape2 : f.shape;
            one.part = parts[Pick(m_Rng, 4)];
            DescentStep(cand, m_Params.Of(cand.design, &one), m_P.stepSize);
        }
    }

    // One rewrite round: score the current design exactly (its diagnostics guide the drawing), draw rewrites, give
    // each a lookahead, and apply every one that improved the objective by more than a small threshold, best first,
    // skipping any that conflict with one already taken: all at once (the paper's greedy selection). Should the
    // combination score worse than the best rewrite alone (rewrites can interact), that one alone is taken. Returns
    // how many rewrites were applied.
    int Search::RewriteRound(State& cur) {
        if (!Score(cur, true)) return 0;
        const float base = Objective(cur.score);
        const float threshold = fmaxf(1e-4f, 1e-3f * base);
        const DrawContext context(cur.design, cur.diag, cur.score);

        // draw and try
        struct Tried {
            Rewrite r;
            State s;
        };
        std::vector<Tried> kept;
        int drawn = 0;
        for (int tries = 0; drawn < m_P.rewritesPerRound && tries < 4 * m_P.rewritesPerRound && Budget(); tries++) {
            Rewrite r;
            if (!m_Grammar.Draw(context, &r)) continue;
            State cand;
            cand.design = cur.design;
            Focus f;
            if (!m_Grammar.Apply(cand.design, r, &f) || !m_Space.Sanitize(cand.design)) continue;
            drawn++;
            if (!Score(cand, false)) continue;
            // a lookahead step costs two scorings: spent on rewrites that landed close enough for it to matter
            bool close = base - Objective(cand.score) > -0.1f * base;
            if (close && (f.shape >= 0 || f.hinge >= 0)) Lookahead(cand, f, Grammar::Patient(r.rule));
            m_Grammar.Remember(r, cand.design, f);
            r.gain = base - Objective(cand.score);
            if (r.gain > threshold) kept.push_back({ std::move(r), std::move(cand) });
            m_RuleStats[(int)r.rule].tried++;
            m_RuleStats[(int)r.rule].helped += r.gain > threshold;
        }
        if (kept.empty()) return 0;

        // the best ones that don't conflict, all at once
        std::sort(kept.begin(), kept.end(), [](const Tried& a, const Tried& b) { return a.r.gain > b.r.gain; });
        std::vector<const Rewrite*> chosen;
        for (const Tried& k : kept) {
            bool clash = false;
            for (const Rewrite* c : chosen) clash = clash || Conflict(k.r, *c);
            if (!clash) chosen.push_back(&k.r);
        }
        if (chosen.size() > 1) {
            State combined;
            combined.design = cur.design;
            int applied = 0;
            for (const Rewrite* r : chosen) {
                Design t = combined.design;
                Focus f;
                if (m_Grammar.Apply(t, *r, &f) && m_Space.Sanitize(t)) {
                    combined.design = std::move(t);
                    applied++;
                }
            }
            if (applied > 1 && Score(combined, false) && Objective(combined.score) <= Objective(kept[0].s.score)) {
                cur.design = std::move(combined.design);
                cur.score = combined.score;
                return applied;
            }
        }
        cur = std::move(kept[0].s);
        return 1;
    }

    // ---- bookkeeping ----------------------------------------------------------------------------------------------------

    bool Search::Score(State& s, bool diagnose) {
        s.score = m_Scorer.Evaluate(s.design, diagnose ? &s.diag : nullptr);
        Consider(s.design, s.score);
        return s.score.valid;
    }

    // Every design scored exactly is a candidate for the result: the start, every probe, every rewrite.
    void Search::Consider(const Design& d, const AutoScore& s) {
        if (!s.valid) return;
        if (m_Best.score.valid && Objective(s) >= Objective(m_Best.score) - 1e-6f) return;
        m_Best.design = d;
        m_Best.score = s;
        if (s.silhouettes) m_BestOutlines = m_Scorer.Outlines().Last();     // the design just scored
        m_Progress.Best(m_Best.design, m_Best.score, m_BestOutlines, m_Scorer.Count(), false);
    }

    AutoResult Search::Result() const {
        AutoResult res;
        res.ok = true;
        res.cancelled = m_C.Cancelled();
        res.design = m_Best.design;
        m_Space.Finalize(res.design);
        res.score = m_Best.score;
        const AutoScore& s = m_Best.score;
        const char* what = res.cancelled ? "Stopped, kept the best so far:" : (m_P.iterations == 0 ? "Scored:" : "Generated");
        std::string shape = Utils::Format("%d bodies, %d shapes, %d hinges", (int)res.design.Bodies().size(), (int)res.design.shapes.size(),
                                          (int)res.design.hinges.size());
        std::string fits = s.silhouettes
            ? Utils::Format("outlines cover %.0f%% of the mesh's (+%.0f%% extra) and %.0f%% of the sphere's (+%.0f%% extra), worst view %.0f%%",
                            s.viewCover * 100.0f, s.viewExtra * 100.0f, s.sphereViewCover * 100.0f, s.sphereViewExtra * 100.0f, s.worstView * 100.0f)
            : Utils::Format("covers %.0f%% of the mesh, %.0f%% outside it, sphere %.0f%%", s.fitCover * 100.0f, s.fitOutside * 100.0f,
                            s.sphereIoU * 100.0f);
        std::string motion = m_P.motionChecks > 0 ? Utils::Format(" / %.1f%% on the way", s.clashMotion * 100.0f) : "";
        res.message = Utils::Format("%s %s: %s, overlapping %.1f%% open / %.1f%% closed%s (score %.3f; %d scorings, %d rewrites in %d rounds).",
                                    what, shape.c_str(), fits.c_str(), s.clashOpen * 100.0f, s.clashClosed * 100.0f, motion.c_str(), s.total,
                                    m_Scorer.Count(), m_Accepted, m_Rounds);
        return res;
    }
}

namespace Bakugan {
    AutoResult AutoDesign(const TriMesh& mesh, const Design& start, const AutoParams& p, JobControl& c, std::shared_ptr<AutoLive> live) {
        Auto::Search s(mesh, p, c, std::move(live));
        return s.Run(start);
    }
}
