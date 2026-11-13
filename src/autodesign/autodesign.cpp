// The optimizer: Adam on every differentiable parameter, and every so often a rewrite round in the manner of
// Stochastic Rewrite Descent (Kodnongbua et al., "Design for Descent", SIGGRAPH Asia 2025):
//
//   1. the target: the mesh (which Adam may move and scale, to wherever the design fits it best); the sphere: at the
//      mesh's centre of volume, with its volume
//   2. the start: the sphere, whole (one piece)
//   3. until the budget runs out:
//        an Adam step: the loss and its gradient, every parameter moved (the cutters', the joints', the placement's)
//        every stepsPerRound steps, a rewrite round: rewrites drawn from the grammar (a cut added or removed), each
//        (and the carving unchanged) given `lookahead` Adam steps; whichever is best afterwards carries on (one that
//        adds only if it is better by rewriteMargin, one that takes away if it is no worse than that)
//   4. the result: the best carving seen, built into a design (a body per piece, a hinge per joint)
//
// Every carving's pieces fill the sphere exactly (carving.hpp), so the closed pose needs no scoring: it's the sphere.
#include "autodesign.hpp"
#include "adam.hpp"
#include "grammar.hpp"
#include "util/utils.hpp"
#include <raymath.h>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <map>

namespace Bakugan {
    namespace {
        // A carving, where the mesh is placed for it, and Adam's memory of their parameters.
        struct State {
            Carving carving;
            Placement target;
            Adam adam;
            AutoScore score;             // its loss before its last step
            const Rule* rule = nullptr;  // the rewrite that made it, in this round
        };

        // One Adam step: the loss and its gradient at the carving, then every parameter moved. Returns the loss of the
        // carving as it was before.
        AutoScore AdamStep(State& s, Loss& loss, const AutoParams& p) {
            std::vector<Param> ps = CarvingParams(s.carving, loss.L());
            std::vector<Param> tp = TargetParams(p.moveTarget, false, loss.L());   // (the scale is set, not optimized)
            ps.insert(ps.end(), tp.begin(), tp.end());
            std::vector<float> grad;
            AutoScore score = loss.Evaluate(s.carving, s.target, ps, &grad);
            std::vector<float> x(ps.size());
            std::vector<uint64_t> keys(ps.size());
            for (size_t i = 0; i < ps.size(); i++) {
                x[i] = Get(s.carving, s.target, ps[i]) / ps[i].unit;     // Adam works in each parameter's unit
                keys[i] = ps[i].Key();
            }
            s.adam.Step(keys, x, grad);
            for (size_t i = 0; i < ps.size(); i++) Set(s.carving, s.target, ps[i], x[i] * ps[i].unit, loss.L());
            Tidy(s.carving);
            s.target = loss.KeepTargetInFrame(s.target);
            s.score = score;
            return score;
        }
    }

    AutoResult AutoDesign(const TriMesh& mesh, const Design& start, const AutoParams& p, JobControl& c, std::shared_ptr<AutoLive> live) {
        AutoResult res;
        if (mesh.indices.empty()) { res.message = "The mesh has no triangles."; return res; }

        // 1. the target
        c.SetStage(p.semanticWeight > 0.0f ? "Drawing the mesh's views, loading DINOv3" : "Drawing the mesh's silhouettes");
        Loss loss(mesh, p);
        if (p.semanticWeight > 0.0f && !loss.Semantic()) {      // asked for, and couldn't be had: say so rather than go without
            res.message = loss.Error();
            return res;
        }
        std::mt19937 rng(p.seed);
        RuleContext rules{ loss, p, rng };
        const int budget = std::max(0, p.steps);

        if (budget == 0) {                                  // only score the design as it is
            AutoScore s = loss.Evaluate(start, Placement{});
            res.ok = s.valid;
            res.design = start;
            res.score = s;
            res.message = Utils::Format("Scored: silhouettes cover %.0f%% of the mesh's (+%.0f%% extra), worst view %.0f%%; overlapping "
                                        "%.1f%% open, %.1f%% on the way (loss %.3f).",
                                        s.viewCover * 100.0f, s.viewExtra * 100.0f, s.worstView * 100.0f, s.clashOpen * 100.0f,
                                        s.clashMotion * 100.0f, s.total);
            if (live) {
                std::lock_guard<std::mutex> l(live->mutex);
                live->best = start;
                live->score = s;
                loss.Picture(&live->views, &live->viewsWidth, &live->viewsHeight);
                loss.SemanticPicture(&live->semanticViews, &live->semanticWidth, &live->semanticHeight);
                live->version++;
            }
            return res;
        }

        // 2. the start: the sphere, whole
        State cur;
        cur.carving = Carving::Sphere(loss.SphereCentre(), loss.SphereRadius());
        // the mesh at the sphere's volume, so the sphere's material fills it exactly (not a parameter: free, it shrank)
        cur.target.scale = p.scaleTarget ? loss.VolumeScale() : 1.0f;
        cur.target = loss.KeepTargetInFrame(cur.target);
        cur.adam.lr = p.learningRate;
        Carving best;
        AutoScore bestScore;
        int steps = 0, rounds = 0, accepted = 0;
        std::map<std::string, int> kept;                     // rewrites kept, by rule
        auto now = std::chrono::steady_clock::now();
        auto lastShown = now - std::chrono::seconds(1);

        // keeps the best carving seen, and shows it now and then
        auto consider = [&](const Carving& k, const AutoScore& s) {
            if (s.valid && (!bestScore.valid || s.total < bestScore.total)) {
                best = k;
                bestScore = s;
            }
            if (!live) return;
            auto t = std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> l(live->mutex);
            live->iteration = steps;
            live->iterations = budget;
            live->rounds = rounds;
            live->accepted = accepted;
            if (live->history.size() < 2000) live->history.push_back(s.total);
            while ((int)live->frames.size() < live->recordFrames &&
                   (long long)steps * std::max(1, live->recordFrames - 1) >= (long long)live->frames.size() * std::max(1, budget))
                live->frames.push_back({ best.Build(), best, bestScore, steps });
            if (t - lastShown < std::chrono::milliseconds(250) && steps < budget) return;
            lastShown = t;
            live->best = best.Build();
            live->bestCarving = best;
            live->score = bestScore;
            loss.Picture(&live->views, &live->viewsWidth, &live->viewsHeight);
            loss.SemanticPicture(&live->semanticViews, &live->semanticWidth, &live->semanticHeight);
            live->version++;
        };

        c.SetStage("Optimizing");
        while (steps < budget && !c.Cancelled()) {
            // 3. an Adam step
            Carving before = cur.carving;
            AutoScore s = AdamStep(cur, loss, p);
            steps++;
            consider(before, s);                             // the score is of the carving before the step
            c.SetProgress((float)steps / budget);

            // ... and now and then a rewrite round
            if (steps % std::max(1, p.stepsPerRound) != 0 || steps + p.lookahead > budget) continue;
            rounds++;
            std::vector<float> weights = p.ruleWeights;      // the UI may have changed them since the start
            if (live) {
                std::lock_guard<std::mutex> l(live->mutex);
                if (!live->ruleWeights.empty()) weights = live->ruleWeights;
            }
            std::vector<State> tries = { cur };              // the carving unchanged ...
            for (int k = 0; k < p.rewritesPerRound; k++) {   // ... and rewritten ones
                State t = cur;
                t.rule = ApplyRandomRule(t.carving, rules, weights);
                if (t.rule) tries.push_back(std::move(t));
            }
            for (State& t : tries)
                for (int k = 0; k < p.lookahead && steps < budget; k++) {
                    Carving before = t.carving;
                    AdamStep(t, loss, p);
                    steps++;
                    consider(before, t.score);
                }
            // each judged against leaving the carving alone, by a margin: a rewrite that adds something has to beat it
            // by rewriteMargin (it adds parameters for good), one that takes something away may be up to rewriteMargin
            // worse (a simpler design that does as well is better)
            auto judged = [&](const State& t) {
                if (!t.rule) return t.score.total;
                return t.score.total / (t.rule->simplifies ? 1.0f + p.rewriteMargin : 1.0f - p.rewriteMargin);
            };
            auto winner = std::min_element(tries.begin(), tries.end(), [&](const State& a, const State& b) { return judged(a) < judged(b); });
            if (winner != tries.begin()) {
                accepted++;
                kept[winner->rule->name]++;
            }
            cur = std::move(*winner);
            cur.rule = nullptr;
        }

        // 4. the result
        AutoScore last = loss.Evaluate(cur.carving, cur.target);
        consider(cur.carving, last);
        Design design = best.Build();
        if (live) {
            std::lock_guard<std::mutex> l(live->mutex);
            if (live->recordFrames > 0) live->frames.push_back({ design, best, bestScore, steps });
            live->best = design;
            live->bestCarving = best;
            live->score = bestScore;
            live->version++;
        }
        res.ok = bestScore.valid;
        res.cancelled = c.Cancelled();
        res.design = design;
        res.carving = best;
        res.score = bestScore;
        const Placement& t = bestScore.target;
        res.message = Utils::Format("%s the sphere into %d pieces on %d joints: silhouettes cover %.0f%% of the mesh's (+%.0f%% extra), "
                                    "worst view %.0f%%; overlapping %.1f%% open, %.1f%% on the way; the mesh moved %.1f mm and scaled %.2fx "
                                    "(loss %.3f; %d steps, %d of %d rewrite rounds changed it).",
                                    res.cancelled ? "Stopped, kept the best so far: carved" : "Carved", (int)best.pieces.size(), (int)best.joints.size(),
                                    bestScore.viewCover * 100.0f, bestScore.viewExtra * 100.0f, bestScore.worstView * 100.0f,
                                    bestScore.clashOpen * 100.0f, bestScore.clashMotion * 100.0f, Vector3Length(t.offset), t.scale,
                                    bestScore.total, steps, accepted, rounds);
        if (loss.Semantic()) {
            static const char* looks[] = { "silhouettes", "depth", "shaded" };
            res.message += Utils::Format(" DINOv3 (%s, %s, on %s): %.3f.", looks[(int)p.semanticLook],
                                         p.patchWeight <= 0.0f ? "global" : (p.patchWeight >= 1.0f ? "patches" : "global + patches"),
                                         loss.SemanticDevice().c_str(), bestScore.semantic);
        }
        if (!kept.empty()) {
            res.message += " Kept:";
            for (const auto& [rule, n] : kept) res.message += Utils::Format(" %s x%d", rule.c_str(), n);
        }
        if (!loss.Error().empty()) res.message += " " + loss.Error();
        return res;
    }
}
