// The optimizer: Adam on every differentiable parameter, and every so often a rewrite round in the manner of
// Stochastic Rewrite Descent (Kodnongbua et al., "Design for Descent", SIGGRAPH Asia 2025):
//
//   1. the target: the mesh (which Adam may move and scale, to wherever the design fits it best); the sphere: at the
//      mesh's centre of volume, with its volume
//   2. the start: the sphere, whole (one piece)
//   3. until the budget runs out:
//        an Adam step: the loss and its gradient, every parameter moved (the cutters', the joints', the placement's);
//        then every hinge put back on its seam, and a cutter whose move left a piece in parts (or too small for its
//        hinges) put back where it was: the carving stays one that can be printed and hinged throughout (snap.cpp)
//        every stepsPerRound steps, a rewrite round: the opening order looked at again (the joints open one after
//        another, in the order that collides least); then rewrites drawn from the grammar (a cut or a split added, a cut removed),
//        each (and the carving unchanged) given `lookahead` Adam steps; whichever is best afterwards carries on (one
//        that adds only if it is better by rewriteMargin, one that takes away if it is no worse than that)
//   4. the result: the best carving seen, built into a design (a body per piece, a hinge per joint), then finished
//      (data/finish.hpp): the hinges' hardware, what still collides carved away, and as asked ledges that make the
//      hinges hold each other shut and a latch on the first hinge; checked for printing (every piece one solid, on a
//      hinge that was built), and its motion checked once more, densely
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
#include <functional>
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

        // Where the k-th step looks for overlaps (Loss::Evaluate's phase): each between the ones before (golden ratio).
        float Phase(int k) { return fmodf(0.5f + 0.6180339887f * k, 1.0f); }

        // One Adam step: the loss and its gradient at the carving, then every parameter moved. Returns the loss of the
        // carving as it was before.
        AutoScore AdamStep(State& s, Loss& loss, const AutoParams& p, float phase, int& held) {
            std::vector<Param> ps = CarvingParams(s.carving, loss.L());
            std::vector<Param> tp = TargetParams(p.moveTarget, p.scaleTarget && p.freeScale, loss.L());   // (the scale: only when free)
            ps.insert(ps.end(), tp.begin(), tp.end());
            std::vector<float> grad;
            AutoScore score = loss.Evaluate(s.carving, s.target, ps, &grad, phase);
            std::vector<float> x(ps.size());
            std::vector<uint64_t> keys(ps.size());
            for (size_t i = 0; i < ps.size(); i++) {
                x[i] = Get(s.carving, s.target, ps[i]) / ps[i].unit;     // Adam works in each parameter's unit
                keys[i] = ps[i].Key();
            }
            const std::vector<float> x0 = x;
            s.adam.Step(keys, x, grad);
            // Every hinge back where its two pieces meet (SnapJoints), and every piece one solid (Solid). A step that
            // parts two hinged pieces, or cuts a piece into parts that aren't joined, is taken again cutter by cutter:
            // each cutter moves only if the carving is still whole with it moved, so the one that would break it
            // stays where it is and the rest carry on.
            const Carving from = s.carving;
            const Placement at = s.target;
            auto move = [&](const std::vector<char>& which) {
                s.carving = from;
                s.target = at;
                for (size_t i = 0; i < ps.size(); i++)
                    if (which[i]) Set(s.carving, s.target, ps[i], x[i] * ps[i].unit, loss.L());
                Tidy(s.carving);
                return loss.SnapJoints(s.carving) == 0 && loss.Solid(s.carving);
            };
            std::vector<char> which(ps.size(), 1);
            if (!move(which)) {
                held++;
                for (size_t i = 0; i < ps.size(); i++) which[i] = ps[i].kind != Param::Cutter;
                if (!move(which)) which.assign(ps.size(), 0);
                for (const Cutter& k : from.cutters) {
                    std::vector<char> with = which;
                    for (size_t i = 0; i < ps.size(); i++)
                        if (ps[i].kind == Param::Cutter && ps[i].id == k.id) with[i] = 1;
                    if (move(with)) which = with;
                }
                move(which);
            }
            s.target = loss.KeepTargetInFrame(s.target);
            s.score = score;
            return score;
        }
    }

    namespace {
        // A design finished (data/finish.hpp), what the passes said, and whether it prints.
        struct Finished {
            Design design;
            std::string said;
            PrintCheck check;
        };

        void Pass(Finished& f, JobControl& c, const char* stage, const std::function<FinishResult()>& pass) {
            if (c.Cancelled()) return;
            c.SetStage(stage);
            FinishResult r = pass();
            if (r.ok) f.design = r.design;
            if (!r.message.empty()) f.said += " " + r.message;
        }

        // After the hinges: a piece without one joined to the piece it hung from, what collides carved away, what
        // that left loose cut away, and the result checked.
        void Refinish(Finished& f, const AutoParams& p, JobControl& c) {
            if (p.buildHinges) Pass(f, c, "Joining what nothing holds", [&] { return FuseUnheld(f.design); });
            if (p.dependencies) Pass(f, c, "Adding ledges", [&] { return AddLedges(f.design, p.ledges, &c); });
            if (p.carveWhenDone) Pass(f, c, "Carving collisions away", [&] { return CarveCollisions(f.design, p.carve, &c); });
            if (p.latchWhenDone) Pass(f, c, "Adding the latch", [&] { return AddLatch(f.design, p.latch, &c); });
            if (!p.buildHinges || c.Cancelled()) return;
            Pass(f, c, "Cutting loose crumbs away", [&] { return RemoveLoose(f.design, &c); });
            c.SetStage("Checking it prints");
            f.check = CheckPrintable(f.design);
        }

        // The carving's design with its hinges built (leaves reaching `reach` mm past the barrel), and the rest.
        Finished Finish(const Design& built, const AutoParams& p, float reach, JobControl& c) {
            Finished f;
            f.design = built;
            HingeParams hinges = p.hinges;
            hinges.reach = reach;
            if (p.buildHinges) Pass(f, c, "Building the hinges", [&] { return AddHinges(f.design, hinges, &c); });
            Refinish(f, p, c);
            return f;
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
            res.checked = MeasureMotion(start, 24, loss.SphereRadius() / 10.0f);
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
        {
            const float R = loss.SphereRadius();
            const float wall = fmaxf(1.2f * p.minThickness, 2.0f * PlanHinge(R, p.hinges).radius + 0.5f);   // the shell's, at least
            const float core = fminf(std::clamp(p.core, 0.0f, 0.95f) * 2.0f * R, 2.0f * (R - wall));
            cur.carving = core > 2.0f * p.minThickness ? Carving::ShellAndCore(loss.SphereCentre(), R, core)
                                                       : Carving::Sphere(loss.SphereCentre(), R);
            cur.carving.coreLocked = core > 2.0f * p.minThickness && p.shellFirst > 0.0f;   // first the shell alone
            loss.SnapJoints(cur.carving);
        }
        // the mesh at the sphere's volume, so the sphere's material fills it exactly (not a parameter: free, it shrank)
        cur.target.scale = p.scaleTarget ? loss.VolumeScale() : 1.0f;
        cur.target = loss.KeepTargetInFrame(cur.target);
        cur.carving.ordered = p.orderHinges || p.dependencies;
        cur.adam.lr = p.learningRate;
        int reordered = 0;
        // the opening order looked at again: taken when the carving does better with it (both scored alike)
        auto reorder = [&](Carving& k, const Placement& target) {
            if (!k.ordered || k.joints.size() < 2) return;
            std::vector<int> order = loss.BestOrder(k);
            if (order == k.Order()) return;
            Carving other = k;
            other.SetOrder(order);
            Tidy(other);
            if (loss.Evaluate(other, target).total < loss.Evaluate(k, target).total - 1e-5f) {
                k = other;
                reordered++;
            }
        };
        Carving best;
        AutoScore bestScore;
        int steps = 0, rounds = 0, accepted = 0, held = 0;     // held: steps in which a cutter stayed put (see AdamStep)
        std::map<std::string, int> kept;                     // rewrites kept, by rule
        auto now = std::chrono::steady_clock::now();
        auto lastShown = now - std::chrono::seconds(1);

        // what overlaps cost rises over the search, from the weight to collisionRamp times it
        const float ramp = std::max(1.0f, p.collisionRamp);
        auto scaleAt = [&](int step) { return powf(ramp, std::clamp((float)step / std::max(1, budget), 0.0f, 1.0f)); };
        // a score as it would be at the end of the ramp: what carvings from different times are compared by
        auto judged = [&](const AutoScore& s) { return s.total + (ramp - s.collisionScale) * p.collisionWeight * s.collision; };
        // keeps the best carving seen, and shows it now and then
        auto consider = [&](const Carving& k, const AutoScore& s) {
            if (s.valid && (!bestScore.valid || judged(s) < judged(bestScore))) {
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
            // the second stage: the core is free (to turn, to be cut, to thicken the shell's pieces)
            if (cur.carving.coreLocked && steps >= p.shellFirst * budget) cur.carving.coreLocked = false;
            // 3. an Adam step
            Carving before = cur.carving;
            loss.SetCollisionScale(scaleAt(steps));
            AutoScore s = AdamStep(cur, loss, p, Phase(steps), held);
            steps++;
            consider(before, s);                             // the score is of the carving before the step
            c.SetProgress((float)steps / budget);

            // ... and now and then a rewrite round
            if (steps % std::max(1, p.stepsPerRound) != 0 || steps + p.lookahead > budget) continue;
            rounds++;
            reorder(cur.carving, cur.target);
            std::vector<float> weights = p.ruleWeights;      // the UI may have changed them since the start
            if (live) {
                std::lock_guard<std::mutex> l(live->mutex);
                if (!live->ruleWeights.empty()) weights = live->ruleWeights;
            }
            std::vector<State> tries = { cur };              // the carving unchanged ...
            for (int k = 0; k < p.rewritesPerRound; k++) {   // ... and rewritten ones
                State t = cur;
                t.rule = ApplyRandomRule(t.carving, rules, weights);
                // only rewrites that leave every hinge on a seam and every piece one solid: nothing may float
                if (t.rule && loss.SnapJoints(t.carving) == 0 && loss.Solid(t.carving)) tries.push_back(std::move(t));
            }
            for (State& t : tries)
                for (int k = 0; k < p.lookahead && steps < budget; k++) {
                    Carving before = t.carving;
                    AdamStep(t, loss, p, Phase(k), held);              // (every try looks in the same places)
                    steps++;
                    consider(before, t.score);
                }
            // each judged against leaving the carving alone, by a margin: a rewrite that adds something has to beat it
            // by rewriteMargin (it adds parameters for good), one that takes something away may be up to rewriteMargin
            // worse (a simpler design that does as well is better)
            auto weighed = [&](const State& t) {
                if (!t.rule) return t.score.total;
                return t.score.total / (t.rule->simplifies ? 1.0f + p.rewriteMargin : 1.0f - p.rewriteMargin);
            };
            auto winner = std::min_element(tries.begin(), tries.end(), [&](const State& a, const State& b) { return weighed(a) < weighed(b); });
            if (winner != tries.begin()) {
                accepted++;
                kept[winner->rule->name]++;
            }
            cur = std::move(*winner);
            cur.rule = nullptr;
        }

        // 4. the result
        loss.SetCollisionScale(ramp);
        reorder(cur.carving, cur.target);
        AutoScore last = loss.Evaluate(cur.carving, cur.target);
        consider(cur.carving, last);
        reorder(best, bestScore.target);
        const int floating = loss.SnapJoints(best);         // (it was snapped when it was made; this says if any can't be)
        bestScore = loss.Evaluate(best, bestScore.target);
        Design design = best.Build();
        // ... finished. With hinges built, the result has to print: every piece one solid, on a hinge whose hardware
        // is part of both of its pieces. The cuts that make room around a hinge can leave its knuckles on a stub that
        // nothing joins to the piece, so the finishing is done again with longer leaves (up to four times as long) (they reach further into the
        // piece, and no cut reaches hardware) until it holds; a hinge that still doesn't is given up, and its piece
        // becomes part of the one it hung from.
        std::string finished;
        if (bestScore.valid && best.joints.size() > 0) {
            const Design built = design;
            const HingeSizes sizes = PlanHinge(loss.SphereRadius(), p.hinges);
            const float reach = sizes.leafLength - sizes.radius;
            Finished done;
            for (int k = 1; k <= 4 && !c.Cancelled(); k++) {
                Finished f = Finish(built, p, k * reach, c);
                if (k > 1 && f.check.ok) f.said += Utils::Format(" (Leaves %.1f mm long: shorter ones left a hinge on a stub.)", k * reach + sizes.radius);
                if (k == 1 || f.check.ok) done = std::move(f);
                if (!p.buildHinges || done.check.ok) break;
            }
            for (int round = 0; p.buildHinges && !done.check.unheld.empty() && round < 8 && !c.Cancelled(); round++) {
                for (Hinge& h : done.design.hinges)
                    if (std::find(done.check.unheld.begin(), done.check.unheld.end(), h.id) != done.check.unheld.end()) h.hardware = false;
                Refinish(done, p, c);
            }
            design = done.design;
            finished = done.said;
            res.printable = done.check;
            if (p.buildHinges && !c.Cancelled()) finished += " " + res.printable.message;
        }
        // ... and its motion checked once more, more closely than the search could afford to
        c.SetStage("Checking the motion");
        const double sphereVolume = 4.0 / 3.0 * PI * pow(loss.SphereRadius(), 3.0);
        res.checked = MeasureMotion(design, 24, loss.SphereRadius() / 10.0f);
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
                                    "worst view %.0f%%; the mesh moved %.1f mm and scaled %.2fx (loss %.3f; %d steps, %d of %d rewrite "
                                    "rounds changed it%s).",
                                    res.cancelled ? "Stopped, kept the best so far: carved" : "Carved", (int)best.pieces.size(), (int)best.joints.size(),
                                    bestScore.viewCover * 100.0f, bestScore.viewExtra * 100.0f, bestScore.worstView * 100.0f,
                                    Vector3Length(t.offset), t.scale, bestScore.total, steps, accepted, rounds,
                                    reordered > 0 ? Utils::Format(", the opening order changed %d times", reordered).c_str() : "");
        res.message += Utils::Format(" Checked at %d poses: overlapping %.2f%% of the sphere open, %.2f%% at worst on the way", res.checked.poses,
                                     100.0 * res.checked.openVolume / sphereVolume, 100.0 * res.checked.worstVolume / sphereVolume);
        if (res.checked.worstHinge >= 0 && res.checked.worstVolume > 0.0)
            res.message += Utils::Format(" (in %s's turn), up to %.2f mm deep", design.hinges[res.checked.worstHinge].name.c_str(), res.checked.deepest);
        res.message += ".";
        res.message += finished;
        if (bestScore.traced) res.message += " Silhouettes drawn by " + p.gpu->Name() + ".";
        if (loss.Semantic()) {
            static const char* looks[] = { "silhouettes", "depth", "shaded" };
            res.message += Utils::Format(" DINOv3 (%s, %s, on %s): %.3f.", looks[(int)p.semanticLook],
                                         p.patchWeight <= 0.0f ? "global" : (p.patchWeight >= 1.0f ? "patches" : "global + patches"),
                                         loss.SemanticDevice().c_str(), bestScore.semantic);
        }
        if (floating > 0)
            res.message += Utils::Format(" %d hinge%s found no seam between pieces to sit on: %s piece%s can float free.", floating,
                                         floating == 1 ? "" : "s", floating == 1 ? "its" : "their", floating == 1 ? "" : "s");
        if (held > 0) res.message += Utils::Format(" In %d steps a cutter was held back: moved, it left a piece in parts.", held);
        if (!kept.empty()) {
            res.message += " Kept:";
            for (const auto& [rule, n] : kept) res.message += Utils::Format(" %s x%d", rule.c_str(), n);
        }
        if (!loss.Error().empty()) res.message += " " + loss.Error();
        return res;
    }
}
