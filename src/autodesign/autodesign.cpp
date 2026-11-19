// The optimizer: Adam on every differentiable parameter, and every so often a rewrite round in the manner of
// Stochastic Rewrite Descent (Kodnongbua et al., "Design for Descent", SIGGRAPH Asia 2025):
//
//   1. the target: the mesh (which Adam may move and scale, to wherever the design fits it best); the sphere: at the
//      mesh's centre of volume, with its volume
//   2. the start: the sphere, whole (one piece)
//   3. until the budget runs out:
//        an Adam step: the loss and its gradient, every parameter moved (the cutters', the joints', the placement's)
//        every stepsPerRound steps, a rewrite round: the opening order looked at again (the joints open one after
//        another, in the order that collides least); then rewrites drawn from the grammar (a cut added or removed),
//        each (and the carving unchanged) given `lookahead` Adam steps; whichever is best afterwards carries on (one
//        that adds only if it is better by rewriteMargin, one that takes away if it is no worse than that)
//   4. the result: the best carving seen, built into a design (a body per piece, a hinge per joint), then finished
//      as asked (data/finish.hpp): ledges that make the hinges hold each other shut, what still collides carved
//      away, a latch on the first hinge; and its motion checked once more, densely
//
// Every carving's pieces fill the sphere exactly (carving.hpp), so the closed pose needs no scoring: it's the sphere.
#include "autodesign.hpp"
#include "adam.hpp"
#include "grammar.hpp"
#include "loss.hpp"
#include "semvideo.hpp"
#include <memory>
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
        AutoScore AdamStep(State& s, Loss& loss, const AutoParams& p, float phase) {
            std::vector<Param> ps = CarvingParams(s.carving, loss.L());
            std::vector<Param> tp = TargetParams(p.moveTarget, p.scaleTarget, loss.L());
            ps.insert(ps.end(), tp.begin(), tp.end());
            std::vector<float> grad;
            AutoScore score = loss.Evaluate(s.carving, s.target, ps, &grad, phase);
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
        // where a step's work ran (said live, so a GPU that isn't being used shows at once, not at the end)
        auto devices = [&](const AutoScore& s, bool* shortfall) {
            *shortfall = false;
            std::string d = "Views drawn ";
            if (!p.gpu) d += "on the CPU";
            else if (s.traced) d += "by " + p.gpu->Name();
            else {
                d += "on the CPU (the GPU didn't answer: see the editor's GPU line)";
                *shortfall = true;
            }
            if (loss.Semantic()) {
                std::string where = loss.SemanticDevice();
                d += "; DINOv3 on " + where;
                if (p.semanticGpu && where != "CUDA") *shortfall = true;
                if (p.gpu) d += s.semanticTraced ? " (its pictures drawn on the GPU)" : " (its pictures drawn on the CPU)";
            }
            if (p.gpu) d += s.gradientTraced ? "; the gradient's field measurements on the GPU" : "; the gradient on the CPU";
            else d += "; the gradient on the CPU";
            d += "; collisions, shading and rebuilding the pieces on the CPU";
            if (s.msViews + s.msPrepare + s.msMeasure > 0.0f)
                d += Utils::Format(". Last step: views %.0f ms, semantic %.0f, collisions %.0f, one piece %.0f, gradient %.0f preparing + %.0f measuring, "
                                   "placement %.0f",
                                   s.msViews, s.msSemantic, s.msCollisions, s.msOnePiece, s.msPrepare, s.msMeasure, s.msPlacement);
            return d;
        };
        if (live) {
            std::lock_guard<std::mutex> l(live->mutex);
            AutoScore none;
            none.traced = p.gpu != nullptr;
            none.semanticTraced = p.gpu != nullptr;
            live->devices = devices(none, &live->deviceShortfall);
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

        // 2. the start: the sphere, whole (or the carving of a design to carry on from)
        State cur;
        cur.carving = Carving::Sphere(loss.SphereCentre(), loss.SphereRadius());
        std::string started;                                 // what the result's message says about the start
        if (p.startFrom) {
            c.SetStage("Reading the design to start from");
            const AutoStart& from = *p.startFrom;
            Carving k;
            std::string err, notes;
            if (!CarvingFromDesign(from.design, &k, &err, &notes)) {
                res.message = "Can't start from " + from.source + ": " + err + ".";
                return res;
            }
            // onto this run's sphere: the same carving, moved and scaled with it (the identity for a design made for this
            // mesh with these settings; another mesh size or sphere size maps over)
            const Vector3 was = k.centre;
            const float s = loss.SphereRadius() / k.radius;
            auto map = [&](Vector3 q) { return Vector3Add(loss.SphereCentre(), Vector3Scale(Vector3Subtract(q, was), s)); };
            for (Cutter& u : k.cutters) {
                u.sf.position = map(u.sf.position);
                u.sf.size = Vector3Scale(u.sf.size, s);
                u.sf.dilation *= s;
                u.sf.onion *= s;
            }
            for (Joint& j : k.joints) j.pivot = map(j.pivot);
            k.centre = loss.SphereCentre();
            k.radius = loss.SphereRadius();
            cur.carving = k;
            // where the mesh was placed for it, carried over: its offset from the sphere grows with the sphere; its scale is
            // relative to the mesh's own size, so it stays unless the sphere changed size against the mesh
            if (from.hasTarget) {
                const float meshRatio = from.meshSize > 0.0f ? loss.L() / from.meshSize : s;
                if (p.moveTarget) cur.target.offset = Vector3Scale(from.target.offset, s);
                if (p.scaleTarget) cur.target.scale = std::clamp(from.target.scale * s / meshRatio, expf(-1.2f), expf(1.2f));
                cur.target = loss.KeepTargetInFrame(cur.target);
            }
            int hollows = k.Hollows();
            started = Utils::Format("Started from %s (%d piece%s, %d joint%s, %d hollow%s", from.source.c_str(), (int)k.pieces.size(),
                                    k.pieces.size() == 1 ? "" : "s", (int)k.joints.size(), k.joints.size() == 1 ? "" : "s", hollows, hollows == 1 ? "" : "s");
            if (fabsf(s - 1.0f) > 1e-3f || Vector3Distance(was, loss.SphereCentre()) > 1e-3f * loss.SphereRadius())
                started += Utils::Format("; its %.1f mm sphere moved and scaled onto this run's %.1f mm one", 2.0f * k.radius / s, 2.0f * loss.SphereRadius());
            started += from.hasTarget ? ")." : "; it had no saved mesh placement, so the mesh starts where it is).";
            if (!notes.empty()) started += " " + notes;
            started += " ";
        }
        cur.carving.ordered = p.orderHinges || p.dependencies;
        Tidy(cur.carving);
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
        int steps = 0, rounds = 0, accepted = 0;
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
            live->devices = devices(s, &live->deviceShortfall);
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

        // the semantic video, frame by frame as the search goes
        std::unique_ptr<SemanticVideo> video;
        std::string videoNote;
        if (!p.semanticVideo.empty()) {
            if (loss.Semantic()) video = std::make_unique<SemanticVideo>(p.semanticVideo, p.semanticVideoFps);
            else videoNote = " No semantic video: the semantic loss is off (it needs semantic weight > 0, the model and ONNX Runtime).";
        }
        const int videoEvery = std::max(1, p.semanticVideoEvery);

        c.SetStage("Optimizing");
        while (steps < budget && !c.Cancelled()) {
            // 3. an Adam step
            Carving before = cur.carving;
            loss.SetCollisionScale(scaleAt(steps));
            AutoScore s = AdamStep(cur, loss, p, Phase(steps));
            steps++;
            consider(before, s);                             // the score is of the carving before the step
            c.SetProgress((float)steps / budget);
            if (video && (steps == 1 || steps % videoEvery == 0))     // (the pictures are the carving's before the step, as the score)
                video->Add(loss, Utils::Format("Step %d / %d", steps, budget), s, (float)steps / budget);

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
                if (t.rule) tries.push_back(std::move(t));
            }
            for (State& t : tries)
                for (int k = 0; k < p.lookahead && steps < budget; k++) {
                    Carving before = t.carving;
                    AdamStep(t, loss, p, Phase(k));              // (every try looks in the same places)
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
        bestScore = loss.Evaluate(best, bestScore.target);
        if (video) {
            c.SetStage("Finishing the semantic video");
            video->Add(loss, Utils::Format("Best design (after %d steps)", steps), bestScore, 1.0f, 2 * video->Fps());
            if (video->Finish(&videoNote)) res.semanticVideo = p.semanticVideo;
            videoNote = " " + videoNote;
            video.reset();
        }
        Design design = best.Build();
        // ... finished
        std::string finished;
        auto finish = [&](const char* stage, const std::function<FinishResult()>& pass) {
            if (c.Cancelled()) return;
            c.SetStage(stage);
            FinishResult f = pass();
            if (f.ok) design = f.design;
            if (!f.message.empty()) finished += " " + f.message;
        };
        if (bestScore.valid && best.joints.size() > 0) {
            if (p.buildHinges) finish("Building the hinges", [&] { return AddHinges(design, p.hinges, &c); });
            if (p.dependencies) finish("Adding ledges", [&] { return AddLedges(design, p.ledges, &c); });
            if (p.carveWhenDone) finish("Carving collisions away", [&] { return CarveCollisions(design, p.carve, &c); });
            if (p.latchWhenDone) finish("Adding the latch", [&] { return AddLatch(design, p.latch, &c); });
        }
        if (bestScore.valid && (p.removeFloatingWhenDone || p.onePiece)) {
            // one piece: whatever is still apart goes, however big (the term above should have left little)
            FloatingParams q = p.floating;
            if (p.onePiece) q.share = 1.0f;
            finish(p.onePiece ? "Making every body one piece" : "Removing floating bits", [&] { return RemoveFloatingBits(design, q, &c); });
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
        res.message = started + Utils::Format("%s the sphere into %d pieces on %d joints: silhouettes cover %.0f%% of the mesh's (+%.0f%% extra), "
                                    "worst view %.0f%%; the mesh moved %.1f mm and scaled %.2fx (loss %.3f; %d steps, %d of %d rewrite "
                                    "rounds changed it%s).",
                                    res.cancelled ? "Stopped, kept the best so far: carved" : "Carved", (int)best.pieces.size(), (int)best.joints.size(),
                                    bestScore.viewCover * 100.0f, bestScore.viewExtra * 100.0f, bestScore.worstView * 100.0f,
                                    Vector3Length(t.offset), t.scale, bestScore.total, steps, accepted, rounds,
                                    reordered > 0 ? Utils::Format(", the opening order changed %d times", reordered).c_str() : "");
        if (p.fitLook == FitLook::Depth) res.message += Utils::Format(" Depth mismatch %.3f.", bestScore.depth);
        if (p.onePiece && bestScore.fragmentParts > 0)
            res.message += Utils::Format(" Before finishing, %d part%s (%.2f%% of the sphere) lay apart from their bodies.", bestScore.fragmentParts,
                                         bestScore.fragmentParts == 1 ? "" : "s", 100.0f * bestScore.fragments);
        res.message += Utils::Format(" Checked at %d poses: overlapping %.2f%% of the sphere open, %.2f%% at worst on the way", res.checked.poses,
                                     100.0 * res.checked.openVolume / sphereVolume, 100.0 * res.checked.worstVolume / sphereVolume);
        if (res.checked.worstHinge >= 0 && res.checked.worstVolume > 0.0)
            res.message += Utils::Format(" (in %s's turn), up to %.2f mm deep", design.hinges[res.checked.worstHinge].name.c_str(), res.checked.deepest);
        res.message += ".";
        res.message += finished;
        if (bestScore.traced && bestScore.semanticTraced) res.message += " Silhouettes and semantic pictures drawn by " + p.gpu->Name() + ".";
        else if (bestScore.traced) res.message += " Silhouettes drawn by " + p.gpu->Name() + ".";
        else if (bestScore.semanticTraced) res.message += " Semantic pictures drawn by " + p.gpu->Name() + ".";
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
        res.message += videoNote;
        if (!loss.Error().empty()) res.message += " " + loss.Error();
        return res;
    }
}
