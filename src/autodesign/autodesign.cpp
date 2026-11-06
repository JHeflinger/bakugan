// The optimizer: Adam on every differentiable parameter, and every so often a rewrite round in the manner of
// Stochastic Rewrite Descent (Kodnongbua et al., "Design for Descent", SIGGRAPH Asia 2025):
//
//   1. the targets: the mesh's silhouettes, and a sphere of its volume (the loss also counts bodies overlapping as the
//      design opens and closes: loss.hpp)
//   2. the start: two boxes, a body each, hinged to fold one onto the other
//   3. until the budget runs out:
//        an Adam step: the loss and its gradient, every parameter moved
//        every stepsPerRound steps, a rewrite round: rewrites drawn from the grammar, each (and the design unchanged)
//        given `lookahead` Adam steps; whichever is best afterwards carries on
//   4. the result: the best design seen
#include "autodesign.hpp"
#include "adam.hpp"
#include "grammar.hpp"
#include "util/utils.hpp"
#include <raymath.h>
#include <algorithm>
#include <cfloat>
#include <chrono>

namespace Bakugan {
    namespace {
        // A design, and Adam's memory of its parameters.
        struct State {
            Design design;
            Adam adam;
            AutoScore score;             // its loss before its last step
        };

        // The start: the mesh's box halved along its longest side, a slightly smaller box in each half, a body each,
        // hinged where the halves meet so the second folds over onto the first.
        Design StartDesign(const TriMesh& mesh) {
            Vector3 lo = { FLT_MAX, FLT_MAX, FLT_MAX }, hi = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
            for (unsigned i : mesh.indices) { lo = Vector3Min(lo, mesh.positions[i]); hi = Vector3Max(hi, mesh.positions[i]); }
            Vector3 size = Vector3Subtract(hi, lo), mid = Vector3Scale(Vector3Add(lo, hi), 0.5f);
            int a = size.x >= size.y && size.x >= size.z ? 0 : (size.y >= size.z ? 1 : 2);     // longest side
            int b = (a + 1) % 3;                                                                   // the fold's axis
            if ((&size.x)[(a + 2) % 3] > (&size.x)[b]) b = (a + 2) % 3;
            Vector3 along = { 0, 0, 0 }, axis = { 0, 0, 0 };
            (&along.x)[a] = 1.0f;
            (&axis.x)[b] = 1.0f;
            float half = 0.5f * (&size.x)[a];

            Design d;
            int body[2];
            for (int k = 0; k < 2; k++) {
                SuperFrustum sf;
                sf.size = Vector3Scale(size, 0.7f);
                (&sf.size.x)[a] = 0.7f * half;
                sf.position = Vector3Add(mid, Vector3Scale(along, (k == 0 ? -0.5f : 0.5f) * half));
                sf.rounding = 0.3f;
                body[k] = d.groups[d.AddGroup("Body", -1)].id;
                d.AddShape(sf, body[k], "Shape");
            }
            Hinge& h = d.hinges[d.AddHinge(body[0], body[1])];
            h.pivot = mid;
            h.axis = axis;
            h.start = 0.0f;
            h.end = 0.9f * PI;
            h.hardware = false;
            return d;
        }

        // One Adam step: the loss and its gradient at the design, then every parameter moved. Returns the loss of the
        // design as it was before the step.
        AutoScore AdamStep(State& s, Loss& loss) {
            std::vector<Param> ps = DesignParams(s.design, loss.L());
            std::vector<float> grad;
            AutoScore score = loss.Evaluate(s.design, ps, &grad);
            std::vector<float> x(ps.size());
            std::vector<uint64_t> keys(ps.size());
            for (size_t i = 0; i < ps.size(); i++) {
                x[i] = *Ref(s.design, ps[i]) / ps[i].unit;       // Adam works in each parameter's unit
                keys[i] = ps[i].Key();
            }
            s.adam.Step(keys, x, grad);
            for (size_t i = 0; i < ps.size(); i++) SetParam(s.design, ps[i], x[i] * ps[i].unit, loss.L());
            Tidy(s.design);
            s.score = score;
            return score;
        }
    }

    AutoResult AutoDesign(const TriMesh& mesh, const Design& start, const AutoParams& p, JobControl& c, std::shared_ptr<AutoLive> live) {
        AutoResult res;
        if (mesh.indices.empty()) { res.message = "The mesh has no triangles."; return res; }

        // 1. the targets
        c.SetStage("Drawing the mesh's silhouettes");
        Loss loss(mesh, p);
        std::mt19937 rng(p.seed);
        RuleContext rules{ loss, p, rng };

        // 2. the start
        State cur;
        cur.design = p.fromCurrent && !start.shapes.empty() ? start : StartDesign(mesh);
        for (Hinge& h : cur.design.hinges) h.hardware = false;
        cur.adam.lr = p.learningRate;
        Design best;
        AutoScore bestScore;
        int steps = 0, rounds = 0, accepted = 0;
        const int budget = std::max(0, p.steps);
        auto now = std::chrono::steady_clock::now();
        auto lastShown = now - std::chrono::seconds(1);

        // keeps the best design seen, and shows it now and then
        auto consider = [&](const Design& d, const AutoScore& s) {
            if (s.valid && (!bestScore.valid || s.total < bestScore.total)) {
                best = d;
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
                live->frames.push_back({ best, bestScore, steps });
            if (t - lastShown < std::chrono::milliseconds(250) && steps < budget) return;
            lastShown = t;
            live->best = best;
            live->score = bestScore;
            loss.Picture(&live->views, &live->viewsWidth, &live->viewsHeight);
            live->version++;
        };

        c.SetStage("Optimizing");
        if (budget == 0) {                                  // only score it
            AutoScore s = loss.Evaluate(cur.design, {}, nullptr);
            consider(cur.design, s);
        }
        while (steps < budget && !c.Cancelled()) {
            // 3. an Adam step
            Design before = cur.design;
            AutoScore s = AdamStep(cur, loss);
            steps++;
            consider(before, s);                             // the score is of the design before the step
            c.SetProgress((float)steps / budget);

            // ... and now and then a rewrite round
            if (steps % std::max(1, p.stepsPerRound) != 0 || steps + p.lookahead > budget) continue;
            rounds++;
            std::vector<State> tries = { cur };              // the design unchanged ...
            for (int k = 0; k < p.rewritesPerRound; k++) {   // ... and rewritten ones
                State t = cur;
                if (ApplyRandomRule(t.design, rules)) tries.push_back(std::move(t));
            }
            for (State& t : tries)
                for (int k = 0; k < p.lookahead && steps < budget; k++) {
                    Design before = t.design;
                    AdamStep(t, loss);
                    steps++;
                    consider(before, t.score);
                }
            auto winner = std::min_element(tries.begin(), tries.end(), [](const State& a, const State& b) {
                return a.score.total < b.score.total;
            });
            accepted += winner != tries.begin();
            cur = std::move(*winner);
        }

        // 4. the result
        AutoScore last = loss.Evaluate(cur.design, {}, nullptr);
        consider(cur.design, last);
        if (live) {
            std::lock_guard<std::mutex> l(live->mutex);
            if (live->recordFrames > 0) live->frames.push_back({ best, bestScore, steps });
            live->best = best;
            live->score = bestScore;
            live->version++;
        }
        res.ok = bestScore.valid;
        res.cancelled = c.Cancelled();
        res.design = best;
        res.score = bestScore;
        res.message = Utils::Format("%s %d bodies, %d hinges: silhouettes cover %.0f%% of the mesh's (+%.0f%% extra) and %.0f%% of the "
                                    "sphere's (+%.0f%% extra), worst view %.0f%%; overlapping %.1f%% open, %.1f%% closed, %.1f%% on the way "
                                    "(loss %.3f; %d steps, %d of %d rewrite rounds changed it).",
                                    res.cancelled ? "Stopped, kept the best so far:" : (budget == 0 ? "Scored:" : "Generated"),
                                    (int)best.Bodies().size(), (int)best.hinges.size(), bestScore.viewCover * 100.0f, bestScore.viewExtra * 100.0f,
                                    bestScore.sphereViewCover * 100.0f, bestScore.sphereViewExtra * 100.0f, bestScore.worstView * 100.0f,
                                    bestScore.clashOpen * 100.0f, bestScore.clashClosed * 100.0f, bestScore.clashMotion * 100.0f, bestScore.total, steps, accepted, rounds);
        return res;
    }
}
