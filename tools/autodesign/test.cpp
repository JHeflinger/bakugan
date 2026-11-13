// Headless run of the generator: ./autotest mesh.obj [steps] [seed] [outPrefix]
// Env: MAXSECONDS (stop after, keeping the best), SNAPSHOT (seconds between progress files), SURFW, MAXHOLLOWS, OTW (transport weight), OTN (its points), SDFW, CHAMW, SHAPEN (3D terms; 0: off), VIEWS, VRES, LR, ROUND (steps per round), REWRITES, LOOKAHEAD, MARGIN, MAXPIECES, WEIGHTS (the rules', e.g. 1,0.5),
// FIXTARGET (the mesh stays put), NOSCALE (it may move but not scale), HINGEW, COLW, CHECKS, CHECKBIAS.
// Semantic loss: semantic_env.hpp.
// Writes <outPrefix>_design.txt (open it in the editor) and <outPrefix>_views.ppm (silhouettes against targets).
#include "autodesign/autodesign.hpp"
#include "autodesign/loss.hpp"
#include "semantic_env.hpp"
#include "autodesign/suite.hpp"
#include "data/field.hpp"
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <raymath.h>
using namespace Bakugan;

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s mesh.obj [steps] [seed] [outPrefix]\n", argv[0]); return 1; }
    TriMesh mesh;
    std::string err;
    if (!LoadObj(argv[1], &mesh, &err)) { printf("can't load %s: %s\n", argv[1], err.c_str()); return 1; }
    AutoParams p;
    if (argc > 2) p.steps = atoi(argv[2]);
    if (argc > 3) p.seed = (unsigned)atoi(argv[3]);
    std::string out = argc > 4 ? argv[4] : "out";
    auto env = [](const char* name, auto& value) { if (const char* s = getenv(name)) value = (std::remove_reference_t<decltype(value)>)atof(s); };
    env("VIEWS", p.views); env("VRES", p.viewResolution); env("LR", p.learningRate); env("ROUND", p.stepsPerRound);
    env("REWRITES", p.rewritesPerRound); env("LOOKAHEAD", p.lookahead); env("MAXPIECES", p.maxPieces); env("MARGIN", p.rewriteMargin);
    if (getenv("FIXTARGET")) p.moveTarget = p.scaleTarget = false;
    if (getenv("NOSCALE")) p.scaleTarget = false;
    env("HINGEW", p.hingeWeight); env("SURFW", p.surfaceWeight); env("MAXHOLLOWS", p.maxHollows);
    env("OTW", p.transportWeight); env("OTN", p.transportPoints); env("SDFW", p.sdfWeight); env("CHAMW", p.chamferWeight); env("SHAPEN", p.shapeSamples);
    env("COLW", p.collisionWeight); env("CHECKS", p.motionChecks); env("CHECKBIAS", p.motionCheckBias);
    SemanticFromEnv(p);
    if (const char* w = getenv("WEIGHTS"))          // the grammar's weights, comma separated, in AutoRules() order
        for (const char* q = w; *q; q++) {
            p.ruleWeights.push_back((float)atof(q));
            while (*q && *q != ',') q++;
            if (!*q) break;
        }

    JobControl c;
    auto live = std::make_shared<AutoLive>();
    auto t0 = std::chrono::steady_clock::now();
    // for long runs: MAXSECONDS stops it (keeping the best so far) after that long; SNAPSHOT=seconds writes the best
    // design so far, its views and a progress line that often (<outPrefix>_progress_design.txt, _progress_views.ppm,
    // _progress.log), so a run cut short still leaves something to open
    double maxSeconds = getenv("MAXSECONDS") ? atof(getenv("MAXSECONDS")) : 0.0, snapshot = getenv("SNAPSHOT") ? atof(getenv("SNAPSHOT")) : 0.0;
    std::atomic<bool> finished{ false };
    std::thread watcher([&] {
        auto lastSnap = std::chrono::steady_clock::now();
        while (!finished) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (maxSeconds > 0.0 && el > maxSeconds) c.Cancel();
            if (snapshot <= 0.0 || std::chrono::duration<double>(std::chrono::steady_clock::now() - lastSnap).count() < snapshot) continue;
            lastSnap = std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> l(live->mutex);
            if (live->best.groups.empty()) continue;
            if (FILE* f = fopen((out + "_progress_design.txt").c_str(), "w")) { fputs(live->best.Serialize().c_str(), f); fclose(f); }
            if (FILE* f = fopen((out + "_progress_views.ppm").c_str(), "wb")) {
                fprintf(f, "P6 %d %d 255\n", live->viewsWidth, live->viewsHeight);
                for (size_t k = 0; k < live->views.size(); k += 4) fwrite(&live->views[k], 1, 3, f);
                fclose(f);
            }
            if (FILE* f = fopen((out + "_progress.log").c_str(), "a")) {
                const AutoScore& sc = live->score;
                fprintf(f, "%6.0f min  step %d/%d  pieces %d  loss %.4f  covered %.0f%% (+%.0f%%)  transport %.1f mm  kept %d of %d rewrite rounds\n",
                        el / 60.0, live->iteration, live->iterations, (int)live->bestCarving.pieces.size(), sc.total, sc.viewCover * 100.0f,
                        sc.viewExtra * 100.0f, sc.transportMove, live->accepted, live->rounds);
                fclose(f);
            }
        }
    });
    AutoResult r = AutoDesign(mesh, Design{}, p, c, live);
    finished = true;
    watcher.join();
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    {
        std::lock_guard<std::mutex> l(live->mutex);
        const std::vector<float>& h = live->history;
        printf("loss:");
        for (size_t k = 0; k < h.size(); k += std::max<size_t>(1, h.size() / 10)) printf(" %.3f", h[k]);
        printf(" -> %.3f\n", r.score.total);
        FILE* f = fopen((out + "_views.ppm").c_str(), "wb");
        fprintf(f, "P6 %d %d 255\n", live->viewsWidth, live->viewsHeight);
        for (size_t k = 0; k < live->views.size(); k += 4) fwrite(&live->views[k], 1, 3, f);
        fclose(f);
        if (!live->semanticViews.empty()) {
            f = fopen((out + "_semantic.ppm").c_str(), "wb");
            fprintf(f, "P6 %d %d 255\n", live->semanticWidth, live->semanticHeight);
            for (size_t k = 0; k < live->semanticViews.size(); k += 4) fwrite(&live->semanticViews[k], 1, 3, f);
            fclose(f);
        }
    }
    printf("%s\n%.1f s (%.1f ms a step)\n", r.message.c_str(), secs, 1000.0 * secs / std::max(1, p.steps));
    {
        // every pivot against its two bodies, in mm, whatever the run's weights
        float worst = 0.0f, sum = 0.0f;
        for (const Hinge& h : r.design.hinges) {
            BodyField a = BodyField::Body(r.design, h.parent), b = BodyField::Body(r.design, h.child);
            float g = fmaxf(fmaxf(a.Distance(h.pivot), b.Distance(h.pivot)), 0.0f);
            worst = fmaxf(worst, g);
            sum += g;
        }
        printf("pivots: farthest %.1f mm outside a body, mean %.1f mm (%d hinges)\n", worst, r.design.hinges.empty() ? 0.0f : sum / r.design.hinges.size(),
               (int)r.design.hinges.size());
    }
    const Placement& t = r.score.target;
    printf("mesh: moved %.1f %.1f %.1f mm, scaled %.3fx; sphere %.1f mm across\n", t.offset.x, t.offset.y, t.offset.z, t.scale, 2.0f * r.score.radius);
    {
        // the result measured in 3D the same way whatever the run used: transport, and the surfaces' gaps
        AutoParams q = p;
        q.transportWeight = q.sdfWeight = q.chamferWeight = 1.0f;
        q.semanticWeight = 0.0f;
        Loss check(mesh, q);
        AutoScore e = check.Evaluate(r.carving, r.score.target);
        printf("3D: material travels %.2f mm (rms) to fill the mesh, %.0f%% of it unfilled; mesh surface %.2f mm from the design, design %.2f mm "
               "from the mesh; sdf %.3f\n", e.transportMove, e.transportUnfilled * 100.0f, e.meshGap, e.designGap, e.sdf);
    }
    {
        // closed, the pieces should be the sphere exactly: points in none of them inside it, in one outside, in two
        const Carving& k = r.carving;
        std::vector<Rigid> xf = r.design.Transforms(1.0f, false);
        std::vector<std::pair<BodyField, int>> pieces;
        for (int g : r.design.Bodies()) pieces.push_back({ BodyField::Body(r.design, r.design.groups[g].id), g });
        int none = 0, outside = 0, two = 0, n = 20000;
        unsigned seed = 5;
        auto u = [&] { seed = seed * 1664525u + 1013904223u; return (seed >> 8) / 16777216.0f * 2.0f - 1.0f; };
        for (int i = 0; i < n; i++) {
            Vector3 q = Vector3Add(k.centre, Vector3Scale({ u(), u(), u() }, 1.2f * k.radius));
            float ds = Vector3Distance(q, k.centre) - k.radius;
            if (fabsf(ds) < 0.002f * k.radius) continue;
            bool hollowed = false;                      // (hollows take material away: no piece there is right)
            for (const Cutter& h : k.cutters) hollowed |= h.hollow && h.sf.Distance(q) < 0.0f;
            if (hollowed) continue;
            int in = 0;
            for (auto& [f, g] : pieces) in += f.Distance(xf[g].ApplyInverse(q)) < 0.0f;
            none += ds < 0.0f && in == 0;
            outside += ds > 0.0f && in > 0;
            two += in > 1;
        }
        printf("closed: of %d points, %d in the sphere (less its hollows) but no piece, %d outside it in a piece, %d in two pieces; "
               "%d hollows, %.1f%% of the surface carved\n", n, none, outside, two, k.Hollows(), r.score.surfaceCarved * 100.0f);
    }
    {
        // the result checked for overlaps the same way whatever the run used: 32 poses between, evenly over the motion
        AutoParams q = p;
        q.collisionWeight = 1.0f;
        q.motionChecks = 32;
        q.motionCheckBias = 0.0f;
        q.semanticWeight = 0.0f;
        Loss check(mesh, q);
        AutoScore e = check.Evaluate(r.carving, t);
        printf("checked: overlapping %.2f%% open, worst between %.2f%% at t=%.2f\n", e.clashOpen * 100.0f, e.clashMotion * 100.0f,
               e.clashMotionT);
    }
    FILE* f = fopen((out + "_design.txt").c_str(), "w");
    fputs(r.design.Serialize().c_str(), f);
    fclose(f);
}
