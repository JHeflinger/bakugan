// Headless run of the generator: ./autotest mesh.obj [steps] [sphereBias] [seed] [outPrefix]
// Env: VIEWS, VRES, LR, ROUND (steps per round), REWRITES, LOOKAHEAD, MAXBODIES, WEIGHTS (the rules', e.g. 1,1,1,0.5,0.5,0.5,0.5,0.5). Semantic loss: semantic_env.hpp.
// Writes <outPrefix>_design.txt (open it in the editor) and <outPrefix>_views.ppm (silhouettes against targets).
#include "autodesign/autodesign.hpp"
#include "semantic_env.hpp"
#include "autodesign/suite.hpp"
#include "data/field.hpp"
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
using namespace Bakugan;

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s mesh.obj [steps] [sphereBias] [seed] [outPrefix]\n", argv[0]); return 1; }
    TriMesh mesh;
    std::string err;
    if (!LoadObj(argv[1], &mesh, &err)) { printf("can't load %s: %s\n", argv[1], err.c_str()); return 1; }
    AutoParams p;
    if (argc > 2) p.steps = atoi(argv[2]);
    if (argc > 3) p.sphereBias = (float)atof(argv[3]);
    if (argc > 4) p.seed = (unsigned)atoi(argv[4]);
    std::string out = argc > 5 ? argv[5] : "out";
    auto env = [](const char* name, auto& value) { if (const char* s = getenv(name)) value = (std::remove_reference_t<decltype(value)>)atof(s); };
    env("VIEWS", p.views); env("VRES", p.viewResolution); env("LR", p.learningRate); env("ROUND", p.stepsPerRound);
    env("REWRITES", p.rewritesPerRound); env("LOOKAHEAD", p.lookahead); env("MAXBODIES", p.maxBodies);
    if (getenv("FIXSPHERE")) p.moveSphere = false;
    env("HINGEW", p.hingeWeight);
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
    AutoResult r = AutoDesign(mesh, Design{}, p, c, live);
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
    printf("sphere: centre %.1f %.1f %.1f, %.1f mm across\n", r.score.centre.x, r.score.centre.y, r.score.centre.z, 2.0f * r.score.radius);
    {
        // the result checked for overlaps the same way whatever the run used: 32 poses between, evenly over the motion
        AutoParams q = p;
        q.collisionWeight = 1.0f;
        q.motionChecks = 32;
        q.motionCheckBias = 0.0f;
        q.steps = 0;
        q.fromCurrent = true;
        JobControl c2;
        AutoResult e = AutoDesign(mesh, r.design, q, c2, nullptr);
        printf("checked: overlapping %.2f%% open, %.2f%% closed, worst between %.2f%% at t=%.2f\n", e.score.clashOpen * 100.0f,
               e.score.clashClosed * 100.0f, e.score.clashMotion * 100.0f, e.score.clashMotionT);
    }
    FILE* f = fopen((out + "_design.txt").c_str(), "w");
    fputs(r.design.Serialize().c_str(), f);
    fclose(f);
}
