// Headless run of the generator: ./autotest mesh.obj [steps] [sphereBias] [seed] [outPrefix]
// Env: VIEWS, VRES, LR, ROUND (steps per round), REWRITES, LOOKAHEAD, MAXBODIES.
// Writes <outPrefix>_design.txt (open it in the editor) and <outPrefix>_views.ppm (silhouettes against targets).
#include "autodesign/autodesign.hpp"
#include "autodesign/suite.hpp"
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
    env("COLW", p.collisionWeight); env("CHECKS", p.motionChecks); env("CHECKBIAS", p.motionCheckBias);

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
    }
    printf("%s\n%.1f s (%.1f ms a step)\n", r.message.c_str(), secs, 1000.0 * secs / std::max(1, p.steps));
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
