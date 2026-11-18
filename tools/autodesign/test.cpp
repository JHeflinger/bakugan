// Headless run of the generator: ./autotest mesh.obj [steps] [seed] [outPrefix]
// Env: VIEWS, VRES, LR, ROUND (steps per round), REWRITES, LOOKAHEAD, MARGIN, MAXPIECES, WEIGHTS (the rules', e.g. 1,0.5),
// FIXTARGET (the mesh stays put), NOSCALE (it may move but not scale), HINGEW, COLW, CHECKS (per hinge), FREETIMING (the
// hinges' timing is Adam's, not an opening order), DEPS (hinges hold each other shut: ledges), DEPW, CARVE (carve what
// still collides when done), LATCH (a press latch on the first hinge), RAMP (what overlaps cost by the end, times COLW),
// HINGES (build the hinges' hardware; PROUD: on the axis as it is, not moved under the surface; REACH: mm a leaf may bridge).
// Semantic loss: semantic_env.hpp.
// Writes <outPrefix>_design.txt (open it in the editor) and <outPrefix>_views.ppm (silhouettes against targets).
#include "autodesign/autodesign.hpp"
#include "autodesign/loss.hpp"
#include "semantic_env.hpp"
#include "autodesign/suite.hpp"
#include "data/field.hpp"
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
    env("HINGEW", p.hingeWeight);
    env("COLW", p.collisionWeight); env("CHECKS", p.hingeChecks); env("DEPW", p.dependencyWeight); env("RAMP", p.collisionRamp);
    if (getenv("HINGES")) p.buildHinges = true;
    if (getenv("PROUD")) p.hinges.flush = false;
    env("REACH", p.hinges.reach);
    if (getenv("FREETIMING")) p.orderHinges = false;
    if (getenv("DEPS")) p.dependencies = true;
    if (getenv("CARVE")) p.carveWhenDone = true;
    if (getenv("LATCH")) p.latchWhenDone = true;
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
        // every hinge against its two bodies (as they are without hardware), at its pivot and 3.3 mm either way along
        // the axis: what a barrel there needs; in mm, whatever the run's weights
        float worst = 0.0f, sum = 0.0f;
        Design bare = r.design;
        for (Hinge& h : bare.hinges) h.hardware = false;
        for (const Hinge& h : bare.hinges) {
            BodyField a = BodyField::Body(bare, h.parent), b = BodyField::Body(bare, h.child);
            float g = 0.0f;
            for (float w : { -3.3f, 0.0f, 3.3f }) {
                Vector3 q = Vector3Add(h.pivot, Vector3Scale(Vector3Normalize(h.axis), w));
                g = fmaxf(g, fmaxf(a.Distance(q), b.Distance(q)));
            }
            worst = fmaxf(worst, g);
            sum += g;
        }
        printf("hinges: farthest %.1f mm off a body along the barrel, mean %.1f mm (%d hinges)\n", worst, r.design.hinges.empty() ? 0.0f : sum / r.design.hinges.size(),
               (int)r.design.hinges.size());
    }
    const Placement& t = r.score.target;
    printf("mesh: moved %.1f %.1f %.1f mm, scaled %.3fx; sphere %.1f mm across\n", t.offset.x, t.offset.y, t.offset.z, t.scale, 2.0f * r.score.radius);
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
        // the finished design's motion checked for overlaps the same way whatever the run used: 48 poses in every
        // hinge's turn, at two phases, on a fine grid
        const double sphere = 4.0 / 3.0 * PI * pow(r.score.radius, 3.0);
        for (float phase : { 0.5f, 0.0f }) {
            MotionOverlap m = MeasureMotion(r.design, 48, r.score.radius / 12.0f, phase);
            printf("checked (%d poses): overlapping %.3f%% open, worst on the way %.3f%% at t=%.2f, up to %.2f mm deep\n", m.poses,
                   100.0 * m.openVolume / sphere, 100.0 * m.worstVolume / sphere, m.worstT, m.deepest);
        }
        // which hinges are held shut, closed (the first to open should be the only one that isn't, with dependencies)
        std::vector<int> order = OpeningOrder(r.design);
        printf("opening order:");
        for (int h : order) {
            double v = 0.0;
            bool held = HeldShut(r.design, h, 0.5f, &v);
            printf(" %s%s", r.design.hinges[h].name.c_str(), held ? " (held)" : " (free)");
        }
        printf("\n");
    }
    FILE* f = fopen((out + "_design.txt").c_str(), "w");
    fputs(r.design.Serialize().c_str(), f);
    fclose(f);
}
