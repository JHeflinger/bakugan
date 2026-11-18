// Headless run of the generator: ./autotest mesh.obj [steps] [seed] [outPrefix]
// Env: MESHSIZE (the mesh centred, its longest side this many mm), SPHERE (the sphere's diameter, mm; 0: the mesh's
// volume), MAXSECONDS (stop after, keeping the best), SNAPSHOT (seconds between progress files), THINW (thin pieces),
// MINTHICK (mm), MINCOVER, REACHW (the reach term), VIEWS, VRES, LR, ROUND (steps per round), REWRITES, LOOKAHEAD, MARGIN, MAXPIECES, WEIGHTS
// (the rules', e.g. 1,0.5), FIXTARGET (the mesh stays put), NOSCALE (it may move but not scale), HINGEW, COLW, CHECKS
// (per hinge), FREETIMING (the hinges' timing is Adam's, not an opening order), DEPS (hinges hold each other shut:
// ledges), DEPW, LATCH (a press latch on the first hinge), RAMP (what overlaps cost by the end, times COLW), NOHINGES
// (don't build the hinges' hardware), NOCARVE (don't carve what still collides when done), PROUD (hinges on the axis
// as it is, not moved under the surface), REACH (mm a leaf may bridge).
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
    if (const char* ms = getenv("MESHSIZE")) {          // centred, its longest side this many mm (as the CLI's mesh_size)
        Vector3 lo = { 1e30f, 1e30f, 1e30f }, hi = { -1e30f, -1e30f, -1e30f };
        for (Vector3 q : mesh.positions) { lo = Vector3Min(lo, q); hi = Vector3Max(hi, q); }
        float k = (float)atof(ms) / fmaxf(fmaxf(hi.x - lo.x, hi.y - lo.y), fmaxf(hi.z - lo.z, 1e-9f));
        for (Vector3& q : mesh.positions) q = Vector3Scale(Vector3Subtract(q, Vector3Scale(Vector3Add(lo, hi), 0.5f)), k);
    }
    AutoParams p;
    if (argc > 2) p.steps = atoi(argv[2]);
    if (argc > 3) p.seed = (unsigned)atoi(argv[3]);
    std::string out = argc > 4 ? argv[4] : "out";
    auto env = [](const char* name, auto& value) { if (const char* s = getenv(name)) value = (std::remove_reference_t<decltype(value)>)atof(s); };
    env("VIEWS", p.views); env("VRES", p.viewResolution); env("LR", p.learningRate); env("ROUND", p.stepsPerRound);
    env("REWRITES", p.rewritesPerRound); env("LOOKAHEAD", p.lookahead); env("MAXPIECES", p.maxPieces); env("MARGIN", p.rewriteMargin);
    env("SPHERE", p.sphereDiameter); env("CORE", p.core); env("SHELLFIRST", p.shellFirst);
    if (getenv("FIXTARGET")) p.moveTarget = p.scaleTarget = false;
    if (getenv("NOSCALE")) p.scaleTarget = false;
    if (getenv("FREESCALE")) p.freeScale = true;
    env("REACHW", p.reachWeight); env("MINCOVER", p.minViewCoverage); env("HINGEW", p.hingeWeight); env("THINW", p.thinWeight); env("MINTHICK", p.minThickness);
    env("COLW", p.collisionWeight); env("CHECKS", p.hingeChecks); env("DEPW", p.dependencyWeight); env("RAMP", p.collisionRamp);
    if (getenv("NOHINGES")) p.buildHinges = false;
    if (getenv("PROUD")) p.hinges.flush = false;
    env("REACH", p.hinges.reach);
    if (getenv("FREETIMING")) p.orderHinges = false;
    if (getenv("DEPS")) p.dependencies = true;
    if (getenv("NOCARVE")) p.carveWhenDone = false;
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
                fprintf(f, "%6.0f min  step %d/%d  pieces %d  loss %.4f  covered %.0f%% (+%.0f%%)  kept %d of %d rewrite rounds\n",
                        el / 60.0, live->iteration, live->iterations, (int)live->bestCarving.pieces.size(), sc.total, sc.viewCover * 100.0f,
                        sc.viewExtra * 100.0f, live->accepted, live->rounds);
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
    printf("pieces: thinnest %.1f mm across (min %.1f mm)\n", r.score.thinnest, p.minThickness);
    const Placement& t = r.score.target;
    printf("mesh: moved %.1f %.1f %.1f mm, scaled %.3fx; sphere %.1f mm across\n", t.offset.x, t.offset.y, t.offset.z, t.scale, 2.0f * r.score.radius);
    printf("SUMMARY pieces %d cover %.3f extra %.3f reach %.3f\n", (int)r.carving.pieces.size(), r.score.viewCover, r.score.viewExtra, r.score.reach);
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
            int in = 0;
            for (auto& [f, g] : pieces) in += f.Distance(xf[g].ApplyInverse(q)) < 0.0f;
            none += ds < 0.0f && in == 0;
            outside += ds > 0.0f && in > 0;
            two += in > 1;
        }
        printf("closed: of %d points, %d in the sphere but no piece, %d outside it in a piece, %d in two pieces\n", n, none, outside, two);
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
    {
        PrintCheck pc = CheckPrintable(r.design), raw = CheckPrintable(r.carving.Build());
        printf("CARVING loose %d loosevol %.1f\n", raw.loose, raw.looseVolume);
        printf("PRINT ok %d bodies %d loose %d loosevol %.1f hinges %d held %d\n%s\n", (int)pc.ok, pc.bodies, pc.loose, pc.looseVolume, pc.hinges, pc.held,
               pc.message.c_str());
    }
    FILE* f = fopen((out + "_design.txt").c_str(), "w");
    fputs(r.design.Serialize().c_str(), f);
    fclose(f);
}
