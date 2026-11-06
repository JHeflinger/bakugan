#include "autodesign/autodesign.hpp"
#include "data/field.hpp"
#include <raymath.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
using namespace Bakugan;

static TriMesh ReadObj(const char* path) {
    TriMesh m; std::ifstream f(path); std::string line;
    while (std::getline(f, line)) {
        std::istringstream ls(line); std::string t; ls >> t;
        if (t == "v") { Vector3 p; ls >> p.x >> p.y >> p.z; m.positions.push_back(p); }
        else if (t == "f") { unsigned a, b, c; ls >> a >> b >> c; m.indices.push_back(a - 1); m.indices.push_back(b - 1); m.indices.push_back(c - 1); }
    }
    return m;
}

// The design's union of bodies on a grid, in the given pose, as raw floats (n^3, x fastest) plus a header line.
static void Dump(const Design& d, float t, const char* path, BoundingBox box, int n) {
    std::vector<BodyField> f; std::vector<int> g;
    for (int b : d.Bodies()) { f.push_back(BodyField::Body(d, d.groups[b].id)); g.push_back(b); }
    std::vector<Rigid> xf = d.Transforms(t, false);
    FILE* o = fopen(path, "wb");
    fprintf(o, "%d %f %f %f %f %f %f\n", n, box.min.x, box.min.y, box.min.z, box.max.x, box.max.y, box.max.z);
    std::vector<float> v((size_t)n * n * n);
    for (int z = 0; z < n; z++) for (int y = 0; y < n; y++) for (int x = 0; x < n; x++) {
        Vector3 p = { box.min.x + (box.max.x - box.min.x) * x / (n - 1), box.min.y + (box.max.y - box.min.y) * y / (n - 1), box.min.z + (box.max.z - box.min.z) * z / (n - 1) };
        float best = 1e6f;
        for (size_t i = 0; i < f.size(); i++) best = fminf(best, f[i].Distance(xf[g[i]].ApplyInverse(p)));
        v[((size_t)z * n + y) * n + x] = best;
    }
    fwrite(v.data(), 4, v.size(), o); fclose(o);
}

int main(int argc, char** argv) {
    TriMesh mesh = ReadObj(argv[1]);
    AutoParams p;
    p.iterations = argc > 2 ? atoi(argv[2]) : 1500;
    p.sphereBias = argc > 3 ? atof(argv[3]) : 0.5f;
    p.quality = argc > 4 ? atoi(argv[4]) : 0;
    p.seed = argc > 5 ? atoi(argv[5]) : 1;
    p.seedBodies = argc > 7 ? atoi(argv[7]) : 3;
    p.maxBodies = argc > 8 ? atoi(argv[8]) : 6;
    JobControl c;
    auto live = std::make_shared<AutoLive>();
    auto t0 = std::chrono::steady_clock::now();
    Design start;
    if (argc > 9 && strstr(argv[9], ".txt")) {
        std::ifstream in(argv[9]); std::stringstream ss; ss << in.rdbuf(); std::string err;
        if (!start.Deserialize(ss.str(), &err)) { printf("bad design %s\n", err.c_str()); return 1; }
        p.fromCurrent = true;
    } else if (argc > 9 && atoi(argv[9]) >= 0) { start = MakeTemplate(atoi(argv[9])); p.fromCurrent = true; }
    if (getenv("PROBE")) {
        // one shape at a given place: PROBE="x y z sx sy sz"
        SuperFrustum f; sscanf(getenv("PROBE"), "%f %f %f %f %f %f", &f.position.x, &f.position.y, &f.position.z, &f.size.x, &f.size.y, &f.size.z);
        f.rounding = 1.0f; start = Design{}; start.AddShape(f, start.groups[start.AddGroup("Body", -1)].id, "Shape"); p.fromCurrent = true;
    }
    if (getenv("SRD_K")) p.rewritesPerRound = atoi(getenv("SRD_K"));
    if (getenv("SRD_STEPS")) p.stepsPerRound = atoi(getenv("SRD_STEPS"));
    if (getenv("LOSS")) p.loss = atoi(getenv("LOSS")) ? AutoLoss::Volume : AutoLoss::Silhouettes;
    if (getenv("VIEWS")) p.views = atoi(getenv("VIEWS"));
    if (getenv("VRES")) p.viewResolution = atoi(getenv("VRES"));
    if (getenv("COLW")) p.collisionWeight = atof(getenv("COLW"));
    if (getenv("SIMP")) p.simplicity = atof(getenv("SIMP"));
    if (getenv("MOTION")) p.motionChecks = atoi(getenv("MOTION"));
    if (getenv("TIMING")) p.allowTiming = atoi(getenv("TIMING")) != 0;
    if (argc > 10) { p.allowCuts = argv[10][0] == '1'; p.allowFeatures = argv[10][1] == '1'; p.maxShapes = atoi(argv[10] + 2); }
    AutoResult r = AutoDesign(mesh, start, p, c, live);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("outside %.3f spill %.3f  clash open %.3f closed %.3f  collision %.3f\n", r.score.fitOutside, r.score.fitSpill, r.score.clashOpen, r.score.clashClosed, r.score.collision);
    printf("%s\n%.1f s  fit %.3f (iou %.3f gap %.3f)  sphere %.3f (iou %.3f gap %.3f)  penalty %.4f  R %.1f\n", r.message.c_str(), s,
           r.score.fit, r.score.fitIoU, r.score.fitGap, r.score.sphere, r.score.sphereIoU, r.score.sphereGap, r.score.penalty, r.score.radius);
    printf("rewrites %d in %d rounds, %d scorings, step %.3f\n", live->accepted, live->rounds, live->iteration, live->stepSize); printf("history:"); for (size_t i = 0; i < live->history.size(); i += std::max<size_t>(1, live->history.size() / 12)) printf(" %.3f", live->history[i]); printf("\n");
    std::string out = std::string(argc > 6 ? argv[6] : "out");
    if (live->viewsWidth > 0) {
        FILE* vf = fopen((out + "_views.ppm").c_str(), "wb");
        fprintf(vf, "P6 %d %d 255\n", live->viewsWidth, live->viewsHeight);
        for (size_t k = 0; k < live->views.size(); k += 4) fwrite(&live->views[k], 1, 3, vf);
        fclose(vf);
    }
    printf("views: cover %.3f extra %.3f | sphere cover %.3f extra %.3f | worst %.3f penalty %.3f offframe %.3f\n", r.score.viewCover, r.score.viewExtra,
           r.score.sphereViewCover, r.score.sphereViewExtra, r.score.worstView, r.score.coveragePenalty, r.score.offFrame);
    FILE* tf = fopen((out + "_design.txt").c_str(), "w"); fputs(r.design.Serialize().c_str(), tf); fclose(tf);
    // re-load to check the design survives a save
    Design back; std::string err; if (!back.Deserialize(r.design.Serialize(), &err)) printf("DESERIALIZE FAILED %s\n", err.c_str());
    BoundingBox box = { { -60, -45, -60 }, { 60, 75, 60 } };
    Dump(r.design, 0.0f, (out + "_open.raw").c_str(), box, 72);
    Dump(r.design, 1.0f, (out + "_closed.raw").c_str(), box, 72);
    {
        // the result rescored with a fine motion check (32 moments, finest samples), the same for every run
        AutoParams q = p;
        q.iterations = 0; q.fromCurrent = true; q.motionChecks = 32; q.quality = 2;
        JobControl c2;
        AutoResult e = AutoDesign(mesh, r.design, q, c2, nullptr);
        int retimed = 0, own = 0;
        for (const Hinge& h : r.design.hinges) { retimed += h.t0 > 0.001f || h.t1 < 0.999f; own += h.ownOpening; }
        printf("fine check: on the way worst %.2f%% at t=%.2f%s, open %.2f%% closed %.2f%% | %d of %d hinges retimed, %d own opening\n",
               e.score.clashMotion * 100.0f, e.score.clashMotionT, e.score.clashMotionOpening ? " (opening)" : "",
               e.score.clashOpen * 100.0f, e.score.clashClosed * 100.0f, retimed, (int)r.design.hinges.size(), own);
    }
    printf("sphere centre %.1f %.1f %.1f\n", r.score.centre.x, r.score.centre.y, r.score.centre.z);
    return 0;
}
