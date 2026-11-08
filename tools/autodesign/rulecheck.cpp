// Checks that splitting a body changes no pose, and merging two keeps the open pose:
//   ./rulecheck mesh.obj [steps]
// It runs the generator for a while (default 600 steps) to get a design, then splits and merges copies of it, and
// compares the design's distance field in the open, half-way and closed poses at points around it, before and after.
#include "autodesign/autodesign.hpp"
#include "autodesign/grammar.hpp"
#include "autodesign/loss.hpp"
#include "autodesign/params.hpp"
#include "autodesign/suite.hpp"
#include "data/field.hpp"
#include "raymath.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace Bakugan;

namespace {
    // the design's distance at each point, in the pose at time t
    std::vector<float> Pose(const Design& d, float t, const std::vector<Vector3>& pts) {
        std::vector<Rigid> xf = d.Transforms(t, false);
        std::vector<float> out(pts.size(), 1e30f);
        for (int b : d.Bodies()) {
            BodyField f = BodyField::Body(d, d.groups[b].id);
            if (f.Empty()) continue;
            for (size_t k = 0; k < pts.size(); k++) out[k] = fminf(out[k], f.Distance(xf[b].ApplyInverse(pts[k])));
        }
        return out;
    }

    // how much it changed outside near the surface (mm), and how many points went from inside to outside or back
    void Compare(const char* what, const std::vector<float>& a, const std::vector<float>& b) {
        float worst = 0.0f;
        int flips = 0;
        for (size_t k = 0; k < a.size(); k++) {
            if ((a[k] < 0.0f) != (b[k] < 0.0f)) flips++;
            if (a[k] >= 0.0f && a[k] < 1.0f) worst = fmaxf(worst, fabsf(a[k] - b[k]));
        }
        printf(" %s %.4f mm/%d flipped", what, worst, flips);
    }

    const Rule* Find(const char* name) {
        for (const Rule& r : Rules())
            if (!strcmp(r.name, name)) return &r;
        return nullptr;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s mesh.obj [steps]\n", argv[0]); return 1; }
    TriMesh mesh;
    std::string err;
    if (!LoadObj(argv[1], &mesh, &err)) { printf("can't load %s: %s\n", argv[1], err.c_str()); return 1; }
    AutoParams p;
    p.steps = argc > 2 ? atoi(argv[2]) : 600;
    JobControl c;
    Design d = AutoDesign(mesh, Design{}, p, c, nullptr).design;
    Loss loss(mesh, p);
    std::mt19937 rng(7);
    RuleContext ctx{ loss, p, rng };
    auto score = [&](const Design& x) { return loss.Evaluate(x, loss.SphereStart(), DesignParams(x, loss.L()), nullptr).total; };
    std::vector<Vector3> pts;
    BoundingBox box = d.Bounds();
    Vector3 mid = Vector3Scale(Vector3Add(box.min, box.max), 0.5f);
    std::uniform_real_distribution<float> u(-0.8f, 0.8f);
    for (int k = 0; k < 60000; k++) pts.push_back(Vector3Add(mid, Vector3Scale({ u(rng), u(rng), u(rng) }, loss.L())));
    const float base = score(d);
    printf("design: %zu bodies, %zu hinges, loss %.4f\n", d.Bodies().size(), d.hinges.size(), base);

    const Rule *split = Find("split a body"), *merge = Find("merge bodies");
    for (int k = 0; k < 6; k++) {
        Design x = d;
        if (!split->apply(x, ctx)) { printf("split: didn't apply\n"); continue; }
        printf("split -> %zu bodies: loss %+.2f%%,", x.Bodies().size(), 100.0f * (score(x) / base - 1.0f));
        Compare("open", Pose(d, 0, pts), Pose(x, 0, pts));
        Compare("half", Pose(d, 0.5f, pts), Pose(x, 0.5f, pts));
        Compare("closed", Pose(d, 1, pts), Pose(x, 1, pts));
        printf("\n");
        Design y = x;
        if (!merge->apply(y, ctx)) continue;
        printf("  then merge -> %zu bodies:", y.Bodies().size());
        Compare("open", Pose(x, 0, pts), Pose(y, 0, pts));
        printf("\n");
    }
}
