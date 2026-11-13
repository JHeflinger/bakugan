// Checks the grammar's two rules:
//   ./rulecheck mesh.obj
// Cuts are added to the sphere one after another (each joint then opened some, as Adam might), and after each:
//   - adding a cut changes nothing (the open pose's distance field at points around it, and the loss)
//   - closed, the pieces are still the sphere: no point of it in no piece or two, none outside it in one
// and then removed again one by one (each removal checked the same way for the closed pose).
#include "autodesign/autodesign.hpp"
#include "autodesign/grammar.hpp"
#include "autodesign/loss.hpp"
#include "autodesign/suite.hpp"
#include "data/field.hpp"
#include "raymath.h"
#include <cmath>
#include <cstdio>
#include <cstring>
using namespace Bakugan;

namespace {
    // the design's distance at each point, in the pose at time t (0 open, 1 closed)
    std::vector<float> Pose(const Carving& c, float t, const std::vector<Vector3>& pts) {
        Design d = c.Build();
        std::vector<Rigid> xf = d.Transforms(t, false);
        std::vector<float> out(pts.size(), 1e30f);
        for (int b : d.Bodies()) {
            BodyField f = BodyField::Body(d, d.groups[b].id);
            for (size_t k = 0; k < pts.size(); k++) out[k] = fminf(out[k], f.Distance(xf[b].ApplyInverse(pts[k])));
        }
        return out;
    }

    // closed, against the sphere: points in it but no piece, outside it but in one, in two pieces
    void Closed(const Carving& c, const std::vector<Vector3>& pts, int* none, int* outside, int* two) {
        Design d = c.Build();
        std::vector<Rigid> xf = d.Transforms(1.0f, false);
        std::vector<std::pair<BodyField, int>> fs;
        for (int b : d.Bodies()) fs.push_back({ BodyField::Body(d, d.groups[b].id), b });
        *none = *outside = *two = 0;
        for (Vector3 p : pts) {
            float ds = Vector3Distance(p, c.centre) - c.radius;
            if (fabsf(ds) < 0.002f * c.radius) continue;
            int in = 0;
            for (auto& [f, b] : fs) in += f.Distance(xf[b].ApplyInverse(p)) < 0.0f;
            *none += ds < 0.0f && in == 0;
            *outside += ds > 0.0f && in > 0;
            *two += in > 1;
        }
    }

    const Rule* Find(const char* name) {
        for (const Rule& r : Rules())
            if (!strcmp(r.name, name)) return &r;
        return nullptr;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s mesh.obj\n", argv[0]); return 1; }
    TriMesh mesh;
    std::string err;
    if (!LoadObj(argv[1], &mesh, &err)) { printf("can't load %s: %s\n", argv[1], err.c_str()); return 1; }
    AutoParams p;
    p.maxPieces = 12;
    Loss loss(mesh, p);
    std::mt19937 rng(7);
    RuleContext ctx{ loss, p, rng };
    Carving c = Carving::Sphere(loss.SphereCentre(), loss.SphereRadius());
    std::vector<Vector3> pts;
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (int k = 0; k < 40000; k++) pts.push_back(Vector3Add(c.centre, Vector3Scale({ u(rng), u(rng), u(rng) }, 2.5f * c.radius)));
    const Rule *add = Find("add a cut"), *remove = Find("remove a cut");

    for (int k = 0; k < 7; k++) {
        float before = loss.Evaluate(c, Placement{}).total;
        std::vector<float> open = Pose(c, 0.0f, pts);
        if (!add->apply(c, ctx)) { printf("add: didn't apply\n"); continue; }
        float after = loss.Evaluate(c, Placement{}).total;
        std::vector<float> open2 = Pose(c, 0.0f, pts);
        float worst = 0.0f;
        int flips = 0;
        for (size_t i = 0; i < pts.size(); i++) {
            if ((open[i] < 0.0f) != (open2[i] < 0.0f)) flips++;
            if (open[i] >= 0.0f && open[i] < 1.0f) worst = fmaxf(worst, fabsf(open[i] - open2[i]));
        }
        int none, outside, two;
        Closed(c, pts, &none, &outside, &two);
        printf("add -> %zu pieces: loss %.4f -> %.4f, open pose changed %.4f mm (%d points in/out); closed: %d / %d / %d\n", c.pieces.size(), before,
               after, worst, flips, none, outside, two);
        c.joints.back().open = std::uniform_real_distribution<float>(-1.5f, 1.5f)(rng);     // as Adam might open it
    }
    while (remove->apply(c, ctx)) {
        int none, outside, two;
        Closed(c, pts, &none, &outside, &two);
        printf("remove -> %zu pieces; closed: %d / %d / %d\n", c.pieces.size(), none, outside, two);
    }
    printf("(closed: points in the sphere but no piece / outside it in a piece / in two pieces)\n");
}
