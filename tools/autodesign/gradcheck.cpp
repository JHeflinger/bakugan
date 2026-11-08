// The envelope-theorem gradient against brute force central differences of the whole loss.
#include "autodesign/loss.hpp"
#include "autodesign/suite.hpp"
#include <raymath.h>
#include <cstdio>
#include <cmath>
using namespace Bakugan;
int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s mesh.obj\n", argv[0]); return 1; }
    TriMesh mesh; std::string err;
    LoadObj(argv[1], &mesh, &err);
    AutoParams p;
    Loss loss(mesh, p);
    // a design with a few bodies: the start, then each second param nudged so nothing sits at a special value
    Design d;
    {
        Vector3 lo = mesh.positions[0], hi = lo;
        for (Vector3 q : mesh.positions) { lo = Vector3Min(lo, q); hi = Vector3Max(hi, q); }
        Vector3 mid = Vector3Scale(Vector3Add(lo, hi), 0.5f), size = Vector3Subtract(hi, lo);
        int b0 = d.groups[d.AddGroup("Body", -1)].id, b1 = d.groups[d.AddGroup("Body", -1)].id;
        SuperFrustum a; a.position = Vector3Add(mid, { -0.2f * size.x, 0, 0 }); a.size = Vector3Scale(size, 0.45f); a.rounding = 0.4f; a.taper = 0.8f;
        SuperFrustum b = a; b.position = Vector3Add(mid, { 0.2f * size.x, 0.05f * size.y, 0 }); b.rotation = QuaternionFromAxisAngle({ 0.3f, 1, 0.2f }, 0.4f);
        d.AddShape(a, b0, "A"); d.AddShape(b, b1, "B");
        // a cut through the edge of A, so it shows in the silhouettes
        SuperFrustum cut; cut.position = Vector3Add(a.position, { -0.2f * size.x, 0.15f * size.y, 0 }); cut.size = Vector3Scale(size, 0.2f);
        cut.size.z = size.z; cut.rotation = QuaternionFromAxisAngle({ 0, 0, 1 }, 0.3f); cut.rounding = 0.3f;
        d.AddShape(cut, b0, "Cut", Op::Cut);
        Hinge& h = d.hinges[d.AddHinge(b0, b1)];
        h.pivot = Vector3Add(mid, { 0, 0, 0.4f * size.z });   // outside both bodies: the hinge term counts
        h.axis = Vector3Normalize({ 0.1f, 1, 0.2f }); h.start = 0.2f; h.end = 2.0f; h.hardware = false; h.t0 = 0.1f; h.t1 = 0.8f;
    }
    std::vector<Param> ps = DesignParams(d, loss.L());
    std::vector<float> g;
    // the sphere a little off the mesh's centre, so its slope isn't zero
    Vector3 sphere = Vector3Add(loss.SphereStart(), { 0.03f * loss.L(), -0.02f * loss.L(), 0.01f * loss.L() });
    Vector3 sg;
    loss.Evaluate(d, sphere, ps, &g, &sg);
    int agree = 0, n = 0;
    for (size_t j = 0; j < ps.size(); j++) {
        double fd = 0;
        float hs[2] = { 1.0f, 3.0f };
        double est[2];
        for (int k = 0; k < 2; k++) {
            Design a = d, b = d;
            SetParam(a, ps[j], *Ref(a, ps[j]) + hs[k] * ps[j].unit, loss.L());
            SetParam(b, ps[j], *Ref(b, ps[j]) - hs[k] * ps[j].unit, loss.L());
            float va = *Ref(a, ps[j]), vb = *Ref(b, ps[j]);    // the nudge as made, before renormalizing
            Tidy(a);
            Tidy(b);
            est[k] = (loss.Evaluate(a, sphere, {}, nullptr).total - loss.Evaluate(b, sphere, {}, nullptr).total) / ((va - vb) / ps[j].unit);
        }
        fd = est[1];  // h = 3 units: smoother over the rays' discrete steps
        bool ok = (fabs(g[j] - fd) <= 0.3 * fmax(fabs(fd), fabs(g[j])) + 2e-5);
        agree += ok; n++;
        printf("%s %2d field %2d: envelope %+.6f  brute force %+.6f (h=1: %+.6f) %s\n", ps[j].kind == Param::Shape ? "shape" : "hinge", ps[j].id, ps[j].field, g[j], fd, est[0], ok ? "" : "  <-- differs");
    }
    for (int a = 0; a < 3; a++) {
        Vector3 e = { 0, 0, 0 };
        (&e.x)[a] = 2.0f * loss.SphereUnit();
        double fd = (loss.Evaluate(d, Vector3Add(sphere, e), {}, nullptr).total - loss.Evaluate(d, Vector3Subtract(sphere, e), {}, nullptr).total) / 4.0;
        printf("sphere centre %c: gradient %+.6f  brute force %+.6f\n", "xyz"[a], (&sg.x)[a], fd);
    }
    AutoScore sc = loss.Evaluate(d, sphere, {}, nullptr);
    printf("%d of %d agree within 30%% (collision term %.3f, hinge term %.3f, of loss %.3f)\n", agree, n, sc.collision, sc.hinges, sc.total);
}
