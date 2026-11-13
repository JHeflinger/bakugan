// The envelope-theorem gradient against brute force central differences of the whole loss, for a carving of three
// pieces and a hollow (cutters', joints' and the mesh placement's parameters).
// With the semantic loss on (see semantic_env.hpp; ONNX Runtime and a model needed), its gradient too; ONLYSEM=1 checks
// it alone. Its shade samples are a random draw, so set SAMPLES high (e.g. 1000000: all of them) for a fair check.
#include "autodesign/loss.hpp"
#include "autodesign/suite.hpp"
#include "semantic_env.hpp"
#include <raymath.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
using namespace Bakugan;
int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s mesh.obj\n", argv[0]); return 1; }
    TriMesh mesh; std::string err;
    LoadObj(argv[1], &mesh, &err);
    AutoParams p;
    SemanticFromEnv(p);
    if (const char* m = getenv("MINCOVER")) p.minViewCoverage = (float)atof(m);   // 0: no coverage penalty (its kink)
    auto envf = [](const char* n, float& v) { if (const char* s = getenv(n)) v = (float)atof(s); };
    envf("OTW", p.transportWeight); envf("SDFW", p.sdfWeight); envf("CHAMW", p.chamferWeight);
    if (getenv("ONLY3D")) p.silhouetteWeight = p.collisionWeight = p.hingeWeight = p.surfaceWeight = 0.0f;   // the 3D terms' gradient alone
    if (getenv("ONLYSEM")) p.silhouetteWeight = p.collisionWeight = p.hingeWeight = 0.0f;     // the semantic gradient alone
    Loss loss(mesh, p);
    if (!loss.Error().empty()) { printf("%s\n", loss.Error().c_str()); return 1; }
    // a carving of three pieces: two caps cut off the sphere, hinged open by different amounts, and the mesh placed a
    // little off where it starts, so no slope is zero by symmetry
    Carving cv = Carving::Sphere(loss.SphereCentre(), loss.SphereRadius());
    {
        const float R = cv.radius;
        Vector3 dirs[2] = { Vector3Normalize({ 1, 0.3f, 0.2f }), Vector3Normalize({ -0.2f, 0.4f, -1 }) };
        float open[2] = { 0.7f, -0.9f };
        int rest = cv.pieces[0].id;
        for (int k = 0; k < 2; k++) {
            Cutter c;
            c.id = cv.nextId++;
            c.sf.size = { 0.8f * R, 0.7f * R, 0.8f * R };
            c.sf.rotation = QuaternionFromVector3ToVector3({ 0, 0, 1 }, dirs[k]);
            c.sf.position = Vector3Add(cv.centre, Vector3Scale(dirs[k], R - 0.3f * R + 0.4f * R));
            c.sf.rounding = 0.3f;
            c.sf.taper = 0.85f;
            cv.cutters.push_back(c);
            Piece cap = { cv.nextId++, { c.id }, {} };
            cv.pieces[0].outside.push_back(c.id);
            cv.pieces.push_back(cap);
            Vector3 side = Vector3Normalize(Vector3CrossProduct(dirs[k], { 0, 1, 0 }));
            Joint j;
            j.id = cv.nextId++;
            j.parent = rest;
            j.child = cap.id;
            j.pivot = Vector3Add(cv.centre, Vector3Add(Vector3Scale(dirs[k], 0.72f * R), Vector3Scale(side, 0.6f * R)));
            j.axis = Vector3Normalize(Vector3CrossProduct(dirs[k], side));
            j.open = open[k];
            j.t0 = 0.1f + 0.2f * k;
            j.t1 = j.t0 + 0.5f;
            cv.joints.push_back(j);
        }
    }
    {   // and a hollow, reaching through the surface a little
        Cutter h;
        h.id = cv.nextId++;
        h.hollow = true;
        h.sf.size = { 0.25f * cv.radius, 0.3f * cv.radius, 0.9f * cv.radius };
        h.sf.rotation = QuaternionFromAxisAngle({ 1, 0, 0 }, 0.3f);
        h.sf.position = Vector3Add(cv.centre, { 0.1f * cv.radius, -0.65f * cv.radius, 0.2f * cv.radius });
        h.sf.rounding = 0.5f;
        cv.cutters.push_back(h);
    }
    Placement target;
    target.offset = { 0.03f * loss.L(), -0.02f * loss.L(), 0.01f * loss.L() };
    target.scale = 0.92f;
    std::vector<Param> ps = CarvingParams(cv, loss.L()), tps = TargetParams(true, true, loss.L());
    ps.insert(ps.end(), tps.begin(), tps.end());
    std::vector<float> g;
    loss.Evaluate(cv, target, ps, &g);
    int agree = 0, n = 0;
    double dot = 0, gg = 0, ff = 0;      // the two as whole vectors: how well they point the same way
    double cdot = 0, cgg = 0, cff = 0;   // ... the carving's alone (the placement's slope is itself a difference of 1 unit)
    double d01 = 0, f00 = 0;             // brute force at h = 1 against h = 3: how noisy brute force itself is
    for (size_t j = 0; j < ps.size(); j++) {
        double fd = 0;
        float hs[2] = { 1.0f, getenv("H") ? (float)atof(getenv("H")) : 3.0f };     // H: the step to judge by, in units
        double est[2];
        for (int k = 0; k < 2; k++) {
            Carving a = cv, b = cv;
            Placement ta = target, tb = target;
            float v = Get(cv, target, ps[j]);
            Set(a, ta, ps[j], v + hs[k] * ps[j].unit, loss.L());
            Set(b, tb, ps[j], v - hs[k] * ps[j].unit, loss.L());
            float va = Get(a, ta, ps[j]), vb = Get(b, tb, ps[j]);    // the nudge as made, before renormalizing
            Tidy(a);
            Tidy(b);
            est[k] = (loss.Evaluate(a, ta).total - loss.Evaluate(b, tb).total) / ((va - vb) / ps[j].unit);
        }
        fd = est[1];  // h = 3 units: smoother over the rays' discrete steps (shading changes faster: H=0.25 there)
        bool ok = (fabs(g[j] - fd) <= 0.3 * fmax(fabs(fd), fabs(g[j])) + 2e-5);
        agree += ok; n++;
        dot += g[j] * fd; gg += (double)g[j] * g[j]; ff += fd * fd;
        if (ps[j].kind != Param::Target) { cdot += g[j] * fd; cgg += (double)g[j] * g[j]; cff += fd * fd; }
        d01 += est[0] * fd; f00 += est[0] * est[0];
        static const char* kinds[] = { "cutter", "joint ", "target" };
        printf("%s %2d field %2d: envelope %+.6f  brute force %+.6f (h=1: %+.6f) %s\n", kinds[ps[j].kind], ps[j].id, ps[j].field, g[j], fd, est[0], ok ? "" : "  <-- differs");
    }
    AutoScore sc = loss.Evaluate(cv, target);
    printf("%d of %d agree within 30%% (collision term %.3f, hinge term %.3f, of loss %.3f)\n", agree, n, sc.collision, sc.hinges, sc.total);
    printf("as vectors: cosine %.3f, length ratio %.3f (gradient / brute force)\n", dot / sqrt(gg * ff + 1e-30), sqrt(gg / (ff + 1e-30)));
    printf("the carving's alone: cosine %.3f, length ratio %.3f\n", cdot / sqrt(cgg * cff + 1e-30), sqrt(cgg / (cff + 1e-30)));
    printf("brute force at h=1 against h=H: cosine %.3f\n", d01 / sqrt(f00 * ff + 1e-30));
}
