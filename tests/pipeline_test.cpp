// Headless end-to-end test of the data pipeline: analyse -> decompose -> fold tree -> SRD -> restore -> export.
// Build from the project root after one `tiny` build (which downloads raylib into build/modules):
//   g++ -std=c++17 -O2 -Isrc -Ibuild/modules/raylib/include tests/pipeline_test.cpp src/data/*.cpp src/util/*.cpp
//       build/modules/raylib/lib/linux_amdx64_libraylib.a -lopenvdb -ltbb -lImath -lglfw -lGL -lpthread -ldl -lm
//       -o build/pipeline_test
// Run: build/pipeline_test [quick|full] [critter|torus|ball]
#include "data/decompose.hpp"
#include "data/fold.hpp"
#include "data/restore.hpp"
#include "util/utils.hpp"
#include <raymath.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace Bakugan;

static float Capsule(Vector3 p, Vector3 a, Vector3 b, float r) {
    Vector3 pa = Vector3Subtract(p, a), ba = Vector3Subtract(b, a);
    float h = Clamp(Vector3DotProduct(pa, ba) / Vector3DotProduct(ba, ba), 0.0f, 1.0f);
    return Vector3Length(Vector3Subtract(pa, Vector3Scale(ba, h))) - r;
}

static float Box(Vector3 p, Vector3 c, Vector3 h, float r) {
    Vector3 q = { fabsf(p.x - c.x) - h.x + r, fabsf(p.y - c.y) - h.y + r, fabsf(p.z - c.z) - h.z + r };
    Vector3 m = { fmaxf(q.x, 0), fmaxf(q.y, 0), fmaxf(q.z, 0) };
    return Vector3Length(m) + fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.0f) - r;
}

// A four legged critter with a head and a tail, as a signed distance function.
static float Critter(Vector3 p) {
    float d = Box(p, { 0, 0.95f, 0 }, { 0.65f, 0.32f, 0.37f }, 0.08f);
    d = fminf(d, Vector3Length(Vector3Subtract(p, { 0.9f, 1.45f, 0 })) - 0.42f);
    for (int sx = -1; sx <= 1; sx += 2)
        for (int sz = -1; sz <= 1; sz += 2)
            d = fminf(d, Capsule(p, { 0.45f * sx, 0.1f, 0.24f * sz }, { 0.45f * sx, 0.8f, 0.24f * sz }, 0.13f));
    d = fminf(d, Capsule(p, { -0.6f, 1.05f, 0 }, { -1.2f, 1.5f, 0 }, 0.09f));
    return d;
}

static float Torus(Vector3 p) {
    Vector3 q = Vector3Subtract(p, { 0, 1, 0 });
    float a = sqrtf(q.x * q.x + q.z * q.z) - 0.8f;
    return sqrtf(a * a + q.y * q.y) - 0.25f;
}

static float BallShape(Vector3 p) { return Vector3Distance(p, { 0, 1, 0 }) - 0.7f; }

int main(int argc, char** argv) {
    bool quick = argc > 1 && strcmp(argv[1], "quick") == 0;
    std::string which = argc > 2 ? argv[2] : "critter";
    TriMesh mesh;
    if (which == "torus") mesh = Utils::MeshImplicit(Torus, { { -1.1f, 0.7f, -1.1f }, { 1.1f, 1.3f, 1.1f } }, 96);
    else if (which == "ball") mesh = Utils::MeshImplicit(BallShape, { { -0.75f, 0.25f, -0.75f }, { 0.75f, 1.75f, 0.75f } }, 64);
    else mesh = Utils::MeshImplicit(Critter, { { -1.4f, -0.1f, -0.6f }, { 1.45f, 1.95f, 0.6f } }, 96);
    if (which != "critter") quick = true;
    printf("mesh: %zu verts, %zu tris\n", mesh.positions.size(), mesh.indices.size() / 3);

    auto shape = std::make_shared<ShapeAnalysis>(AnalyzeShape(mesh, 64, 4000));
    printf("analysis: ok=%d %s  R0=%.3f  interior=%zu detail=%zu\n", shape->ok, shape->message.c_str(),
           shape->equalRadius, shape->interior.size(), shape->detail.size());
    if (!shape->ok) return 1;

    // superfrustum sanity: a unit box primitive
    SuperFrustum box;
    box.size = { 1, 1, 1 };
    printf("sf box: centre %.3f (expect -0.5)  face %.3f (0)  outside %.3f (0.5)\n", box.LocalDistance({ 0, 0, 0 }),
           box.LocalDistance({ 0.5f, 0, 0 }), box.LocalDistance({ 0, 0, 1.0f }));
    SuperFrustum bent = box;
    bent.size = { 0.3f, 0.3f, 2.0f };
    bent.bulge = 0.6f;
    printf("sf bent: centre %.3f  end %.3f\n", bent.LocalDistance({ -bent.Sagitta(), 0, 0 }), bent.LocalDistance({ 0, 0, 1.0f }));

    auto meshPtr = std::make_shared<const TriMesh>(mesh);
    JobControl control;
    DecomposeParams dp;
    if (quick) { dp.rounds = 2; dp.iterations = 25; }
    DecomposeResult dr;
    {
        FILE* cache = fopen("build/decomp_cache.txt", "r");
        if (cache && !quick && which == "critter") {
            char buf[1 << 16]; size_t n = fread(buf, 1, sizeof(buf) - 1, cache); buf[n] = 0; fclose(cache);
            Design cd; std::string e2; cd.Deserialize(buf, &e2);
            for (const Part& p : cd.parts) dr.shapes.push_back(p.shape);
            dr.ok = true; dr.message = "cached";
        } else {
            if (cache) fclose(cache);
            dr = Decompose(meshPtr, dp, {}, control);
            Design cd; for (const SuperFrustum& s2 : dr.shapes) cd.AddPart(s2, -1);
            if (!quick && which == "critter") { FILE* w = fopen("build/decomp_cache.txt", "w"); std::string t = cd.Serialize(); fwrite(t.data(), 1, t.size(), w); fclose(w); }
        }
    }
    printf("decompose: %s\n", dr.message.c_str());
    if (!dr.ok) return 1;
    for (const SuperFrustum& s : dr.shapes)
        printf("  sf pos(%.2f %.2f %.2f) size(%.2f %.2f %.2f) round %.2f taper %.2f bulge %.2f dil %.3f\n", s.position.x, s.position.y,
               s.position.z, s.size.x, s.size.y, s.size.z, s.rounding, s.taper, s.bulge, s.dilation);

    Design design;
    for (const SuperFrustum& s : dr.shapes) design.AddPart(s, -1);
    FoldParams fp;
    if (quick) { fp.iterations = 30; fp.candidates = 8; }
    if (getenv("BK_WCOL")) fp.wCollision = atof(getenv("BK_WCOL"));
    if (getenv("BK_WCOV")) fp.wCoverage = atof(getenv("BK_WCOV"));
    if (getenv("BK_WTRIM")) fp.wTrim = atof(getenv("BK_WTRIM"));
    if (getenv("BK_TOL")) fp.volumeTolerance = atof(getenv("BK_TOL"));
    if (getenv("BK_ITERS")) fp.iterations = atoi(getenv("BK_ITERS"));
    FoldModel model(shape, fp);
    FoldState state = model.MakeState(design);
    BuildFoldTree(state, model);
    FoldMetrics m0;
    model.Evaluate(state, &m0);
    printf("tree: loss %.4f coverage %.2f trimmed %.3f collision %.3f hinges %d\n", m0.loss, m0.coverage, m0.trimmed, m0.collision, m0.hinges);
    for (size_t i = 0; i < state.design.parts.size(); i++) {
        const Part& p = state.design.parts[i];
        printf("  %-8s parent %2d angle %6.1f deg  samples %zu\n", p.name.c_str(), p.hinge.parent, p.hinge.angle * RAD2DEG,
               state.samples[i].lattice.size());
    }

    if (getenv("BK_SKIPFOLD")) return 0;
    JobControl fc;
    FoldResult fr = OptimizeFold(state.design, shape, fp, fc);
    printf("fold: %s\n  loss %.4f -> %.4f  radius x%.3f\n", fr.message.c_str(), m0.loss, fr.metrics.loss, fr.metrics.radiusRatio);
    std::vector<float> hist = fc.History();
    printf("  history:");
    for (size_t i = 0; i < hist.size(); i += std::max<size_t>(1, hist.size() / 10)) printf(" %.4f", hist[i]);
    printf("\n");

    std::string text = fr.design.Serialize();
    Design back;
    std::string err;
    printf("serialize roundtrip: %s\n", back.Deserialize(text, &err) && back.parts.size() == fr.design.parts.size() ? "ok" : err.c_str());

    if (getenv("BK_NORESTORE")) return 0;
    RestoreParams rp;
    if (quick) rp.resolution = 48;
    JobControl rc;
    RestoreResult rr = RestoreDetail(fr.design, meshPtr, rp, rc);
    printf("restore: %s\n", rr.message.c_str());
    std::vector<const TriMesh*> meshes;
    std::vector<Rigid> openXf, closedXf;
    std::vector<std::string> names;
    std::vector<Rigid> xf = rr.design.Transforms(1.0f);
    for (const RestoredPart& p : rr.parts) {
        int idx = rr.design.IndexOf(p.partId);
        meshes.push_back(&p.mesh);
        openXf.push_back(Rigid{});
        closedXf.push_back(xf[idx]);
        names.push_back(rr.design.parts[idx].name);
        printf("  piece %-8s %zu tris, volume %.4f\n", names.back().c_str(), p.mesh.indices.size() / 3, p.volume);
    }
    Utils::WriteOBJ("build/test_open.obj", meshes, openXf, names, &err);
    Utils::WriteOBJ("build/test_closed.obj", meshes, closedXf, names, &err);
    printf("done\n");
    return 0;
}
