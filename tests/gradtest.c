// The analytic gradient against central differences of the loss, on a two-body design with a hollow.
#include "core/design.h"
#include "core/util.h"
#include "opt/eval.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static Design D; static Evaluator* E; static Params P;
static float loss(void) { design_hinge_frames(&D); return eval_run(E, &D, NULL, 1.0f, 0.3f).total; }
static void check(const char* name, float* x, float g, float eps) {
    float x0 = *x; *x = x0 + eps; float a = loss(); *x = x0 - eps; float b = loss(); *x = x0; loss();
    printf("%-14s analytic %+.5f  numeric %+.5f\n", name, g, (a - b) / (2 * eps));
}
int main(int argc, char** argv) {
    fb_threads_init(0);
    TriMesh m; char err[256];
    if (!mesh_load_obj(argv[1], &m, err, sizeof err)) { printf("%s\n", err); return 1; }
    mesh_weld(&m, 1e-6f); mesh_normalize(&m, 2.0f);
    design_init(&D, &m, 60.0f, 2.0f);
    params_default(&P); P.viewPixels = 48; P.views = 6; P.dependencyWeight = 0;
    if (argc > 2) P.collisionWeight = atof(argv[2]);
    if (argc > 3) P.depthWeight = atof(argv[3]);
    if (argc > 4) P.hingeWeight = atof(argv[4]);
    if (argc > 5) P.surfaceWeight = atof(argv[5]);
    if (argc > 6) P.fitLook = atoi(argv[6]);
    Plane pl = { 0 }; pl.id = design_new_id(&D); pl.n = v3norm(v3(0.1f, 1, 0.05f)); pl.d = 0.3f; D.planes[D.nplanes++] = pl;
    Body* a = &D.bodies[0]; a->cells[0].plane[0] = 0; a->cells[0].sign[0] = 1; a->cells[0].n = 1;
    Body* b = &D.bodies[D.nbodies++]; memset(b, 0, sizeof *b); b->id = design_new_id(&D); b->ncells = 1;
    b->cells[0].plane[0] = 0; b->cells[0].sign[0] = -1; b->cells[0].n = 1;
    Hinge* h = &D.hinges[D.nhinges++]; memset(h, 0, sizeof *h); h->id = design_new_id(&D); h->parent = a->id; h->child = b->id;
    h->plane = 0; h->ref = v3(1, 0, 0); h->phi = 1.2f; h->angle = 1.4f; h->order = 0; h->holder = -1;
    Hollow* o = &D.hollows[D.nhollows++]; memset(o, 0, sizeof *o); o->id = design_new_id(&D);
    o->a = v3(-0.3f, -0.5f, 0.6f); o->b = v3(0.3f, -0.4f, 0.75f); o->r = 0.2f;
    design_hinge_frames(&D);
    E = eval_create(&P, &D);
    Grad* g = grad_create(&D);
    double t0 = fb_now();
    Score s = eval_run(E, &D, g, 1.0f, 0.3f);
    double tg = (fb_now() - t0) * 1e3;
    printf("loss %.5f (open %.4f cover %.2f coll %.4f hinge %.4f surface %.4f frag %.4f) in %.0f ms (with gradient %.0f ms)\n", s.total, s.open,
           s.openCover, s.collision, s.hinge, s.surface, s.fragments, eval_run(E, &D, NULL, 1, 0.3f).ms, tg);
    check("hinge angle", &D.hinges[0].angle, g->hinge[0][1], 2e-3f);
    check("hinge phi", &D.hinges[0].phi, g->hinge[0][0], 2e-3f);
    check("plane d", &D.planes[0].d, g->plane[0][3], 2e-3f);
    check("plane n.z", &D.planes[0].n.z, g->plane[0][2], 2e-3f);
    check("scale", &D.scale, g->place[0], 2e-3f);
    check("offset x", &D.offset.x, g->place[1], 2e-3f);
    check("offset y", &D.offset.y, g->place[2], 2e-3f);
    check("hollow r", &D.hollows[0].r, g->hollow[0][6], 2e-3f);
    check("hollow a.z", &D.hollows[0].a.z, g->hollow[0][2], 2e-3f);
    return 0;
}
