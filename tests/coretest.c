// The representation: fields, the closed pose being the sphere, hinges on their seams, save/load.
#include "core/design.h"
#include "core/util.h"
#include "core/config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char** argv) {
    fb_threads_init(0);
    TriMesh m; char err[256];
    if (!mesh_load_obj(argv[1], &m, err, sizeof err)) { printf("%s\n", err); return 1; }
    mesh_weld(&m, 1e-6f); mesh_normalize(&m, 2.0f);
    Design d; design_init(&d, &m, 60.0f, 2.0f);
    printf("R %.3f inset %.4f scale %.3f offset %.2f %.2f %.2f\n", d.R, d.inset, d.scale, d.offset.x, d.offset.y, d.offset.z);
    // two cuts, three bodies
    Plane p0 = { 0 }; p0.n = v3(0, 1, 0); p0.d = 0.2f; d.planes[d.nplanes++] = p0;
    Plane p1 = { 0 }; p1.n = v3norm(v3(1, 0, 0.3f)); p1.d = -0.1f; d.planes[d.nplanes++] = p1;
    d.bodies[0].cells[0].n = 1; d.bodies[0].cells[0].plane[0] = 0; d.bodies[0].cells[0].sign[0] = -1;   // above
    for (int s = 0; s < 2; s++) {
        Body* b = &d.bodies[d.nbodies++]; memset(b, 0, sizeof *b); b->id = design_new_id(&d); b->ncells = 1;
        b->cells[0].n = 2; b->cells[0].plane[0] = 0; b->cells[0].sign[0] = 1; b->cells[0].plane[1] = 1; b->cells[0].sign[1] = s ? 1 : -1;
    }
    for (int k = 0; k < 2; k++) {
        Hinge* h = &d.hinges[d.nhinges++]; memset(h, 0, sizeof *h); h->id = design_new_id(&d);
        h->parent = d.bodies[k == 0 ? 0 : 1].id; h->child = d.bodies[k + 1].id; h->plane = k; h->ref = v3perp(d.planes[k].n);
        h->phi = 0.7f + k; h->angle = 1.0f; h->order = k; h->holder = -1;
    }
    design_hinge_frames(&d);
    // closed: every point of the sphere in exactly one body, nothing outside
    Rng r; rng_seed(&r, 3);
    int bad = 0, n = 20000;
    for (int s = 0; s < n; s++) {
        V3 q = v3(rng_uniform(&r) * 2.2f - 1.1f, rng_uniform(&r) * 2.2f - 1.1f, rng_uniform(&r) * 2.2f - 1.1f);
        int in = 0;
        for (int b = 0; b < d.nbodies; b++) in += body_field(&d, b, q, NULL) < 0;
        bool want = v3len(q) < d.R;
        if (in != (want ? 1 : 0)) bad++;
    }
    printf("closed pose: %d of %d points in the wrong number of bodies\n", bad, n);
    for (int h = 0; h < d.nhinges; h++) {
        const Hinge* hg = &d.hinges[h];
        float pf = body_field(&d, design_body(&d, hg->parent), hg->pivot, NULL), cf = body_field(&d, design_body(&d, hg->child), hg->pivot, NULL);
        printf("hinge %d: |pivot| %.4f (R - inset %.4f), on plane %.2e, axis . n %.2e, axis . pivot %.2e; field parent %.4f child %.4f\n", h,
               v3len(hg->pivot), d.R - d.inset, v3dot(d.planes[hg->plane].n, hg->pivot) - d.planes[hg->plane].d, v3dot(hg->axis, d.planes[hg->plane].n),
               v3dot(hg->axis, hg->pivot), pf, cf);
    }
    if (!design_save(&d, "/tmp/t.fold", err, sizeof err)) printf("%s\n", err);
    Design e; memset(&e, 0, sizeof e);
    if (!design_load(&e, "/tmp/t.fold", err, sizeof err)) printf("%s\n", err);
    else printf("reloaded: %d bodies %d hinges %d planes, target %d verts, hinge 1 phi %.3f\n", e.nbodies, e.nhinges, e.nplanes, e.target.nv, e.hinges[1].phi);
    Params p; params_default(&p); params_save(&p, "/tmp/t.ini"); Params q; params_default(&q);
    if (!params_load(&q, "/tmp/t.ini", err, sizeof err)) printf("%s\n", err); else printf("ini round trip ok\n");
    return 0;
}
