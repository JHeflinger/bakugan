#include "post.h"
#include "mesher.h"
#include "core/util.h"
#include <stdlib.h>
#include <string.h>

int export_stl(const Design* d, const Params* p, const char* dir, const char* stem, char* msg, int msgSize) {
    fb_make_dir(dir);
    const char* pose = p->stlPose == 1 ? "closed" : "open";
    Rigid xf[FB_MAX_BODIES];
    design_transforms(d, p->stlPose == 1 ? 0.0f : 1.0f, xf);
    const float cell = 2.0f * d->R / fb_maxi(p->stlResolution, 16);
    int written = 0, tris = 0;
    for (int b = 0; b < d->nbodies; b++) {
        TriMesh m = mesh_body(d, b, cell, 4 * p->stlResolution);
        if (m.nt == 0) { mesh_free(&m); continue; }
        char* path = fb_format("%s/%s_%s_part%d.stl", dir, stem, pose, b + 1);
        char* name = fb_format("part %d", b + 1);
        if (stl_write(path, &m, d->mmPerUnit, &xf[b], name)) { written++; tris += m.nt; }
        free(path);
        free(name);
        mesh_free(&m);
    }
    snprintf(msg, (size_t)msgSize, "%d STL part%s (%s pose, %d triangles) in %s", written, written == 1 ? "" : "s", pose, tris, dir);
    return written;
}
