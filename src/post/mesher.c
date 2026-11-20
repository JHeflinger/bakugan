#include "mesher.h"
#include "core/util.h"
#include "opt/eval.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    FieldFn f;
    void* ctx;
    V3 origin;
    float cell;
    int nx, ny, nz;
    float* v;
} SampleCtx;

static void sample_task(void* vc, int k) {
    SampleCtx* c = vc;
    for (int j = 0; j < c->ny; j++)
        for (int i = 0; i < c->nx; i++)
            c->v[((size_t)k * c->ny + j) * c->nx + i] = c->f(c->ctx, v3(c->origin.x + i * c->cell, c->origin.y + j * c->cell, c->origin.z + k * c->cell));
}


typedef struct { SampleCtx* c; V3* pos; unsigned char* has; } VertCtx;
static const int kE[12][2] = { { 0, 1 }, { 2, 3 }, { 4, 5 }, { 6, 7 }, { 0, 2 }, { 1, 3 }, { 4, 6 }, { 5, 7 }, { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 } };

// solves the 3x3 system A x = b (symmetric), false if it's singular
static bool solve3(const double A[9], const double b[3], double x[3]) {
    double det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
    if (fabs(det) < 1e-12) return false;
    double inv[9] = { A[4] * A[8] - A[5] * A[7], A[2] * A[7] - A[1] * A[8], A[1] * A[5] - A[2] * A[4],
                      A[5] * A[6] - A[3] * A[8], A[0] * A[8] - A[2] * A[6], A[2] * A[3] - A[0] * A[5],
                      A[3] * A[7] - A[4] * A[6], A[1] * A[6] - A[0] * A[7], A[0] * A[4] - A[1] * A[3] };
    for (int r = 0; r < 3; r++) x[r] = (inv[3 * r] * b[0] + inv[3 * r + 1] * b[1] + inv[3 * r + 2] * b[2]) / det;
    return true;
}

static void vertex_task(void* vp, int k) {
    VertCtx* V = vp;
    SampleCtx* c = V->c;
    const int nx = c->nx, ny = c->ny;
    const float cell = c->cell, e = 0.05f * cell;
    for (int j = 0; j < ny - 1; j++)
        for (int i = 0; i < nx - 1; i++) {
            float cv[8];
            int neg = 0;
            for (int q = 0; q < 8; q++) {
                cv[q] = c->v[((size_t)(k + ((q >> 2) & 1)) * ny + (j + ((q >> 1) & 1))) * nx + (i + (q & 1))];
                neg += cv[q] < 0;
            }
            size_t cid = ((size_t)k * (ny - 1) + j) * (nx - 1) + i;
            if (neg == 0 || neg == 8) { V->has[cid] = 0; continue; }
            V->has[cid] = 1;
            V3 pts[12], nrm[12];
            int cnt = 0;
            V3 mass = v3(0, 0, 0);
            for (int q = 0; q < 12; q++) {
                float a = cv[kE[q][0]], b = cv[kE[q][1]];
                if ((a < 0) == (b < 0)) continue;
                float t = a / (a - b);
                V3 pa = v3((float)(kE[q][0] & 1), (float)((kE[q][0] >> 1) & 1), (float)((kE[q][0] >> 2) & 1));
                V3 pb = v3((float)(kE[q][1] & 1), (float)((kE[q][1] >> 1) & 1), (float)((kE[q][1] >> 2) & 1));
                V3 l = v3lerp(pa, pb, t);
                V3 w = v3(c->origin.x + (i + l.x) * cell, c->origin.y + (j + l.y) * cell, c->origin.z + (k + l.z) * cell);
                V3 g = v3(c->f(c->ctx, v3(w.x + e, w.y, w.z)) - c->f(c->ctx, v3(w.x - e, w.y, w.z)),
                          c->f(c->ctx, v3(w.x, w.y + e, w.z)) - c->f(c->ctx, v3(w.x, w.y - e, w.z)),
                          c->f(c->ctx, v3(w.x, w.y, w.z + e)) - c->f(c->ctx, v3(w.x, w.y, w.z - e)));
                pts[cnt] = l;
                nrm[cnt] = v3len(g) > 1e-9f ? v3norm(g) : v3(0, 0, 0);
                mass = v3add(mass, l);
                cnt++;
            }
            mass = v3scale(mass, 1.0f / cnt);
            // minimize sum (n . (x - p))^2 + lambda |x - mass|^2
            const double lambda = 0.02;
            double A[9] = { lambda, 0, 0, 0, lambda, 0, 0, 0, lambda }, b[3] = { lambda * mass.x, lambda * mass.y, lambda * mass.z }, x[3];
            for (int q = 0; q < cnt; q++) {
                double n[3] = { nrm[q].x, nrm[q].y, nrm[q].z }, d = nrm[q].x * pts[q].x + nrm[q].y * pts[q].y + nrm[q].z * pts[q].z;
                for (int r = 0; r < 3; r++) {
                    for (int s = 0; s < 3; s++) A[3 * r + s] += n[r] * n[s];
                    b[r] += n[r] * d;
                }
            }
            V3 l = mass;
            if (solve3(A, b, x)) {
                V3 dc = v3((float)x[0], (float)x[1], (float)x[2]);
                // (kept in the cell, a little beyond at most: past it, the mean)
                if (dc.x > -0.25f && dc.y > -0.25f && dc.z > -0.25f && dc.x < 1.25f && dc.y < 1.25f && dc.z < 1.25f) l = dc;
            }
            V->pos[cid] = v3(c->origin.x + (i + l.x) * cell, c->origin.y + (j + l.y) * cell, c->origin.z + (k + l.z) * cell);
        }
}

TriMesh mesh_field(FieldFn f, void* ctx, Box3 box, float cell, int maxCells) {
    TriMesh out = { 0 };
    if (!box_valid(box)) return out;
    box = box_grow(box, 1.5f * cell);
    V3 e = v3sub(box.max, box.min);
    float longest = fb_max(e.x, fb_max(e.y, e.z));
    if (longest / cell > maxCells) cell = longest / maxCells;
    SampleCtx c;
    c.f = f;
    c.ctx = ctx;
    // (an odd offset keeps samples off the round coordinates designs are full of)
    c.origin = v3sub(box.min, v3(0.3713f * cell, 0.2847f * cell, 0.4129f * cell));
    c.cell = cell;
    c.nx = (int)ceilf(e.x / cell) + 2;
    c.ny = (int)ceilf(e.y / cell) + 2;
    c.nz = (int)ceilf(e.z / cell) + 2;
    const size_t NN = (size_t)c.nx * c.ny * c.nz;
    c.v = FB_NEW(float, NN);
    fb_parallel_for(c.nz, sample_task, &c);
    const int nx = c.nx, ny = c.ny, nz = c.nz;
    #define AT(i, j, k) c.v[((size_t)(k) * ny + (j)) * nx + (i)]
    // one vertex per cell with a sign change: dual contouring, where the planes the crossings lie on (normals from
    // the field) meet, so the bodies' edges and corners stay sharp; held in the cell, toward the crossings' mean
    const size_t NC = (size_t)(nx - 1) * (ny - 1) * (nz - 1);
    int* vid = FB_NEW(int, NC);
    V3* vpos = FB_NEW(V3, NC);
    unsigned char* has = FB_NEW(unsigned char, NC);
    VertCtx vc = { &c, vpos, has };
    fb_parallel_for(nz - 1, vertex_task, &vc);
    FB_ARRAY(V3) pos = { 0 };
    for (size_t q = 0; q < NC; q++) {
        if (!has[q]) { vid[q] = -1; continue; }
        vid[q] = pos.count;
        fb_push(&pos, vpos[q]);
    }
    free(vpos);
    free(has);
    // one quad per crossed edge, between the four cells around it
    FB_ARRAY(int) idx = { 0 };
    #define CELL(i, j, k) vid[((size_t)(k) * (ny - 1) + (j)) * (nx - 1) + (i)]
    for (int k = 1; k < nz - 1; k++)
        for (int j = 1; j < ny - 1; j++)
            for (int i = 1; i < nx - 1; i++) {
                float v0 = AT(i, j, k);
                for (int axis = 0; axis < 3; axis++) {
                    int i1 = i + (axis == 0), j1 = j + (axis == 1), k1 = k + (axis == 2);
                    if (i1 >= nx || j1 >= ny || k1 >= nz) continue;
                    float v1 = AT(i1, j1, k1);
                    if ((v0 < 0) == (v1 < 0)) continue;
                    int q[4];
                    if (axis == 0) { q[0] = CELL(i, j - 1, k - 1); q[1] = CELL(i, j, k - 1); q[2] = CELL(i, j, k); q[3] = CELL(i, j - 1, k); }
                    else if (axis == 1) { q[0] = CELL(i - 1, j, k - 1); q[1] = CELL(i - 1, j, k); q[2] = CELL(i, j, k); q[3] = CELL(i, j, k - 1); }
                    else { q[0] = CELL(i - 1, j - 1, k); q[1] = CELL(i, j - 1, k); q[2] = CELL(i, j, k); q[3] = CELL(i - 1, j, k); }
                    if (q[0] < 0 || q[1] < 0 || q[2] < 0 || q[3] < 0) continue;
                    bool flip = v0 < 0;                // faces point toward the outside
                    if (flip) {
                        fb_push(&idx, q[0]); fb_push(&idx, q[1]); fb_push(&idx, q[2]);
                        fb_push(&idx, q[0]); fb_push(&idx, q[2]); fb_push(&idx, q[3]);
                    } else {
                        fb_push(&idx, q[0]); fb_push(&idx, q[2]); fb_push(&idx, q[1]);
                        fb_push(&idx, q[0]); fb_push(&idx, q[3]); fb_push(&idx, q[2]);
                    }
                }
            }
    #undef CELL
    #undef AT
    free(vid);
    free(c.v);
    out.pos = pos.data;
    out.nv = pos.count;
    out.idx = idx.data;
    out.nt = idx.count / 3;
    // (the winding: make the signed volume positive)
    if (mesh_volume(&out) < 0)
        for (int t = 0; t < out.nt; t++) {
            int s = out.idx[3 * t + 1];
            out.idx[3 * t + 1] = out.idx[3 * t + 2];
            out.idx[3 * t + 2] = s;
        }
    return out;
}

typedef struct { const Design* d; int b; } BodyCtx;
static float body_fn(void* vc, V3 p) {
    BodyCtx* c = vc;
    return body_field(c->d, c->b, p, NULL);
}

typedef struct { const Design* d; int b; const Grid* g; Box3* slices; } BoundsCtx;
static void bounds_task(void* vc, int k) {
    BoundsCtx* c = vc;
    const Grid* g = c->g;
    Box3 box = box_empty();
    for (int j = 0; j < g->n; j++)
        for (int i = 0; i < g->n; i++) {
            V3 q = grid_node(g, i, j, k);
            if (body_cell(c->d, c->b, q, NULL) < g->h && body_field(c->d, c->b, q, NULL) < g->h) box = box_add(box, q);
        }
    c->slices[k] = box;
}

Box3 body_bounds(const Design* d, int b) {
    // (a lattice over the sphere: every body lies in it, extras aside)
    Grid lat = { 0 };
    lat.n = 64;
    lat.B = 1.02f * d->R;
    lat.h = 2.0f * lat.B / (lat.n - 1);
    const Grid* g = &lat;
    Box3 slices[64];
    BoundsCtx bc = { d, b, g, slices };
    fb_parallel_for(g->n, bounds_task, &bc);
    Box3 box = box_empty();
    for (int k = 0; k < g->n; k++)
        if (box_valid(slices[k])) box = box_valid(box) ? box_add(box_add(box, slices[k].min), slices[k].max) : slices[k];
    // (restored features can reach past the cells: the whole of their grid where it adds anything)
    const BoxGrid* dg = &d->bodies[b].detail;
    if (dg->v && box_valid(box)) {
        Box3 db = { dg->o, v3add(dg->o, v3((dg->nx - 1) * dg->h, (dg->ny - 1) * dg->h, (dg->nz - 1) * dg->h)) };
        box = box_add(box_add(box, db.min), db.max);
    }
    for (int e = 0; e < d->nextras; e++) {
        const Extra* x = &d->extras[e];
        if (x->body != d->bodies[b].id || x->op < 0) continue;
        V3 s = x->shape == EX_BOX ? x->size : v3(x->size.x, x->size.x, x->size.z);
        box = box_add(box_add(box, rapply(x->pose, v3scale(s, -1))), rapply(x->pose, s));
        for (int c = 0; c < 8; c++)
            box = box_add(box, rapply(x->pose, v3((c & 1) ? s.x : -s.x, (c & 2) ? s.y : -s.y, (c & 4) ? s.z : -s.z)));
    }
    return box_valid(box) ? box_grow(box, g->h) : box;
}

TriMesh mesh_body(const Design* d, int b, float cell, int maxCells) {
    BodyCtx c = { d, b };
    return mesh_field(body_fn, &c, body_bounds(d, b), cell, maxCells);
}

bool stl_write(const char* path, const TriMesh* m, float scale, const Rigid* xf, const char* name) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    char header[80];
    memset(header, 0, sizeof(header));
    snprintf(header, sizeof(header), "foldball %s", name ? name : "");
    fwrite(header, 1, 80, f);
    unsigned int n = (unsigned int)m->nt;
    fwrite(&n, 4, 1, f);
    for (int t = 0; t < m->nt; t++) {
        V3 p[3];
        for (int c = 0; c < 3; c++) {
            V3 q = m->pos[m->idx[3 * t + c]];
            if (xf) q = rapply(*xf, q);
            p[c] = v3scale(q, scale);
        }
        V3 nrm = v3norm(v3cross(v3sub(p[1], p[0]), v3sub(p[2], p[0])));
        float rec[12] = { nrm.x, nrm.y, nrm.z, p[0].x, p[0].y, p[0].z, p[1].x, p[1].y, p[1].z, p[2].x, p[2].y, p[2].z };
        unsigned short attr = 0;
        fwrite(rec, 4, 12, f);
        fwrite(&attr, 2, 1, f);
    }
    return fclose(f) == 0;
}
