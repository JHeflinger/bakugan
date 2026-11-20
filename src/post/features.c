// Feature restoration (post processing): the input's own surface, where it lies near a part's in the open pose, put
// back onto that part, so the parts carry the mesh's detail (a face, fingers, ridges) and not only its outline.
//
// Per body, on a grid of feature_resolution points across the sphere (in the body's own frame, around it), the mesh's
// signed distance is measured where the grid point sits in the open pose (through the placement: the semantic
// deformation's target, if one was kept). With f the part's field and t the mesh's there:
//
//   restored   g = f + w (t - f)       w fades from 1 to 0 as |t - f| goes from 0.6 to 1 reach
//
// so the part takes the mesh's surface wherever the two are within reach of each other, and is left alone where they
// aren't (where the part is just a coarse stand-in, or where the mesh is far away).
//
// Taking material away never collides. Adding it may: the parts fill the sphere when closed, so a bump has nowhere to go
// but into the part next to it, past the sphere's surface, or into a part that sweeps by as the design moves. Where
// added material would (closer than the clearance to another part, closed or at any pose of a hinge turn that moves
// the two apart, or past the sphere):
//   1. flatten: the bump (and everything within feature_sink_mm of where it doesn't fit) keeps (1 - flatten) of its
//      height;
//   2. sink: what still sticks out is pressed in: the surface within feature_sink_mm of it sinks by sink times as much,
//      so the feature sits whole in a recess, its top flush (sink 1), or is squared off flush (sink 0);
//   3. whatever still doesn't fit is cut off: the result never adds material where it can't go.
// The sphere's surface so stays a sphere, carrying the mesh's features as a texture (pressed into it).
//
// The result is kept as an offset grid per body (Body.detail), added to its field (core/design.c).
#include "features.h"
#include "mesher.h"
#include "core/util.h"
#include <stdlib.h>
#include <string.h>

// ---- the mesh's signed distance, exact near its surface

typedef struct {
    const TriMesh* m;
    V3* fn;                          // face normals
    V3* vn;                          // vertex pseudo normals (angle weighted)
    int* edgeOf;                     // per face edge (3 t + e): its edge
    V3* en;                          // edge pseudo normals (the two faces' sum)
    // triangles bucketed on a grid
    V3 lo;
    float cell;
    int nx, ny, nz;
    int *start, *tris;
    Grid coarse;                     // far from the surface (and its sign there)
} MeshSdf;

typedef struct { long long key; int id; } EdgeKey;
static int cmp_edge(const void* a, const void* b) {
    long long x = ((const EdgeKey*)a)->key, y = ((const EdgeKey*)b)->key;
    return x < y ? -1 : (x > y);
}

static void sdf_build(MeshSdf* s, const TriMesh* m, float cell) {
    memset(s, 0, sizeof(*s));
    s->m = m;
    s->fn = FB_NEW(V3, m->nt + 1);
    s->vn = FB_NEW(V3, m->nv + 1);
    for (int t = 0; t < m->nt; t++) {
        const int* id = &m->idx[3 * t];
        V3 a = m->pos[id[0]], b = m->pos[id[1]], c = m->pos[id[2]];
        V3 n = v3cross(v3sub(b, a), v3sub(c, a));
        s->fn[t] = v3norm(n);
        V3 P[3] = { a, b, c };
        for (int k = 0; k < 3; k++) {
            V3 e1 = v3sub(P[(k + 1) % 3], P[k]), e2 = v3sub(P[(k + 2) % 3], P[k]);
            float l1 = v3len(e1), l2 = v3len(e2);
            if (l1 < 1e-12f || l2 < 1e-12f) continue;
            float ang = acosf(fb_clamp(v3dot(e1, e2) / (l1 * l2), -1.0f, 1.0f));
            s->vn[id[k]] = v3add(s->vn[id[k]], v3scale(s->fn[t], ang));
        }
    }
    // edges: the two faces' normals, summed
    EdgeKey* keys = FB_NEW(EdgeKey, 3 * m->nt + 1);
    for (int t = 0; t < m->nt; t++)
        for (int e = 0; e < 3; e++) {
            int a = m->idx[3 * t + e], b = m->idx[3 * t + (e + 1) % 3];
            if (a > b) { int x = a; a = b; b = x; }
            keys[3 * t + e].key = (long long)a * m->nv + b;
            keys[3 * t + e].id = 3 * t + e;
        }
    qsort(keys, (size_t)3 * m->nt, sizeof(EdgeKey), cmp_edge);
    s->edgeOf = FB_NEW(int, 3 * m->nt + 1);
    s->en = FB_NEW(V3, 3 * m->nt + 1);
    int ne = 0;
    for (int k = 0; k < 3 * m->nt;) {
        int e = k;
        while (e < 3 * m->nt && keys[e].key == keys[k].key) {
            s->edgeOf[keys[e].id] = ne;
            s->en[ne] = v3add(s->en[ne], s->fn[keys[e].id / 3]);
            e++;
        }
        ne++;
        k = e;
    }
    free(keys);
    // buckets
    Box3 b = mesh_bounds(m);
    s->cell = cell;
    s->lo = v3sub(b.min, v3(cell, cell, cell));
    V3 ext = v3sub(b.max, s->lo);
    s->nx = (int)(ext.x / cell) + 2;
    s->ny = (int)(ext.y / cell) + 2;
    s->nz = (int)(ext.z / cell) + 2;
    const size_t NC = (size_t)s->nx * s->ny * s->nz;
    int* count = FB_NEW(int, NC + 1);
    // (a triangle goes in every cell its box touches)
    for (int pass = 0; pass < 2; pass++) {
        if (pass == 1) {
            s->start = FB_NEW(int, NC + 1);
            for (size_t c = 0; c < NC; c++) s->start[c + 1] = s->start[c] + count[c];
            s->tris = FB_NEW(int, s->start[NC] + 1);
            memset(count, 0, sizeof(int) * NC);
        }
        for (int t = 0; t < m->nt; t++) {
            const int* id = &m->idx[3 * t];
            Box3 tb = box_add(box_add(box_add(box_empty(), m->pos[id[0]]), m->pos[id[1]]), m->pos[id[2]]);
            int i0 = (int)((tb.min.x - s->lo.x) / cell), i1 = (int)((tb.max.x - s->lo.x) / cell);
            int j0 = (int)((tb.min.y - s->lo.y) / cell), j1 = (int)((tb.max.y - s->lo.y) / cell);
            int k0 = (int)((tb.min.z - s->lo.z) / cell), k1 = (int)((tb.max.z - s->lo.z) / cell);
            for (int k = k0; k <= k1; k++)
                for (int j = j0; j <= j1; j++)
                    for (int i = i0; i <= i1; i++) {
                        size_t c = ((size_t)k * s->ny + j) * s->nx + i;
                        if (pass == 1) s->tris[s->start[c] + count[c]] = t;
                        count[c]++;
                    }
        }
    }
    free(count);
    // the coarse field: around the mesh (as the evaluator draws it)
    float B = fb_max(fb_max(fabsf(b.min.x), fabsf(b.max.x)), fb_max(fb_max(fabsf(b.min.y), fabsf(b.max.y)), fb_max(fabsf(b.min.z), fabsf(b.max.z))));
    V3 e = v3sub(b.max, b.min);
    s->coarse = grid_create(96, B + 0.1f * fb_max(e.x, fb_max(e.y, e.z)) + 1e-3f);
    grid_from_mesh(&s->coarse, m);
}

static void sdf_free(MeshSdf* s) {
    free(s->fn);
    free(s->vn);
    free(s->edgeOf);
    free(s->en);
    free(s->start);
    free(s->tris);
    grid_free(&s->coarse);
}

// the point of triangle abc nearest p, and which part of it: 0 face, 1..3 vertex a, b, c, 4..6 edge ab, bc, ca
static V3 closest_on_tri(V3 p, V3 a, V3 b, V3 c, int* part) {
    V3 ab = v3sub(b, a), ac = v3sub(c, a), ap = v3sub(p, a);
    float d1 = v3dot(ab, ap), d2 = v3dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) { *part = 1; return a; }
    V3 bp = v3sub(p, b);
    float d3 = v3dot(ab, bp), d4 = v3dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) { *part = 2; return b; }
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) { *part = 4; return v3mad(a, ab, d1 / (d1 - d3)); }
    V3 cp = v3sub(p, c);
    float d5 = v3dot(ab, cp), d6 = v3dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) { *part = 3; return c; }
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) { *part = 6; return v3mad(a, ac, d2 / (d2 - d6)); }
    float va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) { *part = 5; return v3mad(b, v3sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))); }
    float den = 1.0f / (va + vb + vc);
    *part = 0;
    return v3add(a, v3add(v3scale(ab, vb * den), v3scale(ac, vc * den)));
}

// The signed distance at p: exact (nearest triangle, signed by its pseudo normal) within `near` of the surface, the
// coarse field's beyond
static float sdf_at(const MeshSdf* s, V3 p, float near) {
    const int r = (int)ceilf(near / s->cell);
    int ci = (int)((p.x - s->lo.x) / s->cell), cj = (int)((p.y - s->lo.y) / s->cell), ck = (int)((p.z - s->lo.z) / s->cell);
    float best = near * near;
    int bt = -1, bpart = 0;
    V3 bq = p;
    for (int k = ck - r; k <= ck + r; k++) {
        if (k < 0 || k >= s->nz) continue;
        for (int j = cj - r; j <= cj + r; j++) {
            if (j < 0 || j >= s->ny) continue;
            for (int i = ci - r; i <= ci + r; i++) {
                if (i < 0 || i >= s->nx) continue;
                size_t c = ((size_t)k * s->ny + j) * s->nx + i;
                for (int q = s->start[c]; q < s->start[c + 1]; q++) {
                    int t = s->tris[q];
                    const int* id = &s->m->idx[3 * t];
                    int part;
                    V3 x = closest_on_tri(p, s->m->pos[id[0]], s->m->pos[id[1]], s->m->pos[id[2]], &part);
                    V3 dlt = v3sub(p, x);
                    float d2 = v3dot(dlt, dlt);
                    if (d2 < best) { best = d2; bt = t; bpart = part; bq = x; }
                }
            }
        }
    }
    if (bt < 0) {
        float c = grid_sample(&s->coarse, p);
        return c < 0 ? fb_min(c, -near) : fb_max(c, near);
    }
    V3 n;
    if (bpart == 0) n = s->fn[bt];
    else if (bpart <= 3) n = s->vn[s->m->idx[3 * bt + bpart - 1]];
    else n = s->en[s->edgeOf[3 * bt + (bpart - 4)]];
    float d = sqrtf(best);
    return v3dot(v3sub(p, bq), n) < 0 ? -d : d;
}

// ---- helpers

static float smoothstep01(float x) {
    x = fb_clamp(x, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

// a max filter of radius r nodes (a box), separable, in place
static void max_filter(float* v, int nx, int ny, int nz, int r) {
    if (r <= 0) return;
    const int dims[3] = { nx, ny, nz };
    const size_t strides[3] = { 1, (size_t)nx, (size_t)nx * ny };
    const int longest = fb_maxi(nx, fb_maxi(ny, nz));
    float* line = FB_NEW(float, longest);
    for (int axis = 0; axis < 3; axis++) {
        const int n = dims[axis];
        const int a1 = axis == 0 ? 1 : 0, a2 = axis == 2 ? 1 : 2;
        for (int u = 0; u < dims[a1]; u++)
            for (int w = 0; w < dims[a2]; w++) {
                size_t base = u * strides[a1] + w * strides[a2];
                for (int i = 0; i < n; i++) line[i] = v[base + i * strides[axis]];
                for (int i = 0; i < n; i++) {
                    float m = line[i];
                    for (int k = fb_maxi(0, i - r); k <= fb_mini(n - 1, i + r); k++) m = fb_max(m, line[k]);
                    v[base + i * strides[axis]] = m;
                }
            }
    }
    free(line);
}

// ---- per body

typedef struct {
    const Design* d;
    const MeshSdf* sdf;
    int b;
    const BoxGrid* g;
    const Rigid* open;               // every body's transform, open
    Rigid* const* poses;             // transforms at the poses another part could get in the way at
    const int* moving;               // per pose: the hinge turning (-1 closed)
    int nposes;
    const Box3* box;                 // every body's box (its own frame)
    float reach, band, clear, flatten;
    float *f, *graw, *lower;         // per node: the part's field, restored, the least it may be (NAN: adds nothing)
    float *delta, *w;                // the mesh's offset from the part's field, and how much of it is taken
} NodeCtx;

// how near (field) another part comes to q (the body's frame) at any pose it could get in the way at
static float room_at(const NodeCtx* c, V3 q, float enough, float stop) {
    const Design* d = c->d;
    float room = FLT_MAX;
    for (int P = 0; P < c->nposes; P++) {
        const Rigid* xf = c->poses[P];
        V3 x = rapply(xf[c->b], q);
        for (int o = 0; o < d->nbodies; o++) {
            if (o == c->b) continue;
            if (c->moving[P] >= 0 && design_below(d, c->moving[P], o) == design_below(d, c->moving[P], c->b)) continue;
            V3 qo = rinv_apply(xf[o], x);
            const Box3 bb = c->box[o];
            float out = 0.0f;
            out = fb_max(out, fb_max(bb.min.x - qo.x, qo.x - bb.max.x));
            out = fb_max(out, fb_max(bb.min.y - qo.y, qo.y - bb.max.y));
            out = fb_max(out, fb_max(bb.min.z - qo.z, qo.z - bb.max.z));
            if (out >= room || out > enough) continue;
            room = fb_min(room, body_field(d, o, qo, NULL));
            if (room <= stop) return room;      // (already as close as matters: nothing may be added here)
        }
    }
    return room;
}

// pass 1: the part's field, and the mesh's offset from it with how much of it to take (before smoothing)
static void node_task(void* vc, int k) {
    NodeCtx* c = vc;
    const Design* d = c->d;
    const BoxGrid* g = c->g;
    const float s = fb_max(d->scale, 1e-6f);
    for (int j = 0; j < g->ny; j++)
        for (int i = 0; i < g->nx; i++) {
            size_t id = boxgrid_index(g, i, j, k);
            V3 q = boxgrid_node(g, i, j, k);
            float f = body_field(d, c->b, q, NULL);
            c->f[id] = f;
            c->delta[id] = 0.0f;
            c->w[id] = 0.0f;
            if (fabsf(f) > c->band) continue;
            // the mesh there, in the open pose (design units)
            V3 x = v3add(v3scale(rapply(c->open[c->b], q), s), d->offset);
            float t = sdf_at(c->sdf, x, (c->reach + c->band) * s) / s;
            float delta = t - f;
            c->delta[id] = delta;
            c->w[id] = 1.0f - smoothstep01((fabsf(delta) / c->reach - 0.5f) / 0.5f);
        }
}

// pass 2: restored (flattened on the sphere's surface), and where it adds material, the least the field may be
static void restore_task(void* vc, int k) {
    NodeCtx* c = vc;
    const Design* d = c->d;
    const BoxGrid* g = c->g;
    for (int j = 0; j < g->ny; j++)
        for (int i = 0; i < g->nx; i++) {
            size_t id = boxgrid_index(g, i, j, k);
            V3 q = boxgrid_node(g, i, j, k);
            const float f = c->f[id];
            c->graw[id] = f;
            c->lower[id] = NAN;
            if (fabsf(f) > c->band) continue;
            float w = c->w[id] * (1.0f - smoothstep01((fabsf(f) - (c->band - 2.0f * g->h)) / (2.0f * g->h)));    // (to nothing at the band's edge)
            float delta = c->delta[id];
            // on the sphere's surface, what is taken away is flattened too: it stays a sphere, textured
            float onSphere = 1.0f - smoothstep01((d->R - v3len(q)) / c->reach);
            if (delta > 0.0f) delta *= 1.0f - c->flatten * onSphere;
            float gr = f + w * delta;
            c->graw[id] = gr;
            if (gr >= f) continue;
            // it adds material: the least the field may be there (never less than the part's own)
            // (past the sphere, nothing may be added whatever else is near)
            float sphere = v3len(q) - d->R;
            if (sphere >= f) {
                c->lower[id] = f;
                continue;
            }
            float room = room_at(c, q, c->reach + c->clear + g->h, c->clear - f + 0.5f * g->h);
            float lower = fb_max(sphere, c->clear - room);
            c->lower[id] = fb_min(f, lower);
        }
}

// a box blur of radius 1, in place (a few times: near a Gaussian), so what is restored blends in smoothly
static void blur(float* v, int nx, int ny, int nz, int times) {
    const int dims[3] = { nx, ny, nz };
    const size_t strides[3] = { 1, (size_t)nx, (size_t)nx * ny };
    const int longest = fb_maxi(nx, fb_maxi(ny, nz));
    float* line = FB_NEW(float, longest);
    for (int it = 0; it < times; it++)
        for (int axis = 0; axis < 3; axis++) {
            const int n = dims[axis];
            const int a1 = axis == 0 ? 1 : 0, a2 = axis == 2 ? 1 : 2;
            for (int u = 0; u < dims[a1]; u++)
                for (int w = 0; w < dims[a2]; w++) {
                    size_t base = u * strides[a1] + w * strides[a2];
                    for (int i = 0; i < n; i++) line[i] = v[base + i * strides[axis]];
                    for (int i = 0; i < n; i++) {
                        float l = i > 0 ? line[i - 1] : line[i], r = i < n - 1 ? line[i + 1] : line[i];
                        v[base + i * strides[axis]] = 0.25f * l + 0.5f * line[i] + 0.25f * r;
                    }
                }
        }
    free(line);
}

static bool restore_body(const Design* d, const Params* p, const MeshSdf* sdf, int b, Rigid* const* poses, const int* moving, int nposes,
                         const Box3* boxes, const Rigid* open, BoxGrid* out, double* added, double* clipped) {
    // (added[0]: grid points that became material, added[1]: that stopped being it)
    const float u = 1.0f / fb_max(d->mmPerUnit, 1e-6f);
    const float h = 2.0f * d->R / fb_maxi(p->featureResolution, 8);
    const float reach = fb_max(p->featureReachMm * u, 2.0f * h);
    const float band = reach + 3.0f * h;
    Box3 bb = body_bounds(d, b);
    if (!box_valid(bb)) return false;
    bb = box_grow(bb, band);
    BoxGrid g = boxgrid_create(bb.min, bb.max, h);
    const size_t N = (size_t)g.nx * g.ny * g.nz;
    float* f = FB_NEW(float, N);
    float* graw = FB_NEW(float, N);
    float* lower = FB_NEW(float, N);
    float* delta = FB_NEW(float, N);
    float* wgt = FB_NEW(float, N);
    NodeCtx c = { d, sdf, b, &g, open, poses, moving, nposes, boxes, reach, band, p->clearanceMm * u, fb_clamp(p->featureFlatten, 0.0f, 1.0f),
                  f, graw, lower, delta, wgt };
    double t0 = fb_now();
    fb_parallel_for(g.nz, node_task, &c);
    double t1 = fb_now();
    blur(wgt, g.nx, g.ny, g.nz, 3);
    fb_parallel_for(g.nz, restore_task, &c);
    if (getenv("FOLDBALL_DEBUG"))
        fprintf(stderr, "features, body %d: %dx%dx%d, measuring %.1f s, restoring %.1f s\n", b, g.nx, g.ny, g.nz, t1 - t0, fb_now() - t1);
    free(delta);
    free(wgt);
    const int rr = (int)ceilf(p->featureSinkMm * u / h);
    // 1. flatten: around where an addition doesn't fit, additions keep (1 - flatten) of their height
    float* zone = FB_NEW(float, N);
    for (size_t k = 0; k < N; k++) zone[k] = !isnan(lower[k]) && graw[k] < lower[k] ? 1.0f : 0.0f;
    max_filter(zone, g.nx, g.ny, g.nz, rr);
    blur(zone, g.nx, g.ny, g.nz, 2);
    const float keep = 1.0f - fb_clamp(p->featureFlatten, 0.0f, 1.0f);
    for (size_t k = 0; k < N; k++)
        if (zone[k] > 0.0f && graw[k] < f[k]) graw[k] = f[k] - (f[k] - graw[k]) * (1.0f - zone[k] * (1.0f - keep));
    // 2. sink: the surface around what still sticks out sinks by as much (times sink)
    float* shift = zone;
    for (size_t k = 0; k < N; k++) shift[k] = !isnan(lower[k]) ? fb_max(0.0f, lower[k] - graw[k]) : 0.0f;
    max_filter(shift, g.nx, g.ny, g.nz, rr);
    blur(shift, g.nx, g.ny, g.nz, 2);              // (the recess's walls sloped, not stepped)
    const float sink = fb_clamp(p->featureSink, 0.0f, 1.0f);
    // 3. what still doesn't fit, cut off; the offset kept
    bool any = false;
    for (size_t k = 0; k < N; k++) {
        float v = graw[k];
        if (fabsf(f[k]) <= band && shift[k] > 0.0f) v += sink * shift[k] * (1.0f - smoothstep01((fabsf(f[k]) - (band - 2.0f * h)) / (2.0f * h)));
        if (!isnan(lower[k])) {
            float cut = fb_max(v, lower[k]);
            if (v < 0.0f && cut >= 0.0f && f[k] > 0.0f) *clipped += 1.0;      // (would have been material, and isn't)
            v = cut;
        }
        // (never past the sphere, whatever)
        V3 q = boxgrid_node(&g, (int)(k % g.nx), (int)((k / g.nx) % g.ny), (int)(k / ((size_t)g.nx * g.ny)));
        v = fb_max(v, fb_min(f[k], v3len(q) - d->R));
        g.v[k] = v - f[k];
        if (f[k] > 0.0f && v < 0.0f) added[0] += 1.0;
        if (f[k] < 0.0f && v > 0.0f) added[1] += 1.0;
        if (fabsf(g.v[k]) > 1e-6f) any = true;
    }
    free(f);
    free(graw);
    free(lower);
    free(zone);
    if (!any) {
        boxgrid_free(&g);
        return false;
    }
    *out = g;
    return true;
}

int restore_features(Design* d, const Params* p, char* msg, int msgSize) {
    const double t0 = fb_now();
    if (d->target.nt == 0) {
        snprintf(msg, (size_t)msgSize, "no features restored: the design has no target mesh");
        return 0;
    }
    const float u = 1.0f / fb_max(d->mmPerUnit, 1e-6f);
    const float h = 2.0f * d->R / fb_maxi(p->featureResolution, 8);
    const float reachW = (fb_max(p->featureReachMm * u, 2.0f * h) + 3.0f * h) * fb_max(d->scale, 1e-6f);
    MeshSdf sdf;
    sdf_build(&sdf, &d->target, fb_max(reachW, 1e-4f));
    // the poses another part could get in the way at: closed, and through every hinge's turn
    const int K = 8;
    int order[FB_MAX_BODIES];
    design_order(d, order);
    const int np = 1 + d->nhinges * K;
    Rigid** poses = FB_NEW(Rigid*, np);
    int* moving = FB_NEW(int, np);
    for (int q = 0; q < np; q++) {
        poses[q] = FB_NEW(Rigid, FB_MAX_BODIES);
        float s = q == 0 ? 0.0f : ((q - 1) / K + ((q - 1) % K + 0.5f) / K) / fb_maxi(d->nhinges, 1);
        moving[q] = q == 0 ? -1 : order[(q - 1) / K];
        design_transforms(d, s, poses[q]);
    }
    Rigid open[FB_MAX_BODIES];
    design_transforms(d, 1.0f, open);
    Box3 boxes[FB_MAX_BODIES];
    for (int b = 0; b < d->nbodies; b++) boxes[b] = body_bounds(d, b);
    int restored = 0;
    double added[2] = { 0, 0 }, clipped = 0;
    BoxGrid out[FB_MAX_BODIES];
    bool has[FB_MAX_BODIES] = { 0 };
    // (every body against the others as they were: the results go in together at the end)
    for (int b = 0; b < d->nbodies; b++) has[b] = restore_body(d, p, &sdf, b, poses, moving, np, boxes, open, &out[b], added, &clipped);
    for (int b = 0; b < d->nbodies; b++)
        if (has[b]) {
            boxgrid_free(&d->bodies[b].detail);
            d->bodies[b].detail = out[b];
            restored++;
        }
    for (int q = 0; q < np; q++) free(poses[q]);
    free(poses);
    free(moving);
    sdf_free(&sdf);
    const double cell = (double)h * h * h * d->mmPerUnit * d->mmPerUnit * d->mmPerUnit;
    snprintf(msg, (size_t)msgSize,
             "features restored on %d of %d parts (%.2f mm grid, reach %.1f mm): about %.0f mm^3 added and %.0f mm^3 taken away; where they "
             "didn't fit, flattened %.0f%%, %s, %.0f mm^3 cut off (%.1f s)",
             restored, d->nbodies, h * d->mmPerUnit, p->featureReachMm, added[0] * cell, added[1] * cell, 100.0f * p->featureFlatten,
             p->featureSink >= 0.999f ? "sunk into recesses" : (p->featureSink <= 0.001f ? "squared off flush" : "partly sunk"), clipped * cell,
             fb_now() - t0);
    return restored;
}
