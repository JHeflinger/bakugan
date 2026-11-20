#include "grid.h"
#include "util.h"
#include <stdlib.h>
#include <string.h>

Grid grid_create(int n, float B) {
    Grid g;
    g.n = n;
    g.B = B;
    g.h = 2.0f * B / (n - 1);
    g.v = FB_NEW(float, (size_t)n * n * n);
    return g;
}

Grid grid_copy(const Grid* g) {
    Grid c = *g;
    size_t N = (size_t)g->n * g->n * g->n;
    c.v = FB_NEW(float, N);
    memcpy(c.v, g->v, sizeof(float) * N);
    return c;
}

void grid_free(Grid* g) {
    free(g->v);
    g->v = NULL;
}

BoxGrid boxgrid_create(V3 lo, V3 hi, float h) {
    BoxGrid g;
    g.h = h;
    g.o = lo;
    g.nx = (int)ceilf((hi.x - lo.x) / h) + 2;
    g.ny = (int)ceilf((hi.y - lo.y) / h) + 2;
    g.nz = (int)ceilf((hi.z - lo.z) / h) + 2;
    g.v = FB_NEW(float, (size_t)g.nx * g.ny * g.nz);
    return g;
}

BoxGrid boxgrid_copy(const BoxGrid* g) {
    BoxGrid c = *g;
    size_t N = (size_t)g->nx * g->ny * g->nz;
    c.v = FB_NEW(float, N);
    memcpy(c.v, g->v, sizeof(float) * N);
    return c;
}

void boxgrid_free(BoxGrid* g) {
    free(g->v);
    g->v = NULL;
}

float tri_dist2(V3 p, V3 a, V3 b, V3 c) {
    // Ericson, Real-Time Collision Detection 5.1.5
    V3 ab = v3sub(b, a), ac = v3sub(c, a), ap = v3sub(p, a);
    float d1 = v3dot(ab, ap), d2 = v3dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) return v3dot(ap, ap);
    V3 bp = v3sub(p, b);
    float d3 = v3dot(ab, bp), d4 = v3dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return v3dot(bp, bp);
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        float v = d1 / (d1 - d3);
        V3 q = v3sub(p, v3mad(a, ab, v));
        return v3dot(q, q);
    }
    V3 cp = v3sub(p, c);
    float d5 = v3dot(ab, cp), d6 = v3dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return v3dot(cp, cp);
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        float w = d2 / (d2 - d6);
        V3 q = v3sub(p, v3mad(a, ac, w));
        return v3dot(q, q);
    }
    float va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        V3 q = v3sub(p, v3mad(b, v3sub(c, b), w));
        return v3dot(q, q);
    }
    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom, w = vc * denom;
    V3 q = v3sub(p, v3add(a, v3add(v3scale(ab, v), v3scale(ac, w))));
    return v3dot(q, q);
}

// ---- distances away from a band of known ones: two chamfer sweeps over the 26 neighbours

static void chamfer(Grid* g, float* d) {
    const int n = g->n;
    const float h = g->h, h2 = h * 1.41421356f, h3 = h * 1.73205081f;
    for (int pass = 0; pass < 2; pass++) {
        for (int it = 0; it < 2; it++) {
            int start = it == 0 ? 0 : n - 1, end = it == 0 ? n : -1, step = it == 0 ? 1 : -1;
            for (int k = start; k != end; k += step)
                for (int j = start; j != end; j += step)
                    for (int i = start; i != end; i += step) {
                        float best = d[grid_index(g, i, j, k)];
                        // the 13 neighbours already swept in this direction
                        for (int dz = -1; dz <= 0; dz++)
                            for (int dy = -1; dy <= 1; dy++)
                                for (int dx = -1; dx <= 1; dx++) {
                                    if (dz == 0 && (dy > 0 || (dy == 0 && dx >= 0))) continue;
                                    int ii = i + dx * step, jj = j + dy * step, kk = k + dz * step;
                                    if (ii < 0 || jj < 0 || kk < 0 || ii >= n || jj >= n || kk >= n) continue;
                                    int m = abs(dx) + abs(dy) + abs(dz);
                                    float w = m == 1 ? h : (m == 2 ? h2 : h3);
                                    float c = d[grid_index(g, ii, jj, kk)] + w;
                                    if (c < best) best = c;
                                }
                        d[grid_index(g, i, j, k)] = best;
                    }
        }
    }
}

// ---- the mesh's signed distance

typedef struct {
    const TriMesh* m;
    Grid* g;
    float* dist;                 // per worker slice
    int workers;
    float band;
} BandCtx;

static void band_task(void* vctx, int w) {
    BandCtx* c = vctx;
    const Grid* g = c->g;
    float* d = c->dist + (size_t)w * g->n * g->n * g->n;
    const int n = g->n;
    for (int t = w; t < c->m->nt; t += c->workers) {
        V3 a = c->m->pos[c->m->idx[3 * t]], b = c->m->pos[c->m->idx[3 * t + 1]], cc = c->m->pos[c->m->idx[3 * t + 2]];
        V3 lo = v3sub(v3min(a, v3min(b, cc)), v3(c->band, c->band, c->band)), hi = v3add(v3max(a, v3max(b, cc)), v3(c->band, c->band, c->band));
        int i0 = fb_maxi(0, (int)floorf((lo.x + g->B) / g->h)), i1 = fb_mini(n - 1, (int)ceilf((hi.x + g->B) / g->h));
        int j0 = fb_maxi(0, (int)floorf((lo.y + g->B) / g->h)), j1 = fb_mini(n - 1, (int)ceilf((hi.y + g->B) / g->h));
        int k0 = fb_maxi(0, (int)floorf((lo.z + g->B) / g->h)), k1 = fb_mini(n - 1, (int)ceilf((hi.z + g->B) / g->h));
        for (int k = k0; k <= k1; k++)
            for (int j = j0; j <= j1; j++)
                for (int i = i0; i <= i1; i++) {
                    float d2 = tri_dist2(grid_node(g, i, j, k), a, b, cc);
                    float* o = &d[grid_index(g, i, j, k)];
                    if (d2 < *o) *o = d2;
                }
    }
}

typedef struct { float t; } Hit;
static int cmp_float(const void* a, const void* b) {
    float x = *(const float*)a, y = *(const float*)b;
    return x < y ? -1 : (x > y);
}

typedef struct {
    const TriMesh* m;
    const Grid* g;
    int axis;
    int** bins;                  // per column: triangle ids
    int* binCount;
    unsigned char* votes;        // per node: insides counted
} ParityCtx;

static void parity_task(void* vctx, int col) {
    ParityCtx* c = vctx;
    const Grid* g = c->g;
    const int n = g->n, a = c->axis, b = (a + 1) % 3, e = (a + 2) % 3;
    const int jb = col % n, je = col / n;
    // the column's line, nudged off the node positions (meshes put edges exactly on round coordinates)
    float pb = -g->B + jb * g->h + 1.7e-4f * g->h, pe = -g->B + je * g->h + 2.3e-4f * g->h;
    float hits[4096];
    int nh = 0;
    for (int q = 0; q < c->binCount[col] && nh < 4096; q++) {
        int t = c->bins[col][q];
        V3 P[3];
        for (int k = 0; k < 3; k++) P[k] = c->m->pos[c->m->idx[3 * t + k]];
        float xb[3], xe[3], xa[3];
        for (int k = 0; k < 3; k++) { xb[k] = v3get(P[k], b); xe[k] = v3get(P[k], e); xa[k] = v3get(P[k], a); }
        float area = (xb[1] - xb[0]) * (xe[2] - xe[0]) - (xe[1] - xe[0]) * (xb[2] - xb[0]);
        if (fabsf(area) < 1e-20f) continue;
        float l0 = ((xb[1] - pb) * (xe[2] - pe) - (xe[1] - pe) * (xb[2] - pb)) / area;
        float l1 = ((xb[2] - pb) * (xe[0] - pe) - (xe[2] - pe) * (xb[0] - pb)) / area;
        float l2 = 1 - l0 - l1;
        if (l0 < 0 || l1 < 0 || l2 < 0) continue;
        hits[nh++] = l0 * xa[0] + l1 * xa[1] + l2 * xa[2];
    }
    qsort(hits, (size_t)nh, sizeof(float), cmp_float);
    int h = 0;
    for (int i = 0; i < n; i++) {
        float x = -g->B + i * g->h;
        while (h < nh && hits[h] < x) h++;
        if (h & 1) {
            int ijk[3];
            ijk[a] = i; ijk[b] = jb; ijk[e] = je;
            c->votes[grid_index(g, ijk[0], ijk[1], ijk[2])]++;
        }
    }
}

void grid_from_mesh(Grid* g, const TriMesh* m) {
    const int n = g->n;
    const size_t N = (size_t)n * n * n;
    // 1. exact distances in a band of three nodes around the surface
    int workers = fb_threads_count();
    BandCtx bc = { m, g, FB_NEW(float, N * (size_t)workers), workers, 3.0f * g->h };
    for (size_t k = 0; k < N * (size_t)workers; k++) bc.dist[k] = FLT_MAX;
    fb_parallel_for(workers, band_task, &bc);
    float* d = FB_NEW(float, N);
    for (size_t k = 0; k < N; k++) {
        float best = FLT_MAX;
        for (int w = 0; w < workers; w++) best = fb_min(best, bc.dist[(size_t)w * N + k]);
        d[k] = best < FLT_MAX ? sqrtf(best) : FLT_MAX;
    }
    free(bc.dist);
    // 2. the rest by chamfer sweeps from the band (kept a little under, so ray marching never steps over a surface)
    chamfer(g, d);
    for (size_t k = 0; k < N; k++)
        if (d[k] > 3.0f * g->h) d[k] = 3.0f * g->h + (d[k] - 3.0f * g->h) * 0.92f;
    // 3. the sign: inside by the parity of crossings along each axis, two votes of three
    unsigned char* votes = FB_NEW(unsigned char, N);
    for (int axis = 0; axis < 3; axis++) {
        const int b = (axis + 1) % 3, e = (axis + 2) % 3;
        ParityCtx pc = { m, g, axis, FB_NEW(int*, n * n), FB_NEW(int, n * n), votes };
        int* cap = FB_NEW(int, n * n);
        for (int t = 0; t < m->nt; t++) {
            float lob = FLT_MAX, hib = -FLT_MAX, loe = FLT_MAX, hie = -FLT_MAX;
            for (int k = 0; k < 3; k++) {
                V3 p = m->pos[m->idx[3 * t + k]];
                lob = fb_min(lob, v3get(p, b)); hib = fb_max(hib, v3get(p, b));
                loe = fb_min(loe, v3get(p, e)); hie = fb_max(hie, v3get(p, e));
            }
            int b0 = fb_maxi(0, (int)floorf((lob + g->B) / g->h)), b1 = fb_mini(n - 1, (int)ceilf((hib + g->B) / g->h));
            int e0 = fb_maxi(0, (int)floorf((loe + g->B) / g->h)), e1 = fb_mini(n - 1, (int)ceilf((hie + g->B) / g->h));
            for (int je = e0; je <= e1; je++)
                for (int jb = b0; jb <= b1; jb++) {
                    int col = je * n + jb;
                    if (pc.binCount[col] >= cap[col]) {
                        cap[col] = cap[col] ? cap[col] * 2 : 8;
                        pc.bins[col] = fb_realloc(pc.bins[col], sizeof(int) * (size_t)cap[col]);
                    }
                    pc.bins[col][pc.binCount[col]++] = t;
                }
        }
        fb_parallel_for(n * n, parity_task, &pc);
        for (int k = 0; k < n * n; k++) free(pc.bins[k]);
        free(pc.bins);
        free(pc.binCount);
        free(cap);
    }
    for (size_t k = 0; k < N; k++) g->v[k] = votes[k] >= 2 ? -d[k] : d[k];
    free(votes);
    free(d);
}

void grid_redistance(Grid* g) {
    // The nodes of every cell the surface passes through (a neighbour among the 26 of the other sign) keep their
    // values exactly, so the trilinear surface doesn't move at all; everything beyond them becomes a distance again
    // (chamfer sweeps from them, with their values as their distances).
    const int n = g->n;
    const size_t N = (size_t)n * n * n;
    float* d = FB_NEW(float, N);
    unsigned char* band = FB_NEW(unsigned char, N);
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++) {
                int id = grid_index(g, i, j, k);
                bool neg = g->v[id] < 0;
                bool edge = false;
                for (int dz = -1; dz <= 1 && !edge; dz++)
                    for (int dy = -1; dy <= 1 && !edge; dy++)
                        for (int dx = -1; dx <= 1 && !edge; dx++) {
                            int ii = i + dx, jj = j + dy, kk = k + dz;
                            if (ii < 0 || jj < 0 || kk < 0 || ii >= n || jj >= n || kk >= n) continue;
                            if ((g->v[grid_index(g, ii, jj, kk)] < 0) != neg) edge = true;
                        }
                band[id] = edge;
                d[id] = edge ? fabsf(g->v[id]) : FLT_MAX;
            }
    chamfer(g, d);
    for (size_t k = 0; k < N; k++) {
        if (band[k]) continue;
        float dd = d[k] < FLT_MAX ? d[k] : 2.0f * g->B;
        g->v[k] = g->v[k] < 0 ? -dd : dd;
    }
    free(band);
    free(d);
}
