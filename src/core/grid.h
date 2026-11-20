#ifndef FB_GRID_H
#define FB_GRID_H
// A signed distance grid: n^3 nodes over the cube [-B, B]^3 (negative inside). Sampled trilinearly; outside the cube
// the value at the nearest point of the cube plus the distance to it.
#include "fmath.h"
#include "mesh.h"

typedef struct {
    int n;
    float B, h;          // half size of the cube, node spacing
    float* v;            // n^3, x fastest
} Grid;

Grid grid_create(int n, float B);
Grid grid_copy(const Grid* g);
void grid_free(Grid* g);
static inline int grid_index(const Grid* g, int i, int j, int k) { return (k * g->n + j) * g->n + i; }
static inline V3 grid_node(const Grid* g, int i, int j, int k) { return v3(-g->B + i * g->h, -g->B + j * g->h, -g->B + k * g->h); }

// trilinear sample; idx/w (optional) receive the 8 nodes and their weights (all zero weights outside the cube's
// nodes is impossible: the clamped cell is used); grad (optional) the analytic gradient
static inline float grid_sample_full(const Grid* g, V3 p, int idx[8], float w[8], V3* grad) {
    const float lim = g->B;
    V3 c = v3(fb_clamp(p.x, -lim, lim), fb_clamp(p.y, -lim, lim), fb_clamp(p.z, -lim, lim));
    float outside = v3dist(p, c);
    float fx = (c.x + lim) / g->h, fy = (c.y + lim) / g->h, fz = (c.z + lim) / g->h;
    int n = g->n;
    int i = (int)fx, j = (int)fy, k = (int)fz;
    if (i > n - 2) i = n - 2;
    if (j > n - 2) j = n - 2;
    if (k > n - 2) k = n - 2;
    float a = fx - i, b = fy - j, cc = fz - k;
    int base = (k * n + j) * n + i, sy = n, sz = n * n;
    const float* v = g->v;
    float v000 = v[base], v100 = v[base + 1], v010 = v[base + sy], v110 = v[base + sy + 1];
    float v001 = v[base + sz], v101 = v[base + sz + 1], v011 = v[base + sz + sy], v111 = v[base + sz + sy + 1];
    float x00 = v000 + (v100 - v000) * a, x10 = v010 + (v110 - v010) * a, x01 = v001 + (v101 - v001) * a, x11 = v011 + (v111 - v011) * a;
    float y0 = x00 + (x10 - x00) * b, y1 = x01 + (x11 - x01) * b;
    float val = y0 + (y1 - y0) * cc;
    if (idx) {
        idx[0] = base; idx[1] = base + 1; idx[2] = base + sy; idx[3] = base + sy + 1;
        idx[4] = base + sz; idx[5] = base + sz + 1; idx[6] = base + sz + sy; idx[7] = base + sz + sy + 1;
        float ia = 1 - a, ib = 1 - b, ic = 1 - cc;
        w[0] = ia * ib * ic; w[1] = a * ib * ic; w[2] = ia * b * ic; w[3] = a * b * ic;
        w[4] = ia * ib * cc; w[5] = a * ib * cc; w[6] = ia * b * cc; w[7] = a * b * cc;
    }
    if (grad) {
        float gx = ((v100 - v000) * (1 - b) + (v110 - v010) * b) * (1 - cc) + ((v101 - v001) * (1 - b) + (v111 - v011) * b) * cc;
        float gy = (x10 - x00) * (1 - cc) + (x11 - x01) * cc;
        float gz = y1 - y0;
        *grad = v3(gx / g->h, gy / g->h, gz / g->h);
        if (outside > 0.0f) *grad = v3add(*grad, v3scale(v3sub(p, c), 1.0f / outside));
    }
    return val + outside;
}
static inline float grid_sample(const Grid* g, V3 p) { return grid_sample_full(g, p, NULL, NULL, NULL); }

// A grid of values over a box (not centred): what feature restoration keeps per body. Sampled trilinearly; 0 outside.
typedef struct {
    int nx, ny, nz;
    V3 o;                // the first node
    float h;
    float* v;            // nx*ny*nz, x fastest
} BoxGrid;
BoxGrid boxgrid_create(V3 lo, V3 hi, float h);
BoxGrid boxgrid_copy(const BoxGrid* g);
void boxgrid_free(BoxGrid* g);
static inline size_t boxgrid_index(const BoxGrid* g, int i, int j, int k) { return ((size_t)k * g->ny + j) * g->nx + i; }
static inline V3 boxgrid_node(const BoxGrid* g, int i, int j, int k) { return v3(g->o.x + i * g->h, g->o.y + j * g->h, g->o.z + k * g->h); }
static inline float boxgrid_sample(const BoxGrid* g, V3 p) {
    float fx = (p.x - g->o.x) / g->h, fy = (p.y - g->o.y) / g->h, fz = (p.z - g->o.z) / g->h;
    if (fx < 0 || fy < 0 || fz < 0 || fx > g->nx - 1 || fy > g->ny - 1 || fz > g->nz - 1) return 0.0f;
    int i = (int)fx, j = (int)fy, k = (int)fz;
    if (i > g->nx - 2) i = g->nx - 2;
    if (j > g->ny - 2) j = g->ny - 2;
    if (k > g->nz - 2) k = g->nz - 2;
    float a = fx - i, b = fy - j, c = fz - k;
    const float* v = g->v + boxgrid_index(g, i, j, k);
    const size_t sy = (size_t)g->nx, sz = (size_t)g->nx * g->ny;
    float x00 = v[0] + (v[1] - v[0]) * a, x10 = v[sy] + (v[sy + 1] - v[sy]) * a;
    float x01 = v[sz] + (v[sz + 1] - v[sz]) * a, x11 = v[sz + sy] + (v[sz + sy + 1] - v[sz + sy]) * a;
    float y0 = x00 + (x10 - x00) * b, y1 = x01 + (x11 - x01) * b;
    return y0 + (y1 - y0) * c;
}

// The mesh's signed distance (sign by majority of three axis parity tests, so small holes don't flip it).
void grid_from_mesh(Grid* g, const TriMesh* m);
// Makes the grid a distance field again around its zero level set (which stays where it is, to within a fraction
// of a node spacing): optimization leaves it a level set function that drifts from a distance.
void grid_redistance(Grid* g);
// Exact squared distance from p to triangle abc.
float tri_dist2(V3 p, V3 a, V3 b, V3 c);
#endif
