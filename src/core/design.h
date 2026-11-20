#ifndef FB_DESIGN_H
#define FB_DESIGN_H
// The design: a sphere carved into bodies. Closed is the design as it is: the bodies partition the sphere (cells of
// cut planes), so closed it is exactly the sphere (less any hollows carved from it). The bodies are joined by hinges
// into a tree; opening turns every hinge by its angle, and the open pose is what looks like the mesh.
//
// Everything is an exact distance field (the sphere, half spaces, capsules), with no grid: there is no resolution but
// the one a mesh for drawing or printing is made at.
//
// A hinge sits on its cut plane's seam: where the plane meets the sphere, inset by the knuckle radius, its axis
// along the seam (a flush hinge on the surface, against both of its bodies). Its parameters are where along the seam
// (phi) and how far it opens (angle).
#include "fmath.h"
#include "grid.h"
#include "mesh.h"

#define FB_MAX_BODIES 64
#define FB_MAX_PLANES 192
#define FB_MAX_CELLS 48
#define FB_MAX_HALF 32
#define FB_MAX_EXTRAS 512
#define FB_MAX_HOLLOWS 64

typedef struct {
    short plane[FB_MAX_HALF];        // plane index
    signed char sign[FB_MAX_HALF];   // the half space s (n.x - d) <= 0
    int n;
} Cell;

// A cut: flat (k = 0, the plane n.q = d) or bent into a piece of a sphere of radius 1/|k| (bulging toward -n for
// k > 0, toward +n for k < 0) through the same point n d. Its field is the exact signed distance to that surface,
// positive on the +n side, written so that it stays exact as k goes to 0. A sphere meets the ball's sphere in a circle,
// so a bent cut's seam is still a circle and its hinges still sit exactly on it.
typedef struct {
    int id;
    V3 n;
    float d;
    float k;                         // bend: 1 / radius (0 flat)
    float m[5], v[5];                // Adam's moments (n, d, k)
    int t;
} Plane;

// the cut's field at q: with a = n.q - d and u = q - n d,
//   (2a - k|u|^2) / (1 + sqrt(1 - 2ak + k^2 |u|^2))
// (= a when k = 0; otherwise the distance to the sphere through n d with centre n (d + 1/k))
static inline float cut_field(V3 n, float d, float k, V3 q) {
    float a = v3dot(n, q) - d;
    if (k == 0.0f) return a;
    V3 u = v3sub(q, v3scale(n, d));
    float u2 = v3dot(u, u);
    return (2.0f * a - k * u2) / (1.0f + sqrtf(fmaxf(0.0f, 1.0f - 2.0f * a * k + k * k * u2)));
}
static inline float plane_field(const Plane* p, V3 q) { return cut_field(p->n, p->d, p->k, q); }
// its gradient in q: (n - k u) / sqrt(1 - 2ak + k^2 |u|^2)
static inline V3 plane_grad(const Plane* p, V3 q) {
    if (p->k == 0.0f) return p->n;
    float a = v3dot(p->n, q) - p->d;
    V3 u = v3sub(q, v3scale(p->n, p->d));
    float S = sqrtf(fmaxf(1e-12f, 1.0f - 2.0f * a * p->k + p->k * p->k * v3dot(u, u)));
    return v3scale(v3sub(p->n, v3scale(u, p->k)), 1.0f / S);
}
// its gradient in the cut's parameters (n.x, n.y, n.z, d, k)
void plane_param_grad(const Plane* p, V3 q, float g[5]);
// where the cut meets the sphere of radius r: the circle in the plane n.q = h, of radius rho
void plane_seam(const Plane* p, float r, float* h, float* rho);
// the bend nearest p->k (toward flat) at which the cut still crosses the sphere of radius r properly (its seam's
// plane within max(|d|, 0.9 r) of the centre, and its centre clear: 1 + k d >= 0.2); a cut bent further would close
// into a bubble inside the ball, which no hinge could reach
float plane_bend_ok(const Plane* p, float r);

typedef struct {
    int id;
    Cell cells[FB_MAX_CELLS];
    int ncells;
    Grid carve;                      // post: material taken away where carve(q) < 0 (v == NULL: none)
    BoxGrid detail;                  // post: the input's features restored, added to the field (v == NULL: none)
} Body;

typedef struct {
    int id;
    int parent, child;               // body ids
    int plane;                       // the cut plane it sits on (index)
    V3 ref;                          // which way phi = 0 is, on the plane
    float phi;                       // where along the seam
    float angle;                     // how far it turns, opening
    int order;                       // its place in the opening sequence (0 opens first: the root hinge)
    float m[2], v[2];                // Adam's moments (phi, angle)
    int t;
    int holder;                      // post: the hinge whose opening frees this one (-1 none), by id
    V3 pivot, axis;                  // (derived from the plane and phi: design_hinge_frames)
} Hinge;

// A hollow: material taken from every body it reaches, a capsule (segment a-b, radius r).
#define FB_HOLLOW_PARAMS 7
typedef struct {
    int id;
    V3 a, b;
    float r;
    float m[FB_HOLLOW_PARAMS], v[FB_HOLLOW_PARAMS];
    int t;
} Hollow;
// a hollow's parameters (what Adam moves): a, b, r
static inline int hollow_params(Hollow* h, float** out) {
    float* p[FB_HOLLOW_PARAMS] = { &h->a.x, &h->a.y, &h->a.z, &h->b.x, &h->b.y, &h->b.z, &h->r };
    for (int k = 0; k < FB_HOLLOW_PARAMS; k++) out[k] = p[k];
    return FB_HOLLOW_PARAMS;
}

typedef enum { EX_BOX = 0, EX_CYLINDER = 1 } ExtraShape;
typedef struct {
    int body;                        // body id
    int shape;                       // ExtraShape
    int op;                          // +1 adds material, -1 takes it away, +2 adds it after the cuts
    Rigid pose;                      // the primitive's frame in the body's (cylinders along z)
    V3 size;                         // box: half extents; cylinder: radius, (unused), half length
    float round;
    int tag;                         // 0 hinge, 1 ledge, 2 latch, 3 other
} Extra;

typedef struct {
    float R;                         // the sphere (centred at the origin)
    float inset;                     // hinges' depth under the surface (their knuckles' radius)
    float gap;                       // post: every cut opened by this on each side (the printing clearance), exactly
    Plane planes[FB_MAX_PLANES];
    int nplanes;
    Body bodies[FB_MAX_BODIES];
    int nbodies;
    Hinge hinges[FB_MAX_BODIES];
    int nhinges;
    Hollow hollows[FB_MAX_HOLLOWS];
    int nhollows;
    Extra extras[FB_MAX_EXTRAS];
    int nextras;
    int root;                        // the root body's id
    // where the open pose sits in the mesh's space: world = scale * design + offset
    float scale;
    V3 offset;
    float pm[4], pv[4];
    int pt;
    int nextId;
    float mmPerUnit;                 // export scale (design units)
    bool built;
    TriMesh target;                  // what the open pose is compared with (the input, or its semantic deformation)
    TriMesh target0;                 // the input as loaded (normalized)
} Design;

typedef struct {
    int kind;                        // 0 the sphere, 1 a plane, 2 a hollow, 3 something else (extras, carving)
    int index, sign;                 // the plane (and its sign) or the hollow
} Active;

void design_init(Design* d, const TriMesh* mesh, float sphereMm, float insetMm);
void design_free(Design* d);
void design_copy(Design* dst, const Design* src);
Design* design_clone(const Design* src);

int design_body(const Design* d, int id);
int design_hinge(const Design* d, int id);
int design_parent_hinge(const Design* d, int bodyId);
int design_new_id(Design* d);

float hollow_field(const Hollow* h, V3 q);
float body_cell(const Design* d, int b, V3 q, Active* act);
float body_field(const Design* d, int b, V3 q, Active* act);
V3 body_gradient(const Design* d, int b, V3 q);
float extra_field(const Extra* e, V3 q);
// the open pose as a whole (the union of the bodies, opened), in design units
float design_field_open(const Design* d, V3 q);

// the hinges' pivots and axes from their planes and phi (call after changing either)
void design_hinge_frames(Design* d);
void hinge_frame(const Design* d, const Plane* planes, const Hinge* h, V3* pivot, V3* axis);
// Every body's transform at opening progress s (0 closed: the design; 1 open), indexed like d->bodies.
void design_transforms(const Design* d, float s, Rigid* out);
// the same with other planes and hinges (arrays like d's)
void design_transforms_h(const Design* d, const Plane* planes, const Hinge* hinges, float s, Rigid* out);
float hinge_angle_at(const Design* d, const Hinge* h, float s);
void design_order(const Design* d, int* order);
void design_normalize_orders(Design* d);
bool design_below(const Design* d, int hingeIndex, int bodyIndex);
void design_tidy(Design* d);

bool design_save(const Design* d, const char* path, char* err, int errSize);
bool design_load(Design* d, const char* path, char* err, int errSize);
#endif
