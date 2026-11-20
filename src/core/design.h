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

typedef struct {
    int id;
    V3 n;
    float d;
    float m[4], v[4];                // Adam's moments (n, d)
    int t;
} Plane;

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

typedef struct {
    int id;
    V3 a, b;                         // a capsule: segment and radius
    float r;
    float m[7], v[7];
    int t;
} Hollow;

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
