#ifndef FB_EVAL_H
#define FB_EVAL_H
// The loss and its gradient.
//
// The closed pose is the sphere by construction, so only the open pose is drawn: from several views as soft
// silhouettes (and depth), against the target mesh, the design placed in the mesh's space (world = scale * design +
// offset: where and how big the open pose sits doesn't matter, so those are parameters too). A pixel's coverage is
// sigmoid(-dmin / k) of the nearest its ray comes to the union of the opened bodies, so its derivative only needs the
// field's derivative at that point (the envelope theorem): every term hands out "slopes" (a point of a body in some
// pose, and d loss / d field there) and one pass turns them into the gradient of the cut planes and hollows (through
// whichever decides the field there), the hinges (through the bodies' transforms) and the placement.
#include "core/config.h"
#include "core/design.h"

typedef struct {
    float plane[FB_MAX_PLANES][5];                 // n, d, k
    float hinge[FB_MAX_BODIES][2];       // phi, angle
    float hollow[FB_MAX_HOLLOWS][FB_HOLLOW_PARAMS];     // a, b, r
    float place[4];                      // scale, offset
} Grad;

typedef struct {
    bool valid;
    float total;
    float open;                          // silhouette (and depth) loss, view mean
    float openCover, openExtra;          // share of the target covered, extra outside it
    float collision, collisionOpen, collisionMotion;
    float hinge, dependency, onePiece, surface;
    float surfaceCarved;                 // the share of the sphere's surface a hollow takes away
    float fragments;
    int fragmentParts;
    float hingeCollision[FB_MAX_BODIES];
    float volume;
    float collisionScale;
    double ms;
} Score;

typedef struct Evaluator Evaluator;
Evaluator* eval_create(const Params* p, const Design* d);
void eval_free(Evaluator* e);
void eval_set_target(Evaluator* e, const TriMesh* target);
Grad* grad_create(const Design* d);
void grad_free(Grad* g);
void grad_clear(Grad* g, const Design* d);
Score eval_run(Evaluator* e, const Design* d, Grad* grad, float collisionScale, float phase);

int eval_views(const Evaluator* e);
int eval_resolution(const Evaluator* e);
void eval_picture(const Evaluator* e, int view, float* design, float* target);

// A lattice over the sphere: which body (index) each point is in (-1: none, outside, or in a hollow).
typedef struct {
    int M;
    float cell;
    V3 origin;
    int* label;
} Lattice;
void lattice_build(const Design* d, int M, Lattice* L);
void lattice_free(Lattice* L);
// What of each body lies apart from its largest piece (6-connected) on a lattice of labels (-1: no body): flag
// (optional) 1 there per point; perBody (optional, nlabels long) per body. Returns the count, and the number of
// separate apart pieces in *parts (optional).
int lattice_apart(int M, const int* label, int nlabels, unsigned char* flag, int* perBody, int* parts);
static inline V3 lattice_point(const Lattice* L, int i, int j, int k) {
    return v3(L->origin.x + i * L->cell, L->origin.y + j * L->cell, L->origin.z + k * L->cell);
}
#endif
