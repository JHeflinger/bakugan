#ifndef FB_SEMANTIC_H
#define FB_SEMANTIC_H
// The semantic deformation: the target the open pose is compared with (the input mesh, posed) is deformed as rigidly
// as possible (ARAP) toward the design's open pose, from the input as loaded (so deformations never pile up into
// drift). The input is simplified to a cage of a few thousand vertices; each round pulls the cage's vertices a
// step toward the design's surface (soft constraints) and solves ARAP; the full mesh follows its cage (each vertex
// carried by the rotations of its nearest cage vertices).
//
// The deformed input is kept as the new target when it lowers the loss and is judged the same shape: semantic
// equivalence here is geometric (no learned model): the deformation must be locally rigid (its stretch, what the
// best rotations don't explain, under max_strain), keep its volume (within max_volume_change), and not run through
// itself (the volume inside it by parity must match its signed volume).
#include "eval.h"

bool semantic_try(Evaluator* ev, Design* d, const Params* p, float collisionScale, char* msg, int msgSize);

// The checks alone, for a deformed copy of `rest` (same vertices): mean strain, volume change, self-overlap share.
void semantic_measure(const TriMesh* rest, const TriMesh* deformed, float* strain, float* volumeChange, float* overlap);
#endif
