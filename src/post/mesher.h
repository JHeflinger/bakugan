#ifndef FB_MESHER_H
#define FB_MESHER_H
// Surfaces of distance fields as triangle meshes (naive surface nets: one vertex per cell the surface passes through,
// at the mean of its edges' crossings; one quad per edge the surface crosses), and STL files.
#include "core/design.h"
#include "core/mesh.h"

typedef float (*FieldFn)(void* ctx, V3 p);
// The surface f = 0 in box, cells of about `cell` (at most maxCells a side). Faces point out (toward f > 0).
TriMesh mesh_field(FieldFn f, void* ctx, Box3 box, float cell, int maxCells);
// Body b (index) of the design, in its own frame (the design pose).
TriMesh mesh_body(const Design* d, int b, float cell, int maxCells);
// The body's bounds in its own frame (its material, extras included).
Box3 body_bounds(const Design* d, int b);

bool stl_write(const char* path, const TriMesh* m, float scale, const Rigid* xf, const char* name);
#endif
