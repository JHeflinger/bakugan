#ifndef FB_MESH_H
#define FB_MESH_H
#include "fmath.h"
#include <stdbool.h>

typedef struct {
    V3* pos;
    int nv;
    int* idx;            // 3 per triangle
    int nt;
} TriMesh;

bool mesh_load_obj(const char* path, TriMesh* out, char* err, int errSize);
bool mesh_save_obj(const char* path, const TriMesh* m);
void mesh_free(TriMesh* m);
TriMesh mesh_copy(const TriMesh* m);
Box3 mesh_bounds(const TriMesh* m);
double mesh_volume(const TriMesh* m);           // signed (positive when wound outward)
// Centres the bounding box at the origin and scales the longest side to `size`. Returns the scale applied.
float mesh_normalize(TriMesh* m, float size);
// Vertex clustering down to about `target` vertices (keeps the shape coarsely; for the ARAP target).
TriMesh mesh_simplify(const TriMesh* m, int target);
// Joins vertices closer than eps (OBJ files with split vertices) and drops degenerate triangles.
void mesh_weld(TriMesh* m, float eps);

// An orthographic view: looking along dir, u right, v up, centred at `center`, `half` world units from the centre
// to the frame's edge, res x res pixels.
typedef struct {
    V3 dir, u, v, center;
    float half;
    int res;
} View;
View view_make(V3 dir, V3 center, float half, int res);
// pixel (i, j)'s centre on the view plane through `center`
static inline V3 view_pixel(const View* w, float i, float j) {
    float sx = ((i + 0.5f) / w->res * 2.0f - 1.0f) * w->half, sy = (1.0f - (j + 0.5f) / w->res * 2.0f) * w->half;
    return v3add(w->center, v3add(v3scale(w->u, sx), v3scale(w->v, sy)));
}
// The fixed set of view directions (n of them, spread over the sphere, none opposite another).
void view_directions(int n, V3* out);

// The mesh drawn into a view: coverage (0..1, `ss` x `ss` samples a pixel) and front depth along dir from the view
// plane through center (only where covered).
void mesh_raster(const TriMesh* m, const View* w, int ss, float* cover, float* depth);
#endif
