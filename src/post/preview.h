#ifndef FB_PREVIEW_H
#define FB_PREVIEW_H
// Software pictures of a design (for the CLI's progress and results, no window needed), and a PNG writer.
#include "core/design.h"

typedef struct {
    int w, h;
    unsigned char* rgb;
} FbImage;

FbImage image_create(int w, int h, unsigned char grey);
void image_free(FbImage* im);
bool png_write(const char* path, const FbImage* im);
void image_text(FbImage* im, int x, int y, int scale, const char* s, unsigned char r, unsigned char g, unsigned char b);

// The body meshes of a design (in their own frames), for drawing it in any pose.
typedef struct {
    int n;
    TriMesh* bodies;
} DesignMeshes;
DesignMeshes design_meshes(const Design* d, int cellsAcross);
void design_meshes_free(DesignMeshes* m);
V3 body_color(int index);

// A sheet: the target, the design open, a third and two thirds closed, closed (two view angles, one per row).
FbImage preview_sheet(const Design* d, const DesignMeshes* m, int cell);
// One square tile into im at (x0, y0): the design at opening s (1 open), from a 3/4 view turned by yaw (radians),
// framed around it (and, with ghost, the target, faintly), on a dark background.
void preview_tile(FbImage* im, int x0, int y0, int size, const Design* d, const DesignMeshes* m, float s, float yaw, bool ghost);
#endif
