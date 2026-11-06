#pragma once
// Software drawing for the test suite's video (it runs on a worker thread, away from the GPU): a canvas with text,
// designs meshed from their fields, and a small depth tested rasterizer with outlines.
#include "data/design.hpp"
#include <string>
#include <vector>

namespace Bakugan::Suite {
    struct Rgb {
        unsigned char r = 255, g = 255, b = 255;
    };
    Rgb Mix(Rgb a, Rgb b, float t);
    const Rgb White = { 255, 255, 255 };
    const Rgb Ink = { 40, 42, 48 };
    const Rgb Muted = { 120, 124, 134 };
    const Rgb Border = { 208, 211, 218 };
    const Rgb Accent = { 70, 140, 230 };
    const Rgb Track = { 232, 235, 241 };

    // The video frame, with raylib's default font (10 pixels high, times `scale`).
    struct Canvas {
        int w = 0, h = 0;
        std::vector<Rgb> px;
        void Resize(int width, int height) { w = width; h = height; px.assign((size_t)w * h, White); }
        void Fill(int x0, int y0, int x1, int y1, Rgb c);
        void Frame(int x0, int y0, int x1, int y1, int t, Rgb c);          // a border t pixels thick
        void Text(int x, int y, int scale, const std::string& s, Rgb c);
        void Rgb24(std::vector<unsigned char>* out) const;                  // what ffmpeg is fed
    };
    int TextWidth(const std::string& s, int scale);
    std::string FitText(std::string s, int width, int scale);              // cut to fit, with an ellipsis

    // A design's bodies as triangle meshes, in the design pose (moved per frame by their transforms).
    struct BodyMesh {
        int group = -1;              // group index (its transform)
        Rgb color;
        std::vector<Vector3> pos, nrm;
        std::vector<unsigned> idx;
    };
    using DesignMesh = std::vector<BodyMesh>;
    DesignMesh MeshDesign(const Design& d, float voxel);

    // A fixed orthographic camera, looking down at the model from the front right, framing the given points.
    struct View {
        Vector3 centre = { 0, 0, 0 }, right = { 1, 0, 0 }, up = { 0, 1, 0 }, fwd = { 0, 0, -1 };
        Vector3 light = { 0, 1, 0 };
        float scale = 1.0f;          // pixels per mm
        float ox = 0.0f, oy = 0.0f;  // pixel position of the centre
        Vector3 Project(Vector3 p) const;
        static View Framing(const std::vector<Vector3>& points, int w, int h);
    };

    // One picture, depth tested; ids mark which body covers a pixel (0: none), for the outlines.
    struct Raster {
        int w = 0, h = 0;
        std::vector<Rgb> color;
        std::vector<float> depth;
        std::vector<unsigned char> id;
        void Clear(int width, int height, const std::vector<Rgb>& background);
        void Triangle(Vector3 a, Vector3 b, Vector3 c, Rgb ca, Rgb cb, Rgb cc, unsigned char id);
        bool Edge(size_t i) const { return id[i] != id[i + 1] || id[i] != id[i + w]; }
    };
    // The design in a pose, lit, with ink outlines where the body under a pixel changes or the depth jumps.
    void DrawDesign(Raster& r, const DesignMesh& m, const std::vector<Rigid>& xf, const View& v);
    // A mesh as a faint silhouette with a light outline, on white: the input behind every frame.
    std::vector<Rgb> Ghost(const TriMesh& m, const View& v, int w, int h);
}
