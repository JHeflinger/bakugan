#pragma once
// Drawing the silhouettes somewhere else than on this thread's cores: the GPU (ui/gputrace.cpp implements this with
// the editor's OpenGL context). The loss hands over every piece's field as GLSL and gets back, per pixel of every view,
// the piece that covers it most, how near its ray comes to it and where: exactly what Loss::Trace finds on the CPU.
#include "util/structs.hpp"
#include <string>
#include <vector>

namespace Bakugan {
    struct TracePiece {
        std::string code;                // the body of `float map(vec3 p)` (BodyField::Glsl)
        std::vector<Vector4> params;     // its numbers (uniform vec4 U[])
        BoundingBox box = {};            // design pose
        Rigid xf;                        // where it is
    };

    struct TraceView {
        Vector3 dir = { 0, 0, -1 }, u = { 1, 0, 0 }, v = { 0, 1, 0 };   // looking along dir; u right and v up
    };

    struct TraceRequest {
        std::vector<TracePiece> pieces;
        std::vector<TraceView> views;
        Vector3 centre = { 0, 0, 0 };    // of the frame every view shows
        float half = 1.0f;               // the frame is 2 half across
        int res = 64;                    // pixels across a view
        float tau = 1.0f;                // a pixel, in mm
    };

    // Per pixel (views x res x res; rows from the top, as the loss keeps them):
    struct TraceResult {
        std::vector<float> nearest;      // the least distance its ray comes to the piece that covers it most (< 0: inside)
        std::vector<float> along;        // ... and how far along the ray that is (from the ray's start: see Loss::Trace)
        std::vector<int> piece;          // that piece (index), or -1 when none covers the pixel at all
    };

    class SilhouetteTracer {
    public:
        virtual ~SilhouetteTracer() = default;
        // Called from the search's thread; false when it couldn't be drawn (the caller draws it on the CPU instead).
        virtual bool Trace(const TraceRequest& request, TraceResult* result) = 0;
        virtual std::string Name() const = 0;
    };
}
