#pragma once
// Drawing the silhouettes somewhere else than on this thread's cores: the GPU (ui/gputrace.cpp implements this with
// the editor's OpenGL context). The loss hands over every piece's field as GLSL and gets back, per pixel of every view,
// the piece that covers it most, how near its ray comes to it and where: exactly what Loss::Trace finds on the CPU.
// Or (everyPiece, for the semantic pictures) the same for every piece, and where the ray enters it.
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
        bool everyPiece = false;         // every piece's own results, not just the one covering a pixel most
    };

    // Per pixel (views x res x res; rows from the top, as the loss keeps them). With everyPiece, per piece and pixel
    // (pieces x views x res x res), and `piece` is the piece itself where it covers the pixel at all, else -1.
    struct TraceResult {
        std::vector<float> nearest;      // the least distance its ray comes to the piece that covers it most (< 0: inside)
        std::vector<float> along;        // ... and how far along the ray that is (from the ray's start: see Loss::Trace)
        std::vector<int> piece;          // that piece (index), or -1 when none covers the pixel at all
        std::vector<float> enter;        // everyPiece: how far along the ray it enters the piece (-1: it doesn't)
    };

    // Weighted sums of fields at points: what the gradient measures (Loss::Gradient). Each job is one field (one of
    // `codes`, with its own numbers), placed by xf, at points [start, start + count) of `points` (world xyz; w: the
    // point's weight); its answer is sum over them of w · field(xf⁻¹ · point).
    struct EvalJob {
        int code = 0;
        std::vector<Vector4> params;
        Rigid xf;
        int start = 0, count = 0;
    };
    struct EvalRequest {
        std::vector<std::string> codes;  // bodies of `float map(vec3 p)` (BodyField::Glsl)
        std::vector<Vector4> points;
        std::vector<EvalJob> jobs;
    };

    class SilhouetteTracer {
    public:
        virtual ~SilhouetteTracer() = default;
        // Called from the search's thread; false when it couldn't be drawn (the caller draws it on the CPU instead).
        virtual bool Trace(const TraceRequest& request, TraceResult* result) = 0;
        // The same for fields at points: false (the caller evaluates them itself) when it can't.
        virtual bool Evaluate(const EvalRequest& request, std::vector<double>* sums) { (void)request; (void)sums; return false; }
        virtual std::string Name() const = 0;
    };
}
