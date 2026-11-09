#pragma once
// DINOv3 as a loss: runs the graph tools/autodesign/export_dinov3.py writes (one view and its target in; their loss,
// and its gradient with respect to every pixel of both, out) with ONNX Runtime.
//
// ONNX Runtime is loaded when first needed, from a shared library (onnxruntime.dll / libonnxruntime.so /
// libonnxruntime.dylib, from its release downloads or a pip install of onnxruntime), so the editor builds and runs
// without it: only the semantic loss needs it. Its C API is the only thing used (vendor/onnxruntime).
#include <memory>
#include <string>
#include <vector>

namespace Bakugan {
    class Dino {
    public:
        // Loads the library (empty: the platform's default name, found the usual way) and the model. Check Ok().
        Dino(const std::string& library, const std::string& model, bool gpu);
        ~Dino();
        Dino(const Dino&) = delete;
        Dino& operator=(const Dino&) = delete;

        bool Ok() const { return m_Error.empty(); }
        const std::string& Error() const { return m_Error; }
        const std::string& Device() const { return m_Device; }    // "CPU", or the GPU it runs on
        int Size() const { return m_Size; }                        // the views' size, in pixels (they're square)

        // One view: render and target are Size() x Size() grey pixels (0..1, row by row). Gives their loss, and
        // d loss / d each pixel of both (either may be null when not needed). patchWeight: 0 compares the global
        // embeddings, 1 the patch embeddings. False (and Error() says why) if it couldn't run.
        bool Run(const float* render, const float* target, float patchWeight, float* loss, float* gradRender, float* gradTarget);

    private:
        struct Runtime;
        std::unique_ptr<Runtime> m_R;
        std::string m_Error, m_Device = "CPU";
        int m_Size = 0;
    };
}
