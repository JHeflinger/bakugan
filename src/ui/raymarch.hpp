#pragma once
#include "util/structs.hpp"
#include "data/field.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace Bakugan {
    // GLSL sources for the signed distance renderer. The library mirrors the C++ field (superfrustum.cpp, field.cpp)
    // line for line; BodyField::Glsl writes the `map` that calls it.
    namespace Sdf {
        // A whole fragment shader: uniforms, the library, `float map(vec3 p) { body }` and the raymarching main.
        std::string FragmentSource(const std::string& mapBody, int uniforms);
        const char* VertexSource();
        // Same map, but each pixel evaluates it at a point read from a float texture (for checking against the CPU).
        std::string ProbeFragmentSource(const std::string& mapBody, int uniforms);
        const char* ProbeVertexSource();
        // One superfrustum on its own (the design view's shapes).
        std::string ShapeMap();
        void ShapeParams(const SuperFrustum& f, std::vector<Vector4>* U);
    }

    // Where the camera is, for the raymarcher.
    struct SdfView {
        Camera3D camera = {};
        bool ortho = false;
        float pixel = 0.001f;        // perspective: radians per pixel; ortho: world units per pixel
    };

    // Draws bodies and shapes as exact signed distance fields: a box around each is rasterised (back faces, so it works
    // from inside too) and every pixel marches its ray through the field, writing depth, so the result sits correctly
    // among ordinary geometry. A body's shader is generated from its field and recompiled only when its structure
    // changes; its numbers are uniforms, so edits and motion cost nothing but an upload.
    class SdfRenderer {
    public:
        bool Init();
        void Unload();
        bool Ok() const { return m_Ok; }
        // (Re)generates the program in `slot` from a field. False if its shader didn't compile.
        bool SetBody(const std::string& slot, const BodyField& f);
        bool HasBody(const std::string& slot) const;
        void DrawBody(const std::string& slot, const Rigid& xf, Color color, float highlight, const SdfView& view);
        void DrawShape(const SuperFrustum& f, const Rigid& xf, Color color, float highlight, const SdfView& view);
        // Drops the bodies not listed.
        void Retain(const std::vector<std::string>& slots);
        int Compiles() const { return m_Compiles; }
    private:
        struct Program {
            std::string code;
            Shader shader = {};
            bool ok = false;
            int locU = -1, locCam = -1, locDir = -1, locOrtho = -1, locMin = -1, locMax = -1, locRot = -1, locColor = -1,
                locHighlight = -1, locPixel = -1;
            std::vector<Vector4> params;
            BoundingBox box = {};
        };
        bool Compile(Program& p, const std::string& code, int uniforms);
        void Draw(Program& p, const Rigid& xf, Color color, float highlight, const SdfView& view);
    private:
        bool m_Ok = false;
        Mesh m_Cube = {};
        Material m_Material = {};
        Program m_Shape;
        std::unordered_map<std::string, Program> m_Bodies;
        int m_Compiles = 0;
    };
}
