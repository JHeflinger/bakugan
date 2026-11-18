#pragma once
#include "data/design.hpp"
#include <climits>
#include <memory>
#include <vector>

namespace Bakugan {
    // The door hinge hardware one body carries for one hinge, in the design pose. The hinge frame has its origin on the
    // pivot and z along the axis; knuckle ranges are z intervals.
    struct HingeHardware {
        int hinge = -1;              // hinge id
        Vector3 pivot = { 0, 0, 0 };
        Vector3 axis = { 0, 0, 1 };
        Vector3 leaf = { 1, 0, 0 };  // leaf direction (perpendicular to the axis)
        Vector3 side = { 0, 1, 0 };  // axis x leaf
        float radius = 3.0f;
        float pinRadius = 1.2f;
        float clearance = 0.4f;
        float leafLength = 8.0f;
        float leafThickness = 2.0f;
        float lo = 0.0f, hi = 0.0f;  // whole knuckle span (a separate pin: shaft and head)
        float headFrom = 0.0f;       // separate pin: its head (a body gets a pocket there)
        float headTo = 0.0f;
        float headRadius = 0.0f;
        std::vector<Knuckle> own;    // knuckles this body carries
        std::vector<Knuckle> other;  // the other body's knuckles (cleared around)
        bool pin = false;            // this body carries the pin (fused, or a separate pin part)
        bool bore = false;           // this body's knuckles are bored for the pin
        bool pinOnly = false;        // a separate pin: nothing but the pin
        Vector3 centre = { 0, 0, 0 };   // bounding sphere of everything above (for skipping far points)
        float reach = 0.0f;
        void Bound();
    };

    // How far past a leaf's end a body's material next to its own knuckles is kept from clearances (Clearance).
    constexpr float KnuckleZone = 1.0f;

    // Signed distance of one printed part (a body with its hinge hardware, or a separate pin) in the design pose.
    // Negative inside. Built once from a design snapshot; evaluation is thread safe.
    // A body's clearances (Design::clearances) are cut from its material before its hinge hardware goes in.
    class BodyField {
    public:
        int body = -1;               // group id (-1 for a pin)
        int pinHinge = -1;           // hinge id when this is a separate pin
        BoundingBox bounds = {};

        bool Empty() const { return m_Root < 0 && m_Hardware.empty(); }
        float Distance(Vector3 p) const;
        Vector3 Gradient(Vector3 p, float h) const;
        // Which primitive the surface point p lies on (smallest |distance|): a shape id, or a hinge id for hardware.
        void Nearest(Vector3 p, int* shape, int* hinge) const;
        // Which primitive the inside point p is deepest in (smallest distance among the material adding ones).
        void Deepest(Vector3 p, int* shape, int* hinge) const;

        static BodyField Body(const Design& d, int bodyId);
        static BodyField Pin(const Design& d, int hingeIndex);
        int Clearances() const { return (int)m_Sweeps.size(); }
        int ClearanceSamples() const;           // what its clearances sample their tools at, all told

        // The same field as GLSL: the body of `float map(vec3 p)` reading its numbers from `uniform vec4 U[]`, which are
        // appended to `params`. The code depends only on the structure (which shapes, groups, ops, knuckles), so
        // editing values only changes the parameters. See ui/raymarch.cpp for the functions it calls.
        std::string Glsl(std::vector<Vector4>* params) const;
    private:
        // What a body is when it is a clearance's tool: the whole body less its hingeOnly clearances; or (hinge) its
        // hardware and its material next to its own knuckles. Either as it was up to an id.
        enum class Kind { Full, Tool, HingeTool };
        static BodyField Make(const Design& d, int bodyId, Kind kind, int upTo);
        // A stretch of a sweep where the tool turns about one line, as this body sees it: a point p of this body is,
        // at sample i of n, at base(c + R(u, angle * i / n)(p - c)) in the tool's frame. n = 0: one pose, base(p).
        struct Segment {
            Rigid base;
            Vector3 c = { 0, 0, 0 }, u = { 0, 0, 1 };
            float angle = 0.0f;
            int n = 0;
            Vector3 centre = { 0, 0, 0 };        // a sphere around the tool all the way, in this body's frame
            float radius = 0.0f;
        };
        struct Sweep {
            std::shared_ptr<const BodyField> tool;
            std::vector<Segment> segments;
            float margin = 0.0f;                 // the clearance's, and half a step
            Rigid rest;                          // this body's frame to the tool's, closed
            Vector3 centre = { 0, 0, 0 };        // around all of them
            float radius = 0.0f;
        };
        float ZoneDistance(Vector3 p) const;     // < 0 next to its own knuckles
        float SweepDistance(float d, Vector3 p) const;
        void GlslInto(std::string& code, std::vector<Vector4>& U, int& vars) const;
        void GlslZone(const std::string& out, std::string& code, std::vector<Vector4>& U) const;
        struct Node {
            int prim = -1;           // index into m_Prims, or -1 for a group
            Op op = Op::Union;
            float blend = 0.0f;
            Vector3 centre = { 0, 0, 0 };   // bounding sphere of the material it adds
            float radius = 0.0f;
            std::vector<int> children;
        };
        int Compile(const Design& d, int groupId, Op op, float blend);
        float Eval(int node, Vector3 p) const;
        float HardwareDistance(float d, Vector3 p) const;
        void GlslNode(int node, const std::string& acc, const std::string& blend, std::string& code, std::vector<Vector4>& U, int& vars) const;
        void GlslHardware(std::string& code, std::vector<Vector4>& U) const;
        static void HardwareParts(const HingeHardware& h, Vector3 p, std::vector<float>* out);
    private:
        std::vector<SuperFrustum> m_Prims;
        std::vector<int> m_PrimShape;          // shape id per primitive
        std::vector<Op> m_PrimOp;              // effective op (inside a cut feature counts as cut)
        std::vector<Node> m_Nodes;
        int m_Root = -1;
        std::vector<HingeHardware> m_Hardware;
        std::vector<Sweep> m_Sweeps;
        bool m_Mask = false;                     // HingeTool: material only next to its own knuckles
        int m_UpTo = INT_MAX;                    // compiling: only shapes and features with ids below this
        int m_LateFrom = INT_MAX;                // compiling: the body's own shapes and features from this id on are late
        bool IsLate(const Design& d, int groupIndex) const;   // a top level feature with nothing older in it
        std::vector<int> m_Late;                 // nodes added to the body after its clearances: they go in after them
    };

    // Same shapes, groups and hinge hardware (angles and timing may differ): the parts would come out the same.
    bool SameGeometry(const Design& a, const Design& b);
}
