#pragma once
#include <string>
#include <vector>
#include <raylib.h>

namespace Bakugan {
    enum class GizmoOp {
        Translate,
        Rotate,
        Scale
    };

    // What a click in the design view selects.
    enum class PickMode {
        Shape,
        Feature,
        Body
    };

    // How a shape or a feature group combines with the rest of its group.
    enum class Op {
        Union,
        Cut
    };

    // Door hinge pin: printed in place inside the knuckles, or a separate pin pushed through after printing.
    enum class PinStyle {
        Fused,
        Separate
    };

    // How the reference mesh is drawn in the design view.
    enum class ReferenceMode {
        Ghost,                       // see-through, behind everything
        Solid,                       // opaque: shapes show only where they poke out of it
        SolidXray                    // opaque, with the shapes inside it shown through as ghosts
    };

    // The pose parts are exported in.
    // Which schedule the hinges follow. The timeline runs from open (0) to closed (1) either way; opening plays it
    // from closed back to open, with each hinge's own opening timing when it has one.
    enum class Motion {
        Closing,
        Opening
    };

    enum class ExportPose {
        Design,
        Open,
        Closed
    };

    // Indexed triangle mesh in plain arrays (no GPU state).
    struct TriMesh {
        std::vector<Vector3> positions;
        std::vector<Vector3> normals;
        std::vector<unsigned> indices;
        BoundingBox box = {};
        bool Empty() const { return indices.empty(); }
    };

    // Rigid transform: rotate, then translate.
    struct Rigid {
        Quaternion rotation = { 0, 0, 0, 1 };
        Vector3 position = { 0, 0, 0 };

        Vector3 Apply(Vector3 p) const;
        Vector3 ApplyVector(Vector3 v) const;
        Vector3 ApplyInverse(Vector3 p) const;
        Rigid Then(const Rigid& next) const;      // this first, then next
        Rigid Inverse() const;
        Matrix ToMatrix() const;
        static Rigid AboutAxis(Vector3 pivot, Vector3 axis, float angle);
    };
}
