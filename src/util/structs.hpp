#pragma once
#include <string>
#include <vector>
#include <raylib.h>

namespace Bakugan {
    enum class Primitive {
        None,
        Cube,
        Sphere,
        Cylinder,
        Torus,
        Knot,
        Critter,
        Dragon
    };

    enum class GizmoOp {
        Translate,
        Rotate,
        Scale
    };

    enum class Mode {
        Automatic,
        Manual
    };

    enum class EditTarget {
        Part,
        Hinge,
        Sphere
    };

    struct Asset {
        std::string name;
        std::string path;
        Primitive primitive = Primitive::None;
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
