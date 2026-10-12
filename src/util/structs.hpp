#pragma once
#include <string>
#include <raylib.h>

namespace Bakugan {
    enum class Primitive {
        None,
        Cube,
        Sphere,
        Cylinder,
        Torus,
        Knot
    };

    enum class GizmoOp {
        Translate,
        Rotate
    };

    struct Asset {
        std::string name;
        std::string path;
        Primitive primitive = Primitive::None;
    };

    struct Joint { // Ball joint, TODO:different types of joints
        int id = -1;
        std::string name;
        int parent = -1;
        Vector3 offset = { 0, 0, 0 };
        Quaternion rotation = { 0, 0, 0, 1 };
        float swingLimit = 180.0f;
        float twistLimit = 180.0f;
    };
}
