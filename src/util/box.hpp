#pragma once
// Axis aligned box helpers shared by the field, part building, collision and generator code.
#include "util/structs.hpp"
#include <raymath.h>

namespace Bakugan {
    inline Vector3 BoxCorner(const BoundingBox& b, int k) {          // k = 0..7
        return { (k & 1) ? b.max.x : b.min.x, (k & 2) ? b.max.y : b.min.y, (k & 4) ? b.max.z : b.min.z };
    }
    inline BoundingBox Grow(BoundingBox b, float m) { return { Vector3SubtractValue(b.min, m), Vector3AddValue(b.max, m) }; }
    inline BoundingBox Merge(BoundingBox a, BoundingBox b) { return { Vector3Min(a.min, b.min), Vector3Max(a.max, b.max) }; }
    inline bool Overlap(BoundingBox a, BoundingBox b) {
        return a.min.x <= b.max.x && a.max.x >= b.min.x && a.min.y <= b.max.y && a.max.y >= b.min.y && a.min.z <= b.max.z && a.max.z >= b.min.z;
    }
    inline bool InBox(Vector3 p, BoundingBox b) {
        return p.x >= b.min.x && p.x <= b.max.x && p.y >= b.min.y && p.y <= b.max.y && p.z >= b.min.z && p.z <= b.max.z;
    }
    // The box around a box moved by r.
    inline BoundingBox TransformBox(BoundingBox b, const Rigid& r) {
        Vector3 lo = { 1e30f, 1e30f, 1e30f }, hi = { -1e30f, -1e30f, -1e30f };
        for (int i = 0; i < 8; i++) {
            Vector3 w = r.Apply(BoxCorner(b, i));
            lo = Vector3Min(lo, w);
            hi = Vector3Max(hi, w);
        }
        return { lo, hi };
    }
}
