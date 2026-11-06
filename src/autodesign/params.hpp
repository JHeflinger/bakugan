#pragma once
// The design's differentiable parameters: what Adam optimizes.
//
// Every shape has position (3), rotation (a quaternion, 4), size (3), rounding, taper, bulge and dilation; every hinge
// has pivot (3), axis (3), its open and closed angles, and when in the closing motion it turns (t0..t1). Each parameter has a unit (the size of a typical step, so
// one learning rate suits them all) and a key that names it for as long as its shape or hinge exists.
//
// To make another number differentiable: a row in the table for its kind in params.cpp, and a case in Ref.
#include "data/design.hpp"
#include <cstdint>
#include <vector>

namespace Bakugan {
    struct Param {
        enum Kind : unsigned char { Shape, Hinge } kind;
        int id;                  // shape or hinge id
        int field;               // which of its numbers (a row of its table in params.cpp)
        float unit;              // a typical step, in the number's own units (mm, radians, ...)
        uint64_t Key() const { return ((uint64_t)kind << 56) | ((uint64_t)(uint32_t)id << 8) | (uint64_t)field; }
    };

    // The parameters of a design, in a fixed order. L is the mesh's size (units of lengths scale with it).
    std::vector<Param> DesignParams(const Design& d, float L);
    // The number a parameter stands for (nullptr when its shape or hinge is gone).
    float* Ref(Design& d, const Param& p);
    // Sets it, kept inside its range (sizes stay positive, rounding within 0..1, ...).
    void SetParam(Design& d, const Param& p, float value, float L);
    // After changing parameters: rotations and axes back to unit length, hinges' timing windows kept in order.
    void Tidy(Design& d);
}
