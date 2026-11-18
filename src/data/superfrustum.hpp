#pragma once
#include "util/structs.hpp"

namespace Bakugan {
    // A SuperFrustum (Ganeshan et al., "SuperFit") style primitive: a rounded-rectangle cross section swept along the
    // local Z axis, with taper, an arc bend ("bulge"), a hollow shell ("onion") and a uniform dilation.
    //   size      full extents (x, y: cross section, z: height along the axis)
    //   rounding  0 = sharp rectangle, 1 = fully rounded (ellipse-like / circle) cross section
    //   taper     1 = prism, 0 = the top cross section shrinks to a ridge or point (cone / pyramid / wedge)
    //   bulge     -1..1 bends the axis into an arc in the local XZ plane (1 = half circle)
    //   dilation  rounds every edge by growing the solid outward (world units)
    //   onion     0 = solid, otherwise a shell whose wall is (1 - onion) of the inner half width
    // The signed distance is evaluated in the primitive's local frame (position + rotation).
    struct SuperFrustum {
        Vector3 position = { 0, 0, 0 };
        Quaternion rotation = { 0, 0, 0, 1 };
        Vector3 size = { 0.5f, 0.5f, 0.5f };
        float rounding = 0.0f;
        float taper = 1.0f;
        float bulge = 0.0f;
        float dilation = 0.0f;
        float onion = 0.0f;

        static constexpr int ParamCount = 11;   // everything but the rotation (handled as a delta)

        float LocalDistance(Vector3 local) const;
        float Distance(Vector3 world) const { return LocalDistance(ToLocal(world)); }
        Vector3 Gradient(Vector3 world, float h) const;
        Vector3 ToLocal(Vector3 world) const;
        Vector3 ToWorld(Vector3 local) const;
        Vector3 Axis(int i) const;                // world direction of local axis i
        Rigid Pose() const { return { rotation, position }; }
        void SetPose(const Rigid& r) { rotation = r.rotation; position = r.position; }
        float BoundingRadius() const;
        BoundingBox LocalBounds() const;
        BoundingBox WorldBounds() const;
        float Sagitta() const;
        void Repair(float minSize);
        void Pack(float* p) const;
        void Unpack(const float* p);
        bool SameShape(const SuperFrustum& o) const;   // ignores the pose
        bool operator==(const SuperFrustum& o) const;
        bool operator!=(const SuperFrustum& o) const { return !(*this == o); }
    };
}
