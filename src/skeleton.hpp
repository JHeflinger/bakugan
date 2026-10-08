#pragma once
#include "raylib.h"
#include "raymath.h"
#include <string>
#include <vector>

// A ball joint: 3 rotational degrees of freedom, constrained by a swing cone
// around the joint's local +Y axis and a twist range about that axis.
struct Joint {
    int id = -1;                                  // stable across inserts/removes (indices are not)
    std::string name;
    int parent = -1;                              // index into Skeleton::joints, -1 = root
    Vector3 offset = { 0, 0, 0 };                 // translation from parent joint, in parent's rotated frame
    Quaternion rotation = { 0, 0, 0, 1 };         // local ball-joint rotation (rotates the subtree)
    float swingLimit = 180.0f;                    // cone half-angle in degrees
    float twistLimit = 180.0f;                    // +/- degrees about local +Y
};

struct Skeleton {
    std::string name;
    Vector3 origin = { 0, 0, 0 };
    Color color = ORANGE;
    bool visible = true;
    std::vector<Joint> joints;                    // parents always precede their children
    int nextId = 0;

    int AddJoint(int parent, Vector3 worldPos, const std::string& name = "");
    void RemoveJoint(int index);                  // removes the joint and its whole subtree
    void InsertJoint(int index, const Joint& j);  // shifts later joints and fixes parent indices
    // Splits the bone parent->child into `segments` equal bones by inserting joints along it.
    // Nothing moves or rotates. Returns the first inserted joint (or child if it can't split).
    int SplitBone(int child, int segments);
    std::vector<Matrix> WorldTransforms() const;
    Matrix ParentWorld(int index, const std::vector<Matrix>& world) const;
    Vector3 WorldToLocalOffset(int parent, Vector3 worldPos) const;
    void ResetPose();
};

// Inverse kinematics (CCD): rotates the ancestors of `effector` so it reaches toward `target`
// (skeleton space). Bone lengths never change, and ball-joint limits are respected.
// chainLength = how many ancestors may rotate (0 = all the way to the root).
// Returns the remaining distance to the target.
float SolveIK(Skeleton& s, int effector, Vector3 target, int chainLength, int iterations = 32);

bool SameSkeleton(const Skeleton& a, const Skeleton& b);   // structure, rest pose and limits

// Clamp a local rotation to a ball joint's swing cone / twist limits.
Quaternion ClampBallJoint(Quaternion q, float swingLimitDeg, float twistLimitDeg);

// Simple humanoid rig sized to fit the given bounds.
Skeleton MakeHumanoidSkeleton(BoundingBox bounds);

inline Vector3 MatrixPosition(const Matrix& m) { return { m.m12, m.m13, m.m14 }; }
