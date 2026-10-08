#include "skeleton.hpp"
#include <cmath>

int Skeleton::AddJoint(int parent, Vector3 worldPos, const std::string& jointName) {
    Joint j;
    j.parent = parent;
    j.offset = WorldToLocalOffset(parent, worldPos);
    j.name = jointName.empty() ? "joint_" + std::to_string(joints.size()) : jointName;
    joints.push_back(j);
    return (int)joints.size() - 1;
}

void Skeleton::RemoveJoint(int index) {
    if (index < 0 || index >= (int)joints.size()) return;
    std::vector<bool> doomed(joints.size(), false);
    doomed[index] = true;
    for (size_t i = index + 1; i < joints.size(); i++)
        if (joints[i].parent >= 0 && doomed[joints[i].parent]) doomed[i] = true;

    std::vector<int> remap(joints.size(), -1);
    std::vector<Joint> kept;
    for (size_t i = 0; i < joints.size(); i++) {
        if (doomed[i]) continue;
        remap[i] = (int)kept.size();
        kept.push_back(joints[i]);
    }
    for (Joint& j : kept)
        if (j.parent >= 0) j.parent = remap[j.parent];
    joints = std::move(kept);
}

std::vector<Matrix> Skeleton::WorldTransforms() const {
    std::vector<Matrix> world(joints.size());
    Matrix root = MatrixTranslate(origin.x, origin.y, origin.z);
    for (size_t i = 0; i < joints.size(); i++) {
        const Joint& j = joints[i];
        // raymath: MatrixMultiply(a, b) applies a first, then b
        Matrix local = MatrixMultiply(QuaternionToMatrix(j.rotation),
                                      MatrixTranslate(j.offset.x, j.offset.y, j.offset.z));
        world[i] = MatrixMultiply(local, j.parent >= 0 ? world[j.parent] : root);
    }
    return world;
}

Matrix Skeleton::ParentWorld(int index, const std::vector<Matrix>& world) const {
    int p = joints[index].parent;
    return p >= 0 ? world[p] : MatrixTranslate(origin.x, origin.y, origin.z);
}

Vector3 Skeleton::WorldToLocalOffset(int parent, Vector3 worldPos) const {
    if (parent < 0) return Vector3Subtract(worldPos, origin);
    return Vector3Transform(worldPos, MatrixInvert(WorldTransforms()[parent]));
}

void Skeleton::ResetPose() {
    for (Joint& j : joints) j.rotation = QuaternionIdentity();
}

Quaternion ClampBallJoint(Quaternion q, float swingLimitDeg, float twistLimitDeg) {
    q = QuaternionNormalize(q);
    if (q.w < 0) q = { -q.x, -q.y, -q.z, -q.w };

    // Swing-twist decomposition about local +Y: q = swing * twist
    Quaternion twist = { 0, q.y, 0, q.w };
    float tlen = sqrtf(twist.y * twist.y + twist.w * twist.w);
    twist = tlen < 1e-6f ? QuaternionIdentity() : Quaternion{ 0, twist.y / tlen, 0, twist.w / tlen };
    Quaternion swing = QuaternionMultiply(q, QuaternionInvert(twist));

    float twistAngle = 2.0f * atan2f(twist.y, twist.w);
    if (twistAngle > PI) twistAngle -= 2 * PI;
    if (twistAngle < -PI) twistAngle += 2 * PI;
    float twistMax = twistLimitDeg * DEG2RAD;
    twistAngle = Clamp(twistAngle, -twistMax, twistMax);
    twist = QuaternionFromAxisAngle({ 0, 1, 0 }, twistAngle);

    if (swing.w < 0) swing = { -swing.x, -swing.y, -swing.z, -swing.w };
    float swingAngle = 2.0f * acosf(Clamp(swing.w, -1.0f, 1.0f));
    float swingMax = swingLimitDeg * DEG2RAD;
    if (swingAngle > swingMax) {
        Vector3 axis = { swing.x, swing.y, swing.z };
        if (Vector3Length(axis) > 1e-6f)
            swing = QuaternionFromAxisAngle(Vector3Normalize(axis), swingMax);
    }
    return QuaternionNormalize(QuaternionMultiply(swing, twist));
}

Skeleton MakeHumanoidSkeleton(BoundingBox b) {
    float h = b.max.y - b.min.y;
    if (h <= 0) h = 2.0f;
    Vector3 base = { (b.min.x + b.max.x) * 0.5f, b.min.y, (b.min.z + b.max.z) * 0.5f };
    auto P = [&](float x, float y) { return Vector3{ base.x + x * h, base.y + y * h, base.z }; };

    Skeleton s;
    s.name = "Humanoid";
    int hips  = s.AddJoint(-1, P(0, 0.50f), "hips");
    int spine = s.AddJoint(hips, P(0, 0.60f), "spine");
    int chest = s.AddJoint(spine, P(0, 0.72f), "chest");
    int neck  = s.AddJoint(chest, P(0, 0.85f), "neck");
    s.AddJoint(neck, P(0, 0.93f), "head");
    for (int side = -1; side <= 1; side += 2) {
        const char* sfx = side < 0 ? "_R" : "_L";
        float x = (float)side;
        int sh = s.AddJoint(chest, P(0.10f * x, 0.82f), std::string("shoulder") + sfx);
        int el = s.AddJoint(sh, P(0.25f * x, 0.82f), std::string("elbow") + sfx);
        s.AddJoint(el, P(0.40f * x, 0.82f), std::string("wrist") + sfx);
        int hp = s.AddJoint(hips, P(0.07f * x, 0.48f), std::string("hip") + sfx);
        int kn = s.AddJoint(hp, P(0.07f * x, 0.27f), std::string("knee") + sfx);
        s.AddJoint(kn, P(0.07f * x, 0.04f), std::string("ankle") + sfx);
    }
    return s;
}
