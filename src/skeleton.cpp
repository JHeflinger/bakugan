#include "skeleton.hpp"
#include <cmath>
#include <cstdio>

int Skeleton::AddJoint(int parent, Vector3 worldPos, const std::string& jointName) {
    Joint j;
    j.id = nextId++;
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

void Skeleton::InsertJoint(int index, const Joint& j) {
    for (Joint& k : joints)
        if (k.parent >= index) k.parent++;
    joints.insert(joints.begin() + index, j);
}

int Skeleton::SplitBone(int child, int segments) {
    if (child <= 0 || child >= (int)joints.size() || segments < 2) return child;
    int parent = joints[child].parent;
    std::vector<Matrix> world = WorldTransforms();
    Vector3 a = MatrixPosition(world[parent]), b = MatrixPosition(world[child]);
    std::string base = joints[child].name;
    int first = child;

    for (int s = 1; s < segments; s++) {
        // New joint goes right before `child` (parents must precede children), with no
        // rotation of its own, so the original child's world transform is unchanged.
        Joint j;
        j.id = nextId++;
        j.parent = parent;
        j.offset = WorldToLocalOffset(parent, Vector3Lerp(a, b, (float)s / segments));
        char suffix[16];
        snprintf(suffix, sizeof(suffix), ".%03d", s);
        j.name = base + suffix;
        InsertJoint(child, j);
        parent = child++;
        joints[child].parent = parent;
        joints[child].offset = WorldToLocalOffset(parent, b);
    }
    return first;
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

float SolveIK(Skeleton& s, int effector, Vector3 target, int chainLength, int iterations) {
    if (effector <= 0 || effector >= (int)s.joints.size()) return 0.0f;
    std::vector<int> chain;   // nearest ancestor first
    for (int j = s.joints[effector].parent; j >= 0; j = s.joints[j].parent) {
        if (chainLength > 0 && (int)chain.size() >= chainLength) break;
        chain.push_back(j);
    }

    float dist = 0.0f;
    for (int it = 0; it < iterations; it++) {
        for (int j : chain) {
            // Rotate joint j so the direction (j -> effector) turns toward (j -> target).
            std::vector<Matrix> world = s.WorldTransforms();
            Vector3 pj = MatrixPosition(world[j]), pe = MatrixPosition(world[effector]);
            Vector3 toEnd = Vector3Subtract(pe, pj), toTarget = Vector3Subtract(target, pj);
            if (Vector3Length(toEnd) < 1e-6f || Vector3Length(toTarget) < 1e-6f) continue;
            Quaternion delta = QuaternionFromVector3ToVector3(Vector3Normalize(toEnd), Vector3Normalize(toTarget));
            // world delta -> local: local' = parent^-1 * delta * parent * local
            Quaternion parentRot = QuaternionFromMatrix(s.ParentWorld(j, world));
            Quaternion localDelta = QuaternionMultiply(QuaternionInvert(parentRot), QuaternionMultiply(delta, parentRot));
            Joint& jt = s.joints[j];
            jt.rotation = ClampBallJoint(QuaternionMultiply(localDelta, jt.rotation), jt.swingLimit, jt.twistLimit);
        }
        dist = Vector3Distance(MatrixPosition(s.WorldTransforms()[effector]), target);
        if (dist < 1e-4f) break;
    }
    return dist;
}

bool SameSkeleton(const Skeleton& a, const Skeleton& b) {
    if (a.joints.size() != b.joints.size() || a.visible != b.visible || a.name != b.name ||
        a.color.r != b.color.r || a.color.g != b.color.g || a.color.b != b.color.b)
        return false;
    for (size_t i = 0; i < a.joints.size(); i++) {
        const Joint &x = a.joints[i], &y = b.joints[i];
        if (x.id != y.id || x.name != y.name || x.parent != y.parent ||
            !Vector3Equals(x.offset, y.offset) || !QuaternionEquals(x.rotation, y.rotation) ||
            x.swingLimit != y.swingLimit || x.twistLimit != y.twistLimit)
            return false;
    }
    return true;
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
