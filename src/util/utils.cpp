#include "utils.hpp"
#include <raymath.h>

namespace Bakugan {
    float Utils::SolveIK(Skeleton& s, int effector, Vector3 target, int chainLength, int iterations) {
        if (effector <= 0 || effector >= (int)s.Joints().size()) return 0.0f;
        std::vector<int> chain;
        for (int j = s.Joints()[effector].parent; j >= 0; j = s.Joints()[j].parent) {
            if (chainLength > 0 && (int)chain.size() >= chainLength) break;
            chain.push_back(j);
        }
        float dist = 0.0f;
        for (int it = 0; it < iterations; it++) {
            for (int j : chain) {
                std::vector<Matrix> world = s.WorldTransforms();
                Vector3 pj = MatrixPosition(world[j]), pe = MatrixPosition(world[effector]);
                Vector3 toEnd = Vector3Subtract(pe, pj), toTarget = Vector3Subtract(target, pj);
                if (Vector3Length(toEnd) < 1e-6f || Vector3Length(toTarget) < 1e-6f) continue;
                Quaternion delta = QuaternionFromVector3ToVector3(Vector3Normalize(toEnd), Vector3Normalize(toTarget));
                Quaternion parentRot = QuaternionFromMatrix(s.ParentWorld(j, world));
                Quaternion localDelta = QuaternionMultiply(QuaternionInvert(parentRot), QuaternionMultiply(delta, parentRot));
                Joint& jt = s.JointsRef()[j];
                jt.rotation = ClampBallJoint(QuaternionMultiply(localDelta, jt.rotation), jt.swingLimit, jt.twistLimit);
            }
            dist = Vector3Distance(MatrixPosition(s.WorldTransforms()[effector]), target);
            if (dist < 1e-4f) break;
        }
        return dist;
    }

    bool Utils::SameSkeleton(const Skeleton& a, const Skeleton& b) {
        if (a.Joints().size() != b.Joints().size() || a.IsVisible() != b.IsVisible() || a.GetName() != b.GetName() ||
            a.GetColor().r != b.GetColor().r || a.GetColor().g != b.GetColor().g || a.GetColor().b != b.GetColor().b)
            return false;
        for (size_t i = 0; i < a.Joints().size(); i++) {
            const Joint &x = a.Joints()[i], &y = b.Joints()[i];
            if (x.id != y.id || x.name != y.name || x.parent != y.parent ||
                !Vector3Equals(x.offset, y.offset) || !QuaternionEquals(x.rotation, y.rotation) ||
                x.swingLimit != y.swingLimit || x.twistLimit != y.twistLimit)
                return false;
        }
        return true;
    }

    Quaternion Utils::ClampBallJoint(Quaternion q, float swingLimitDeg, float twistLimitDeg) {
        q = QuaternionNormalize(q);
        if (q.w < 0) q = { -q.x, -q.y, -q.z, -q.w };
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

    Skeleton Utils::MakeHumanoidSkeleton(BoundingBox b) {
        float h = b.max.y - b.min.y;
        if (h <= 0) h = 2.0f;
        Vector3 base = { (b.min.x + b.max.x) * 0.5f, b.min.y, (b.min.z + b.max.z) * 0.5f };
        auto P = [&](float x, float y) { return Vector3{ base.x + x * h, base.y + y * h, base.z }; };
        Skeleton s;
        s.SetName("Humanoid");
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

    const char* Utils::GetFragmentShader() {
        return R"(#version 330
            in vec3 vertexPosition;
            in vec2 vertexTexCoord;
            in vec3 vertexNormal;
            in vec4 vertexColor;
            uniform mat4 mvp;
            uniform mat4 matModel;
            uniform mat4 matNormal;
            out vec3 fragPos;
            out vec2 fragTexCoord;
            out vec3 fragNormal;
            out vec4 fragColor;
            void main() {
                fragPos = vec3(matModel * vec4(vertexPosition, 1.0));
                fragTexCoord = vertexTexCoord;
                fragColor = vertexColor;
                fragNormal = vec3(matNormal * vec4(vertexNormal, 0.0));
                gl_Position = mvp * vec4(vertexPosition, 1.0);
            })";
    }

    const char* Utils::GetVertexShader() {
        return R"(#version 330
            in vec3 fragPos;
            in vec2 fragTexCoord;
            in vec3 fragNormal;
            in vec4 fragColor;
            uniform sampler2D texture0;
            uniform vec4 colDiffuse;
            uniform vec3 viewPos;
            out vec4 finalColor;
            void main() {
                vec4 base = texture(texture0, fragTexCoord) * colDiffuse * fragColor;
                float light = 1.0;
                if (length(fragNormal) > 0.01) {
                    vec3 n = normalize(fragNormal);
                    if (dot(n, viewPos - fragPos) < 0.0) n = -n;   // two-sided
                    vec3 key = normalize(vec3(0.4, 1.0, 0.6));
                    light = 0.35 + 0.65 * max(dot(n, key), 0.0);
                }
                finalColor = vec4(base.rgb * light, base.a);
            })";
    }

    bool Utils::SameEdit(const EditState& a, const EditState& b) {
        if (!SameSkeleton(a.skeleton, b.skeleton) || !Vector3Equals(a.offset, b.offset) ||
            a.pose.size() != b.pose.size())
            return false;
        for (const auto& [id, q] : a.pose) {
            auto it = b.pose.find(id);
            if (it == b.pose.end() || !QuaternionEquals(q, it->second)) return false;
        }
        return true;
    }

    bool Utils::IsModelFile(const char* path) {
        return IsFileExtension(path, ".obj;.gltf;.glb;.iqm;.vox;.m3d");
    }

    bool Utils::LoadAsset(const Asset& a, Model* out, std::string* err) {
        Mesh mesh = {};
        switch (a.primitive) {
            case Primitive::Cube:     mesh = GenMeshCube(1, 1, 1); break;
            case Primitive::Sphere:   mesh = GenMeshSphere(0.5f, 24, 24); break;
            case Primitive::Cylinder: mesh = GenMeshCylinder(0.3f, 1.5f, 24); break;
            case Primitive::Torus:    mesh = GenMeshTorus(0.3f, 1.0f, 24, 32); break;
            case Primitive::Knot:     mesh = GenMeshKnot(0.5f, 1.5f, 64, 128); break;
            case Primitive::None: {
                if (!FileExists(a.path.c_str())) { *err = "File not found: " + a.path; return false; }
                Model m = LoadModel(a.path.c_str());
                if (m.meshCount == 0 || !IsModelValid(m)) {
                    UnloadModel(m);
                    *err = "Could not load " + a.name;
                    return false;
                }
                *out = m;
                return true;
            }
        }
        *out = LoadModelFromMesh(mesh);
        return true;
    }
}
