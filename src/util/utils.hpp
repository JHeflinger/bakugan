#pragma once
#include "data/skeleton.hpp"
#include "core/app.hpp"

namespace Bakugan {
    namespace Utils {
        float SolveIK(Skeleton& s, int effector, Vector3 target, int chainLength, int iterations = 32);

        bool SameSkeleton(const Skeleton& a, const Skeleton& b);

        Quaternion ClampBallJoint(Quaternion q, float swingLimitDeg, float twistLimitDeg);

        Skeleton MakeHumanoidSkeleton(BoundingBox bounds);

        inline Vector3 MatrixPosition(const Matrix& m) { return { m.m12, m.m13, m.m14 }; }

        const char* GetFragmentShader();

        const char* GetVertexShader();

        bool SameEdit(const EditState& a, const EditState& b);

        bool LoadAsset(const Asset& a, Model* out, std::string* err);

        bool IsModelFile(const char* path);
    }
}
