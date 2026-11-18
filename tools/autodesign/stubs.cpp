// Just enough of util/utils.cpp for the data code, without raylib's library or TBB.
#include "util/utils.hpp"
#include <raymath.h>
#include <cstdarg>
#include <cstdio>
#include <vector>
namespace Bakugan {
    Vector3 Rigid::Apply(Vector3 p) const { return Vector3Add(Vector3RotateByQuaternion(p, rotation), position); }
    Vector3 Rigid::ApplyVector(Vector3 v) const { return Vector3RotateByQuaternion(v, rotation); }
    Vector3 Rigid::ApplyInverse(Vector3 p) const { return Vector3RotateByQuaternion(Vector3Subtract(p, position), { -rotation.x, -rotation.y, -rotation.z, rotation.w }); }
    Rigid Rigid::Then(const Rigid& next) const { return { QuaternionNormalize(QuaternionMultiply(next.rotation, rotation)), next.Apply(position) }; }
    Rigid Rigid::Inverse() const { Quaternion inv = { -rotation.x, -rotation.y, -rotation.z, rotation.w }; return { inv, Vector3Negate(Vector3RotateByQuaternion(position, inv)) }; }
    Rigid Rigid::AboutAxis(Vector3 pivot, Vector3 axis, float angle) {
        Quaternion q = QuaternionFromAxisAngle(Vector3Normalize(axis), angle);
        return { q, Vector3Subtract(pivot, Vector3RotateByQuaternion(pivot, q)) };
    }
    Color Utils::PartColor(int i) { return Color{ (unsigned char)(50 + i * 40), 120, 200, 255 }; }
    Vector3 Utils::AnyPerpendicular(Vector3 v) {
        Vector3 a = fabsf(v.x) < 0.8f ? Vector3{ 1, 0, 0 } : Vector3{ 0, 1, 0 };
        return Vector3Normalize(Vector3CrossProduct(v, a));
    }
    Quaternion Utils::EulerDegreesToQuaternion(Vector3 d) { return QuaternionNormalize(QuaternionFromEuler(d.x * DEG2RAD, d.y * DEG2RAD, d.z * DEG2RAD)); }
    std::string Utils::Format(const char* fmt, ...) {
        va_list a; va_start(a, fmt); char buf[1024]; vsnprintf(buf, sizeof buf, fmt, a); va_end(a); return buf;
    }
}
