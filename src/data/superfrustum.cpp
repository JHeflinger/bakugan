#include "superfrustum.hpp"
#include <raymath.h>
#include <algorithm>
#include <cmath>

namespace Bakugan {
    namespace {
        inline float Cross2(float ax, float ay, float bx, float by) { return ax * by - ay * bx; }

        // Squared distance from p to segment ab, and the side of p (positive = left of a->b).
        inline void Segment(float px, float py, float ax, float ay, float bx, float by, float* d2, float* side) {
            float ex = bx - ax, ey = by - ay, wx = px - ax, wy = py - ay;
            float t = std::clamp((wx * ex + wy * ey) / (ex * ex + ey * ey + 1e-20f), 0.0f, 1.0f);
            float dx = wx - ex * t, dy = wy - ey * t;
            *d2 = dx * dx + dy * dy;
            *side = Cross2(ex, ey, wx, wy);
        }

        // Exact signed distance to a convex quad with counter-clockwise vertices.
        float ConvexQuad(float px, float py, const float (&v)[4][2]) {
            float best = 1e30f;
            bool inside = true;
            for (int i = 0; i < 4; i++) {
                const float* a = v[i];
                const float* b = v[(i + 1) % 4];
                float d2, side;
                Segment(px, py, a[0], a[1], b[0], b[1], &d2, &side);
                best = fminf(best, d2);
                inside &= side >= 0.0f;
            }
            float d = sqrtf(best);
            return inside ? -d : d;
        }

        // Bends the (x, z) plane so the straight axis segment z in [-h, h] becomes an arc through its two end points
        // with half angle theta = |bulge| * pi / 2. Past the ends the space continues along the end tangents.
        void BendMap(float& x, float& z, float h, float bulge) {
            float s = bulge < 0.0f ? -1.0f : 1.0f;
            x *= s;
            float theta = fmaxf(fminf(fabsf(bulge), 1.0f) * (PI * 0.5f), 1e-4f);
            float st = sinf(theta), ct = cosf(theta);
            float c = h * ct / st;                           // arc centre at (c, 0)
            float rho = sqrtf(c * c + h * h);
            float phi = atan2f(z, c - x);
            float nx, nz;
            if (phi > theta) {
                float dx = x, dz = z - h;
                nz = h + dx * st + dz * ct;
                nx = dx * ct - dz * st;
            } else if (phi < -theta) {
                float dx = x, dz = z + h;
                nz = -h - (dx * st - dz * ct);
                nx = dx * ct + dz * st;
            } else {
                float vx = x - c;
                nz = phi / theta * h;
                nx = rho - sqrtf(vx * vx + z * z);
            }
            x = nx * s;
            z = nz;
        }
    }

    float SuperFrustum::LocalDistance(Vector3 p) const {
        float hx = size.x * 0.5f, hy = size.y * 0.5f, h = size.z * 0.5f;
        float x = p.x, z = p.z;
        if (fabsf(bulge) > 1e-4f) BendMap(x, z, h, bulge);

        // rounded rectangle cross section
        float inner = fminf(hx, hy);
        float r = rounding * inner;
        float qx = fabsf(x) - hx + r, qy = fabsf(p.y) - hy + r;
        float mx = fmaxf(qx, 0.0f), my = fmaxf(qy, 0.0f);
        float u = sqrtf(mx * mx + my * my) + fminf(fmaxf(qx, qy), 0.0f) - r;

        // revolve a trapezoid in (u, z): the right edge leans inward by (1 - taper) * inner over the height.
        float x3 = -(1.0f - taper) * inner;
        float leftBottom, leftTop;
        if (onion > 1e-3f) {
            leftBottom = -inner * (1.0f - onion);
            leftTop = -inner + (x3 + inner) * onion;
        } else {
            float far = -inner - 4.0f * (h + inner) - 1.0f;    // solid: push the inner edge out of reach
            leftBottom = far;
            leftTop = far;
        }
        const float quad[4][2] = { { leftBottom, -h }, { 0.0f, -h }, { x3, h }, { leftTop, h } };
        return ConvexQuad(u, z, quad) - dilation;
    }

    Vector3 SuperFrustum::Gradient(Vector3 w, float e) const {
        Vector3 g = {
            Distance({ w.x + e, w.y, w.z }) - Distance({ w.x - e, w.y, w.z }),
            Distance({ w.x, w.y + e, w.z }) - Distance({ w.x, w.y - e, w.z }),
            Distance({ w.x, w.y, w.z + e }) - Distance({ w.x, w.y, w.z - e })
        };
        return Vector3Scale(g, 0.5f / e);
    }

    Vector3 SuperFrustum::ToLocal(Vector3 w) const {
        Vector3 d = Vector3Subtract(w, position);
        return Vector3RotateByQuaternion(d, { -rotation.x, -rotation.y, -rotation.z, rotation.w });
    }

    Vector3 SuperFrustum::ToWorld(Vector3 l) const {
        return Vector3Add(Vector3RotateByQuaternion(l, rotation), position);
    }

    Vector3 SuperFrustum::Axis(int i) const {
        Vector3 a = { 0, 0, 0 };
        (&a.x)[i] = 1.0f;
        return Vector3RotateByQuaternion(a, rotation);
    }

    float SuperFrustum::Sagitta() const {
        if (fabsf(bulge) <= 1e-4f) return 0.0f;
        float h = size.z * 0.5f;
        float theta = fminf(fabsf(bulge), 1.0f) * (PI * 0.5f);
        float c = h * cosf(theta) / sinf(theta);
        return sqrtf(c * c + h * h) - c;
    }

    float SuperFrustum::BoundingRadius() const {
        Vector3 hs = Vector3Scale(size, 0.5f);
        return Vector3Length(hs) + Sagitta() + dilation;
    }

    BoundingBox SuperFrustum::LocalBounds() const {
        float sag = Sagitta();
        float tilt = fabsf(bulge) > 1e-4f ? sinf(fminf(fabsf(bulge), 1.0f) * PI * 0.5f) * size.x * 0.5f : 0.0f;
        Vector3 e = { size.x * 0.5f + sag + dilation, size.y * 0.5f + dilation, size.z * 0.5f + tilt + dilation };
        return { Vector3Negate(e), e };
    }

    BoundingBox SuperFrustum::WorldBounds() const {
        BoundingBox lb = LocalBounds();
        Vector3 lo = { 1e30f, 1e30f, 1e30f }, hi = { -1e30f, -1e30f, -1e30f };
        for (int i = 0; i < 8; i++) {
            Vector3 c = { (i & 1) ? lb.max.x : lb.min.x, (i & 2) ? lb.max.y : lb.min.y, (i & 4) ? lb.max.z : lb.min.z };
            Vector3 w = ToWorld(c);
            lo = Vector3Min(lo, w);
            hi = Vector3Max(hi, w);
        }
        return { lo, hi };
    }

    void SuperFrustum::Repair(float minSize) {
        size.x = fmaxf(size.x, minSize);
        size.y = fmaxf(size.y, minSize);
        size.z = fmaxf(size.z, minSize);
        rounding = std::clamp(rounding, 0.0f, 1.0f);
        taper = std::clamp(taper, 0.0f, 1.0f);
        bulge = std::clamp(bulge, -1.0f, 1.0f);
        dilation = std::clamp(dilation, 0.0f, 100.0f);
        onion = std::clamp(onion, 0.0f, 0.95f);
        rotation = QuaternionNormalize(rotation);
    }

    void SuperFrustum::Pack(float* p) const {
        p[0] = position.x; p[1] = position.y; p[2] = position.z;
        p[3] = size.x; p[4] = size.y; p[5] = size.z;
        p[6] = rounding; p[7] = taper; p[8] = bulge; p[9] = dilation; p[10] = onion;
    }

    void SuperFrustum::Unpack(const float* p) {
        position = { p[0], p[1], p[2] };
        size = { p[3], p[4], p[5] };
        rounding = p[6]; taper = p[7]; bulge = p[8]; dilation = p[9]; onion = p[10];
    }

    bool SuperFrustum::SameShape(const SuperFrustum& o) const {
        return Vector3Equals(size, o.size) && rounding == o.rounding && taper == o.taper && bulge == o.bulge &&
               dilation == o.dilation && onion == o.onion;
    }

    bool SuperFrustum::operator==(const SuperFrustum& o) const {
        return SameShape(o) && position.x == o.position.x && position.y == o.position.y && position.z == o.position.z &&
               rotation.x == o.rotation.x && rotation.y == o.rotation.y && rotation.z == o.rotation.z &&
               rotation.w == o.rotation.w;
    }
}
