#ifndef FB_FMATH_H
#define FB_FMATH_H
// Small vector, quaternion and rigid-transform math. Everything inline; no dependency on raylib so the optimizer
// core can be used headless.
#include <math.h>
#include <float.h>
#include <stddef.h>
#include <stdbool.h>

#ifndef FB_PI
#define FB_PI 3.14159265358979323846f
#endif

typedef struct { float x, y, z; } V3;
typedef struct { float x, y, z, w; } Quat;
typedef struct { Quat q; V3 t; } Rigid;   // x -> q x + t

static inline float fb_min(float a, float b) { return a < b ? a : b; }
static inline float fb_max(float a, float b) { return a > b ? a : b; }
static inline float fb_clamp(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
static inline int fb_mini(int a, int b) { return a < b ? a : b; }
static inline int fb_maxi(int a, int b) { return a > b ? a : b; }

static inline V3 v3(float x, float y, float z) { V3 r = { x, y, z }; return r; }
static inline V3 v3add(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline V3 v3sub(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline V3 v3scale(V3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static inline V3 v3mad(V3 a, V3 b, float s) { return v3(a.x + b.x * s, a.y + b.y * s, a.z + b.z * s); }
static inline float v3dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 v3cross(V3 a, V3 b) { return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static inline float v3len(V3 a) { return sqrtf(v3dot(a, a)); }
static inline float v3dist(V3 a, V3 b) { return v3len(v3sub(a, b)); }
static inline V3 v3norm(V3 a) { float l = v3len(a); return l > 1e-20f ? v3scale(a, 1.0f / l) : v3(0, 0, 1); }
static inline V3 v3min(V3 a, V3 b) { return v3(fb_min(a.x, b.x), fb_min(a.y, b.y), fb_min(a.z, b.z)); }
static inline V3 v3max(V3 a, V3 b) { return v3(fb_max(a.x, b.x), fb_max(a.y, b.y), fb_max(a.z, b.z)); }
static inline V3 v3lerp(V3 a, V3 b, float t) { return v3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t); }
static inline float v3get(V3 a, int i) { return i == 0 ? a.x : (i == 1 ? a.y : a.z); }
static inline void v3set(V3* a, int i, float s) { if (i == 0) a->x = s; else if (i == 1) a->y = s; else a->z = s; }
// some unit vector perpendicular to a (a unit)
static inline V3 v3perp(V3 a) {
    V3 o = fabsf(a.x) < 0.9f ? v3(1, 0, 0) : v3(0, 1, 0);
    return v3norm(v3cross(a, o));
}

static inline Quat qident(void) { Quat q = { 0, 0, 0, 1 }; return q; }
static inline Quat qaxis(V3 axis, float angle) {
    V3 a = v3norm(axis);
    float s = sinf(0.5f * angle);
    Quat q = { a.x * s, a.y * s, a.z * s, cosf(0.5f * angle) };
    return q;
}
static inline Quat qmul(Quat a, Quat b) {
    Quat r = { a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y, a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
               a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w, a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
    return r;
}
static inline Quat qconj(Quat a) { Quat r = { -a.x, -a.y, -a.z, a.w }; return r; }
static inline Quat qnorm(Quat a) {
    float l = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
    if (l < 1e-20f) return qident();
    Quat r = { a.x / l, a.y / l, a.z / l, a.w / l };
    return r;
}
static inline V3 qrot(Quat q, V3 v) {
    V3 u = v3(q.x, q.y, q.z);
    V3 t = v3scale(v3cross(u, v), 2.0f);
    return v3add(v3add(v, v3scale(t, q.w)), v3cross(u, t));
}
// axis and angle (angle in (-pi, pi])
static inline void qtoaxis(Quat q, V3* axis, float* angle) {
    q = qnorm(q);
    if (q.w < 0) { q.x = -q.x; q.y = -q.y; q.z = -q.z; q.w = -q.w; }
    float s = sqrtf(fb_max(0.0f, 1.0f - q.w * q.w));
    *angle = 2.0f * atan2f(s, q.w);
    *axis = s > 1e-7f ? v3(q.x / s, q.y / s, q.z / s) : v3(0, 0, 1);
}

static inline Rigid rident(void) { Rigid r = { { 0, 0, 0, 1 }, { 0, 0, 0 } }; return r; }
static inline V3 rapply(Rigid r, V3 p) { return v3add(qrot(r.q, p), r.t); }
static inline V3 rinv_apply(Rigid r, V3 p) { return qrot(qconj(r.q), v3sub(p, r.t)); }
static inline V3 rrot(Rigid r, V3 d) { return qrot(r.q, d); }
static inline V3 rinv_rot(Rigid r, V3 d) { return qrot(qconj(r.q), d); }
// (a then b): x -> b(a(x))
static inline Rigid rthen(Rigid a, Rigid b) {
    Rigid r;
    r.q = qnorm(qmul(b.q, a.q));
    r.t = v3add(qrot(b.q, a.t), b.t);
    return r;
}
// rotation by angle about the line through p along axis
static inline Rigid rabout(V3 p, V3 axis, float angle) {
    Rigid r;
    r.q = qaxis(axis, angle);
    r.t = v3sub(p, qrot(r.q, p));
    return r;
}
static inline Rigid rinverse(Rigid r) {
    Rigid i;
    i.q = qconj(r.q);
    i.t = v3scale(qrot(i.q, r.t), -1.0f);
    return i;
}

typedef struct { V3 min, max; } Box3;
static inline Box3 box_empty(void) { Box3 b = { { FLT_MAX, FLT_MAX, FLT_MAX }, { -FLT_MAX, -FLT_MAX, -FLT_MAX } }; return b; }
static inline bool box_valid(Box3 b) { return b.min.x <= b.max.x && b.min.y <= b.max.y && b.min.z <= b.max.z; }
static inline Box3 box_add(Box3 b, V3 p) { b.min = v3min(b.min, p); b.max = v3max(b.max, p); return b; }
static inline Box3 box_grow(Box3 b, float g) { b.min = v3sub(b.min, v3(g, g, g)); b.max = v3add(b.max, v3(g, g, g)); return b; }
static inline bool box_overlap(Box3 a, Box3 b) {
    return a.min.x <= b.max.x && b.min.x <= a.max.x && a.min.y <= b.max.y && b.min.y <= a.max.y && a.min.z <= b.max.z && b.min.z <= a.max.z;
}
static inline Box3 box_transform(Box3 b, Rigid r) {
    Box3 o = box_empty();
    if (!box_valid(b)) return o;
    for (int k = 0; k < 8; k++) {
        V3 c = v3((k & 1) ? b.max.x : b.min.x, (k & 2) ? b.max.y : b.min.y, (k & 4) ? b.max.z : b.min.z);
        o = box_add(o, rapply(r, c));
    }
    return o;
}
// the ray o + t d (t in [*tn, *tf]) clipped to the box; false when it misses
static inline bool box_ray(Box3 b, V3 o, V3 d, float* tn, float* tf) {
    for (int a = 0; a < 3; a++) {
        float oa = v3get(o, a), da = v3get(d, a), lo = v3get(b.min, a), hi = v3get(b.max, a);
        if (fabsf(da) < 1e-12f) {
            if (oa < lo || oa > hi) return false;
            continue;
        }
        float t0 = (lo - oa) / da, t1 = (hi - oa) / da;
        if (t0 > t1) { float s = t0; t0 = t1; t1 = s; }
        if (t0 > *tn) *tn = t0;
        if (t1 < *tf) *tf = t1;
        if (*tn > *tf) return false;
    }
    return true;
}

static inline float fb_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// a small fast deterministic RNG (xorshift64*)
typedef struct { unsigned long long s; } Rng;
static inline void rng_seed(Rng* r, unsigned long long seed) { r->s = seed * 0x9E3779B97F4A7C15ull + 0x1234567ull; if (!r->s) r->s = 1; }
static inline unsigned long long rng_next(Rng* r) {
    r->s ^= r->s >> 12; r->s ^= r->s << 25; r->s ^= r->s >> 27;
    return r->s * 2685821657736338717ull;
}
static inline float rng_uniform(Rng* r) { return (float)((rng_next(r) >> 40) * (1.0 / 16777216.0)); }
static inline int rng_int(Rng* r, int n) { return n <= 0 ? 0 : (int)(rng_next(r) % (unsigned long long)n); }
static inline V3 rng_dir(Rng* r) {
    float z = 2.0f * rng_uniform(r) - 1.0f, a = 2.0f * FB_PI * rng_uniform(r), s = sqrtf(fb_max(0.0f, 1.0f - z * z));
    return v3(s * cosf(a), s * sinf(a), z);
}
#endif
