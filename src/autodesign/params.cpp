#include "params.hpp"
#include <raymath.h>
#include <algorithm>
#include <cfloat>

namespace Bakugan {
    namespace {
        struct Field {
            float unit;          // a typical step ...
            bool length;         // ... times L when it is a length
            float lo, hi;        // its range (lengths: times L)
        };
        // in Ref's order
        const Field ShapeFields[] = {
            { 0.005f, true, -FLT_MAX, FLT_MAX }, { 0.005f, true, -FLT_MAX, FLT_MAX }, { 0.005f, true, -FLT_MAX, FLT_MAX },   // position
            { 0.01f, false, -1, 1 }, { 0.01f, false, -1, 1 }, { 0.01f, false, -1, 1 }, { 0.01f, false, -1, 1 },              // rotation
            { 0.005f, true, 0.004f, 2.0f }, { 0.005f, true, 0.004f, 2.0f }, { 0.005f, true, 0.004f, 2.0f },                  // size
            { 0.01f, false, 0, 1 },          // rounding
            { 0.01f, false, 0, 1 },          // taper
            { 0.01f, false, -1, 1 },         // bulge
            { 0.002f, true, 0, 0.2f },       // dilation
        };
        const Field HingeFields[] = {
            { 0.005f, true, -FLT_MAX, FLT_MAX }, { 0.005f, true, -FLT_MAX, FLT_MAX }, { 0.005f, true, -FLT_MAX, FLT_MAX },   // pivot
            { 0.01f, false, -1, 1 }, { 0.01f, false, -1, 1 }, { 0.01f, false, -1, 1 },                                       // axis
            { 0.01f, false, -2 * PI, 2 * PI },   // open angle (start)
            { 0.01f, false, -2 * PI, 2 * PI },   // closed angle (end)
            { 0.01f, false, 0, 1 },              // when it starts turning, as the design closes (t0) ...
            { 0.01f, false, 0, 1 },              // ... and when it stops (t1)
        };
        const int ShapeCount = sizeof(ShapeFields) / sizeof(Field), HingeCount = sizeof(HingeFields) / sizeof(Field);

        const Field& FieldOf(const Param& p) { return p.kind == Param::Shape ? ShapeFields[p.field] : HingeFields[p.field]; }
    }

    std::vector<Param> DesignParams(const Design& d, float L) {
        std::vector<Param> out;
        for (const Shape& s : d.shapes)
            for (int f = 0; f < ShapeCount; f++) out.push_back({ Param::Shape, s.id, f, ShapeFields[f].unit * (ShapeFields[f].length ? L : 1.0f) });
        for (const Hinge& h : d.hinges)
            for (int f = 0; f < HingeCount; f++) out.push_back({ Param::Hinge, h.id, f, HingeFields[f].unit * (HingeFields[f].length ? L : 1.0f) });
        return out;
    }

    float* Ref(Design& d, const Param& p) {
        if (p.kind == Param::Shape) {
            int s = d.ShapeIndex(p.id);
            if (s < 0) return nullptr;
            SuperFrustum& f = d.shapes[s].sf;
            switch (p.field) {
                case 0: return &f.position.x;
                case 1: return &f.position.y;
                case 2: return &f.position.z;
                case 3: return &f.rotation.x;
                case 4: return &f.rotation.y;
                case 5: return &f.rotation.z;
                case 6: return &f.rotation.w;
                case 7: return &f.size.x;
                case 8: return &f.size.y;
                case 9: return &f.size.z;
                case 10: return &f.rounding;
                case 11: return &f.taper;
                case 12: return &f.bulge;
                case 13: return &f.dilation;
            }
            return nullptr;
        }
        int h = d.HingeIndex(p.id);
        if (h < 0) return nullptr;
        Hinge& hg = d.hinges[h];
        switch (p.field) {
            case 0: return &hg.pivot.x;
            case 1: return &hg.pivot.y;
            case 2: return &hg.pivot.z;
            case 3: return &hg.axis.x;
            case 4: return &hg.axis.y;
            case 5: return &hg.axis.z;
            case 6: return &hg.start;
            case 7: return &hg.end;
            case 8: return &hg.t0;
            case 9: return &hg.t1;
        }
        return nullptr;
    }

    void SetParam(Design& d, const Param& p, float value, float L) {
        float* v = Ref(d, p);
        if (!v) return;
        const Field& f = FieldOf(p);
        float k = f.length ? L : 1.0f;
        *v = std::clamp(value, f.lo == -FLT_MAX ? -FLT_MAX : f.lo * k, f.hi == FLT_MAX ? FLT_MAX : f.hi * k);
    }

    void Tidy(Design& d) {
        for (Shape& s : d.shapes) {
            Quaternion& q = s.sf.rotation;
            q = QuaternionLength(q) > 1e-6f ? QuaternionNormalize(q) : QuaternionIdentity();
        }
        for (Hinge& h : d.hinges) {
            h.axis = Vector3Length(h.axis) > 1e-6f ? Vector3Normalize(h.axis) : Vector3{ 0, 1, 0 };
            // its turn takes a twentieth of the motion at least, inside it
            float w = std::clamp(h.t1 - h.t0, 0.05f, 1.0f);
            h.t0 = std::clamp(h.t0, 0.0f, 1.0f - w);
            h.t1 = h.t0 + w;
        }
    }
}
