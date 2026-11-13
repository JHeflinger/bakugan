#include "carving.hpp"
#include <raymath.h>
#include <algorithm>
#include <cfloat>
#include <string>

namespace Bakugan {
    Carving Carving::Sphere(Vector3 centre, float radius) {
        Carving c;
        c.centre = centre;
        c.radius = radius;
        c.pieces.push_back({ c.nextId++, {}, {} });
        return c;
    }

    int Carving::PieceIndex(int id) const {
        for (size_t i = 0; i < pieces.size(); i++)
            if (pieces[i].id == id) return (int)i;
        return -1;
    }

    int Carving::CutterIndex(int id) const {
        for (size_t i = 0; i < cutters.size(); i++)
            if (cutters[i].id == id) return (int)i;
        return -1;
    }

    int Carving::JointIndex(int id) const {
        for (size_t i = 0; i < joints.size(); i++)
            if (joints[i].id == id) return (int)i;
        return -1;
    }

    int Carving::ParentJoint(int pieceId) const {
        for (size_t i = 0; i < joints.size(); i++)
            if (joints[i].child == pieceId) return (int)i;
        return -1;
    }

    int Carving::Hollows() const {
        int n = 0;
        for (const Cutter& k : cutters) n += k.hollow;
        return n;
    }

    std::vector<int> Carving::PiecesUsing(int cutterId) const {
        std::vector<int> out;
        int k = CutterIndex(cutterId);
        if (k >= 0 && cutters[k].hollow) {
            for (size_t i = 0; i < pieces.size(); i++) out.push_back((int)i);
            return out;
        }
        for (size_t i = 0; i < pieces.size(); i++) {
            const Piece& p = pieces[i];
            if (std::find(p.inside.begin(), p.inside.end(), cutterId) != p.inside.end() ||
                std::find(p.outside.begin(), p.outside.end(), cutterId) != p.outside.end())
                out.push_back((int)i);
        }
        return out;
    }

    BoundingBox Carving::PieceBounds(const Piece& p) const {
        Vector3 r = { radius, radius, radius };
        BoundingBox b = { Vector3Subtract(centre, r), Vector3Add(centre, r) };
        for (int id : p.inside) {
            int k = CutterIndex(id);
            if (k < 0) continue;
            BoundingBox w = cutters[k].sf.WorldBounds();
            b.min = Vector3Max(b.min, w.min);
            b.max = Vector3Min(b.max, w.max);
        }
        b.max = Vector3Max(b.max, b.min);               // an empty piece: a point
        return b;
    }

    Design Carving::Build(std::vector<int>* bodyOf) const {
        Design d;
        // the sphere: a speck rounded out to the radius (an exact sphere, to a thousandth of it)
        SuperFrustum sphere;
        const float speck = 0.002f * radius;
        sphere.position = centre;
        sphere.size = { speck, speck, speck };
        sphere.rounding = 1.0f;
        sphere.dilation = radius - 0.5f * speck;
        // around the sphere, for "outside a cutter": this box, less the cutter
        SuperFrustum box;
        box.position = centre;
        box.size = Vector3Scale({ 1, 1, 1 }, 2.2f * radius);

        std::vector<int> body(pieces.size());
        for (size_t k = 0; k < pieces.size(); k++) {
            const Piece& p = pieces[k];
            body[k] = d.groups[d.AddGroup("Piece " + std::to_string(k + 1), -1)].id;
            d.AddShape(sphere, body[k], "Sphere");
            for (int id : p.outside) {
                int c = CutterIndex(id);
                if (c >= 0) d.AddShape(cutters[c].sf, body[k], "Cutter " + std::to_string(id), Op::Cut);
            }
            for (const Cutter& h : cutters)
                if (h.hollow) d.AddShape(h.sf, body[k], "Hollow " + std::to_string(h.id), Op::Cut);
            for (int id : p.inside) {
                int c = CutterIndex(id);
                if (c < 0) continue;
                int g = d.AddGroup("Outside cutter " + std::to_string(id), body[k]);
                Group& grp = d.groups[g];
                grp.op = Op::Cut;
                int gid = grp.id;
                d.AddShape(box, gid, "Around");
                d.AddShape(cutters[c].sf, gid, "Cutter " + std::to_string(id), Op::Cut);
            }
        }
        for (const Joint& j : joints) {
            int a = PieceIndex(j.parent), b = PieceIndex(j.child);
            if (a < 0 || b < 0 || !d.CanHinge(body[a], body[b])) continue;
            Hinge& h = d.hinges[d.AddHinge(body[a], body[b])];
            h.name = "Joint " + std::to_string(j.id);
            h.pivot = j.pivot;
            h.axis = j.axis;
            h.start = j.open;
            h.end = 0.0f;                                // closed: back in the sphere
            h.t0 = j.t0;
            h.t1 = j.t1;
            h.hardware = false;
        }
        if (bodyOf) *bodyOf = body;
        return d;
    }

    // ---- parameters ---------------------------------------------------------------------------------------------------

    namespace {
        struct Field {
            float unit;          // a typical step ...
            bool length;         // ... times L when it is a length
            float lo, hi;        // its range (lengths: times L)
        };
        const Field CutterFields[] = {
            { 0.005f, true, -FLT_MAX, FLT_MAX }, { 0.005f, true, -FLT_MAX, FLT_MAX }, { 0.005f, true, -FLT_MAX, FLT_MAX },   // position
            { 0.01f, false, -1, 1 }, { 0.01f, false, -1, 1 }, { 0.01f, false, -1, 1 }, { 0.01f, false, -1, 1 },              // rotation
            { 0.005f, true, 0.004f, 2.0f }, { 0.005f, true, 0.004f, 2.0f }, { 0.005f, true, 0.004f, 2.0f },                  // size
            { 0.01f, false, 0, 1 },          // rounding
            { 0.01f, false, 0, 1 },          // taper
            { 0.01f, false, -1, 1 },         // bulge
            { 0.002f, true, 0, 0.2f },       // dilation
        };
        const Field JointFields[] = {
            { 0.005f, true, -FLT_MAX, FLT_MAX }, { 0.005f, true, -FLT_MAX, FLT_MAX }, { 0.005f, true, -FLT_MAX, FLT_MAX },   // pivot
            { 0.01f, false, -1, 1 }, { 0.01f, false, -1, 1 }, { 0.01f, false, -1, 1 },                                       // axis
            { 0.01f, false, -2 * PI, 2 * PI },   // open angle
            { 0.01f, false, 0, 1 },              // when it starts turning, as the design closes (t0) ...
            { 0.01f, false, 0, 1 },              // ... and when it stops (t1)
        };
        const Field TargetFields[] = {
            { 0.005f, true, -FLT_MAX, FLT_MAX }, { 0.005f, true, -FLT_MAX, FLT_MAX }, { 0.005f, true, -FLT_MAX, FLT_MAX },   // offset
            { 0.01f, false, -1.2f, 1.2f },       // log scale (0.3x .. 3.3x)
        };
        const int CutterCount = sizeof(CutterFields) / sizeof(Field), JointCount = sizeof(JointFields) / sizeof(Field);

        const Field& FieldOf(const Param& p) {
            return p.kind == Param::Cutter ? CutterFields[p.field] : (p.kind == Param::Joint ? JointFields[p.field] : TargetFields[p.field]);
        }

        float* Ref(Carving& c, Placement& t, const Param& p) {
            if (p.kind == Param::Cutter) {
                int k = c.CutterIndex(p.id);
                if (k < 0) return nullptr;
                SuperFrustum& f = c.cutters[k].sf;
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
            if (p.kind == Param::Joint) {
                int k = c.JointIndex(p.id);
                if (k < 0) return nullptr;
                Joint& j = c.joints[k];
                switch (p.field) {
                    case 0: return &j.pivot.x;
                    case 1: return &j.pivot.y;
                    case 2: return &j.pivot.z;
                    case 3: return &j.axis.x;
                    case 4: return &j.axis.y;
                    case 5: return &j.axis.z;
                    case 6: return &j.open;
                    case 7: return &j.t0;
                    case 8: return &j.t1;
                }
                return nullptr;
            }
            switch (p.field) {
                case 0: return &t.offset.x;
                case 1: return &t.offset.y;
                case 2: return &t.offset.z;
            }
            return nullptr;                              // the scale: not stored as its log
        }
    }

    std::vector<Param> CarvingParams(const Carving& c, float L) {
        std::vector<Param> out;
        for (const Cutter& k : c.cutters)
            for (int f = 0; f < CutterCount; f++) out.push_back({ Param::Cutter, k.id, f, CutterFields[f].unit * (CutterFields[f].length ? L : 1.0f) });
        for (const Joint& j : c.joints)
            for (int f = 0; f < JointCount; f++) out.push_back({ Param::Joint, j.id, f, JointFields[f].unit * (JointFields[f].length ? L : 1.0f) });
        return out;
    }

    std::vector<Param> TargetParams(bool move, bool scale, float L) {
        std::vector<Param> out;
        if (move)
            for (int f = 0; f < 3; f++) out.push_back({ Param::Target, 0, f, TargetFields[f].unit * L });
        if (scale) out.push_back({ Param::Target, 0, 3, TargetFields[3].unit });
        return out;
    }

    float Get(const Carving& c, const Placement& t, const Param& p, bool* ok) {
        if (ok) *ok = true;
        if (p.kind == Param::Target && p.field == 3) return logf(fmaxf(t.scale, 1e-6f));
        float* v = Ref(const_cast<Carving&>(c), const_cast<Placement&>(t), p);
        if (!v) {
            if (ok) *ok = false;
            return 0.0f;
        }
        return *v;
    }

    void Set(Carving& c, Placement& t, const Param& p, float value, float L) {
        const Field& f = FieldOf(p);
        float k = f.length ? L : 1.0f;
        value = std::clamp(value, f.lo == -FLT_MAX ? -FLT_MAX : f.lo * k, f.hi == FLT_MAX ? FLT_MAX : f.hi * k);
        if (p.kind == Param::Target && p.field == 3) {
            t.scale = expf(value);
            return;
        }
        if (float* v = Ref(c, t, p)) *v = value;
    }

    void Tidy(Carving& c) {
        for (Cutter& k : c.cutters) {
            Quaternion& q = k.sf.rotation;
            q = QuaternionLength(q) > 1e-6f ? QuaternionNormalize(q) : QuaternionIdentity();
        }
        for (Joint& j : c.joints) {
            j.axis = Vector3Length(j.axis) > 1e-6f ? Vector3Normalize(j.axis) : Vector3{ 0, 1, 0 };
            // its turn takes a twentieth of the motion at least, inside it
            float w = std::clamp(j.t1 - j.t0, 0.05f, 1.0f);
            j.t0 = std::clamp(j.t0, 0.0f, 1.0f - w);
            j.t1 = j.t0 + w;
        }
    }
}
