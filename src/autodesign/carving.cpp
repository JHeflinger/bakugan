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

    Carving Carving::ShellAndCore(Vector3 centre, float radius, float core) {
        Carving c = Sphere(centre, radius);
        const float r = 0.5f * core;
        Cutter k;
        k.id = c.nextId++;
        k.core = true;
        k.sf.position = centre;
        k.sf.size = { 0.1f * r, 0.1f * r, 0.1f * r };    // a speck rounded out to the radius: a ball (so the shell is as
        k.sf.rounding = 1.0f;                             // thick everywhere), which Adam can grow, move and stretch
        k.sf.dilation = 0.95f * r;
        c.cutters.push_back(k);
        Piece shell = c.pieces[0];
        shell.outside.push_back(k.id);
        Piece inner;
        inner.id = c.nextId++;
        inner.inside.push_back(k.id);
        c.pieces = { shell, inner };
        Joint j;
        j.id = c.nextId++;
        j.parent = shell.id;
        j.child = inner.id;
        j.pivot = Vector3Add(centre, { r, 0, 0 });
        j.axis = { 0, 0, 1 };
        j.open = 0.0f;
        j.order = 0;
        c.joints.push_back(j);
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

    std::vector<int> Carving::Order() const {
        std::vector<const Joint*> js;
        for (const Joint& j : joints) js.push_back(&j);
        std::stable_sort(js.begin(), js.end(), [](const Joint* a, const Joint* b) { return a->order < b->order; });
        std::vector<int> ids;
        for (const Joint* j : js) ids.push_back(j->id);
        return ids;
    }

    void Carving::SetOrder(const std::vector<int>& ids) {
        std::vector<int> all = ids;
        for (int id : Order())
            if (std::find(all.begin(), all.end(), id) == all.end()) all.push_back(id);
        for (size_t r = 0; r < all.size(); r++) {
            int k = JointIndex(all[r]);
            if (k >= 0) joints[k].order = (int)r;
        }
    }

    namespace {
        bool Has(const std::vector<int>& v, int x) { return std::find(v.begin(), v.end(), x) != v.end(); }
    }

    std::vector<int> Carving::PiecesUsing(int cutterId) const {
        std::vector<int> out;
        for (size_t i = 0; i < pieces.size(); i++) {
            const Piece& p = pieces[i];
            bool uses = Has(p.inside, cutterId) || Has(p.outside, cutterId);
            for (const Region& r : p.extra) uses |= Has(r.inside, cutterId) || Has(r.outside, cutterId);
            if (uses) out.push_back((int)i);
        }
        return out;
    }

    int Carving::Split(int pi, int cutterId) {
        Piece part;
        part.id = nextId++;
        part.inside = pieces[pi].inside;
        part.outside = pieces[pi].outside;
        part.inside.push_back(cutterId);
        for (Region r : pieces[pi].extra) {
            r.inside.push_back(cutterId);
            part.extra.push_back(r);
        }
        pieces[pi].outside.push_back(cutterId);
        for (Region& r : pieces[pi].extra) r.outside.push_back(cutterId);
        pieces.push_back(part);
        return (int)pieces.size() - 1;
    }

    bool Carving::InCore(const Piece& p) const {
        for (const Cutter& k : cutters) {
            if (!k.core) continue;
            if (Has(p.inside, k.id)) return true;
            for (const Region& r : p.extra)
                if (Has(r.inside, k.id)) return true;
        }
        return false;
    }

    bool Carving::CoreJoint(const Joint& j) const {
        int a = PieceIndex(j.parent), b = PieceIndex(j.child);
        return (a >= 0 && InCore(pieces[a])) || (b >= 0 && InCore(pieces[b]));
    }

    BoundingBox Carving::PieceBounds(const Piece& p) const {
        Vector3 r = { radius, radius, radius };
        const BoundingBox sphere = { Vector3Subtract(centre, r), Vector3Add(centre, r) };
        auto region = [&](const std::vector<int>& inside) {
            BoundingBox b = sphere;
            for (int id : inside) {
                int k = CutterIndex(id);
                if (k < 0) continue;
                BoundingBox w = cutters[k].sf.WorldBounds();
                b.min = Vector3Max(b.min, w.min);
                b.max = Vector3Min(b.max, w.max);
            }
            b.max = Vector3Max(b.max, b.min);           // an empty region: a point
            return b;
        };
        BoundingBox b = region(p.inside);
        for (const Region& e : p.extra) {
            BoundingBox w = region(e.inside);
            b.min = Vector3Min(b.min, w.min);
            b.max = Vector3Max(b.max, w.max);
        }
        return b;
    }

    Design Carving::Build(std::vector<int>* bodyOf) const {
        Design d;
        // the sphere: a speck rounded out to the radius (an exact sphere, to a thousandth of it)
        SuperFrustum sphere;
        const float speck = fmaxf(0.002f * radius, 0.05f);     // (0.05: the smallest size a saved design keeps)
        sphere.position = centre;
        sphere.size = { speck, speck, speck };
        sphere.rounding = 1.0f;
        sphere.dilation = radius - 0.5f * speck;
        // around the sphere, for "outside a cutter": this box, less the cutter
        SuperFrustum box;
        box.position = centre;
        box.size = Vector3Scale({ 1, 1, 1 }, 2.2f * radius);

        // a region into a group: the sphere, cut by the cutters it's outside of and by "outside" each it's inside of
        auto addRegion = [&](int group, const std::vector<int>& inside, const std::vector<int>& outside) {
            d.AddShape(sphere, group, "Sphere");
            for (int id : outside) {
                int c = CutterIndex(id);
                if (c >= 0) d.AddShape(cutters[c].sf, group, "Cutter " + std::to_string(id), Op::Cut);
            }
            for (int id : inside) {
                int c = CutterIndex(id);
                if (c < 0) continue;
                int g = d.AddGroup("Outside cutter " + std::to_string(id), group);
                Group& grp = d.groups[g];
                grp.op = Op::Cut;
                int gid = grp.id;
                d.AddShape(box, gid, "Around");
                d.AddShape(cutters[c].sf, gid, "Cutter " + std::to_string(id), Op::Cut);
            }
        };
        std::vector<int> body(pieces.size());
        for (size_t k = 0; k < pieces.size(); k++) {
            const Piece& p = pieces[k];
            body[k] = d.groups[d.AddGroup("Piece " + std::to_string(k + 1), -1)].id;
            if (p.extra.empty()) {
                addRegion(body[k], p.inside, p.outside);
                continue;
            }
            // several regions: each a feature of the body, joined (a cut in one mustn't reach the others)
            addRegion(d.groups[d.AddGroup("Region 1", body[k])].id, p.inside, p.outside);
            for (size_t e = 0; e < p.extra.size(); e++)
                addRegion(d.groups[d.AddGroup("Region " + std::to_string(e + 2), body[k])].id, p.extra[e].inside, p.extra[e].outside);
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
        for (const Cutter& k : c.cutters) {
            if (k.core && c.coreLocked) continue;
            for (int f = 0; f < CutterCount; f++) out.push_back({ Param::Cutter, k.id, f, CutterFields[f].unit * (CutterFields[f].length ? L : 1.0f) });
        }
        for (const Joint& j : c.joints) {
            if (c.coreLocked && c.CoreJoint(j)) continue;
            for (int f = 0; f < JointCount; f++) {
                if (c.ordered && f >= 7) continue;       // timing comes from the opening order
                out.push_back({ Param::Joint, j.id, f, JointFields[f].unit * (JointFields[f].length ? L : 1.0f) });
            }
        }
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
        if (c.ordered) c.SetOrder({});                   // ranks 0 .. n - 1, as they stand
        const float n = (float)std::max<size_t>(1, c.joints.size());
        for (Joint& j : c.joints) {
            j.axis = Vector3Length(j.axis) > 1e-6f ? Vector3Normalize(j.axis) : Vector3{ 0, 1, 0 };
            if (c.ordered) {
                // one after another: the first to open is the last to close
                j.t0 = 1.0f - (j.order + 1) / n;
                j.t1 = 1.0f - j.order / n;
                continue;
            }
            // its turn takes a twentieth of the motion at least, inside it
            float w = std::clamp(j.t1 - j.t0, 0.05f, 1.0f);
            j.t0 = std::clamp(j.t0, 0.0f, 1.0f - w);
            j.t1 = j.t0 + w;
        }
    }
}
