#include "params.hpp"
#include <algorithm>
#include <unordered_map>

namespace Bakugan::Auto {
    namespace {
        struct DofInfo {
            Dof dof;
            Part part;
            float unit;              // one step's size ...
            bool length;             // ... times the mesh's size
        };
        const DofInfo Dofs[] = {
            { Dof::Pos,      Part::Position, 0.01f,  true },
            { Dof::LogSize,  Part::Size,     0.05f,  false },    // 5% bigger or smaller
            { Dof::Rot,      Part::Rotation, 0.05f,  false },    // radians
            { Dof::Rounding, Part::Profile,  0.05f,  false },
            { Dof::Taper,    Part::Profile,  0.05f,  false },
            { Dof::Bulge,    Part::Profile,  0.05f,  false },
            { Dof::Dilation, Part::Profile,  0.004f, true },
            { Dof::Pivot,    Part::Pivot,    0.01f,  true },
            { Dof::Axis,     Part::Axis,     0.05f,  false },    // radians
            { Dof::Angle,    Part::Angle,    0.06f,  false },    // radians
            { Dof::Timing,   Part::Timing,   0.04f,  false },    // share of the motion
            { Dof::Blend,    Part::Blend,    0.004f, true },
        };
        const DofInfo& Info(Dof k) { return Dofs[(int)k]; }
    }

    float Parameters::Unit(Dof k) const { return Info(k).unit * (Info(k).length ? m_Space.L : 1.0f); }
    Part Parameters::PartOf(Dof k) const { return Info(k).part; }

    // A shape's shell (onion) is left alone: it is all or nothing in effect, so a small step of it has no useful slope.
    std::vector<Param> Parameters::Of(const Design& d, const Focus* only) const {
        std::vector<Param> ps;
        auto add = [&](Dof k, int id, int axes) {
            if (only && only->part != Part::All && PartOf(k) != only->part) return;
            for (int a = 0; a < axes; a++) ps.push_back({ k, id, a });
        };
        for (const Shape& sh : d.shapes) {
            if (only && sh.id != only->shape && sh.id != only->shape2) continue;
            add(Dof::Pos, sh.id, 3);
            add(Dof::LogSize, sh.id, 3);
            add(Dof::Rot, sh.id, 3);
            add(Dof::Rounding, sh.id, 1);
            add(Dof::Taper, sh.id, 1);
            add(Dof::Bulge, sh.id, 1);
            add(Dof::Dilation, sh.id, 1);
        }
        for (const Hinge& h : d.hinges) {
            if (only && h.id != only->hinge) continue;
            add(Dof::Pivot, h.id, 3);
            add(Dof::Axis, h.id, 3);
            add(Dof::Angle, h.id, 1);
            if (m_Space.timing) add(Dof::Timing, h.id, h.ownOpening ? 4 : 2);
        }
        for (const Group& g : d.groups) {
            if (only && g.id != only->group) continue;
            if (d.GroupShapes(g.id).size() >= 2) add(Dof::Blend, g.id, 1);
        }
        return ps;
    }

    // Rotations collect their three components first and turn once, about the combined axis.
    void Parameters::Step(Design& d, const std::vector<Param>& ps, const std::vector<float>& x) const {
        std::unordered_map<int, Vector3> turnShape, turnAxis;
        for (size_t i = 0; i < ps.size(); i++) {
            const Param& p = ps[i];
            float v = x[i] * Unit(p.dof);
            if (v == 0.0f) continue;
            int s = p.dof <= Dof::Dilation ? d.ShapeIndex(p.id) : -1;
            int h = p.dof >= Dof::Pivot && p.dof <= Dof::Timing ? d.HingeIndex(p.id) : -1;
            SuperFrustum* f = s >= 0 ? &d.shapes[s].sf : nullptr;
            Hinge* hg = h >= 0 ? &d.hinges[h] : nullptr;
            switch (p.dof) {
                case Dof::Rot: Coord(turnShape[p.id], p.axis) += v; break;
                case Dof::Axis: Coord(turnAxis[p.id], p.axis) += v; break;
                case Dof::Pos: if (f) Coord(f->position, p.axis) += v; break;
                case Dof::LogSize: if (f) Coord(f->size, p.axis) *= expf(v); break;
                case Dof::Rounding: if (f) f->rounding += v; break;
                case Dof::Taper: if (f) f->taper += v; break;
                case Dof::Bulge: if (f) f->bulge += v; break;
                case Dof::Dilation: if (f) f->dilation += v; break;
                case Dof::Pivot: if (hg) Coord(hg->pivot, p.axis) += v; break;
                case Dof::Angle: if (hg) hg->end += v; break;
                case Dof::Timing:
                    if (hg) {
                        float& a = p.axis < 2 ? hg->t0 : hg->o0;
                        float& b = p.axis < 2 ? hg->t1 : hg->o1;
                        if (p.axis % 2 == 0) { a += v; b += v; }               // slide the window
                        else { a -= 0.5f * v; b += 0.5f * v; }                 // stretch it about its middle
                    }
                    break;
                case Dof::Blend: {
                    int g = d.GroupIndex(p.id);
                    if (g >= 0) d.groups[g].blend += v;
                    break;
                }
            }
        }
        auto turn = [](Vector3 r) {
            float a = Vector3Length(r);
            return a > 1e-9f ? QuaternionFromAxisAngle(Vector3Scale(r, 1.0f / a), a) : QuaternionIdentity();
        };
        for (const auto& [id, r] : turnShape) {
            int s = d.ShapeIndex(id);
            if (s >= 0) d.shapes[s].sf.rotation = QuaternionNormalize(QuaternionMultiply(turn(r), d.shapes[s].sf.rotation));
        }
        for (const auto& [id, r] : turnAxis) {
            int h = d.HingeIndex(id);
            if (h >= 0) d.hinges[h].axis = Vector3Normalize(Vector3RotateByQuaternion(d.hinges[h].axis, turn(r)));
        }
        m_Space.Project(d);
    }

    Focus Parameters::RandomBlock(const Design& d, Rng& rng) const {
        int blends = 0;
        for (const Group& g : d.groups) blends += d.GroupShapes(g.id).size() >= 2;
        int n = (int)d.shapes.size() + (int)d.hinges.size() + blends;
        Focus f;
        if (n == 0) return f;
        int k = Pick(rng, n);
        bool all = Chance(rng, 0.2f);
        if (k < (int)d.shapes.size()) {
            f.shape = d.shapes[k].id;
            const Part parts[] = { Part::Position, Part::Size, Part::Rotation, Part::Profile };
            if (!all) f.part = parts[Pick(rng, 4)];
            return f;
        }
        k -= (int)d.shapes.size();
        if (k < (int)d.hinges.size()) {
            f.hinge = d.hinges[k].id;
            const Part parts[] = { Part::Pivot, Part::Axis, Part::Angle, Part::Timing };
            if (!all) f.part = parts[Pick(rng, m_Space.timing ? 4 : 3)];
            return f;
        }
        k -= (int)d.hinges.size();
        for (const Group& g : d.groups)
            if (d.GroupShapes(g.id).size() >= 2 && k-- == 0) { f.group = g.id; break; }
        return f;
    }
}
