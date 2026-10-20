#include "field.hpp"
#include "util/utils.hpp"
#include <raymath.h>
#include <algorithm>
#include <cfloat>
#include <cmath>

namespace Bakugan {
    namespace {
        inline float SMin(float a, float b, float k) {
            float h = fmaxf(k - fabsf(a - b), 0.0f) / k;
            return fminf(a, b) - h * h * k * 0.25f;
        }

        // Capped cylinder about the frame's z axis: radius r, z in [lo, hi].
        inline float Cylinder(float x, float y, float z, float r, float lo, float hi) {
            float dr = sqrtf(x * x + y * y) - r;
            float dz = fmaxf(lo - z, z - hi);
            float ox = fmaxf(dr, 0.0f), oz = fmaxf(dz, 0.0f);
            return fminf(fmaxf(dr, dz), 0.0f) + sqrtf(ox * ox + oz * oz);
        }

        inline float Box(float x, float y, float z, Vector3 lo, Vector3 hi) {
            float cx = (lo.x + hi.x) * 0.5f, cy = (lo.y + hi.y) * 0.5f, cz = (lo.z + hi.z) * 0.5f;
            float qx = fabsf(x - cx) - (hi.x - lo.x) * 0.5f;
            float qy = fabsf(y - cy) - (hi.y - lo.y) * 0.5f;
            float qz = fabsf(z - cz) - (hi.z - lo.z) * 0.5f;
            float mx = fmaxf(qx, 0.0f), my = fmaxf(qy, 0.0f), mz = fmaxf(qz, 0.0f);
            return sqrtf(mx * mx + my * my + mz * mz) + fminf(fmaxf(qx, fmaxf(qy, qz)), 0.0f);
        }

        // Point in the hinge frame: x along the leaf, y along the side, z along the axis.
        inline void ToFrame(const HingeHardware& h, Vector3 p, float* x, float* y, float* z) {
            Vector3 q = Vector3Subtract(p, h.pivot);
            *x = Vector3DotProduct(q, h.leaf);
            *y = Vector3DotProduct(q, h.side);
            *z = Vector3DotProduct(q, h.axis);
        }

        BoundingBox Grow(BoundingBox b, float m) {
            return { Vector3Subtract(b.min, { m, m, m }), Vector3Add(b.max, { m, m, m }) };
        }

        BoundingBox Merge(BoundingBox a, BoundingBox b) {
            return { Vector3Min(a.min, b.min), Vector3Max(a.max, b.max) };
        }

        bool EmptyBox(BoundingBox b) { return b.max.x <= b.min.x && b.max.y <= b.min.y && b.max.z <= b.min.z; }

        BoundingBox HardwareBounds(const HingeHardware& h) {
            float rad = h.pinOnly ? fmaxf(h.pinRadius, h.headRadius) : fmaxf(h.radius + h.clearance, h.leafLength + h.leafThickness);
            Vector3 lo = { 1e30f, 1e30f, 1e30f }, hi = { -1e30f, -1e30f, -1e30f };
            for (int a = 0; a < 3; a++) {
                float w = (&h.axis.x)[a];
                float c = (&h.pivot.x)[a] + w * (h.lo + h.hi) * 0.5f;
                float e = fabsf(w) * (h.hi - h.lo) * 0.5f + sqrtf(fmaxf(0.0f, 1.0f - w * w)) * rad + h.clearance;
                (&lo.x)[a] = c - e;
                (&hi.x)[a] = c + e;
            }
            return { lo, hi };
        }
    }

    // ---- compile -----------------------------------------------------------------------------------------------------

    int BodyField::Compile(const Design& d, int groupId, Op op, float blend) {
        int self = (int)m_Nodes.size();
        m_Nodes.push_back(Node{});
        m_Nodes[self].op = op;
        m_Nodes[self].blend = blend;
        std::vector<int> kids;
        for (int s : d.DirectShapes(groupId)) {
            int n = (int)m_Nodes.size();
            Node leaf;
            leaf.prim = (int)m_Prims.size();
            leaf.op = d.shapes[s].op;
            leaf.centre = d.shapes[s].sf.position;
            leaf.radius = Vector3Length(d.shapes[s].sf.LocalBounds().max);
            m_Prims.push_back(d.shapes[s].sf);
            m_PrimShape.push_back(d.shapes[s].id);
            m_PrimOp.push_back(leaf.op);
            m_Nodes.push_back(leaf);
            kids.push_back(n);
        }
        for (int g : d.ChildGroups(groupId)) {
            const Group& grp = d.groups[g];
            size_t firstPrim = m_Prims.size();
            int n = Compile(d, grp.id, grp.op, grp.blend);
            if (grp.op == Op::Cut)
                for (size_t k = firstPrim; k < m_Prims.size(); k++) m_PrimOp[k] = Op::Cut;
            kids.push_back(n);
        }
        // bounding sphere of what the node adds
        Vector3 lo = { 1e30f, 1e30f, 1e30f }, hi = { -1e30f, -1e30f, -1e30f };
        bool any = false;
        for (int k : kids) {
            const Node& c = m_Nodes[k];
            if (c.op != Op::Union || c.radius <= 0.0f) continue;
            lo = Vector3Min(lo, Vector3SubtractValue(c.centre, c.radius));
            hi = Vector3Max(hi, Vector3AddValue(c.centre, c.radius));
            any = true;
        }
        Node& me = m_Nodes[self];
        me.children = kids;
        if (any) {
            me.centre = Vector3Scale(Vector3Add(lo, hi), 0.5f);
            me.radius = Vector3Distance(lo, hi) * 0.5f + blend;
        }
        return self;
    }

    BodyField BodyField::Body(const Design& d, int bodyId) {
        BodyField f;
        f.body = bodyId;
        int gi = d.GroupIndex(bodyId);
        if (gi < 0) return f;
        f.m_Root = f.Compile(d, bodyId, Op::Union, d.groups[gi].blend);
        if (f.m_Prims.empty()) { f.m_Root = -1; f.m_Nodes.clear(); }
        BoundingBox box = d.BodyBounds(bodyId);
        bool hasBox = f.m_Root >= 0 && !EmptyBox(box);
        if (hasBox) box = Grow(box, d.groups[gi].blend + 0.5f);

        for (int h = 0; h < (int)d.hinges.size(); h++) {
            const Hinge& hg = d.hinges[h];
            bool parent = hg.parent == bodyId, child = hg.child == bodyId;
            if (!hg.hardware || (!parent && !child)) continue;
            HingeHardware hw;
            hw.hinge = hg.id;
            hw.pivot = hg.pivot;
            hw.axis = Vector3Normalize(hg.axis);
            hw.leaf = d.LeafDirection(h, child);
            hw.side = Vector3CrossProduct(hw.axis, hw.leaf);
            hw.radius = hg.radius;
            hw.pinRadius = hg.pinRadius;
            hw.clearance = hg.clearance;
            hw.leafLength = hg.leafLength;
            hw.leafThickness = hg.leafThickness;
            std::vector<Knuckle> ks = d.Knuckles(h);
            hw.lo = ks.front().lo;
            hw.hi = ks.back().hi;
            for (const Knuckle& k : ks) (k.child == child ? hw.own : hw.other).push_back(k);
            hw.pin = parent && hg.pin == PinStyle::Fused;
            hw.bore = !hw.pin;
            if (hg.pin == PinStyle::Separate) d.PinHead(h, &hw.headFrom, &hw.headTo, &hw.headRadius);
            f.m_Hardware.push_back(hw);
            BoundingBox hb = HardwareBounds(hw);
            box = hasBox ? Merge(box, hb) : hb;
            hasBox = true;
        }
        f.bounds = hasBox ? Grow(box, 0.5f) : BoundingBox{ { 0, 0, 0 }, { 0, 0, 0 } };
        return f;
    }

    BodyField BodyField::Pin(const Design& d, int h) {
        BodyField f;
        const Hinge& hg = d.hinges[h];
        f.pinHinge = hg.id;
        HingeHardware hw;
        hw.hinge = hg.id;
        hw.pivot = hg.pivot;
        hw.axis = Vector3Normalize(hg.axis);
        hw.leaf = Utils::AnyPerpendicular(hw.axis);
        hw.side = Vector3CrossProduct(hw.axis, hw.leaf);
        hw.pinRadius = hg.pinRadius;
        hw.clearance = hg.clearance;
        std::vector<Knuckle> ks = d.Knuckles(h);
        // the shaft runs through every knuckle; a head past the last one keeps it from dropping out
        d.PinHead(h, &hw.headFrom, &hw.headTo, &hw.headRadius);
        hw.lo = ks.front().lo;
        hw.hi = hw.headTo;
        hw.pin = true;
        hw.pinOnly = true;
        f.m_Hardware.push_back(hw);
        f.bounds = Grow(HardwareBounds(hw), 0.5f);
        return f;
    }

    // ---- evaluate ----------------------------------------------------------------------------------------------------

    float BodyField::Eval(int n, Vector3 p) const {
        const Node& nd = m_Nodes[n];
        if (nd.prim >= 0) return m_Prims[nd.prim].Distance(p);
        float u = FLT_MAX;
        bool any = false;
        for (int c : nd.children) {
            const Node& ch = m_Nodes[c];
            if (ch.op != Op::Union) continue;
            if (nd.blend <= 0.0f && any && ch.radius > 0.0f && Vector3Distance(p, ch.centre) - ch.radius > u) continue;
            float d = Eval(c, p);
            u = (any && nd.blend > 0.0f) ? SMin(u, d, nd.blend) : fminf(u, d);
            any = true;
        }
        if (!any) return FLT_MAX;
        for (int c : nd.children) {
            const Node& ch = m_Nodes[c];
            if (ch.op != Op::Cut) continue;
            // a cut only matters where it reaches deeper than -u
            if (ch.radius > 0.0f && Vector3Distance(p, ch.centre) - ch.radius >= -u) continue;
            u = fmaxf(u, -Eval(c, p));
        }
        return u;
    }

    float BodyField::HardwareDistance(float d, Vector3 p) const {
        // leaves join the body, then the other body's knuckles are cleared, then this body's knuckles and pin go in,
        // and finally the pin bore is drilled
        for (const HingeHardware& h : m_Hardware) {
            if (h.pinOnly || h.leafLength <= 0.0f) continue;
            float x, y, z;
            ToFrame(h, p, &x, &y, &z);
            for (const Knuckle& k : h.own)
                d = fminf(d, Box(x, y, z, { 0.0f, -h.leafThickness * 0.5f, k.lo }, { h.leafLength, h.leafThickness * 0.5f, k.hi }));
        }
        for (const HingeHardware& h : m_Hardware) {
            if (h.pinOnly) continue;
            float x, y, z;
            ToFrame(h, p, &x, &y, &z);
            for (const Knuckle& k : h.other)
                d = fmaxf(d, -Cylinder(x, y, z, h.radius + h.clearance, k.lo - h.clearance, k.hi + h.clearance));
        }
        for (const HingeHardware& h : m_Hardware) {
            float x, y, z;
            ToFrame(h, p, &x, &y, &z);
            if (!h.pinOnly)
                for (const Knuckle& k : h.own) d = fminf(d, Cylinder(x, y, z, h.radius, k.lo, k.hi));
            if (h.pin) d = fminf(d, Cylinder(x, y, z, h.pinRadius, h.lo, h.hi));
            if (h.pinOnly && h.headRadius > 0.0f) d = fminf(d, Cylinder(x, y, z, h.headRadius, h.headFrom, h.headTo));
        }
        for (const HingeHardware& h : m_Hardware) {
            if (!h.bore) continue;
            float x, y, z;
            ToFrame(h, p, &x, &y, &z);
            d = fmaxf(d, -Cylinder(x, y, z, h.pinRadius + h.clearance, h.lo - h.clearance, h.hi + h.clearance));
            // a pocket for a separate pin's head
            if (h.headRadius > 0.0f) d = fmaxf(d, -Cylinder(x, y, z, h.headRadius + h.clearance, h.headFrom - h.clearance, h.headTo + h.clearance));
        }
        return d;
    }

    float BodyField::Distance(Vector3 p) const {
        float d = m_Root >= 0 ? Eval(m_Root, p) : FLT_MAX;
        if (!m_Hardware.empty()) d = HardwareDistance(d, p);
        return fminf(d, 1e6f);
    }

    Vector3 BodyField::Gradient(Vector3 p, float e) const {
        Vector3 g = {
            Distance({ p.x + e, p.y, p.z }) - Distance({ p.x - e, p.y, p.z }),
            Distance({ p.x, p.y + e, p.z }) - Distance({ p.x, p.y - e, p.z }),
            Distance({ p.x, p.y, p.z + e }) - Distance({ p.x, p.y, p.z - e })
        };
        return Vector3Scale(g, 0.5f / e);
    }

    void BodyField::HardwareParts(const HingeHardware& h, Vector3 p, std::vector<float>* out) {
        float x, y, z;
        ToFrame(h, p, &x, &y, &z);
        float best = FLT_MAX;
        if (!h.pinOnly) {
            for (const Knuckle& k : h.own) best = fminf(best, Cylinder(x, y, z, h.radius, k.lo, k.hi));
            if (h.leafLength > 0.0f)
                for (const Knuckle& k : h.own)
                    best = fminf(best, Box(x, y, z, { 0.0f, -h.leafThickness * 0.5f, k.lo }, { h.leafLength, h.leafThickness * 0.5f, k.hi }));
        }
        if (h.pin) best = fminf(best, Cylinder(x, y, z, h.pinRadius, h.lo, h.hi));
        if (h.pinOnly && h.headRadius > 0.0f) best = fminf(best, Cylinder(x, y, z, h.headRadius, h.headFrom, h.headTo));
        out->push_back(best);
    }

    void BodyField::Nearest(Vector3 p, int* shape, int* hinge) const {
        float best = FLT_MAX;
        *shape = -1;
        *hinge = -1;
        for (size_t i = 0; i < m_Prims.size(); i++) {
            float d = fabsf(m_Prims[i].Distance(p));
            if (d < best) { best = d; *shape = m_PrimShape[i]; *hinge = -1; }
        }
        std::vector<float> parts;
        for (const HingeHardware& h : m_Hardware) {
            parts.clear();
            HardwareParts(h, p, &parts);
            if (fabsf(parts[0]) < best) { best = fabsf(parts[0]); *shape = -1; *hinge = h.hinge; }
        }
    }

    void BodyField::Deepest(Vector3 p, int* shape, int* hinge) const {
        float best = FLT_MAX;
        *shape = -1;
        *hinge = -1;
        for (size_t i = 0; i < m_Prims.size(); i++) {
            if (m_PrimOp[i] != Op::Union) continue;
            float d = m_Prims[i].Distance(p);
            if (d < best) { best = d; *shape = m_PrimShape[i]; *hinge = -1; }
        }
        std::vector<float> parts;
        for (const HingeHardware& h : m_Hardware) {
            parts.clear();
            HardwareParts(h, p, &parts);
            if (parts[0] < best) { best = parts[0]; *shape = -1; *hinge = h.hinge; }
        }
    }

    bool SameGeometry(const Design& a, const Design& b) {
        if (!(a.shapes == b.shapes) || !(a.groups == b.groups) || a.hinges.size() != b.hinges.size()) return false;
        for (size_t i = 0; i < a.hinges.size(); i++) {
            Hinge x = a.hinges[i], y = b.hinges[i];
            // motion only settings do not change the parts
            x.start = y.start;
            x.end = y.end;
            x.t0 = y.t0;
            x.t1 = y.t1;
            x.name = y.name;
            if (!(x == y)) return false;
        }
        return true;
    }
}
