#include "util/fastmath.hpp"
#include "field.hpp"
#include "util/utils.hpp"
#include "util/box.hpp"
#include <raymath.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <string>

namespace Bakugan {
    namespace {
        inline float SMin(float a, float b, float k) {
            float h = Fmax(k - fabsf(a - b), 0.0f) / k;
            return Fmin(a, b) - h * h * k * 0.25f;
        }

        // Capped cylinder about the frame's z axis: radius r, z in [lo, hi].
        inline float Cylinder(float x, float y, float z, float r, float lo, float hi) {
            float dr = sqrtf(x * x + y * y) - r;
            float dz = Fmax(lo - z, z - hi);
            float ox = Fmax(dr, 0.0f), oz = Fmax(dz, 0.0f);
            return Fmin(Fmax(dr, dz), 0.0f) + sqrtf(ox * ox + oz * oz);
        }

        inline float Box(float x, float y, float z, Vector3 lo, Vector3 hi) {
            float cx = (lo.x + hi.x) * 0.5f, cy = (lo.y + hi.y) * 0.5f, cz = (lo.z + hi.z) * 0.5f;
            float qx = fabsf(x - cx) - (hi.x - lo.x) * 0.5f;
            float qy = fabsf(y - cy) - (hi.y - lo.y) * 0.5f;
            float qz = fabsf(z - cz) - (hi.z - lo.z) * 0.5f;
            float mx = Fmax(qx, 0.0f), my = Fmax(qy, 0.0f), mz = Fmax(qz, 0.0f);
            return sqrtf(mx * mx + my * my + mz * mz) + Fmin(Fmax(qx, Fmax(qy, qz)), 0.0f);
        }

        // Point in the hinge frame: x along the leaf, y along the side, z along the axis.
        inline void ToFrame(const HingeHardware& h, Vector3 p, float* x, float* y, float* z) {
            Vector3 q = Vector3Subtract(p, h.pivot);
            *x = Vector3DotProduct(q, h.leaf);
            *y = Vector3DotProduct(q, h.side);
            *z = Vector3DotProduct(q, h.axis);
        }

        bool EmptyBox(BoundingBox b) { return b.max.x <= b.min.x && b.max.y <= b.min.y && b.max.z <= b.min.z; }

        BoundingBox HardwareBounds(const HingeHardware& h) {
            float rad = h.pinOnly ? Fmax(h.pinRadius, h.headRadius) : Fmax(h.radius + h.clearance, h.leafLength + h.leafThickness);
            Vector3 lo = { 1e30f, 1e30f, 1e30f }, hi = { -1e30f, -1e30f, -1e30f };
            for (int a = 0; a < 3; a++) {
                float w = (&h.axis.x)[a];
                float c = (&h.pivot.x)[a] + w * (h.lo + h.hi) * 0.5f;
                float e = fabsf(w) * (h.hi - h.lo) * 0.5f + sqrtf(Fmax(0.0f, 1.0f - w * w)) * rad + h.clearance;
                (&lo.x)[a] = c - e;
                (&hi.x)[a] = c + e;
            }
            return { lo, hi };
        }
    }

    // A sphere around the knuckles, leaves, clearances, bore and head pocket: a point outside it by more than the
    // current distance can't be changed by this hardware.
    void HingeHardware::Bound() {
        float lo2 = lo - clearance, hi2 = Fmax(hi, headTo) + clearance;
        float rad = Fmax(Fmax(radius + clearance, leafLength + leafThickness), headRadius + clearance);
        if (pinOnly) rad = Fmax(pinRadius, headRadius);
        centre = Vector3Add(pivot, Vector3Scale(axis, (lo2 + hi2) * 0.5f));
        float half = (hi2 - lo2) * 0.5f;
        reach = sqrtf(half * half + rad * rad) + 0.01f;
    }

    // ---- compile -----------------------------------------------------------------------------------------------------

    int BodyField::Compile(const Design& d, int groupId, Op op, float blend) {
        int self = (int)m_Nodes.size();
        m_Nodes.push_back(Node{});
        m_Nodes[self].op = op;
        m_Nodes[self].blend = blend;
        std::vector<int> kids;
        const bool top = d.GroupIndex(groupId) >= 0 && d.groups[d.GroupIndex(groupId)].parent < 0;
        for (int s : d.DirectShapes(groupId)) {
            if (d.shapes[s].id >= m_UpTo || (top && d.shapes[s].id >= m_LateFrom)) continue;
            int n = (int)m_Nodes.size();
            Node leaf;
            leaf.prim = (int)m_Prims.size();
            leaf.op = d.shapes[s].op;
            leaf.centre = d.shapes[s].sf.position;
            leaf.radius = Vector3Length(d.shapes[s].sf.LocalBounds().max);
            m_Prims.push_back(d.shapes[s].sf);
            m_Fast.emplace_back(d.shapes[s].sf);
            m_PrimShape.push_back(d.shapes[s].id);
            m_PrimOp.push_back(leaf.op);
            m_Nodes.push_back(leaf);
            kids.push_back(n);
        }
        for (int g : d.ChildGroups(groupId)) {
            const Group& grp = d.groups[g];
            if (grp.id >= m_UpTo || (top && IsLate(d, g))) continue;
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

    BodyField BodyField::Body(const Design& d, int bodyId) { return Make(d, bodyId, Kind::Full, INT_MAX); }

    // (a feature that only wraps older material, as Core does for a latch, is not late: what is in it was there)
    bool BodyField::IsLate(const Design& d, int g) const {
        if (d.groups[g].id < m_LateFrom) return false;
        for (int s : d.GroupShapes(d.groups[g].id))
            if (d.shapes[s].id < m_LateFrom) return false;
        return true;
    }

    int BodyField::ClearanceSamples() const {
        int n = 0;
        for (const Sweep& s : m_Sweeps)
            for (const Segment& g : s.segments) n += g.n + 1;
        return n;
    }

    namespace {
        // p turned by `angle` about the line through c along u (unit)
        inline Vector3 Turn(Vector3 p, Vector3 c, Vector3 u, float angle) {
            Vector3 x = Vector3Subtract(p, c);
            float cs = cosf(angle), sn = sinf(angle), along = Vector3DotProduct(u, x);
            Vector3 r = Vector3Add(Vector3Add(Vector3Scale(x, cs), Vector3Scale(Vector3CrossProduct(u, x), sn)), Vector3Scale(u, along * (1.0f - cs)));
            return Vector3Add(c, r);
        }
    }

    BodyField BodyField::Make(const Design& d, int bodyId, Kind kind, int upTo) {
        BodyField f;
        f.body = bodyId;
        int gi = d.GroupIndex(bodyId);
        if (gi < 0) return f;
        f.m_UpTo = upTo;
        f.m_Mask = kind == Kind::HingeTool;
        // what the body got after its clearances were made goes in after them (they don't cut it)
        if (kind != Kind::HingeTool)
            for (const Clearance& c : d.clearances)
                if (c.body == bodyId && (kind == Kind::Full || c.hingeOnly)) f.m_LateFrom = std::min(f.m_LateFrom, c.upTo);
        f.m_Root = f.Compile(d, bodyId, Op::Union, d.groups[gi].blend);
        if (f.m_LateFrom < INT_MAX) {
            for (int s : d.DirectShapes(bodyId)) {
                if (d.shapes[s].id < f.m_LateFrom || d.shapes[s].id >= upTo) continue;
                Node leaf;
                leaf.prim = (int)f.m_Prims.size();
                leaf.op = d.shapes[s].op;
                leaf.centre = d.shapes[s].sf.position;
                leaf.radius = Vector3Length(d.shapes[s].sf.LocalBounds().max);
                f.m_Prims.push_back(d.shapes[s].sf);
                f.m_Fast.emplace_back(d.shapes[s].sf);
                f.m_PrimShape.push_back(d.shapes[s].id);
                f.m_PrimOp.push_back(leaf.op);
                f.m_Late.push_back((int)f.m_Nodes.size());
                f.m_Nodes.push_back(leaf);
            }
            for (int g : d.ChildGroups(bodyId)) {
                const Group& grp = d.groups[g];
                if (!f.IsLate(d, g) || grp.id >= upTo) continue;
                size_t firstPrim = f.m_Prims.size();
                f.m_Late.push_back(f.Compile(d, grp.id, grp.op, grp.blend));
                if (grp.op == Op::Cut)
                    for (size_t k = firstPrim; k < f.m_Prims.size(); k++) f.m_PrimOp[k] = Op::Cut;
            }
        }
        f.m_UpTo = INT_MAX;
        if (f.m_Prims.empty()) { f.m_Root = -1; f.m_Nodes.clear(); f.m_Late.clear(); }
        BoundingBox box = d.BodyBounds(bodyId);
        bool hasBox = f.m_Root >= 0 && !EmptyBox(box);
        if (hasBox) box = Grow(box, d.groups[gi].blend + 0.5f);

        for (int h = 0; h < (int)d.hinges.size(); h++) {
            const Hinge& hg = d.hinges[h];
            bool parent = hg.parent == bodyId, child = hg.child == bodyId;
            if (!hg.hardware || (!parent && !child) || hg.id >= upTo) continue;
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
            // a removed knuckle is cleared in both bodies, leaving an open gap around the pin
            for (const Knuckle& k : ks) (k.child == child && !k.removed ? hw.own : hw.other).push_back(k);
            hw.pin = parent && hg.pin == PinStyle::Fused;
            hw.bore = !hw.pin;
            if (hg.pin == PinStyle::Separate) d.PinHead(h, &hw.headFrom, &hw.headTo, &hw.headRadius);
            hw.Bound();
            f.m_Hardware.push_back(hw);
            BoundingBox hb = HardwareBounds(hw);
            box = hasBox ? Merge(box, hb) : hb;
            hasBox = true;
        }
        f.bounds = hasBox ? Grow(box, 0.5f) : BoundingBox{ { 0, 0, 0 }, { 0, 0, 0 } };

        // clearances: the tool as it was, turning between its poses, in this body's frame
        if (kind != Kind::HingeTool && f.m_Root >= 0) {
            const BoundingBox own = f.bounds;
            for (const Clearance& c : d.clearances) {
                if (c.body != bodyId || (kind == Kind::Tool && !c.hingeOnly)) continue;
                const int ti = d.GroupIndex(c.tool);
                if (ti < 0 || d.groups[ti].parent >= 0 || c.tool == bodyId || c.poses.empty()) continue;
                auto tool = std::make_shared<BodyField>(Make(d, c.tool, c.hingeOnly ? Kind::HingeTool : Kind::Tool, c.upTo));
                if (tool->Empty()) continue;
                const float step = std::clamp(c.step, 0.01f, 5.0f);
                Sweep s;
                s.tool = tool;
                s.margin = c.margin + 0.5f * step;
                {
                    std::vector<float> shut(d.hinges.size());
                    for (size_t h = 0; h < d.hinges.size(); h++) shut[h] = d.hinges[h].end;
                    std::vector<Rigid> xf = d.TransformsAt(shut);
                    s.rest = xf[gi].Then(xf[ti].Inverse());
                }
                const Vector3 tc = Vector3Scale(Vector3Add(tool->bounds.min, tool->bounds.max), 0.5f);
                const float tr = 0.5f * Vector3Distance(tool->bounds.min, tool->bounds.max);
                // how far from a line this body reaches: what a turn about it moves its farthest point by
                auto reach = [&](Vector3 lc, Vector3 lu) {
                    float far = 0.0f;
                    for (int k = 0; k < 8; k++) {
                        Vector3 q = Vector3Subtract({ k & 1 ? own.max.x : own.min.x, k & 2 ? own.max.y : own.min.y, k & 4 ? own.max.z : own.min.z }, lc);
                        far = Fmax(far, Vector3Length(Vector3Subtract(q, Vector3Scale(lu, Vector3DotProduct(q, lu)))));
                    }
                    return far;
                };
                auto toToolAt = [&](const std::vector<float>& angles) {
                    std::vector<Rigid> xf = d.TransformsAt(angles);
                    return xf[gi].Then(xf[ti].Inverse());
                };
                auto single = [&](const Rigid& base) {
                    Segment g;
                    g.base = base;
                    s.segments.push_back(g);
                };
                std::vector<Rigid> keys;
                for (size_t k = 0; k < c.poses.size(); k++) keys.push_back(toToolAt(d.ClearanceAngles(c, (int)k)));
                bool open = false;                   // the last segment can be carried on
                for (size_t k = 0; k + 1 < keys.size(); k++) {
                    // this body's frame to itself: from pose k to pose k + 1, as the tool moves
                    const Rigid K = keys[k + 1].Then(keys[k].Inverse());
                    Quaternion q = QuaternionNormalize(K.rotation);
                    if (q.w < 0.0f) q = { -q.x, -q.y, -q.z, -q.w };
                    const float sh = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z);
                    const float angle = 2.0f * atan2f(sh, q.w);            // (acos is all rounding near w = 1)
                    float moved = 0.0f;                                    // how far K moves this body's farthest corner
                    for (int j = 0; j < 8; j++) {
                        Vector3 cq = { j & 1 ? own.max.x : own.min.x, j & 2 ? own.max.y : own.min.y, j & 4 ? own.max.z : own.min.z };
                        moved = Fmax(moved, Vector3Distance(K.Apply(cq), cq));
                    }
                    if (moved < 0.01f * step) continue;                    // still (as far as a sample could tell): nothing new
                    if (angle < 1e-5f) {
                        // (a slide, which hinges don't make: its two ends)
                        single(keys[k]);
                        single(keys[k + 1]);
                        open = false;
                        continue;
                    }
                    const Vector3 u = Vector3Scale({ q.x, q.y, q.z }, 1.0f / sh), t = K.position;
                    const float along = Vector3DotProduct(u, t);
                    if (fabsf(along) > 1e-3f) {
                        // not a turn about one line (two hinges at once): samples a step apart, each a pose of its own
                        const std::vector<float> a0 = d.ClearanceAngles(c, (int)k), a1 = d.ClearanceAngles(c, (int)k + 1);
                        const int m = std::clamp((int)ceilf(moved / step), 1, 512);
                        for (int i = 0; i <= m; i++) {
                            std::vector<float> ai(a0.size());
                            for (size_t h = 0; h < ai.size(); h++) ai[h] = a0[h] + (a1[h] - a0[h]) * i / m;
                            single(toToolAt(ai));
                        }
                        open = false;
                        continue;
                    }
                    // the line: (I - R) c = t, across u
                    const Vector3 tp = Vector3Subtract(t, Vector3Scale(u, along));
                    const Vector3 lc = Vector3Add(Vector3Scale(tp, 0.5f), Vector3Scale(Vector3CrossProduct(u, tp), 0.5f * cosf(0.5f * angle) / sh));
                    if (open) {
                        Segment& g = s.segments.back();
                        Vector3 off = Vector3Subtract(lc, g.c);
                        off = Vector3Subtract(off, Vector3Scale(g.u, Vector3DotProduct(off, g.u)));
                        if (Vector3DotProduct(g.u, u) > 1.0f - 1e-5f && Vector3Length(off) < 0.01f) { g.angle += angle; continue; }
                    }
                    Segment g;
                    g.base = keys[k];
                    g.c = lc;
                    g.u = u;
                    g.angle = angle;
                    s.segments.push_back(g);
                    open = true;
                }
                if (s.segments.empty()) single(keys[0]);
                // samples a step apart where this body is farthest from the line; spheres around the tool all the way
                Vector3 lo = { 1e30f, 1e30f, 1e30f }, hi = { -1e30f, -1e30f, -1e30f };
                for (Segment& g : s.segments) {
                    if (g.angle > 0.0f) g.n = std::clamp((int)ceilf(g.angle * reach(g.c, g.u) / step), 1, 1 << 16);
                    const Vector3 x0 = g.base.ApplyInverse(tc);      // the tool's centre at the segment's start, in this frame
                    Vector3 blo = x0, bhi = x0;
                    const int parts = g.n > 0 ? 16 : 0;
                    float sag = 0.0f;
                    for (int k = 1; k <= parts; k++) {
                        // at sample s the point p of this body is at base(turn(p)): the tool's centre at turn^-1
                        Vector3 x = Turn(x0, g.c, g.u, -g.angle * k / parts);
                        blo = Vector3Min(blo, x);
                        bhi = Vector3Max(bhi, x);
                    }
                    if (parts > 0) {
                        Vector3 r = Vector3Subtract(x0, g.c);
                        float rad = Vector3Length(Vector3Subtract(r, Vector3Scale(g.u, Vector3DotProduct(r, g.u))));
                        sag = rad * (1.0f - cosf(0.5f * g.angle / parts));
                    }
                    g.centre = Vector3Scale(Vector3Add(blo, bhi), 0.5f);
                    g.radius = 0.5f * Vector3Distance(blo, bhi) + sag + tr;
                    lo = Vector3Min(lo, Vector3SubtractValue(g.centre, g.radius));
                    hi = Vector3Max(hi, Vector3AddValue(g.centre, g.radius));
                }
                s.centre = Vector3Scale(Vector3Add(lo, hi), 0.5f);
                s.radius = 0.5f * Vector3Distance(lo, hi);
                f.m_Sweeps.push_back(std::move(s));
            }
        }
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
        hw.Bound();
        f.m_Hardware.push_back(hw);
        f.bounds = Grow(HardwareBounds(hw), 0.5f);
        return f;
    }

    // ---- evaluate ----------------------------------------------------------------------------------------------------

    float BodyField::Eval(int n, Vector3 p) const {
        const Node& nd = m_Nodes[n];
        if (nd.prim >= 0) return m_Fast[nd.prim].Distance(p);
        float u = FLT_MAX;
        bool any = false;
        for (int c : nd.children) {
            const Node& ch = m_Nodes[c];
            if (ch.op != Op::Union) continue;
            if (nd.blend <= 0.0f && any && ch.radius > 0.0f && Vector3Distance(p, ch.centre) - ch.radius > u) continue;
            float d = Eval(c, p);
            u = (any && nd.blend > 0.0f) ? SMin(u, d, nd.blend) : Fmin(u, d);
            any = true;
        }
        if (!any) return FLT_MAX;
        for (int c : nd.children) {
            const Node& ch = m_Nodes[c];
            if (ch.op != Op::Cut) continue;
            // a cut only matters where it reaches deeper than -u
            if (ch.radius > 0.0f && Vector3Distance(p, ch.centre) - ch.radius >= -u) continue;
            u = Fmax(u, -Eval(c, p));
        }
        return u;
    }

    void BodyField::Leaves(Vector3 p, float* out) const {
        for (size_t i = 0; i < m_Fast.size(); i++) out[i] = m_Fast[i].Distance(p);
    }

    bool BodyField::Differences(const BodyField& o, std::vector<int>* changed) const {
        changed->clear();
        if (!Simple() || !o.Simple() || m_Root != o.m_Root || m_Prims.size() != o.m_Prims.size() || m_Nodes.size() != o.m_Nodes.size()) return false;
        for (size_t n = 0; n < m_Nodes.size(); n++) {
            const Node &a = m_Nodes[n], &b = o.m_Nodes[n];
            if (a.prim != b.prim || a.op != b.op || a.blend != b.blend || a.children != b.children) return false;
        }
        for (size_t i = 0; i < m_Prims.size(); i++)
            if (m_Prims[i] != o.m_Prims[i]) changed->push_back((int)i);
        return true;
    }

    // Eval from the primitives' distances, with no culling (the culling only ever skips what can't change the answer:
    // a union's child farther than the union so far, a cut that can't reach as deep as it is inside)
    float BodyField::EvalLeaves(int n, const float* leaves) const {
        const Node& nd = m_Nodes[n];
        if (nd.prim >= 0) return leaves[nd.prim];
        float u = FLT_MAX;
        bool any = false;
        for (int c : nd.children) {
            if (m_Nodes[c].op != Op::Union) continue;
            float d = EvalLeaves(c, leaves);
            u = (any && nd.blend > 0.0f) ? SMin(u, d, nd.blend) : Fmin(u, d);
            any = true;
        }
        if (!any) return FLT_MAX;
        for (int c : nd.children)
            if (m_Nodes[c].op == Op::Cut) u = Fmax(u, -EvalLeaves(c, leaves));
        return u;
    }

    float BodyField::Recombine(const float* leaves, Vector3 p, const std::vector<int>& changed, float* scratch) const {
        std::copy(leaves, leaves + m_Fast.size(), scratch);
        for (int i : changed) scratch[i] = m_Fast[i].Distance(p);
        return Fmin(EvalLeaves(m_Root, scratch), 1e6f);
    }

    float BodyField::HardwareDistance(float d, Vector3 p) const {
        // leaves join the body, then the other body's knuckles are cleared, then this body's knuckles and pin go in,
        // and finally the pin bore is drilled. Each hinge is skipped where its bounding sphere shows it can't matter:
        // material only where it lies closer than d, cuts only where they reach deeper than -d.
        float lb[16];
        size_t n = std::min(m_Hardware.size(), (size_t)16);
        for (size_t i = 0; i < m_Hardware.size(); i++)
            if (i < n) lb[i] = Vector3Distance(p, m_Hardware[i].centre) - m_Hardware[i].reach;
        auto far = [&](size_t i) { return i < n ? lb[i] : -FLT_MAX; };
        for (size_t i = 0; i < m_Hardware.size(); i++) {
            const HingeHardware& h = m_Hardware[i];
            if (h.pinOnly || h.leafLength <= 0.0f || far(i) > d) continue;
            float x, y, z;
            ToFrame(h, p, &x, &y, &z);
            for (const Knuckle& k : h.own)
                d = Fmin(d, Box(x, y, z, { 0.0f, -h.leafThickness * 0.5f, k.lo }, { h.leafLength, h.leafThickness * 0.5f, k.hi }));
        }
        for (size_t i = 0; i < m_Hardware.size(); i++) {
            const HingeHardware& h = m_Hardware[i];
            if (h.pinOnly || far(i) >= -d) continue;
            float x, y, z;
            ToFrame(h, p, &x, &y, &z);
            for (const Knuckle& k : h.other)
                d = Fmax(d, -Cylinder(x, y, z, h.radius + h.clearance, k.lo - h.clearance, k.hi + h.clearance));
        }
        for (size_t i = 0; i < m_Hardware.size(); i++) {
            const HingeHardware& h = m_Hardware[i];
            if (far(i) > d) continue;
            float x, y, z;
            ToFrame(h, p, &x, &y, &z);
            if (!h.pinOnly)
                for (const Knuckle& k : h.own) d = Fmin(d, Cylinder(x, y, z, h.radius, k.lo, k.hi));
            if (h.pin) d = Fmin(d, Cylinder(x, y, z, h.pinRadius, h.lo, h.hi));
            if (h.pinOnly && h.headRadius > 0.0f) d = Fmin(d, Cylinder(x, y, z, h.headRadius, h.headFrom, h.headTo));
        }
        for (size_t i = 0; i < m_Hardware.size(); i++) {
            const HingeHardware& h = m_Hardware[i];
            if (!h.bore || far(i) >= -d) continue;
            float x, y, z;
            ToFrame(h, p, &x, &y, &z);
            d = Fmax(d, -Cylinder(x, y, z, h.pinRadius + h.clearance, h.lo - h.clearance, h.hi + h.clearance));
            // a pocket for a separate pin's head
            if (h.headRadius > 0.0f) d = Fmax(d, -Cylinder(x, y, z, h.headRadius + h.clearance, h.headFrom - h.clearance, h.headTo + h.clearance));
        }
        return d;
    }

    float BodyField::ZoneDistance(Vector3 p) const {
        float z = FLT_MAX;
        for (const HingeHardware& h : m_Hardware) {
            if (h.pinOnly || h.leafLength <= 0.0f) continue;
            if (Vector3Distance(p, h.centre) - h.reach - KnuckleZone > z) continue;
            float x, y, w;
            ToFrame(h, p, &x, &y, &w);
            for (const Knuckle& k : h.own) z = Fmin(z, Cylinder(x, y, w, h.leafLength + KnuckleZone, k.lo, k.hi));
        }
        return z;
    }

    // Each clearance cuts where its tool, anywhere on its way, is nearer than its margin; but not next to this body's
    // own knuckles. The margin is room around what the tool really cuts into, not around everything it passes:
    //  - it is the clearance only where the point has moved that far from where it rests against the tool when closed
    //    (less, nearer that): faces that just touch closed get no gap;
    //  - and only where the tool comes nearer than the point is deep in the body (then it cuts into the body within
    //    that distance): a face the tool just slides past keeps its edge. With the point d deep and the tool v away,
    //    the cut is where v < min(margin, d): w = max(v - margin, (v - d) / 2) < 0 (halved, as v - d changes twice
    //    as fast as a distance). Full clearance from twice its depth in, none at the body's surface. (So where the
    //    tool cuts in right at the surface, between two samples it may reach up to half a step past the cut there:
    //    0.05 mm at most, where giving that much room would cost the edge of every face it slides past.) Along a segment the samples are a fixed turn apart, so the point moves (relative to the tool) by
    // at most r * angle / n from one to the next, r its distance from the line; the tool's field changes no faster
    // than that, so from a sample where the tool was v away, the next that could be nearer than the bar (the margin
    // reached, or the nearest so far) is (v - bar) / that further on. Segments and whole clearances whose spheres are
    // too far are skipped.
    // Exact only near the cut's surface: within SweepBand of the margin. Further in or out, the search stops once that
    // much is certain, and the value is then nearer zero than the truth (never further), which is all a field needs.
    static constexpr float SweepBand = 1.0f;

    float BodyField::SweepDistance(float d, Vector3 p) const {
        // (in terms of w = the tool's distance less the margin there: cut where w < -d)
        const float d0 = d;
        float zone = FLT_MAX;
        bool zoned = false;
        for (const Sweep& s : m_Sweeps) {
            const float needed = -d, far = SweepBand, deep = -SweepBand;
            if (Vector3Distance(p, s.centre) - s.radius - s.margin >= needed) continue;
            const Vector3 rest = s.rest.Apply(p);         // where the point is against the tool when closed
            float w = FLT_MAX;
            for (const Segment& g : s.segments) {
                if (w <= deep) break;
                if (Vector3Distance(p, g.centre) - g.radius - s.margin >= Fmin(Fmin(w, needed), far)) continue;
                const Vector3 x = Vector3Subtract(p, g.c);
                const float r = Vector3Length(Vector3Subtract(x, Vector3Scale(g.u, Vector3DotProduct(x, g.u))));
                // per sample, the tool's distance changes by at most ds, and the margin (how far the point has moved
                // from where it rests, up to the clearance) by as much again, until it has moved the clearance
                const float ds = g.n > 0 ? r * g.angle / g.n : 0.0f;
                for (int i = 0; i <= g.n;) {
                    const Vector3 q = g.base.Apply(g.n > 0 ? Turn(p, g.c, g.u, g.angle * i / g.n) : p);
                    const float fromRest = Vector3Distance(q, rest);
                    const float v = s.tool->Distance(q);
                    const float wi = Fmax(v - Fmin(s.margin, fromRest), 0.5f * (v + d0));
                    w = Fmin(w, wi);
                    if (ds <= 1e-7f || w <= deep) break;
                    const float room = Fmax(0.0f, wi - Fmin(Fmin(w, needed), far));
                    const float ahead = room <= fromRest - s.margin ? room / ds : room / (2.0f * ds);
                    i += 1 + (int)Fmin(ahead, 1e7f);
                }
            }
            w = Fmin(w, far);                          // what was skipped is at least this far
            if (w >= needed) continue;
            if (!zoned) { zone = ZoneDistance(p); zoned = true; }
            d = Fmax(d, Fmin(-w, zone));
        }
        return d;
    }

    float BodyField::Distance(Vector3 p) const {
        float d = m_Root >= 0 ? Eval(m_Root, p) : FLT_MAX;
        if (m_Mask) d = Fmax(d, ZoneDistance(p));
        // (outside the body, clearances can only take it further away: its distance is a safe answer as it is)
        if (!m_Sweeps.empty() && d < 0.0f) d = SweepDistance(d, p);
        if (!m_Late.empty()) {
            const float blend = m_Root >= 0 ? m_Nodes[m_Root].blend : 0.0f;
            for (int n : m_Late)
                if (m_Nodes[n].op == Op::Union) {
                    float v = Eval(n, p);
                    d = blend > 0.0f && d < 1e30f ? SMin(d, v, blend) : Fmin(d, v);
                }
            for (int n : m_Late)
                if (m_Nodes[n].op == Op::Cut) d = Fmax(d, -Eval(n, p));
        }
        if (!m_Hardware.empty()) d = HardwareDistance(d, p);
        return Fmin(d, 1e6f);
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
            for (const Knuckle& k : h.own) best = Fmin(best, Cylinder(x, y, z, h.radius, k.lo, k.hi));
            if (h.leafLength > 0.0f)
                for (const Knuckle& k : h.own)
                    best = Fmin(best, Box(x, y, z, { 0.0f, -h.leafThickness * 0.5f, k.lo }, { h.leafLength, h.leafThickness * 0.5f, k.hi }));
        }
        if (h.pin) best = Fmin(best, Cylinder(x, y, z, h.pinRadius, h.lo, h.hi));
        if (h.pinOnly && h.headRadius > 0.0f) best = Fmin(best, Cylinder(x, y, z, h.headRadius, h.headFrom, h.headTo));
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
        if (!(a.shapes == b.shapes) || !(a.groups == b.groups) || !(a.clearances == b.clearances) || a.hinges.size() != b.hinges.size()) return false;
        for (size_t i = 0; i < a.hinges.size(); i++) {
            Hinge x = a.hinges[i], y = b.hinges[i];
            // motion only settings do not change the parts
            x.start = y.start;
            x.end = y.end;
            x.t0 = y.t0;
            x.t1 = y.t1;
            x.ownOpening = y.ownOpening;
            x.o0 = y.o0;
            x.o1 = y.o1;
            x.name = y.name;
            if (!(x == y)) return false;
        }
        return true;
    }

    // ---- GLSL ----------------------------------------------------------------------------------------------------------

    static int Push(std::vector<Vector4>& U, Vector4 v) {
        U.push_back(v);
        return (int)U.size() - 1;
    }

    static std::string Ref(int i) { return "U[" + std::to_string(i) + "]"; }

    // One child of a group, folded into the group's accumulator `acc`. A child whose bounding sphere is further away
    // than |acc| (+ the group's blend) can't change it, so its lower bound stands in for it (exactly like Eval).
    void BodyField::GlslNode(int n, const std::string& acc, const std::string& blend, std::string& code, std::vector<Vector4>& U,
                             int& vars) const {
        const Node& nd = m_Nodes[n];
        bool cut = nd.op == Op::Cut;
        auto combine = [&](const std::string& x) {
            return cut ? acc + " = max(" + acc + ", -" + x + ");" : acc + " = opU(" + acc + ", " + x + ", " + blend + ");";
        };
        if (nd.prim >= 0) {
            const SuperFrustum& f = m_Prims[nd.prim];
            int i = Push(U, { f.position.x, f.position.y, f.position.z, nd.radius });
            Push(U, { f.rotation.x, f.rotation.y, f.rotation.z, f.rotation.w });
            Push(U, { f.size.x, f.size.y, f.size.z, f.rounding });
            Push(U, { f.taper, f.bulge, f.dilation, f.onion });
            code += "{ float lb = length(p - " + Ref(i) + ".xyz) - " + Ref(i) + ".w; float x = lb > abs(" + acc + ") + " + blend +
                    " ? lb : sf(p, " + std::to_string(i) + "); " + combine("x") + " }\n";
            return;
        }
        std::string g = "g" + std::to_string(vars++);
        int c = Push(U, { nd.centre.x, nd.centre.y, nd.centre.z, nd.radius });
        int k = Push(U, { nd.blend, 0, 0, 0 });
        code += "{ float " + g + " = 1e9; float lb = length(p - " + Ref(c) + ".xyz) - " + Ref(c) + ".w;\n";
        code += "if (" + Ref(c) + ".w > 0.0 && lb > abs(" + acc + ") + " + blend + ") " + g + " = lb; else {\n";
        for (int ch : nd.children)
            if (m_Nodes[ch].op == Op::Union) GlslNode(ch, g, Ref(k) + ".x", code, U, vars);
        for (int ch : nd.children)
            if (m_Nodes[ch].op == Op::Cut) GlslNode(ch, g, Ref(k) + ".x", code, U, vars);
        code += "}\n" + combine(g) + " }\n";
    }

    void BodyField::GlslHardware(std::string& code, std::vector<Vector4>& U) const {
        // per hinge: [0] bounding sphere, [1] pivot + radius, [2] leaf + pin radius, [3] side + clearance,
        // [4] axis + leaf length, [5] leaf thickness, lo, hi, head from, [6] head to, head radius, then one vec4 per knuckle
        struct Slots { int base; std::vector<int> own, other; };
        std::vector<Slots> slots;
        for (const HingeHardware& h : m_Hardware) {
            Slots s;
            s.base = Push(U, { h.centre.x, h.centre.y, h.centre.z, h.reach });
            Push(U, { h.pivot.x, h.pivot.y, h.pivot.z, h.radius });
            Push(U, { h.leaf.x, h.leaf.y, h.leaf.z, h.pinRadius });
            Push(U, { h.side.x, h.side.y, h.side.z, h.clearance });
            Push(U, { h.axis.x, h.axis.y, h.axis.z, h.leafLength });
            Push(U, { h.leafThickness, h.lo, h.hi, h.headFrom });
            Push(U, { h.headTo, h.headRadius, 0, 0 });
            for (const Knuckle& k : h.own) s.own.push_back(Push(U, { k.lo, k.hi, 0, 0 }));
            for (const Knuckle& k : h.other) s.other.push_back(Push(U, { k.lo, k.hi, 0, 0 }));
            slots.push_back(s);
        }
        auto far = [&](const Slots& s) { return "(length(p - " + Ref(s.base) + ".xyz) - " + Ref(s.base) + ".w)"; };
        auto frame = [&](const Slots& s) { return "vec3 f = hwFrame(p, " + std::to_string(s.base) + ");"; };
        // leaves
        for (size_t i = 0; i < m_Hardware.size(); i++) {
            const HingeHardware& h = m_Hardware[i];
            const Slots& s = slots[i];
            if (h.pinOnly || h.leafLength <= 0.0f || h.own.empty()) continue;
            std::string b = std::to_string(s.base);
            code += "if (!(" + far(s) + " > d)) { " + frame(s) + "\n";
            for (int k : s.own)
                code += "  d = min(d, sdBox(f, vec3(0.0, -U[" + b + "+5].x * 0.5, " + Ref(k) + ".x), vec3(U[" + b + "+4].w, U[" + b + "+5].x * 0.5, " +
                        Ref(k) + ".y)));\n";
            code += "}\n";
        }
        // clearance around the other body's knuckles
        for (size_t i = 0; i < m_Hardware.size(); i++) {
            const HingeHardware& h = m_Hardware[i];
            const Slots& s = slots[i];
            if (h.pinOnly || h.other.empty()) continue;
            std::string b = std::to_string(s.base);
            code += "if (!(" + far(s) + " >= -d)) { " + frame(s) + "\n";
            for (int k : s.other)
                code += "  d = max(d, -sdCyl(f, U[" + b + "+1].w + U[" + b + "+3].w, " + Ref(k) + ".x - U[" + b + "+3].w, " + Ref(k) + ".y + U[" + b +
                        "+3].w));\n";
            code += "}\n";
        }
        // own knuckles, pin, separate pin head
        for (size_t i = 0; i < m_Hardware.size(); i++) {
            const HingeHardware& h = m_Hardware[i];
            const Slots& s = slots[i];
            bool knuckles = !h.pinOnly && !h.own.empty(), head = h.pinOnly && h.headRadius > 0.0f;
            if (!knuckles && !h.pin && !head) continue;
            std::string b = std::to_string(s.base);
            code += "if (!(" + far(s) + " > d)) { " + frame(s) + "\n";
            if (knuckles)
                for (int k : s.own) code += "  d = min(d, sdCyl(f, U[" + b + "+1].w, " + Ref(k) + ".x, " + Ref(k) + ".y));\n";
            if (h.pin) code += "  d = min(d, sdCyl(f, U[" + b + "+2].w, U[" + b + "+5].y, U[" + b + "+5].z));\n";
            if (head) code += "  d = min(d, sdCyl(f, U[" + b + "+6].y, U[" + b + "+5].w, U[" + b + "+6].x));\n";
            code += "}\n";
        }
        // bore and head pocket
        for (size_t i = 0; i < m_Hardware.size(); i++) {
            const HingeHardware& h = m_Hardware[i];
            const Slots& s = slots[i];
            if (!h.bore) continue;
            std::string b = std::to_string(s.base);
            code += "if (!(" + far(s) + " >= -d)) { " + frame(s) + "\n";
            code += "  d = max(d, -sdCyl(f, U[" + b + "+2].w + U[" + b + "+3].w, U[" + b + "+5].y - U[" + b + "+3].w, U[" + b + "+5].z + U[" + b +
                    "+3].w));\n";
            if (h.headRadius > 0.0f)
                code += "  d = max(d, -sdCyl(f, U[" + b + "+6].y + U[" + b + "+3].w, U[" + b + "+5].w - U[" + b + "+3].w, U[" + b + "+6].x + U[" + b +
                        "+3].w));\n";
            code += "}\n";
        }
    }

    // The zone next to its own knuckles (ZoneDistance) into `out`, which starts at 1e9.
    void BodyField::GlslZone(const std::string& out, std::string& code, std::vector<Vector4>& U) const {
        for (const HingeHardware& h : m_Hardware) {
            if (h.pinOnly || h.leafLength <= 0.0f || h.own.empty()) continue;
            int b = Push(U, { h.centre.x, h.centre.y, h.centre.z, h.reach + KnuckleZone });
            Push(U, { h.pivot.x, h.pivot.y, h.pivot.z, h.leafLength + KnuckleZone });
            Push(U, { h.leaf.x, h.leaf.y, h.leaf.z, 0 });
            Push(U, { h.side.x, h.side.y, h.side.z, 0 });
            Push(U, { h.axis.x, h.axis.y, h.axis.z, 0 });
            std::string bs = std::to_string(b);
            code += "if (!(length(p - " + Ref(b) + ".xyz) - " + Ref(b) + ".w > " + out + ")) { vec3 f = hwFrame(p, " + bs + ");\n";
            for (const Knuckle& k : h.own) {
                int r = Push(U, { k.lo, k.hi, 0, 0 });
                code += "  " + out + " = min(" + out + ", sdCyl(f, U[" + bs + "+1].w, " + Ref(r) + ".x, " + Ref(r) + ".y));\n";
            }
            code += "}\n";
        }
    }

    std::string BodyField::Glsl(std::vector<Vector4>* params) const {
        std::vector<Vector4>& U = *params;
        std::string code = "float d = 1e9;\n";
        int vars = 0;
        GlslInto(code, U, vars);
        code += "return min(d, 1e6);\n";
        return code;
    }

    // The field into `d` (declared by the caller, 1e9), at `p`. A clearance's tool goes in once, inside a loop over its
    // poses with p moved into the tool's frame (shadowing p and d in an inner block).
    void BodyField::GlslInto(std::string& code, std::vector<Vector4>& U, int& vars) const {
        if (m_Root >= 0) {
            // the root group is the body itself: its children fold straight into d
            const Node& root = m_Nodes[m_Root];
            int k = Push(U, { root.blend, 0, 0, 0 });
            for (int ch : root.children)
                if (m_Nodes[ch].op == Op::Union) GlslNode(ch, "d", Ref(k) + ".x", code, U, vars);
            for (int ch : root.children)
                if (m_Nodes[ch].op == Op::Cut) GlslNode(ch, "d", Ref(k) + ".x", code, U, vars);
        }
        if (m_Mask) {
            std::string z = "z" + std::to_string(vars++);
            code += "{ float " + z + " = 1e9;\n";
            GlslZone(z, code, U);
            code += "d = max(d, " + z + "); }\n";
        }
        if (!m_Sweeps.empty()) {
            std::string zone = "zn" + std::to_string(vars++);
            code += "if (d < 0.0) { float " + zone + " = 1e9; float d0s = d;\n";
            GlslZone(zone, code, U);
            for (const Sweep& s : m_Sweeps) {
                const std::string id = std::to_string(vars++), w = "w" + id, need = "nd" + id, sg = "sg" + id, b = "b" + id, x = "x" + id,
                                  i = "i" + id, far = "fr" + id, deep = "dp" + id, rest = "rs" + id;
                // [0] centre, radius around it all; [1] margin; [2] rest rotation; [3] rest translation; per segment:
                // [0] centre, radius, [1] base rotation, [2] base translation, n, [3] c, angle, [4] u
                int c = Push(U, { s.centre.x, s.centre.y, s.centre.z, s.radius });
                Push(U, { s.margin, 0, 0, 0 });
                Push(U, { s.rest.rotation.x, s.rest.rotation.y, s.rest.rotation.z, s.rest.rotation.w });
                Push(U, { s.rest.position.x, s.rest.position.y, s.rest.position.z, 0 });
                int first = (int)U.size();
                for (const Segment& g : s.segments) {
                    U.push_back({ g.centre.x, g.centre.y, g.centre.z, g.radius });
                    U.push_back({ g.base.rotation.x, g.base.rotation.y, g.base.rotation.z, g.base.rotation.w });
                    U.push_back({ g.base.position.x, g.base.position.y, g.base.position.z, (float)g.n });
                    U.push_back({ g.c.x, g.c.y, g.c.z, g.angle });
                    U.push_back({ g.u.x, g.u.y, g.u.z, 0 });
                }
                const std::string M = Ref(c + 1) + ".x", band = std::to_string(SweepBand);
                code += "{ float " + need + " = -d, " + far + " = " + band + ", " + deep + " = -" + band + ";\n";
                code += "if (length(p - " + Ref(c) + ".xyz) - " + Ref(c) + ".w - " + M + " < " + need + ") { float " + w + " = 1e9;\n";
                code += "vec3 " + rest + " = p + 2.0 * cross(" + Ref(c + 2) + ".xyz, cross(" + Ref(c + 2) + ".xyz, p) + " + Ref(c + 2) + ".w * p) + " + Ref(c + 3) +
                        ".xyz;\n";
                code += "for (int " + sg + " = 0; " + sg + " < " + std::to_string(s.segments.size()) + "; " + sg + "++) { int " + b + " = " +
                        std::to_string(first) + " + 5 * " + sg + ";\n";
                code += "  if (" + w + " <= " + deep + ") break;\n";
                code += "  if (length(p - U[" + b + "].xyz) - U[" + b + "].w - " + M + " >= min(min(" + w + ", " + need + "), " + far + ")) continue;\n";
                code += "  int n" + id + " = int(U[" + b + " + 2].w); float an" + id + " = U[" + b + " + 3].w; vec3 c" + id + " = U[" + b + " + 3].xyz, u" + id +
                        " = U[" + b + " + 4].xyz; vec4 r" + id + " = U[" + b + " + 1];\n";
                code += "  vec3 " + x + " = p - c" + id + "; float al" + id + " = dot(u" + id + ", " + x + "); float ds" + id + " = n" + id +
                        " > 0 ? length(" + x + " - u" + id + " * al" + id + ") * an" + id + " / float(n" + id + ") : 0.0;\n";
                code += "  int " + i + " = 0;\n";
                code += "  while (" + i + " <= n" + id + ") {\n";
                code += "    float a" + id + " = n" + id + " > 0 ? an" + id + " * float(" + i + ") / float(n" + id + ") : 0.0; float cs" + id + " = cos(a" + id +
                        "), sn" + id + " = sin(a" + id + ");\n";
                code += "    vec3 tq" + id + " = c" + id + " + " + x + " * cs" + id + " + cross(u" + id + ", " + x + ") * sn" + id + " + u" + id + " * al" + id +
                        " * (1.0 - cs" + id + ");\n";
                code += "    vec3 q" + id + " = tq" + id + " + 2.0 * cross(r" + id + ".xyz, cross(r" + id + ".xyz, tq" + id + ") + r" + id + ".w * tq" + id + ") + U[" + b +
                        " + 2].xyz;\n";
                code += "    float v" + id + ", fr" + id + "x = length(q" + id + " - " + rest + ");\n";
                code += "    { vec3 p = q" + id + "; float d = 1e9;\n";
                s.tool->GlslInto(code, U, vars);
                code += "    v" + id + " = max(min(d, 1e6) - min(" + M + ", fr" + id + "x), 0.5 * (min(d, 1e6) + d0s)); }\n";
                code += "    " + w + " = min(" + w + ", v" + id + ");\n";
                code += "    if (ds" + id + " <= 1e-7 || " + w + " <= " + deep + ") break;\n";
                code += "    float rm" + id + " = max(v" + id + " - min(min(" + w + ", " + need + "), " + far + "), 0.0);\n";
                code += "    " + i + " += 1 + int(min((rm" + id + " <= fr" + id + "x - " + M + " ? rm" + id + " : 0.5 * rm" + id + ") / ds" + id + ", 1e7));\n";
                code += "  }\n}\n";
                code += w + " = min(" + w + ", " + far + ");\n";
                code += "if (" + w + " < " + need + ") d = max(d, min(-" + w + ", " + zone + "));\n";
                code += "} }\n";
            }
            code += "}\n";
        }
        if (!m_Late.empty()) {
            int k = Push(U, { m_Root >= 0 ? m_Nodes[m_Root].blend : 0.0f, 0, 0, 0 });
            for (int n : m_Late)
                if (m_Nodes[n].op == Op::Union) GlslNode(n, "d", Ref(k) + ".x", code, U, vars);
            for (int n : m_Late)
                if (m_Nodes[n].op == Op::Cut) GlslNode(n, "d", Ref(k) + ".x", code, U, vars);
        }
        if (!m_Hardware.empty()) GlslHardware(code, U);
    }
}
