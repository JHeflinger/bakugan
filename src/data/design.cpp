#include "design.hpp"
#include "util/utils.hpp"
#include <raymath.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <sstream>

namespace Bakugan {
    static bool SameVec(Vector3 a, Vector3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

    static bool SameColor(Color a, Color b) { return a.r == b.r && a.g == b.g && a.b == b.b && a.a == b.a; }

    bool Shape::operator==(const Shape& o) const {
        return id == o.id && name == o.name && group == o.group && sf == o.sf && op == o.op && SameColor(color, o.color);
    }

    bool Group::operator==(const Group& o) const {
        return id == o.id && name == o.name && parent == o.parent && op == o.op && blend == o.blend && SameColor(color, o.color);
    }

    bool Hinge::operator==(const Hinge& o) const {
        return id == o.id && name == o.name && parent == o.parent && child == o.child && SameVec(pivot, o.pivot) &&
               SameVec(axis, o.axis) && start == o.start && end == o.end && t0 == o.t0 && t1 == o.t1 && ownOpening == o.ownOpening &&
               o0 == o.o0 && o1 == o.o1 && hardware == o.hardware &&
               length == o.length && radius == o.radius && pinRadius == o.pinRadius && clearance == o.clearance &&
               knuckles == o.knuckles && pin == o.pin && leafLength == o.leafLength && leafThickness == o.leafThickness &&
               autoLeaves == o.autoLeaves && leafAngle[0] == o.leafAngle[0] && leafAngle[1] == o.leafAngle[1] &&
               removed == o.removed && swapKnuckles == o.swapKnuckles && headDiameter == o.headDiameter;
    }

    bool Design::operator==(const Design& o) const {
        return shapes == o.shapes && groups == o.groups && hinges == o.hinges && nextId == o.nextId;
    }

    // ---- lookups -----------------------------------------------------------------------------------------------------

    int Design::ShapeIndex(int id) const {
        for (int i = 0; i < (int)shapes.size(); i++)
            if (shapes[i].id == id) return i;
        return -1;
    }

    int Design::GroupIndex(int id) const {
        for (int i = 0; i < (int)groups.size(); i++)
            if (groups[i].id == id) return i;
        return -1;
    }

    int Design::HingeIndex(int id) const {
        for (int i = 0; i < (int)hinges.size(); i++)
            if (hinges[i].id == id) return i;
        return -1;
    }

    int Design::BodyOf(int groupId) const {
        int g = GroupIndex(groupId), guard = 0;
        while (g >= 0 && groups[g].parent >= 0 && guard++ <= (int)groups.size()) {
            int up = GroupIndex(groups[g].parent);
            if (up < 0) break;
            g = up;
        }
        return g >= 0 ? groups[g].id : -1;
    }

    int Design::ShapeBody(int i) const {
        return i >= 0 && i < (int)shapes.size() ? BodyOf(shapes[i].group) : -1;
    }

    std::vector<int> Design::Bodies() const {
        std::vector<int> out;
        for (int i = 0; i < (int)groups.size(); i++)
            if (groups[i].parent < 0) out.push_back(i);
        return out;
    }

    bool Design::Inside(int groupId, int ancestorId) const {
        int g = groupId, guard = 0;
        while (g >= 0 && guard++ <= (int)groups.size()) {
            if (g == ancestorId) return true;
            int i = GroupIndex(g);
            if (i < 0) return false;
            g = groups[i].parent;
        }
        return false;
    }

    std::vector<int> Design::DirectShapes(int groupId) const {
        std::vector<int> out;
        for (int i = 0; i < (int)shapes.size(); i++)
            if (shapes[i].group == groupId) out.push_back(i);
        return out;
    }

    std::vector<int> Design::ChildGroups(int groupId) const {
        std::vector<int> out;
        for (int i = 0; i < (int)groups.size(); i++)
            if (groups[i].parent == groupId && groups[i].id != groupId) out.push_back(i);
        return out;
    }

    std::vector<int> Design::GroupShapes(int groupId) const {
        std::vector<int> out;
        for (int i = 0; i < (int)shapes.size(); i++)
            if (Inside(shapes[i].group, groupId)) out.push_back(i);
        return out;
    }

    std::string Design::GroupPath(int groupId) const {
        std::string path;
        int g = groupId, guard = 0;
        while (g >= 0 && guard++ <= (int)groups.size()) {
            int i = GroupIndex(g);
            if (i < 0) break;
            path = path.empty() ? groups[i].name : groups[i].name + " / " + path;
            g = groups[i].parent;
        }
        return path;
    }

    int Design::ParentHinge(int bodyId) const {
        for (int i = 0; i < (int)hinges.size(); i++)
            if (hinges[i].child == bodyId) return i;
        return -1;
    }

    bool Design::CanHinge(int parent, int child, int ignore) const {
        int pi = GroupIndex(parent), ci = GroupIndex(child);
        if (pi < 0 || ci < 0 || parent == child) return false;
        if (groups[pi].parent >= 0 || groups[ci].parent >= 0) return false;   // bodies only
        for (int i = 0; i < (int)hinges.size(); i++)
            if (i != ignore && hinges[i].child == child) return false;          // one hinge turns each body
        // the parent must not hang from the child
        int b = parent, guard = 0;
        while (b >= 0 && guard++ <= (int)hinges.size()) {
            if (b == child) return false;
            int h = -1;
            for (int i = 0; i < (int)hinges.size(); i++)
                if (i != ignore && hinges[i].child == b) h = i;
            b = h >= 0 ? hinges[h].parent : -1;
        }
        return true;
    }

    // ---- kinematics --------------------------------------------------------------------------------------------------

    static float Window(float t, float a, float b, bool ease) {
        float span = b - a;
        float f = span > 1e-4f ? (t - a) / span : (t >= b ? 1.0f : 0.0f);
        f = Clamp(f, 0.0f, 1.0f);
        if (ease) f = f * f * (3.0f - 2.0f * f);
        return f;
    }

    float Design::HingeFraction(int h, float t, bool ease, Motion m) const {
        const Hinge& hg = hinges[h];
        if (m == Motion::Opening && hg.ownOpening) return 1.0f - Window(1.0f - t, hg.o0, hg.o1, ease);
        return Window(t, hg.t0, hg.t1, ease);
    }

    float Design::HingeAngle(int h, float t, bool ease, Motion m) const {
        const Hinge& hg = hinges[h];
        return hg.start + (hg.end - hg.start) * HingeFraction(h, t, ease, m);
    }

    bool Design::HasOwnOpening() const {
        for (const Hinge& h : hinges)
            if (h.ownOpening) return true;
        return false;
    }

    std::vector<Rigid> Design::Transforms(float t, bool ease, Motion m) const {
        int n = (int)groups.size();
        std::vector<Rigid> body(n);
        std::vector<char> state(n, 0);
        std::function<Rigid(int)> solve = [&](int g) -> Rigid {
            if (state[g] == 2) return body[g];
            if (state[g] == 1) return Rigid{};    // a loop (Repair removes those)
            state[g] = 1;
            Rigid r;
            int h = ParentHinge(groups[g].id);
            if (h >= 0) {
                const Hinge& hg = hinges[h];
                int p = GroupIndex(hg.parent);
                Rigid parent = p >= 0 ? solve(p) : Rigid{};
                Vector3 axis = Vector3Length(hg.axis) > 1e-6f ? hg.axis : Vector3{ 1, 0, 0 };
                r = Rigid::AboutAxis(hg.pivot, axis, HingeAngle(h, t, ease, m)).Then(parent);
            }
            state[g] = 2;
            body[g] = r;
            return r;
        };
        std::vector<Rigid> xf(n);
        for (int g = 0; g < n; g++) {
            int b = GroupIndex(BodyOf(groups[g].id));
            xf[g] = b >= 0 ? solve(b) : Rigid{};
        }
        return xf;
    }

    std::vector<Rigid> Design::PoseTransforms(ExportPose pose, bool ease) const {
        if (pose == ExportPose::Open) return Transforms(0.0f, ease);
        if (pose == ExportPose::Closed) return Transforms(1.0f, ease);
        return std::vector<Rigid>(groups.size());
    }

    void Design::HingeInWorld(int h, const std::vector<Rigid>& xf, Vector3* pivot, Vector3* axis) const {
        const Hinge& hg = hinges[h];
        int p = GroupIndex(hg.parent);
        if (p < 0 || p >= (int)xf.size()) { *pivot = hg.pivot; *axis = hg.axis; return; }
        *pivot = xf[p].Apply(hg.pivot);
        *axis = xf[p].ApplyVector(hg.axis);
    }

    float Design::AutoLeafAngle(int h, bool child) const {
        const Hinge& hg = hinges[h];
        Vector3 a = Vector3Normalize(hg.axis);
        Vector3 ref = Utils::AnyPerpendicular(a);
        BoundingBox b = BodyBounds(child ? hg.child : hg.parent);
        Vector3 v = Vector3Subtract(Vector3Scale(Vector3Add(b.min, b.max), 0.5f), hg.pivot);
        v = Vector3Subtract(v, Vector3Scale(a, Vector3DotProduct(v, a)));
        if (Vector3Length(v) < 1e-4f) return child ? PI : 0.0f;
        v = Vector3Normalize(v);
        return atan2f(Vector3DotProduct(Vector3CrossProduct(ref, v), a), Vector3DotProduct(ref, v));
    }

    Vector3 Design::LeafDirection(int h, bool child) const {
        const Hinge& hg = hinges[h];
        Vector3 a = Vector3Normalize(hg.axis);
        Vector3 ref = Utils::AnyPerpendicular(a);
        float angle = hg.autoLeaves ? AutoLeafAngle(h, child) : hg.leafAngle[child ? 1 : 0];
        return Vector3RotateByQuaternion(ref, QuaternionFromAxisAngle(a, angle));
    }

    std::vector<Knuckle> Design::Knuckles(int h) const {
        const Hinge& hg = hinges[h];
        int n = std::max(2, hg.knuckles);
        float gap = hg.clearance;
        float seg = fmaxf((hg.length - (n - 1) * gap) / n, 0.1f);
        float total = n * seg + (n - 1) * gap;
        std::vector<Knuckle> out(n);
        for (int k = 0; k < n; k++) {
            out[k].lo = -total * 0.5f + k * (seg + gap);
            out[k].hi = out[k].lo + seg;
            out[k].child = ((k % 2) == 1) != hg.swapKnuckles;
            out[k].removed = k < 32 && (hg.removed >> k) & 1u;
        }
        return out;
    }

    bool Design::KnuckleRemovable(int h, int k) const {
        std::vector<Knuckle> ks = Knuckles(h);
        if (k < 0 || k >= (int)ks.size()) return false;
        int left = 0;
        for (int i = 0; i < (int)ks.size(); i++)
            if (i != k && !ks[i].removed && ks[i].child == ks[k].child) left++;
        return left > 0;
    }

    std::vector<SpringSpace> Design::SpringSpaces(int h) const {
        const Hinge& hg = hinges[h];
        std::vector<Knuckle> ks = Knuckles(h);
        std::vector<SpringSpace> out;
        for (int k = 0; k < (int)ks.size(); k++) {
            if (!ks[k].removed) continue;
            if (!out.empty() && out.back().last == k - 1) {
                out.back().last = k;
                out.back().hi = ks[k].hi + hg.clearance;
                continue;
            }
            SpringSpace s;
            s.first = s.last = k;
            s.lo = ks[k].lo - hg.clearance;
            s.hi = ks[k].hi + hg.clearance;
            // both bodies are cleared to the knuckle radius + clearance; keep the clearance around the coil
            s.outer = 2.0f * hg.radius;
            s.inner = 2.0f * hg.pinRadius + hg.clearance;
            out.push_back(s);
        }
        return out;
    }

    void Design::PinHead(int h, float* from, float* to, float* radius) const {
        const Hinge& hg = hinges[h];
        std::vector<Knuckle> ks = Knuckles(h);
        *from = ks.back().hi + hg.clearance;
        *to = *from + fmaxf(1.0f, hg.pinRadius);
        float dia = hg.headDiameter > 0.0f ? hg.headDiameter : AutoPinHeadDiameter(h);
        *radius = fmaxf(dia, MinPinHeadDiameter(h)) * 0.5f;
    }

    float Design::AutoPinHeadDiameter(int h) const {
        const Hinge& hg = hinges[h];
        return 2.0f * fmaxf(hg.radius - hg.clearance, hg.pinRadius + 0.4f);
    }

    float Design::MinPinHeadDiameter(int h) const {
        const Hinge& hg = hinges[h];
        return 2.0f * (hg.pinRadius + hg.clearance + 0.2f);
    }

    bool Design::AddsMaterial(int s) const {
        if (shapes[s].op != Op::Union) return false;
        int g = shapes[s].group, guard = 0;
        while (g >= 0 && guard++ <= (int)groups.size()) {
            int i = GroupIndex(g);
            if (i < 0) break;
            if (groups[i].parent >= 0 && groups[i].op == Op::Cut) return false;
            g = groups[i].parent;
        }
        return true;
    }

    BoundingBox Design::BodyBounds(int bodyId) const {
        Vector3 lo = { 1e30f, 1e30f, 1e30f }, hi = { -1e30f, -1e30f, -1e30f };
        bool any = false;
        for (int s : GroupShapes(bodyId)) {
            if (!AddsMaterial(s)) continue;
            BoundingBox b = shapes[s].sf.WorldBounds();
            lo = Vector3Min(lo, b.min);
            hi = Vector3Max(hi, b.max);
            any = true;
        }
        if (!any) return { { 0, 0, 0 }, { 0, 0, 0 } };
        return { lo, hi };
    }

    BoundingBox Design::Bounds() const {
        Vector3 lo = { 1e30f, 1e30f, 1e30f }, hi = { -1e30f, -1e30f, -1e30f };
        bool any = false;
        for (const Shape& s : shapes) {
            BoundingBox b = s.sf.WorldBounds();
            lo = Vector3Min(lo, b.min);
            hi = Vector3Max(hi, b.max);
            any = true;
        }
        if (!any) return { { -40, 0, -40 }, { 40, 40, 40 } };
        return { lo, hi };
    }

    // ---- editing -----------------------------------------------------------------------------------------------------

    std::string Design::UniqueName(const std::string& base) const {
        auto used = [&](const std::string& n) {
            for (const Shape& s : shapes) if (s.name == n) return true;
            for (const Group& g : groups) if (g.name == n) return true;
            for (const Hinge& h : hinges) if (h.name == n) return true;
            return false;
        };
        if (!used(base)) return base;
        for (int k = 2;; k++) {
            std::string n = base + " " + std::to_string(k);
            if (!used(n)) return n;
        }
    }

    Color Design::NextColor() const {
        return Utils::PartColor((int)Bodies().size());
    }

    int Design::AddGroup(const std::string& name, int parent) {
        Group g;
        g.id = nextId++;
        g.parent = GroupIndex(parent) >= 0 ? parent : -1;
        g.name = UniqueName(name.empty() ? (g.parent < 0 ? "Body" : "Feature") : name);
        g.color = g.parent < 0 ? NextColor() : groups[GroupIndex(parent)].color;
        groups.push_back(g);
        return (int)groups.size() - 1;
    }

    int Design::AddShape(const SuperFrustum& sf, int groupId, const std::string& name, Op op) {
        int g = GroupIndex(groupId);
        if (g < 0) g = AddGroup("Body", -1);
        Shape s;
        s.id = nextId++;
        s.group = groups[g].id;
        s.sf = sf;
        s.op = op;
        s.name = UniqueName(name.empty() ? "Shape" : name);
        s.color = op == Op::Cut ? Color{ 225, 90, 90, 255 } : groups[g].color;
        shapes.push_back(s);
        return (int)shapes.size() - 1;
    }

    int Design::AddHinge(int parent, int child) {
        Hinge h;
        h.id = nextId++;
        h.parent = parent;
        h.child = child;
        h.name = UniqueName("Hinge");
        hinges.push_back(h);
        return (int)hinges.size() - 1;
    }

    void Design::RemoveShape(int index) {
        if (index < 0 || index >= (int)shapes.size()) return;
        shapes.erase(shapes.begin() + index);
    }

    void Design::RemoveHinge(int index) {
        if (index < 0 || index >= (int)hinges.size()) return;
        hinges.erase(hinges.begin() + index);
    }

    void Design::RemoveGroup(int index) {
        if (index < 0 || index >= (int)groups.size()) return;
        int id = groups[index].id;
        std::vector<int> gone;
        for (const Group& g : groups)
            if (Inside(g.id, id)) gone.push_back(g.id);
        auto isGone = [&](int gid) { return std::find(gone.begin(), gone.end(), gid) != gone.end(); };
        shapes.erase(std::remove_if(shapes.begin(), shapes.end(), [&](const Shape& s) { return isGone(s.group); }), shapes.end());
        hinges.erase(std::remove_if(hinges.begin(), hinges.end(), [&](const Hinge& h) { return isGone(h.parent) || isGone(h.child); }),
                     hinges.end());
        groups.erase(std::remove_if(groups.begin(), groups.end(), [&](const Group& g) { return isGone(g.id); }), groups.end());
    }

    bool Design::MoveShape(int index, int groupId) {
        if (index < 0 || index >= (int)shapes.size() || GroupIndex(groupId) < 0) return false;
        shapes[index].group = groupId;
        return true;
    }

    bool Design::MoveGroup(int index, int parentId) {
        if (index < 0 || index >= (int)groups.size()) return false;
        Group& g = groups[index];
        if (parentId >= 0 && (GroupIndex(parentId) < 0 || Inside(parentId, g.id))) return false;
        bool wasBody = g.parent < 0;
        g.parent = parentId;
        if (parentId < 0) {
            g.op = Op::Union;
        } else if (wasBody) {
            int id = g.id;
            hinges.erase(std::remove_if(hinges.begin(), hinges.end(), [&](const Hinge& h) { return h.parent == id || h.child == id; }),
                         hinges.end());
        }
        return true;
    }

    int Design::GroupItems(const std::vector<int>& shapeIds, const std::vector<int>& groupIds) {
        std::vector<int> containers, sources;
        for (int id : shapeIds) {
            int s = ShapeIndex(id);
            if (s >= 0) containers.push_back(shapes[s].group);
        }
        for (int id : groupIds) {
            int g = GroupIndex(id);
            if (g >= 0) containers.push_back(groups[g].parent);
        }
        if (containers.empty()) return -1;
        sources = containers;
        // deepest group that contains every item's container
        int common = -1;
        int g = containers[0], guard = 0;
        while (g >= 0 && guard++ <= (int)groups.size()) {
            bool all = true;
            for (int c : containers) all &= Inside(c, g);
            if (all) { common = g; break; }
            int i = GroupIndex(g);
            g = i >= 0 ? groups[i].parent : -1;
        }
        int ng = AddGroup(common < 0 ? "Body" : "Feature", common);
        int newId = groups[ng].id;
        for (int id : shapeIds) MoveShape(ShapeIndex(id), newId);
        for (int id : groupIds) {
            if (id == newId) continue;
            MoveGroup(GroupIndex(id), newId);
        }
        // drop the groups the items came from if they are now empty
        for (int src : sources) {
            if (src < 0 || src == newId || Inside(newId, src)) continue;
            int i = GroupIndex(src);
            if (i >= 0 && DirectShapes(src).empty() && ChildGroups(src).empty()) RemoveGroup(i);
        }
        return newId;
    }

    void Design::Ungroup(int index) {
        if (index < 0 || index >= (int)groups.size()) return;
        Group g = groups[index];
        bool cut = g.parent >= 0 && g.op == Op::Cut;
        std::vector<int> shapeIds, groupIds;
        for (int s : DirectShapes(g.id)) shapeIds.push_back(shapes[s].id);
        for (int c : ChildGroups(g.id)) groupIds.push_back(groups[c].id);
        if (g.parent >= 0) {
            for (int id : shapeIds) {
                int s = ShapeIndex(id);
                MoveShape(s, g.parent);
                if (cut) shapes[s].op = Op::Cut;
            }
            for (int id : groupIds) {
                MoveGroup(GroupIndex(id), g.parent);
                if (cut) groups[GroupIndex(id)].op = Op::Cut;
            }
        } else {
            for (int id : groupIds) MoveGroup(GroupIndex(id), -1);
            for (int id : shapeIds) {
                int s = ShapeIndex(id);
                int b = AddGroup(shapes[s].name, -1);
                MoveShape(ShapeIndex(id), groups[b].id);
            }
        }
        RemoveGroup(GroupIndex(g.id));
    }

    int Design::DuplicateShape(int index, Vector3 offset) {
        if (index < 0 || index >= (int)shapes.size()) return -1;
        Shape s = shapes[index];
        s.id = nextId++;
        s.name = UniqueName(s.name);
        s.sf.position = Vector3Add(s.sf.position, offset);
        shapes.insert(shapes.begin() + index + 1, s);
        return index + 1;
    }

    int Design::DuplicateGroup(int index, int parentId, Vector3 offset) {
        if (index < 0 || index >= (int)groups.size()) return -1;
        Group src = groups[index];
        Group g = src;
        g.id = nextId++;
        g.parent = parentId;
        g.name = UniqueName(src.name);
        if (parentId < 0) g.color = NextColor();
        groups.push_back(g);
        for (int s : DirectShapes(src.id)) {
            Shape c = shapes[s];
            c.id = nextId++;
            c.group = g.id;
            c.name = UniqueName(c.name);
            c.sf.position = Vector3Add(c.sf.position, offset);
            if (parentId < 0 && c.op == Op::Union && SameColor(c.color, src.color)) c.color = g.color;
            shapes.push_back(c);
        }
        for (int c : ChildGroups(src.id)) {
            int k = GroupIndex(groups[c].id);   // indices shift as groups grow; look it up again
            if (k >= 0) DuplicateGroup(k, g.id, offset);
        }
        return GroupIndex(g.id);
    }

    void Design::Repair() {
        for (Group& g : groups)
            if (g.parent >= 0 && (GroupIndex(g.parent) < 0 || g.parent == g.id)) g.parent = -1;
        // break group loops
        for (Group& g : groups) {
            int p = g.parent, guard = 0;
            while (p >= 0 && guard++ <= (int)groups.size()) {
                if (p == g.id) { g.parent = -1; break; }
                int i = GroupIndex(p);
                p = i >= 0 ? groups[i].parent : -1;
            }
        }
        for (size_t s = 0; s < shapes.size(); s++) {
            if (GroupIndex(shapes[s].group) >= 0) continue;
            int b = AddGroup("Body", -1);
            shapes[s].group = groups[b].id;
        }
        for (Shape& s : shapes) s.sf.Repair(0.05f);
        for (Group& g : groups) g.blend = Clamp(g.blend, 0.0f, 50.0f);
        std::vector<Hinge> all = hinges;
        hinges.clear();
        for (Hinge h : all) {
            if (!CanHinge(h.parent, h.child)) continue;
            float len = Vector3Length(h.axis);
            h.axis = len > 1e-6f ? Vector3Scale(h.axis, 1.0f / len) : Vector3{ 1, 0, 0 };
            h.start = Clamp(h.start, -2 * PI, 2 * PI);
            h.end = Clamp(h.end, -2 * PI, 2 * PI);
            if (fabsf(h.end - h.start) > 2 * PI) h.end = h.start + (h.end > h.start ? 2 * PI : -2 * PI);
            h.t0 = Clamp(h.t0, 0.0f, 1.0f);
            h.t1 = Clamp(h.t1, h.t0, 1.0f);
            h.o0 = Clamp(h.o0, 0.0f, 1.0f);
            h.o1 = Clamp(h.o1, h.o0, 1.0f);
            h.knuckles = std::clamp(h.knuckles, 2, 15);
            h.length = fmaxf(h.length, 1.0f);
            h.clearance = Clamp(h.clearance, 0.05f, 3.0f);
            h.pinRadius = fmaxf(h.pinRadius, 0.2f);
            h.radius = fmaxf(h.radius, h.pinRadius + h.clearance + 0.4f);
            h.leafLength = fmaxf(h.leafLength, 0.0f);
            h.headDiameter = Clamp(h.headDiameter, 0.0f, 500.0f);
            // forget knuckles past the count, and never leave a body without one
            h.removed &= h.knuckles >= 32 ? ~0u : (1u << h.knuckles) - 1u;
            for (int side = 0; side < 2; side++) {
                uint32_t mine = 0;
                for (int k = side; k < h.knuckles; k += 2) mine |= 1u << k;
                if ((h.removed & mine) == mine) h.removed &= ~mine;
            }
            h.leafThickness = fmaxf(h.leafThickness, 0.2f);
            hinges.push_back(h);
        }
        int maxId = 0;
        for (const Shape& s : shapes) maxId = std::max(maxId, s.id);
        for (const Group& g : groups) maxId = std::max(maxId, g.id);
        for (const Hinge& h : hinges) maxId = std::max(maxId, h.id);
        nextId = std::max(nextId, maxId + 1);
    }

    // ---- files -------------------------------------------------------------------------------------------------------

    static std::string RestOfLine(std::istringstream& ls) {
        std::string rest;
        std::getline(ls, rest);
        size_t a = rest.find_first_not_of(" \t\r");
        size_t b = rest.find_last_not_of(" \t\r");
        return a == std::string::npos ? "" : rest.substr(a, b - a + 1);
    }

    std::string Design::Serialize() const {
        std::ostringstream s;
        s.precision(9);
        s << "bakugan-hinges 1\n";
        s << "# lengths in millimetres, angles in radians, poses are the design pose\n";
        s << "nextid " << nextId << "\n";
        for (const Group& g : groups)
            s << "group " << g.id << " " << g.parent << " " << (int)g.op << " " << g.blend << " " << (int)g.color.r << " "
              << (int)g.color.g << " " << (int)g.color.b << " " << g.name << "\n";
        for (const Shape& p : shapes) {
            const SuperFrustum& f = p.sf;
            s << "shape " << p.id << " " << p.group << " " << (int)p.op << " " << (int)p.color.r << " " << (int)p.color.g << " "
              << (int)p.color.b << " " << f.position.x << " " << f.position.y << " " << f.position.z << " "
              << f.rotation.x << " " << f.rotation.y << " " << f.rotation.z << " " << f.rotation.w << " "
              << f.size.x << " " << f.size.y << " " << f.size.z << " "
              << f.rounding << " " << f.taper << " " << f.bulge << " " << f.dilation << " " << f.onion << " " << p.name << "\n";
        }
        for (const Hinge& h : hinges)
            s << "hinge " << h.id << " " << h.parent << " " << h.child << " " << h.pivot.x << " " << h.pivot.y << " " << h.pivot.z
              << " " << h.axis.x << " " << h.axis.y << " " << h.axis.z << " " << h.start << " " << h.end << " " << h.t0 << " "
              << h.t1 << " " << (h.hardware ? 1 : 0) << " " << h.length << " " << h.radius << " " << h.pinRadius << " "
              << h.clearance << " " << h.knuckles << " " << (int)h.pin << " " << h.leafLength << " " << h.leafThickness << " "
              << (h.autoLeaves ? 1 : 0) << " " << h.leafAngle[0] << " " << h.leafAngle[1] << " " << h.name << "\n";
        for (const Hinge& h : hinges)
            if (h.ownOpening) s << "opening " << h.id << " " << h.o0 << " " << h.o1 << "\n";
        for (const Hinge& h : hinges)
            if (h.removed) s << "removed " << h.id << " " << h.removed << "\n";
        for (const Hinge& h : hinges)
            if (h.swapKnuckles) s << "swapknuckles " << h.id << "\n";
        for (const Hinge& h : hinges)
            if (h.headDiameter > 0.0f) s << "pinhead " << h.id << " " << h.headDiameter << "\n";
        return s.str();
    }

    bool Design::Deserialize(const std::string& text, std::string* err) {
        std::istringstream in(text);
        std::string line;
        Design d;
        bool header = false;
        struct Opening { int hinge; float o0, o1; };
        std::vector<Opening> openings;
        std::vector<std::pair<int, uint32_t>> removals;
        std::vector<int> swaps;
        std::vector<std::pair<int, float>> heads;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string tag;
            ls >> tag;
            if (tag.empty() || tag[0] == '#') continue;
            if (tag == "bakugan-hinges") header = true;
            else if (tag == "nextid") ls >> d.nextId;
            else if (tag == "group") {
                Group g;
                int op = 0, r = 0, gr = 0, b = 0;
                ls >> g.id >> g.parent >> op >> g.blend >> r >> gr >> b;
                if (ls.fail()) { *err = "Malformed group line: " + line.substr(0, 60); return false; }
                g.op = op == 1 ? Op::Cut : Op::Union;
                g.color = { (unsigned char)r, (unsigned char)gr, (unsigned char)b, 255 };
                g.name = RestOfLine(ls);
                if (g.name.empty()) g.name = "Group";
                d.groups.push_back(g);
            } else if (tag == "shape") {
                Shape p;
                int op = 0, r = 0, g = 0, b = 0;
                SuperFrustum& f = p.sf;
                ls >> p.id >> p.group >> op >> r >> g >> b
                   >> f.position.x >> f.position.y >> f.position.z
                   >> f.rotation.x >> f.rotation.y >> f.rotation.z >> f.rotation.w
                   >> f.size.x >> f.size.y >> f.size.z
                   >> f.rounding >> f.taper >> f.bulge >> f.dilation >> f.onion;
                if (ls.fail()) { *err = "Malformed shape line: " + line.substr(0, 60); return false; }
                p.op = op == 1 ? Op::Cut : Op::Union;
                p.color = { (unsigned char)r, (unsigned char)g, (unsigned char)b, 255 };
                p.name = RestOfLine(ls);
                if (p.name.empty()) p.name = "Shape";
                d.shapes.push_back(p);
            } else if (tag == "hinge") {
                Hinge h;
                int hw = 1, pin = 0, autoLeaves = 1;
                ls >> h.id >> h.parent >> h.child >> h.pivot.x >> h.pivot.y >> h.pivot.z >> h.axis.x >> h.axis.y >> h.axis.z
                   >> h.start >> h.end >> h.t0 >> h.t1 >> hw >> h.length >> h.radius >> h.pinRadius >> h.clearance >> h.knuckles
                   >> pin >> h.leafLength >> h.leafThickness >> autoLeaves >> h.leafAngle[0] >> h.leafAngle[1];
                if (ls.fail()) { *err = "Malformed hinge line: " + line.substr(0, 60); return false; }
                h.hardware = hw != 0;
                h.pin = pin == 1 ? PinStyle::Separate : PinStyle::Fused;
                h.autoLeaves = autoLeaves != 0;
                h.name = RestOfLine(ls);
                if (h.name.empty()) h.name = "Hinge";
                d.hinges.push_back(h);
            } else if (tag == "opening") {
                Opening o;
                ls >> o.hinge >> o.o0 >> o.o1;
                if (ls.fail()) { *err = "Malformed opening line: " + line.substr(0, 60); return false; }
                openings.push_back(o);
            } else if (tag == "removed") {
                int id = 0;
                uint32_t mask = 0;
                ls >> id >> mask;
                if (ls.fail()) { *err = "Malformed removed line: " + line.substr(0, 60); return false; }
                removals.push_back({ id, mask });
            } else if (tag == "swapknuckles") {
                int id = 0;
                ls >> id;
                if (ls.fail()) { *err = "Malformed swapknuckles line: " + line.substr(0, 60); return false; }
                swaps.push_back(id);
            } else if (tag == "pinhead") {
                int id = 0;
                float dia = 0.0f;
                ls >> id >> dia;
                if (ls.fail()) { *err = "Malformed pinhead line: " + line.substr(0, 60); return false; }
                heads.push_back({ id, dia });
            }
        }
        for (const Opening& o : openings)
            for (Hinge& h : d.hinges)
                if (h.id == o.hinge) { h.ownOpening = true; h.o0 = o.o0; h.o1 = o.o1; }
        for (const auto& r : removals)
            for (Hinge& h : d.hinges)
                if (h.id == r.first) h.removed = r.second;
        for (int id : swaps)
            for (Hinge& h : d.hinges)
                if (h.id == id) h.swapKnuckles = true;
        for (const auto& hd : heads)
            for (Hinge& h : d.hinges)
                if (h.id == hd.first) h.headDiameter = hd.second;
        if (!header) { *err = "Not a bakugan hinge design file"; return false; }
        d.Repair();
        *this = d;
        return true;
    }
}
