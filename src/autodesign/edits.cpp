#include "edits.hpp"
#include "data/field.hpp"
#include <algorithm>
#include <cfloat>

namespace Bakugan::Auto {
    std::vector<int> Roots(const Design& d) {
        std::vector<int> out;
        for (int g : d.Bodies())
            if (d.ParentHinge(d.groups[g].id) < 0) out.push_back(d.groups[g].id);
        return out;
    }

    int MaterialShapes(const Design& d, int bodyId) {
        int n = 0;
        for (int s : d.GroupShapes(bodyId)) n += d.AddsMaterial(s);
        return n;
    }

    int NearestBody(const Design& d, Vector3 p, int except) {
        int best = -1;
        float bd = FLT_MAX;
        for (int g : d.Bodies()) {
            if (d.groups[g].id == except) continue;
            BodyField f = BodyField::Body(d, d.groups[g].id);
            if (f.Empty()) continue;
            float dist = f.Distance(p);
            if (dist < bd) { bd = dist; best = g; }
        }
        return best;
    }

    Vector3 BodyHint(const Design& d, int bodyId) {
        float big = -1.0f;
        Vector3 at = { 0, 0, 0 };
        for (int s : d.GroupShapes(bodyId)) {
            if (!d.AddsMaterial(s)) continue;
            const SuperFrustum& f = d.shapes[s].sf;
            float v = f.size.x * f.size.y * f.size.z + f.dilation * f.dilation * f.dilation * 4.0f;
            if (v > big) { big = v; at = f.position; }
        }
        return at;
    }

    Vector3 PivotBetween(const Design& d, int bodyA, int bodyB, Vector3 a, Vector3 b) {
        BodyField fa = BodyField::Body(d, bodyA), fb = BodyField::Body(d, bodyB);
        if (fa.Empty() || fb.Empty()) return Vector3Lerp(a, b, 0.5f);
        float lo = 0.0f, hi = 1.0f;
        for (int k = 0; k < 14; k++) {
            float mid = (lo + hi) * 0.5f;
            Vector3 p = Vector3Lerp(a, b, mid);
            if (fa.Distance(p) < fb.Distance(p)) lo = mid;
            else hi = mid;
        }
        return Vector3Lerp(a, b, (lo + hi) * 0.5f);
    }

    void RemoveBody(Design& d, int bodyId) {
        int ph = d.ParentHinge(bodyId);
        int up = ph >= 0 ? d.hinges[ph].parent : -1;
        std::vector<int> kids;
        for (int h = 0; h < (int)d.hinges.size(); h++)
            if (d.hinges[h].parent == bodyId) kids.push_back(h);
        if (up >= 0) {
            for (int h : kids) d.hinges[h].parent = up;
        } else if (!kids.empty()) {
            int root = d.hinges[kids[0]].child;
            for (size_t k = 1; k < kids.size(); k++) d.hinges[kids[k]].parent = root;
        }
        d.RemoveGroup(d.GroupIndex(bodyId));
    }

    bool MergeBody(Design& d, int bodyId) {
        int ph = d.ParentHinge(bodyId);
        if (ph < 0) return false;
        int into = d.hinges[ph].parent;
        for (int s : d.DirectShapes(bodyId)) {
            d.MoveShape(s, into);
            if (d.shapes[s].op == Op::Union) d.shapes[s].color = d.groups[d.GroupIndex(into)].color;
        }
        std::vector<int> features;
        for (int g : d.ChildGroups(bodyId)) features.push_back(d.groups[g].id);
        for (int id : features) d.MoveGroup(d.GroupIndex(id), into);
        for (Hinge& h : d.hinges)
            if (h.parent == bodyId) h.parent = into;
        d.RemoveGroup(d.GroupIndex(bodyId));
        return true;
    }

    bool SplitShape(const SuperFrustum& f, int axis, float t, SuperFrustum* a, SuperFrustum* b) {
        if (axis == 2 && fabsf(f.bulge) > 1e-4f) axis = f.size.x >= f.size.y ? 0 : 1;
        float S = Coord(f.size, axis);
        if (S <= 0.0f || t <= 0.0f || t >= 1.0f) return false;
        *a = f;
        *b = f;
        Coord(a->size, axis) = t * S;
        Coord(b->size, axis) = (1.0f - t) * S;
        Vector3 dir = { 0, 0, 0 };
        Coord(dir, axis) = 1.0f;
        dir = Vector3RotateByQuaternion(dir, f.rotation);
        a->position = Vector3Add(f.position, Vector3Scale(dir, (t - 1.0f) * S * 0.5f));
        b->position = Vector3Add(f.position, Vector3Scale(dir, t * S * 0.5f));
        if (axis == 2 && f.taper < 1.0f) {
            // The cross-section shrinks steadily up the shape: its outline is inset by (1 - taper) * inner at the top
            // (inner: the smaller half side), and an inset rounded rectangle keeps its sides but loses that much of
            // its corner radius. The lower half ends where the cut is; the upper half starts there.
            float hx = f.size.x * 0.5f, hy = f.size.y * 0.5f, inner = fminf(hx, hy), k = 1.0f - f.taper;
            float inset = k * inner * t;                     // at the cut
            float inner2 = inner - inset;
            if (inner2 <= 1e-6f) return false;
            a->taper = 1.0f - k * t;
            b->size.x = f.size.x - 2.0f * inset;
            b->size.y = f.size.y - 2.0f * inset;
            b->taper = std::clamp(1.0f - k * inner * (1.0f - t) / inner2, 0.0f, 1.0f);
            b->rounding = std::clamp(fmaxf(f.rounding * inner - inset, 0.0f) / inner2, 0.0f, 1.0f);
        }
        return true;
    }

    void SetupHinge(Design& d, int index, Vector3 pivot, Vector3 axis, float angle, bool fresh, float L) {
        Hinge& hg = d.hinges[index];
        if (fresh) {
            hg.start = 0.0f;
            hg.hardware = false;
            hg.radius = std::clamp(0.03f * L, 1.6f, 4.0f);
            hg.pinRadius = hg.radius * 0.4f;
            hg.length = std::clamp(0.2f * L, 6.0f, 30.0f);
            hg.leafLength = std::clamp(0.07f * L, 2.0f, 10.0f);
            hg.leafThickness = std::clamp(0.025f * L, 1.2f, 3.0f);
            hg.knuckles = 3;
        }
        hg.pivot = pivot;
        hg.axis = Vector3Length(axis) > 1e-6f ? Vector3Normalize(axis) : Vector3{ 0, 1, 0 };
        hg.end = hg.start + angle;
    }

    void ClampTiming(Hinge& h) {
        auto window = [](float& a, float& b) {
            float w = std::clamp(b - a, 0.1f, 1.0f);
            a = std::clamp(a, 0.0f, 1.0f - w);
            b = a + w;
        };
        window(h.t0, h.t1);
        if (h.ownOpening) window(h.o0, h.o1);
    }
}
