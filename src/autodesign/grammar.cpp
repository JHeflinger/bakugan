#include "grammar.hpp"
#include "data/field.hpp"
#include <raymath.h>
#include <algorithm>
#include <cfloat>

namespace Bakugan {
    namespace {
        float Uniform(std::mt19937& rng, float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); }

        // Add a superfrustum: a very small box, in a body of its own, at a random point on the mesh, hinged to the
        // body nearest it (in the open pose) at the point of that body nearest it. Adam grows and moves it from there.
        bool AddBody(Design& d, const RuleContext& c) {
            if ((int)d.Bodies().size() >= c.p.maxBodies || c.loss.Surface().empty()) return false;
            Vector3 q = c.loss.Surface()[std::uniform_int_distribution<size_t>(0, c.loss.Surface().size() - 1)(c.rng)];
            // the nearest body, in the open pose
            std::vector<Rigid> open = d.Transforms(0.0f, false);
            int parent = -1;
            float best = FLT_MAX;
            BodyField field;
            for (int g : d.Bodies()) {
                BodyField f = BodyField::Body(d, d.groups[g].id);
                if (f.Empty()) continue;
                float dist = f.Distance(open[g].ApplyInverse(q));
                if (dist < best) { best = dist; parent = g; field = std::move(f); }
            }
            if (parent < 0) return false;
            // in the parent's design pose: the new body moves with it, so it goes where the parent's frame puts it
            Vector3 at = open[parent].ApplyInverse(q);
            // the hinge's pivot: the point on the parent's surface nearest the new shape (a few projection steps)
            Vector3 pivot = at;
            const float e = 1e-3f * c.loss.L();
            for (int k = 0; k < 8; k++) {
                float dist = field.Distance(pivot);
                Vector3 grad = { field.Distance(Vector3Add(pivot, { e, 0, 0 })) - dist, field.Distance(Vector3Add(pivot, { 0, e, 0 })) - dist,
                                 field.Distance(Vector3Add(pivot, { 0, 0, e })) - dist };
                if (Vector3Length(grad) < 1e-12f) break;
                pivot = Vector3Subtract(pivot, Vector3Scale(Vector3Normalize(grad), dist));
            }

            SuperFrustum sf;
            sf.position = at;
            float size = 0.01f * c.loss.L();
            sf.size = { size, size, size };
            sf.rounding = 0.5f;
            int body = d.groups[d.AddGroup("Body", -1)].id;
            d.AddShape(sf, body, "Shape");
            int parentId = d.groups[parent].id;
            if (!d.CanHinge(parentId, body)) return false;
            Hinge& h = d.hinges[d.AddHinge(parentId, body)];
            h.pivot = pivot;
            Vector3 axis = { Uniform(c.rng, -1, 1), Uniform(c.rng, -1, 1), Uniform(c.rng, -1, 1) };
            h.axis = Vector3Length(axis) > 1e-3f ? Vector3Normalize(axis) : Vector3{ 0, 1, 0 };
            h.start = 0.0f;
            h.end = Uniform(c.rng, -PI, PI);           // some turn: with none, the axis would have no gradient
            h.hardware = false;
            return true;
        }

        bool AddUnion(Design& d, const RuleContext& c) {
            const std::vector<Loss::Spot>& extra = c.loss.ExtraSpots();
            const std::vector<Loss::Spot>& overlaps = c.loss.OverlapSpots();
            if (extra.empty() && overlaps.empty()) return false;
            bool overlap = extra.empty() || (!overlaps.empty() && std::uniform_int_distribution<int>(0, 1)(c.rng));
            const std::vector<Loss::Spot>& from = overlap ? overlaps : extra;
            const Loss::Spot& s = from[std::uniform_int_distribution<size_t>(0, from.size() - 1)(c.rng)];
            if (s.group < 0 || s.group >= (int)d.groups.size() || d.groups[s.group].parent >= 0) return false;
            int body = d.groups[s.group].id;
            SuperFrustum sf;
            sf.position = s.at;
            sf.rounding = 0.5f;
            float thin = 0.03f * c.loss.L();
            if (overlap) {
                sf.size = { thin, thin, thin };
            } else {
                BoundingBox b = d.BodyBounds(body);
                Vector3 dir = Vector3Normalize(s.dir);
                // the superfrustum's z axis along the view; long enough to go through the whole body
                sf.rotation = QuaternionFromVector3ToVector3({ 0, 0, 1 }, dir);
                sf.size = { thin, thin, 2.0f * Vector3Distance(b.min, b.max) };
            }
            d.AddShape(sf, body, "Shape", Op::Union);
            return true;
        }

        bool AddCut(Design& d, const RuleContext& c) {
            const std::vector<Loss::Spot>& extra = c.loss.ExtraSpots();
            const std::vector<Loss::Spot>& overlaps = c.loss.OverlapSpots();
            if (extra.empty() && overlaps.empty()) return false;
            bool overlap = extra.empty() || (!overlaps.empty() && std::uniform_int_distribution<int>(0, 1)(c.rng));
            const std::vector<Loss::Spot>& from = overlap ? overlaps : extra;
            const Loss::Spot& s = from[std::uniform_int_distribution<size_t>(0, from.size() - 1)(c.rng)];
            if (s.group < 0 || s.group >= (int)d.groups.size() || d.groups[s.group].parent >= 0) return false;
            int body = d.groups[s.group].id;
            SuperFrustum sf;
            sf.position = s.at;
            sf.rounding = 0.5f;
            float thin = 0.03f * c.loss.L();
            if (overlap) {
                sf.size = { thin, thin, thin };
            } else {
                BoundingBox b = d.BodyBounds(body);
                Vector3 dir = Vector3Normalize(s.dir);
                // the superfrustum's z axis along the view; long enough to go through the whole body
                sf.rotation = QuaternionFromVector3ToVector3({ 0, 0, 1 }, dir);
                sf.size = { thin, thin, 2.0f * Vector3Distance(b.min, b.max) };
            }
            d.AddShape(sf, body, "Cut", Op::Cut);
            return true;
        }

        // ---- removals: each takes one thing away, picked at random among what can go -------------------------------

        // Remove a body: any body on a hinge (the root stays, and so do at least two bodies, so the design still folds).
        // The bodies hanging from it hang from its parent instead, on their own hinges as they were.
        bool RemoveBody(Design& d, const RuleContext& c) {
            if (d.Bodies().size() <= 2) return false;
            if (d.hinges.empty()) return false;
            const Hinge& h = d.hinges[std::uniform_int_distribution<size_t>(0, d.hinges.size() - 1)(c.rng)];
            int body = h.child, parent = h.parent;          // every hinge's child is a body that can go
            for (Hinge& k : d.hinges)
                if (k.parent == body) k.parent = parent;
            d.RemoveGroup(d.GroupIndex(body));                   // its shapes, and its own hinge, go with it
            return true;
        }

        // Pick a random shape of the given kind; for an added shape, only one whose body keeps another (a body's last
        // material shape can only go with the body).
        int PickShape(const Design& d, Op op, std::mt19937& rng) {
            std::vector<int> can;
            for (int s = 0; s < (int)d.shapes.size(); s++) {
                if (d.shapes[s].op != op) continue;
                if (op == Op::Union) {
                    int others = 0, body = d.ShapeBody(s);
                    for (int k = 0; k < (int)d.shapes.size(); k++) others += k != s && d.shapes[k].op == Op::Union && d.ShapeBody(k) == body;
                    if (others == 0) continue;
                }
                can.push_back(s);
            }
            return can.empty() ? -1 : can[std::uniform_int_distribution<size_t>(0, can.size() - 1)(rng)];
        }

        // Remove a union: a shape that adds material, from a body that has another.
        bool RemoveUnion(Design& d, const RuleContext& c) {
            int s = PickShape(d, Op::Union, c.rng);
            if (s < 0) return false;
            d.RemoveShape(s);
            return true;
        }

        // Remove a cut.
        bool RemoveCut(Design& d, const RuleContext& c) {
            int s = PickShape(d, Op::Cut, c.rng);
            if (s < 0) return false;
            d.RemoveShape(s);
            return true;
        }

        // ---- splitting and merging bodies ------------------------------------------------------------------------------
        int CondenseContents(Design& d, int bodyId) {
            int b = d.GroupIndex(bodyId);
            if (b < 0) return -1;
            std::vector<int> shapes = d.DirectShapes(bodyId), features;
            for (int k : d.ChildGroups(bodyId)) features.push_back(d.groups[k].id);
            int id = d.groups[d.AddGroup(d.groups[b].name, bodyId)].id;
            Group& g = d.groups[d.GroupIndex(id)];
            g.blend = d.groups[b].blend;
            d.groups[b].blend = 0.0f;
            for (int sh : shapes) d.MoveShape(sh, id);
            for (int f : features) d.MoveGroup(d.GroupIndex(f), id);
            return id;
        }

        // A copy of a group (and everything in it) inside newParent (-1: as a body of its own). Returns its id.
        int CopyGroup(Design& d, int groupId, int newParent) {
            const Group src = d.groups[d.GroupIndex(groupId)];
            int id = d.groups[d.AddGroup(src.name, newParent)].id;
            Group& g = d.groups[d.GroupIndex(id)];
            g.op = src.op;
            g.blend = src.blend;
            if (newParent >= 0) g.color = src.color;            // a new body keeps its own new colour
            for (int s : d.DirectShapes(groupId)) {
                const Shape sh = d.shapes[s];
                d.shapes[d.AddShape(sh.sf, id, sh.name, sh.op)].color = sh.color;
            }
            std::vector<int> children;
            for (int k : d.ChildGroups(groupId)) children.push_back(d.groups[k].id);
            for (int child : children) CopyGroup(d, child, id);
            return id;
        }

        // A box that cuts away everything on the far side of a plane (through `at`, facing `normal`) within `big`.
        SuperFrustum HalfSpaceCut(Vector3 at, Vector3 normal, float big) {
            SuperFrustum sf;
            sf.position = Vector3Add(at, Vector3Scale(normal, 0.5f * big));
            sf.rotation = QuaternionFromVector3ToVector3({ 0, 0, 1 }, normal);
            sf.size = { big, big, big };
            sf.rounding = 0.0f;
            sf.taper = 1.0f;
            return sf;
        }

        // The silhouettes see a body's field, and the two halves of a split each end in a flat face where they meet. A
        // view looking along that face sees each half only graze the rays running inside it, near the face (the field
        // there is no deeper than the distance to the face), so a split exactly edge-on to a view shows a hairline seam
        // there that the joined body didn't have. So the cut leans away from the axis, a little (up to ~25 degrees,
        // in the open and closed poses alike) if that's what it takes to face no view edge-on.
        Vector3 Unaligned(const Design& d, int body, Vector3 axis, const RuleContext& c) {
            std::vector<Vector3> views = c.loss.ViewDirections();
            int g = d.GroupIndex(body);
            std::vector<Rigid> open = d.Transforms(0.0f, true), closed = d.Transforms(1.0f, true);
            auto score = [&](Vector3 n) {                   // how far from edge-on its most edge-on view is
                float worst = 1.0f;
                for (const Rigid* xf : { &open[g], &closed[g] }) {
                    Vector3 w = xf->ApplyVector(n);
                    for (Vector3 dir : views) worst = fminf(worst, fabsf(Vector3DotProduct(w, dir)));
                }
                return worst;
            };
            const float enough = 0.35f;                     // ~20 degrees from edge-on
            Vector3 best = axis;
            float bestScore = score(axis);
            std::normal_distribution<float> gauss(0.0f, 1.0f);
            for (int k = 0; k < 300 && bestScore < enough; k++) {
                Vector3 r = { gauss(c.rng), gauss(c.rng), gauss(c.rng) };
                Vector3 n = Vector3Normalize(Vector3Add(axis, Vector3Scale(Vector3Normalize(r), Uniform(c.rng, 0.0f, 0.45f))));
                float sc = score(n);
                if (sc > bestScore + 0.02f || (sc >= enough && bestScore < enough)) { best = n; bestScore = sc; }
            }
            return best;
        }

        // Split a body: in half across the middle of its box (along an axis drawn by the box's sides, longer ones
        // likelier), as two bodies, each the whole body with a cut taking away the other half. They are joined by a
        // hinge on the cut, at the point there nearest both, that doesn't turn yet (closed angle = open angle = 0),
        // so nothing looks or scores different until Adam turns it. The body's own hinge and the hinges hanging from it
        // go to whichever half their pivot is nearer; the half on the body's own hinge is the new hinge's parent.
        bool SplitBody(Design& d, const RuleContext& c) {
            if ((int)d.Bodies().size() >= c.p.maxBodies) return false;
            std::vector<int> bodies = d.Bodies();
            int a = d.groups[bodies[std::uniform_int_distribution<size_t>(0, bodies.size() - 1)(c.rng)]].id;
            if (BodyField::Body(d, a).Empty()) return false;
            BoundingBox box = d.BodyBounds(a);
            Vector3 ext = Vector3Subtract(box.max, box.min), mid = Vector3Scale(Vector3Add(box.min, box.max), 0.5f);
            float pick = Uniform(c.rng, 0.0f, ext.x + ext.y + ext.z);
            int axis = pick < ext.x ? 0 : (pick < ext.x + ext.y ? 1 : 2);
            Vector3 n = { 0, 0, 0 };
            (&n.x)[axis] = 1.0f;
            n = Unaligned(d, a, n, c);
            Vector3 u = Vector3Normalize(Vector3CrossProduct(n, fabsf(n.x) < 0.9f ? Vector3{ 1, 0, 0 } : Vector3{ 0, 1, 0 }));
            Vector3 v = Vector3CrossProduct(n, u);
            const float big = 1.95f * c.loss.L();           // as big as a shape may be (params.cpp), past any body

            // the two halves: a keeps the -n side, b (its copy) the +n side
            int b = CopyGroup(d, a, -1);
            d.AddShape(HalfSpaceCut(mid, n, big), a, "Split", Op::Cut);
            d.AddShape(HalfSpaceCut(mid, Vector3Negate(n), big), b, "Split", Op::Cut);
            BodyField fa = BodyField::Body(d, a), fb = BodyField::Body(d, b);
            if (fa.Empty() || fb.Empty()) return false;
            CondenseContents(d, a);
            CondenseContents(d, b);

            // the new hinge's pivot: the point on the cut nearest both halves (a grid over the cut's face)
            Vector3 pivot = mid;
            float best = FLT_MAX;
            for (int i = 0; i <= 20; i++)
                for (int j = 0; j <= 20; j++) {
                    float span = Vector3Length(ext);
                    Vector3 p = Vector3Add(mid, Vector3Add(Vector3Scale(u, (i / 20.0f - 0.5f) * span), Vector3Scale(v, (j / 20.0f - 0.5f) * span)));
                    float dist = fmaxf(fa.Distance(p), fb.Distance(p));
                    if (dist < best) { best = dist; pivot = p; }
                }

            // the hinges the body had, to the nearer half
            auto nearerB = [&](Vector3 p) { return fb.Distance(p) < fa.Distance(p); };
            int anchor = a;                                     // the half on the body's own hinge
            for (Hinge& h : d.hinges) {
                if (h.child == a && nearerB(h.pivot)) { h.child = b; anchor = b; }
                else if (h.parent == a && nearerB(h.pivot)) h.parent = b;
            }
            int other = anchor == a ? b : a;
            if (!d.CanHinge(anchor, other)) return false;
            Hinge& h = d.hinges[d.AddHinge(anchor, other)];
            h.pivot = pivot;
            h.axis = u;
            h.start = h.end = 0.0f;
            h.hardware = false;
            return true;
        }

        // Merge bodies: a body (likelier the less its hinge turns) folded into its parent as a feature of it. Its
        // shapes and features become one feature of the parent; the bodies hanging from it hang from the parent. Where
        // the hinge held it in the open pose is baked into it (and into everything hanging from it), so the open pose
        // stays as it was; the closed pose changes only as much as the hinge used to turn.
        bool MergeBodies(Design& d, const RuleContext& c) {
            if (d.Bodies().size() <= 2 || d.hinges.empty()) return false;
            std::vector<double> w;
            for (const Hinge& h : d.hinges) w.push_back(1.0 / (fabs(h.end - h.start) + 0.1));
            const Hinge h = d.hinges[std::discrete_distribution<size_t>(w.begin(), w.end())(c.rng)];
            int parent = h.parent, child = h.child;
            int pg = d.GroupIndex(parent), cg = d.GroupIndex(child);
            if (pg < 0 || cg < 0) return false;

            // the child, and everything hanging from it
            std::vector<int> below = { child };
            for (size_t k = 0; k < below.size(); k++)
                for (const Hinge& g : d.hinges)
                    if (g.parent == below[k]) below.push_back(g.child);
            auto isBelow = [&](int body) { return std::find(below.begin(), below.end(), body) != below.end(); };
            // where the child sits against its parent in the open pose, baked in
            std::vector<Rigid> open = d.Transforms(0.0f, false);
            Rigid rel = open[cg].Then(open[pg].Inverse());
            for (Shape& s : d.shapes)
                if (isBelow(d.ShapeBody(d.ShapeIndex(s.id)))) {
                    s.sf.position = rel.Apply(s.sf.position);
                    s.sf.rotation = QuaternionNormalize(QuaternionMultiply(rel.rotation, s.sf.rotation));
                }
            for (Hinge& g : d.hinges)
                if (g.id != h.id && isBelow(g.parent)) {
                    g.pivot = rel.Apply(g.pivot);
                    g.axis = Vector3Normalize(rel.ApplyVector(g.axis));
                }

            // its contents into a new feature of the parent, its children onto the parent. The parent's own cuts and
            // blend would reach into that feature too, so then the parent's own contents go into a feature of their own
            // first, and the two only meet as they did before: side by side
            bool parentCuts = d.groups[pg].blend > 0.0f;
            for (int s : d.DirectShapes(parent)) parentCuts |= d.shapes[s].op == Op::Cut;
            for (int k : d.ChildGroups(parent)) parentCuts |= d.groups[k].op == Op::Cut;
            if (parentCuts) {
                std::vector<int> shapes = d.DirectShapes(parent), features;
                for (int k : d.ChildGroups(parent)) features.push_back(d.groups[k].id);
                int own = d.groups[d.AddGroup(d.groups[pg].name, parent)].id;
                d.groups[d.GroupIndex(own)].blend = d.groups[d.GroupIndex(parent)].blend;
                d.groups[d.GroupIndex(parent)].blend = 0.0f;
                for (int s : shapes) d.MoveShape(s, own);
                for (int f : features) d.MoveGroup(d.GroupIndex(f), own);
            }
            int feature = d.groups[d.AddGroup(d.groups[d.GroupIndex(child)].name, parent)].id;
            d.groups[d.GroupIndex(feature)].blend = d.groups[d.GroupIndex(child)].blend;
            for (int s : d.DirectShapes(child)) d.MoveShape(s, feature);
            std::vector<int> features;
            for (int k : d.ChildGroups(child)) features.push_back(d.groups[k].id);
            for (int f : features) d.MoveGroup(d.GroupIndex(f), feature);
            for (Hinge& g : d.hinges)
                if (g.parent == child) g.parent = parent;
            d.RemoveHinge(d.HingeIndex(h.id));
            d.RemoveGroup(d.GroupIndex(child));                  // empty now
            return true;
        }
    }

    const std::vector<Rule>& Rules() {
        static const std::vector<Rule> rules = {
            // name              weight  apply        simplifies
            { "add a body",      1.0f,   AddBody,     false },
            { "add a union",     1.0f,   AddUnion,    false },
            { "add a cut",       1.0f,   AddCut,      false },
            { "remove a body",   0.5f,   RemoveBody,  true },
            { "remove a union",  0.5f,   RemoveUnion, true },
            { "remove a cut",    0.5f,   RemoveCut,   true },
            { "split a body",    0.5f,   SplitBody,   false },
            { "merge bodies",    0.5f,   MergeBodies, true },
        };
        return rules;
    }

    const Rule* ApplyRandomRule(Design& d, const RuleContext& c, const std::vector<float>& weights) {
        const std::vector<Rule>& rules = Rules();
        auto weight = [&](size_t i) { return std::max(0.0f, i < weights.size() ? weights[i] : rules[i].weight); };
        float total = 0.0f;
        for (size_t i = 0; i < rules.size(); i++) total += weight(i);
        if (total <= 0.0f) return nullptr;                  // every rule switched off
        float pick = Uniform(c.rng, 0.0f, total);
        for (size_t i = 0; i < rules.size(); i++) {
            if (weight(i) <= 0.0f) continue;
            if (pick <= weight(i) || i + 1 == rules.size()) return rules[i].apply(d, c) ? &rules[i] : nullptr;
            pick -= weight(i);
        }
        return nullptr;
    }

    std::vector<AutoRuleInfo> AutoRules() {
        std::vector<AutoRuleInfo> out;
        for (const Rule& r : Rules()) out.push_back({ r.name, r.weight, r.simplifies });
        return out;
    }
}
