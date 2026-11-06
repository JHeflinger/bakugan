#include "grammar.hpp"
#include "data/field.hpp"
#include <raymath.h>
#include <cfloat>

namespace Bakugan {
    namespace {
        float Uniform(std::mt19937& rng, float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); }

        // Add a superfrustum: a very small box, in a body of its own, at a random point on the mesh, hinged to the
        // body nearest it (in the open pose) at the point of that body nearest it. Adam grows and moves it from there.
        bool AddSuperfrustum(Design& d, const RuleContext& c) {
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
            float size = 0.03f * c.loss.L();
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
    }

    const std::vector<Rule>& Rules() {
        static const std::vector<Rule> rules = {
            { "add a superfrustum", 1.0f, AddSuperfrustum },
        };
        return rules;
    }

    const char* ApplyRandomRule(Design& d, const RuleContext& c) {
        const std::vector<Rule>& rules = Rules();
        float total = 0.0f;
        for (const Rule& r : rules) total += r.weight;
        float pick = Uniform(c.rng, 0.0f, total);
        for (const Rule& r : rules) {
            if (pick <= r.weight || &r == &rules.back()) return r.apply(d, c) ? r.name : nullptr;
            pick -= r.weight;
        }
        return nullptr;
    }
}
