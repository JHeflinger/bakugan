// TBB first: raylib's macros (PI, EPSILON, ...) must not leak into its headers.
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include "collide.hpp"
#include <raymath.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <map>
#include <tuple>

namespace Bakugan {
    namespace {
        struct Hit {
            int a = -1, b = -1;              // part indices
            int shapeA = -1, shapeB = -1;
            int hingeA = -1, hingeB = -1;
            float depth = 0.0f;
            Vector3 point = { 0, 0, 0 };
            Vector3 local = { 0, 0, 0 };
        };

        BoundingBox TransformBox(BoundingBox b, const Rigid& r) {
            Vector3 lo = { 1e30f, 1e30f, 1e30f }, hi = { -1e30f, -1e30f, -1e30f };
            for (int i = 0; i < 8; i++) {
                Vector3 c = { (i & 1) ? b.max.x : b.min.x, (i & 2) ? b.max.y : b.min.y, (i & 4) ? b.max.z : b.min.z };
                Vector3 w = r.Apply(c);
                lo = Vector3Min(lo, w);
                hi = Vector3Max(hi, w);
            }
            return { lo, hi };
        }

        bool Overlap(BoundingBox a, BoundingBox b) {
            return a.min.x <= b.max.x && a.max.x >= b.min.x && a.min.y <= b.max.y && a.max.y >= b.min.y && a.min.z <= b.max.z &&
                   a.max.z >= b.min.z;
        }

        bool InBox(Vector3 p, BoundingBox b) {
            return p.x >= b.min.x && p.x <= b.max.x && p.y >= b.min.y && p.y <= b.max.y && p.z >= b.min.z && p.z <= b.max.z;
        }

        void AddUnique(std::vector<int>& v, int x) {
            if (x >= 0 && std::find(v.begin(), v.end(), x) == v.end()) v.push_back(x);
        }
    }

    int SweepResult::StepAt(float t) const {
        if (steps.empty()) return -1;
        int k = (int)roundf(Clamp(t, 0.0f, 1.0f) * (steps.size() - 1));
        return std::clamp(k, 0, (int)steps.size() - 1);
    }

    SweepResult SweepCollisions(const Design& d, std::shared_ptr<const std::vector<BuiltPart>> parts, const SweepParams& p,
                                JobControl& c) {
        SweepResult r;
        r.design = d;
        r.params = p;
        c.SetStage("Checking the closing motion");
        // one field per body that has surface samples
        std::vector<BodyField> fields;
        std::vector<const std::vector<Vector3>*> samples;
        std::vector<int> groupIndex;
        for (const BuiltPart& part : *parts) {
            if (part.body < 0 || part.samples.empty()) continue;
            int g = d.GroupIndex(part.body);
            if (g < 0) continue;
            fields.push_back(BodyField::Body(d, part.body));
            samples.push_back(&part.samples);
            groupIndex.push_back(g);
        }
        int n = (int)fields.size();
        int steps = std::max(p.steps, 2);
        r.steps.resize(steps);
        std::vector<std::vector<Hit>> hits(steps);
        std::atomic<int> done{ 0 };
        tbb::parallel_for(tbb::blocked_range<int>(0, steps, 1), [&](const tbb::blocked_range<int>& range) {
            for (int k = range.begin(); k != range.end(); k++) {
                if (c.Cancelled()) return;
                float t = k / (float)(steps - 1);
                std::vector<Rigid> xf = d.Transforms(t, p.ease);
                std::vector<Hit>& out = hits[k];
                for (int a = 0; a < n; a++)
                    for (int b = 0; b < n; b++) {
                        if (a == b) continue;
                        const Rigid& xa = xf[groupIndex[a]];
                        const Rigid& xb = xf[groupIndex[b]];
                        Rigid rel = xa.Then(xb.Inverse());   // body a's design frame -> body b's design frame
                        if (!Overlap(TransformBox(fields[a].bounds, rel), fields[b].bounds)) continue;
                        for (const Vector3& s : *samples[a]) {
                            Vector3 q = rel.Apply(s);
                            if (!InBox(q, fields[b].bounds)) continue;
                            float db = fields[b].Distance(q);
                            if (db >= -p.tolerance) continue;
                            Hit h;
                            h.a = a;
                            h.b = b;
                            h.depth = -db;
                            h.point = xa.Apply(s);
                            h.local = s;
                            fields[a].Nearest(s, &h.shapeA, &h.hingeA);
                            fields[b].Deepest(q, &h.shapeB, &h.hingeB);
                            out.push_back(h);
                        }
                    }
                c.SetProgress(++done / (float)steps);
            }
        });
        if (c.Cancelled()) { r.cancelled = true; r.message = "Stopped."; return r; }

        // per step summaries and per pair of colliding primitives
        using Key = std::tuple<int, int, int, int, int, int>;
        std::map<Key, Collision> merged;
        for (int k = 0; k < steps; k++) {
            SweepStep& st = r.steps[k];
            st.t = k / (float)(steps - 1);
            std::map<Key, bool> seen;
            size_t stride = std::max<size_t>(1, hits[k].size() / 300);
            for (size_t i = 0; i < hits[k].size(); i++) {
                const Hit& h = hits[k][i];
                int ba = d.groups[groupIndex[h.a]].id, bb = d.groups[groupIndex[h.b]].id;
                st.depth = fmaxf(st.depth, h.depth);
                if (i % stride == 0) st.points.push_back(h.point);
                AddUnique(st.shapes, h.shapeA);
                AddUnique(st.shapes, h.shapeB);
                AddUnique(st.bodies, ba);
                AddUnique(st.bodies, bb);
                AddUnique(st.hinges, h.hingeA);
                AddUnique(st.hinges, h.hingeB);
                // order the pair so a hits b and b hits a merge
                bool swap = ba > bb;
                Key key = swap ? Key{ bb, ba, h.shapeB, h.shapeA, h.hingeB, h.hingeA } : Key{ ba, bb, h.shapeA, h.shapeB, h.hingeA, h.hingeB };
                Collision& col = merged[key];
                col.bodyA = std::get<0>(key);
                col.bodyB = std::get<1>(key);
                col.shapeA = std::get<2>(key);
                col.shapeB = std::get<3>(key);
                col.hingeA = std::get<4>(key);
                col.hingeB = std::get<5>(key);
                col.firstT = fminf(col.firstT, st.t);
                col.lastT = fmaxf(col.lastT, st.t);
                if (h.depth > col.depth) { col.depth = h.depth; col.worstT = st.t; col.point = h.point; col.local = h.local; }
                if (!seen[key]) { seen[key] = true; col.steps++; }
            }
        }
        for (auto& kv : merged) r.collisions.push_back(kv.second);
        std::sort(r.collisions.begin(), r.collisions.end(), [](const Collision& a, const Collision& b) { return a.depth > b.depth; });
        r.ok = true;
        if (n < 2) r.message = "Fewer than two bodies: nothing can collide.";
        else if (r.collisions.empty()) r.message = "The closing motion is clear.";
        else r.message = std::to_string(r.collisions.size()) + " collision" + (r.collisions.size() == 1 ? "" : "s") + " while closing.";
        return r;
    }
}
