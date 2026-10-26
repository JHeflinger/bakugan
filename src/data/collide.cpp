// TBB first: raylib's macros (PI, EPSILON, ...) must not leak into its headers.
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include "collide.hpp"
#include <raymath.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <tuple>

namespace Bakugan {
    namespace {
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

    int SweepResult::Count(Motion m) const {
        int n = 0;
        for (const Collision& c : collisions) n += c.motion == m;
        return n;
    }

    // Hits of this many samples per pair and moment are worked out to the shapes involved (the deepest always): enough
    // to name every shape that collides, without paying for the attribution of thousands of hits.
    static const size_t NamedPerStep = 48;

    SweepResult SweepCollisions(const Design& d, std::shared_ptr<const std::vector<BuiltPart>> parts, const SweepParams& p,
                                JobControl& c, std::shared_ptr<SweepCache> cache) {
        using PairHits = std::vector<std::vector<SweepHit>>;
        SweepResult r;
        r.design = d;
        r.params = p;
        bool opens = d.HasOwnOpening();
        c.SetStage(opens ? "Checking the closing and opening motions" : "Checking the closing motion");
        // one field per body that has surface samples
        std::vector<BodyField> fields;
        std::vector<const BuiltPart*> built;
        std::vector<int> groupIndex;
        for (const BuiltPart& part : *parts) {
            if (part.body < 0 || part.samples.empty()) continue;
            int g = d.GroupIndex(part.body);
            if (g < 0) continue;
            fields.push_back(BodyField::Body(d, part.body));
            built.push_back(&part);
            groupIndex.push_back(g);
        }
        int n = (int)fields.size();
        // checked moments: the closing motion, then the opening one when it differs
        int steps = std::max(p.steps, 2);
        int moments = opens ? 2 * steps : steps;
        auto motionOf = [&](int k) { return k >= steps ? Motion::Opening : Motion::Closing; };
        auto timeOf = [&](int k) { return (k % steps) / (float)(steps - 1); };
        std::vector<std::vector<Rigid>> xf(moments);
        for (int k = 0; k < moments; k++) xf[k] = d.Transforms(timeOf(k), p.ease, motionOf(k));

        // every ordered pair, keyed by both parts and their relative motion
        struct Pair { int a, b; std::string key; std::vector<Rigid> rel; std::shared_ptr<const PairHits> hits; };
        std::vector<Pair> pairs;
        char head[64];
        snprintf(head, sizeof(head), "n%d tol%.6g|", steps, p.tolerance);
        for (int a = 0; a < n; a++)
            for (int b = 0; b < n; b++) {
                if (a == b) continue;
                Pair pr{ a, b, std::string(head) + built[a]->key + "|" + built[b]->key + "|", {}, nullptr };
                pr.rel.resize(moments);
                std::string motion;
                motion.reserve(moments * 7 * 4);
                for (int k = 0; k < moments; k++) {
                    pr.rel[k] = xf[k][groupIndex[a]].Then(xf[k][groupIndex[b]].Inverse());   // a's frame -> b's frame
                    const float v[7] = { pr.rel[k].rotation.x, pr.rel[k].rotation.y, pr.rel[k].rotation.z, pr.rel[k].rotation.w,
                                         pr.rel[k].position.x, pr.rel[k].position.y, pr.rel[k].position.z };
                    for (int i = 0; i < 7; i++) {
                        int32_t q = (int32_t)lroundf(v[i] * (i < 4 ? 1e5f : 1e4f));
                        motion.append((const char*)&q, 4);
                    }
                }
                pr.key += motion;
                if (cache) {
                    std::lock_guard<std::mutex> l(cache->mutex);
                    auto it = cache->pairs.find(pr.key);
                    if (it != cache->pairs.end()) pr.hits = it->second;
                }
                pairs.push_back(std::move(pr));
            }

        // check the pairs that aren't known yet, one moment at a time
        std::vector<int> todo;
        for (int i = 0; i < (int)pairs.size(); i++)
            if (!pairs[i].hits) todo.push_back(i);
        r.pairsChecked = (int)todo.size();
        r.pairsReused = (int)pairs.size() - r.pairsChecked;
        std::vector<PairHits> fresh(todo.size(), PairHits(moments));
        size_t tasks = todo.size() * moments;
        std::atomic<size_t> done{ 0 };
        tbb::parallel_for(tbb::blocked_range<size_t>(0, tasks, 1), [&](const tbb::blocked_range<size_t>& range) {
            for (size_t task = range.begin(); task != range.end(); task++) {
                if (c.Cancelled()) return;
                size_t t = task / moments;
                int k = (int)(task % moments);
                const Pair& pr = pairs[todo[t]];
                const BodyField& fa = fields[pr.a];
                const BodyField& fb = fields[pr.b];
                const Rigid& rel = pr.rel[k];
                std::vector<SweepHit>& out = fresh[t][k];
                if (Overlap(TransformBox(fa.bounds, rel), fb.bounds)) {
                    std::vector<Vector3> inside;   // the sample in b's frame, for naming
                    // b's level set rules out the samples clearly outside it; only the rest get the exact field.
                    // Trilinear interpolation of a distance field is off by less than a voxel, so a margin of one
                    // voxel keeps every real hit.
                    std::unique_ptr<LevelSetProbe> probe;
                    if (built[pr.b]->levelSet) probe = std::make_unique<LevelSetProbe>(*built[pr.b]->levelSet);
                    float margin = probe ? probe->Voxel() + p.tolerance : 0.0f;
                    for (const Vector3& s : built[pr.a]->samples) {
                        Vector3 q = rel.Apply(s);
                        if (!InBox(q, fb.bounds)) continue;
                        if (probe && (*probe)(q) > margin) continue;
                        float db = fb.Distance(q);
                        if (db >= -p.tolerance) continue;
                        SweepHit h;
                        h.local = s;
                        h.depth = -db;
                        out.push_back(h);
                        inside.push_back(q);
                    }
                    if (!out.empty()) {
                        size_t deepest = 0;
                        for (size_t i = 1; i < out.size(); i++)
                            if (out[i].depth > out[deepest].depth) deepest = i;
                        size_t stride = std::max<size_t>(1, out.size() / NamedPerStep);
                        for (size_t i = 0; i < out.size(); i++) {
                            if (i % stride != 0 && i != deepest) continue;
                            SweepHit& h = out[i];
                            fa.Nearest(h.local, &h.shapeA, &h.hingeA);
                            fb.Deepest(inside[i], &h.shapeB, &h.hingeB);
                            h.named = true;
                        }
                    }
                }
                c.SetProgress(++done / (float)std::max<size_t>(tasks, 1));
            }
        });
        if (c.Cancelled()) { r.cancelled = true; r.message = "Stopped."; return r; }
        for (size_t t = 0; t < todo.size(); t++) {
            Pair& pr = pairs[todo[t]];
            pr.hits = std::make_shared<const PairHits>(std::move(fresh[t]));
        }
        if (cache) {
            std::lock_guard<std::mutex> l(cache->mutex);
            if (cache->pairs.size() > 512) cache->pairs.clear();   // old designs: start over
            for (int t : todo) cache->pairs[pairs[t].key] = pairs[t].hits;
        }

        // per step summaries and per pair of colliding primitives
        using Key = std::tuple<int, int, int, int, int, int, int>;
        std::map<Key, Collision> merged;
        r.steps.resize(steps);
        if (opens) r.openSteps.resize(steps);
        for (int k = 0; k < moments; k++) {
            SweepStep& st = k >= steps ? r.openSteps[k - steps] : r.steps[k];
            st.motion = motionOf(k);
            st.t = timeOf(k);
            int m = (int)st.motion;
            size_t total = 0;
            for (const Pair& pr : pairs) total += (*pr.hits)[k].size();
            size_t stride = std::max<size_t>(1, total / 300), counter = 0;
            std::map<Key, bool> seen;
            for (const Pair& pr : pairs) {
                const std::vector<SweepHit>& hits = (*pr.hits)[k];
                if (hits.empty()) continue;
                const Rigid& xa = xf[k][groupIndex[pr.a]];
                int ba = d.groups[groupIndex[pr.a]].id, bb = d.groups[groupIndex[pr.b]].id;
                AddUnique(st.bodies, ba);
                AddUnique(st.bodies, bb);
                for (const SweepHit& h : hits) {
                    st.depth = fmaxf(st.depth, h.depth);
                    if (counter++ % stride == 0) st.points.push_back(xa.Apply(h.local));
                    if (!h.named) continue;
                    AddUnique(st.shapes, h.shapeA);
                    AddUnique(st.shapes, h.shapeB);
                    AddUnique(st.hinges, h.hingeA);
                    AddUnique(st.hinges, h.hingeB);
                    // order the pair so a hits b and b hits a merge
                    bool swap = ba > bb;
                    Key key = swap ? Key{ m, bb, ba, h.shapeB, h.shapeA, h.hingeB, h.hingeA } : Key{ m, ba, bb, h.shapeA, h.shapeB, h.hingeA, h.hingeB };
                    Collision& col = merged[key];
                    col.motion = st.motion;
                    col.bodyA = std::get<1>(key);
                    col.bodyB = std::get<2>(key);
                    col.shapeA = std::get<3>(key);
                    col.shapeB = std::get<4>(key);
                    col.hingeA = std::get<5>(key);
                    col.hingeB = std::get<6>(key);
                    col.firstT = fminf(col.firstT, st.t);
                    col.lastT = fmaxf(col.lastT, st.t);
                    if (h.depth > col.depth) { col.depth = h.depth; col.worstT = st.t; col.point = xa.Apply(h.local); col.local = h.local; }
                    if (!seen[key]) { seen[key] = true; col.steps++; }
                }
            }
        }
        for (auto& kv : merged) r.collisions.push_back(kv.second);
        std::sort(r.collisions.begin(), r.collisions.end(), [](const Collision& a, const Collision& b) { return a.depth > b.depth; });
        r.ok = true;
        if (n < 2) r.message = "Fewer than two bodies: nothing can collide.";
        else if (r.collisions.empty()) r.message = opens ? "The closing and opening motions are clear." : "The closing motion is clear.";
        else r.message = std::to_string(r.collisions.size()) + " collision" + (r.collisions.size() == 1 ? "" : "s") +
                         (opens ? " while closing or opening." : " while closing.");
        return r;
    }
}
