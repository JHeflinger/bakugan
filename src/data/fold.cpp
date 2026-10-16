// TBB first: raylib's macros must not leak into its headers.
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include "fold.hpp"
#include "util/optim.hpp"
#include "util/utils.hpp"
#include <raymath.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <unordered_map>

namespace Bakugan {
    namespace {
        using Clock = std::chrono::steady_clock;

        // Uniform-grid hash over a point set (counting sort into buckets). Every item remembers its integer cell,
        // so a query visits the 27 neighbour cells and skips items that only share a bucket by hash collision.
        class PointHash {
        public:
            void Build(const std::vector<Vector3>& pts, float cell) {
                m_Inv = 1.0f / cell;
                unsigned size = 64;
                while (size < pts.size() * 4) size <<= 1;
                m_Mask = size - 1;
                m_Start.assign(size + 1, 0);
                m_Items.resize(pts.size());
                m_Cells.resize(pts.size());
                m_Key.resize(pts.size());
                std::vector<unsigned>& key = m_Key;
                for (size_t i = 0; i < pts.size(); i++) {
                    Cell3 c = { Cell(pts[i].x), Cell(pts[i].y), Cell(pts[i].z) };
                    m_Cells[i] = c;
                    key[i] = Bucket(c.x, c.y, c.z);
                    m_Start[key[i] + 1]++;
                }
                for (unsigned b = 0; b < size; b++) m_Start[b + 1] += m_Start[b];
                m_Fill.assign(m_Start.begin(), m_Start.end() - 1);
                m_Sorted.resize(pts.size());
                for (size_t i = 0; i < pts.size(); i++) {
                    int at = m_Fill[key[i]]++;
                    m_Items[at] = (int)i;
                    m_Sorted[at] = m_Cells[i];
                }
                m_Cells.swap(m_Sorted);
            }

            template <class F>
            void Query(Vector3 p, F&& fn) const {
                int cx = Cell(p.x), cy = Cell(p.y), cz = Cell(p.z);
                for (int dz = -1; dz <= 1; dz++)
                    for (int dy = -1; dy <= 1; dy++)
                        for (int dx = -1; dx <= 1; dx++) {
                            int x = cx + dx, y = cy + dy, z = cz + dz;
                            unsigned b = Bucket(x, y, z);
                            for (int k = m_Start[b]; k < m_Start[b + 1]; k++) {
                                const Cell3& c = m_Cells[k];
                                if (c.x == x && c.y == y && c.z == z) fn(m_Items[k]);
                            }
                        }
            }
        private:
            struct Cell3 { int x, y, z; };
            int Cell(float v) const { return (int)floorf(v * m_Inv); }
            unsigned Bucket(int x, int y, int z) const {
                return ((unsigned)x * 73856093u ^ (unsigned)y * 19349663u ^ (unsigned)z * 83492791u) & m_Mask;
            }
            float m_Inv = 1.0f;
            unsigned m_Mask = 0;
            std::vector<int> m_Start;
            std::vector<int> m_Items;
            std::vector<Cell3> m_Cells;   // parallel to m_Items (sorted by bucket)
            std::vector<Cell3> m_Sorted;
            std::vector<unsigned> m_Key;
            std::vector<int> m_Fill;
        };

        bool HasCycle(const Design& d) {
            int n = (int)d.parts.size();
            for (int i = 0; i < n; i++) {
                int guard = 0;
                for (int p = d.parts[i].hinge.parent; p >= 0; p = d.parts[p].hinge.parent)
                    if (++guard > n) return true;
            }
            return false;
        }

        float MaxAngle(const Design& d, int i, const FoldParams& p) {
            return fminf(d.parts[i].hinge.maxAngle, p.maxHingeAngle);
        }

        // 1D search of a hinge angle over [0, max]: coarse scan, then golden section refinement.
        template <class F>
        float SearchAngle(float maxAngle, int coarse, F&& objective) {
            float best = 0.0f, bestValue = objective(0.0f);
            if (maxAngle <= 1e-4f) return 0.0f;
            for (int k = 1; k <= coarse; k++) {
                float a = maxAngle * k / coarse;
                float v = objective(a);
                if (v < bestValue) { bestValue = v; best = a; }
            }
            float step = maxAngle / coarse;
            float lo = fmaxf(0.0f, best - step), hi = fminf(maxAngle, best + step);
            const float g = 0.618034f;
            float x1 = hi - g * (hi - lo), x2 = lo + g * (hi - lo);
            float f1 = objective(x1), f2 = objective(x2);
            for (int it = 0; it < 8; it++) {
                if (f1 < f2) { hi = x2; x2 = x1; f2 = f1; x1 = hi - g * (hi - lo); f1 = objective(x1); }
                else { lo = x1; x1 = x2; f1 = f2; x2 = lo + g * (hi - lo); f2 = objective(x2); }
            }
            float mid = f1 < f2 ? x1 : x2, fm = fminf(f1, f2);
            return fm < bestValue ? mid : best;
        }

        // Closed positions of a subtree's lattice samples (subsampled), used to keep rewrites jump-continuous.
        std::vector<Vector3> SubtreeClosed(const FoldState& s, int part, int maxCount) {
            std::vector<int> sub = s.design.Subtree(part);
            std::vector<Rigid> xf = s.design.Transforms(1.0f);
            size_t total = 0;
            for (int i : sub) total += s.samples[i].lattice.size();
            size_t stride = std::max<size_t>(1, total / std::max(1, maxCount));
            std::vector<Vector3> out;
            size_t c = 0;
            for (int i : sub)
                for (const Vector3& p : s.samples[i].lattice)
                    if (c++ % stride == 0) out.push_back(xf[i].Apply(p));
            return out;
        }

        float Displacement(const FoldState& s, int part, const std::vector<Vector3>& before) {
            std::vector<Vector3> now = SubtreeClosed(s, part, 300);
            if (now.size() != before.size() || now.empty()) return 0.0f;
            double acc = 0;
            for (size_t i = 0; i < now.size(); i++) acc += Vector3DistanceSqr(now[i], before[i]);
            return (float)(acc / now.size());
        }

        // Re-chooses the hinge angle so the subtree lands as close as possible to where it was.
        void KeepPose(FoldState& s, int part, const FoldParams& p, const std::vector<Vector3>& before) {
            float maxA = MaxAngle(s.design, part, p);
            s.design.parts[part].hinge.angle = SearchAngle(maxA, 24, [&](float a) {
                s.design.parts[part].hinge.angle = a;
                return Displacement(s, part, before);
            });
        }

        std::vector<Vector3> CandidateAxes(const FoldState& s, int part) {
            const Design& d = s.design;
            const Part& p = d.parts[part];
            std::vector<Vector3> axes;
            for (int a = 0; a < 3; a++) axes.push_back(p.shape.Axis(a));
            int parent = p.hinge.parent;
            if (parent >= 0)
                for (int a = 0; a < 3; a++) axes.push_back(d.parts[parent].shape.Axis(a));
            Vector3 out = Vector3Subtract(p.shape.position, p.hinge.pivot);
            Vector3 in = Vector3Subtract(d.ball.center, p.hinge.pivot);
            Vector3 n = Vector3CrossProduct(out, in);
            if (Vector3Length(n) > 1e-5f) axes.push_back(Vector3Normalize(n));
            size_t base = axes.size();
            for (size_t i = 0; i < base; i++) axes.push_back(Vector3Negate(axes[i]));
            return axes;
        }

        enum PKind { CenterX, CenterY, CenterZ, Radius, Angle, AxisU, AxisV, PivotX, PivotY, PivotZ, KindCount };

        struct Param {
            PKind kind;
            int part;   // index, -1 for the sphere
        };

        void ApplyParam(Design& d, const Param& p, float delta) {
            if (p.part < 0) {
                if (p.kind == Radius) d.ball.radius += delta;
                else (&d.ball.center.x)[p.kind - CenterX] += delta;
                return;
            }
            Hinge& h = d.parts[p.part].hinge;
            switch (p.kind) {
                case Angle: h.angle += delta; break;
                case AxisU:
                case AxisV: {
                    Vector3 e1 = Utils::AnyPerpendicular(h.axis);
                    Vector3 e = p.kind == AxisU ? e1 : Vector3CrossProduct(h.axis, e1);
                    h.axis = Vector3Normalize(Vector3Add(h.axis, Vector3Scale(e, delta)));
                    break;
                }
                case PivotX: case PivotY: case PivotZ: (&h.pivot.x)[p.kind - PivotX] += delta; break;
                default: break;
            }
        }

        struct AdamCell { float m = 0, v = 0; int t = 0; };

        // ---- the fold grammar ----------------------------------------------------------------------------------

        enum Rule { RuleReparent, RuleSplit, RuleMerge, RuleLock, RuleUnlock, RuleAxisSnap, RuleFlip, RuleReseat, RuleCount };
        const char* RuleNames[RuleCount] = { "reparent", "split", "merge", "lock", "unlock", "axis snap", "flip", "reseat" };

        bool Structural(Rule r) { return r == RuleSplit || r == RuleMerge; }

        int RandomHingedPart(const Design& d, std::mt19937& rng, bool includeLoose) {
            std::vector<int> c;
            int root = d.Root();
            for (int i = 0; i < (int)d.parts.size(); i++)
                if (i != root && (d.parts[i].hinge.parent >= 0 || includeLoose)) c.push_back(i);
            if (c.empty()) return -1;
            return c[std::uniform_int_distribution<int>(0, (int)c.size() - 1)(rng)];
        }

        bool ApplyReparent(FoldState& s, const FoldModel& m, std::mt19937& rng, std::vector<int>& affected) {
            Design& d = s.design;
            int i = RandomHingedPart(d, rng, true);
            if (i < 0) return false;
            std::vector<int> js;
            for (int j = 0; j < (int)d.parts.size(); j++)
                if (j != i && j != d.parts[i].hinge.parent && !d.IsDescendant(j, i)) js.push_back(j);
            std::shuffle(js.begin(), js.end(), rng);
            for (int j : js) {
                float rij = d.parts[i].shape.BoundingRadius() + d.parts[j].shape.BoundingRadius() + 2 * m.Spacing();
                if (Vector3DistanceSqr(d.parts[i].shape.position, d.parts[j].shape.position) > rij * rij) continue;
                Vector3 contact;
                int count;
                if (!m.ContactPoint(s, i, j, &contact, &count) || count < 3) continue;
                std::vector<Vector3> before = SubtreeClosed(s, i, 300);
                int old = d.parts[i].hinge.parent;
                d.parts[i].hinge.parent = j;
                d.parts[i].hinge.pivot = contact;
                d.parts[i].hinge.locked = false;
                KeepPose(s, i, m.Params(), before);
                affected = { d.parts[i].id, d.parts[j].id };
                if (old >= 0) affected.push_back(d.parts[old].id);
                return true;
            }
            return false;
        }

        bool ApplySplit(FoldState& s, const FoldModel& m, std::mt19937& rng, std::vector<int>& affected) {
            Design& d = s.design;
            if ((int)d.parts.size() >= m.Params().maxParts) return false;
            std::vector<int> c;
            for (int i = 0; i < (int)d.parts.size(); i++) {
                const SuperFrustum& f = d.parts[i].shape;
                if (fabsf(f.bulge) < 0.05f && f.onion < 1e-3f && f.size.z >= 1.2f * fmaxf(f.size.x, f.size.y) &&
                    f.size.z >= 5.0f * m.Spacing())
                    c.push_back(i);
            }
            if (c.empty()) return false;
            int i = c[std::uniform_int_distribution<int>(0, (int)c.size() - 1)(rng)];
            SuperFrustum f = d.parts[i].shape;
            float h = f.size.z * 0.5f;
            float inner = 0.5f * fminf(f.size.x, f.size.y);
            float off = (1.0f - f.taper) * inner;
            SuperFrustum lower = f, upper = f;
            lower.size.z = upper.size.z = h;
            lower.position = f.ToWorld({ 0, 0, -h * 0.5f });
            upper.position = f.ToWorld({ 0, 0, h * 0.5f });
            lower.taper = inner > 0 ? 1.0f - 0.5f * off / inner : 1.0f;
            upper.size.x = fmaxf(f.size.x - off, m.Spacing());
            upper.size.y = fmaxf(f.size.y - off, m.Spacing());
            float innerU = inner - 0.5f * off;
            upper.taper = innerU > 1e-6f ? 1.0f - 0.5f * off / innerU : f.taper;
            float r = f.rounding * inner;
            upper.rounding = innerU > 1e-6f ? std::clamp((r - 0.5f * off) / innerU, 0.0f, 1.0f) : f.rounding;
            lower.Repair(m.Spacing() * 0.5f);
            upper.Repair(m.Spacing() * 0.5f);

            // the half holding this part's own hinge keeps the part
            bool keepLower = true;
            if (d.parts[i].hinge.parent >= 0) keepLower = f.ToLocal(d.parts[i].hinge.pivot).z <= 0.0f;
            const SuperFrustum& kept = keepLower ? lower : upper;
            const SuperFrustum& moved = keepLower ? upper : lower;
            float movedSign = keepLower ? 1.0f : -1.0f;

            std::vector<int> children = d.Children(i);
            d.parts[i].shape = kept;
            int k = d.AddPart(moved, i, d.parts[i].name + "b");
            Hinge& hk = d.parts[k].hinge;
            hk.pivot = f.ToWorld({ 0, 0, 0 });
            hk.axis = f.Axis(0);
            hk.angle = 0.0f;
            hk.maxAngle = d.parts[i].hinge.maxAngle;
            for (int ch : children)
                if (f.ToLocal(d.parts[ch].hinge.pivot).z * movedSign > 0.0f) d.parts[ch].hinge.parent = k;

            // hand the material on the moved side to the new part
            s.samples.resize(d.parts.size());
            PartSamples& a = s.samples[i];
            PartSamples& b = s.samples[k];
            PartSamples keepA;
            for (const Vector3& p : a.lattice) (f.ToLocal(p).z * movedSign > 0 ? b.lattice : keepA.lattice).push_back(p);
            for (const DetailSample& p : a.detail) (f.ToLocal(p.position).z * movedSign > 0 ? b.detail : keepA.detail).push_back(p);
            a = std::move(keepA);
            affected = { d.parts[i].id, d.parts[k].id };
            for (int ch : children) affected.push_back(d.parts[ch].id);
            return true;
        }

        bool ApplyMerge(FoldState& s, const FoldModel& m, std::mt19937& rng, std::vector<int>& affected) {
            Design& d = s.design;
            std::vector<int> c;
            for (int k = 0; k < (int)d.parts.size(); k++) {
                int i = d.parts[k].hinge.parent;
                if (i < 0 || d.FoldAngle(k) > 0.08f) continue;
                const SuperFrustum& a = d.parts[i].shape;
                const SuperFrustum& b = d.parts[k].shape;
                if (fabsf(a.bulge) > 0.05f || fabsf(b.bulge) > 0.05f || a.onion > 1e-3f || b.onion > 1e-3f) continue;
                if (fabsf(Vector3DotProduct(a.Axis(2), b.Axis(2))) < 0.96f) continue;
                Vector3 local = a.ToLocal(b.position);
                float inner = 0.5f * fminf(a.size.x, a.size.y);
                if (sqrtf(local.x * local.x + local.y * local.y) > 0.6f * inner) continue;
                float ra = fminf(a.size.x, a.size.y), rb = fminf(b.size.x, b.size.y);
                if (rb > 1.7f * ra || ra > 1.7f * rb) continue;
                c.push_back(k);
            }
            if (c.empty()) return false;
            int k = c[std::uniform_int_distribution<int>(0, (int)c.size() - 1)(rng)];
            int i = d.parts[k].hinge.parent;
            SuperFrustum a = d.parts[i].shape;
            const SuperFrustum& b = d.parts[k].shape;
            float ha = a.size.z * 0.5f, hb = b.size.z * 0.5f;
            float zb = a.ToLocal(b.position).z;
            float lo = fminf(-ha, zb - hb), hi = fmaxf(ha, zb + hb);
            float la = a.size.z, lb = b.size.z, wa = la / (la + lb), wb = lb / (la + lb);
            SuperFrustum merged = a;
            merged.position = a.ToWorld({ 0, 0, 0.5f * (lo + hi) });
            merged.size.z = hi - lo;
            merged.size.x = a.size.x * wa + b.size.x * wb;
            merged.size.y = a.size.y * wa + b.size.y * wb;
            merged.taper = a.taper * wa + b.taper * wb;
            merged.rounding = a.rounding * wa + b.rounding * wb;
            merged.dilation = a.dilation * wa + b.dilation * wb;
            merged.Repair(m.Spacing() * 0.5f);

            affected = { d.parts[i].id, d.parts[k].id };
            for (int ch : d.Children(k)) affected.push_back(d.parts[ch].id);
            d.parts[i].shape = merged;
            PartSamples& dst = s.samples[i];
            dst.lattice.insert(dst.lattice.end(), s.samples[k].lattice.begin(), s.samples[k].lattice.end());
            dst.detail.insert(dst.detail.end(), s.samples[k].detail.begin(), s.samples[k].detail.end());
            s.samples.erase(s.samples.begin() + k);
            d.RemovePart(k);   // children of k move up to i, keeping their pivots
            return true;
        }

        bool ApplyLock(FoldState& s, std::mt19937& rng, std::vector<int>& affected, bool lock) {
            Design& d = s.design;
            std::vector<int> c;
            for (int i = 0; i < (int)d.parts.size(); i++) {
                const Hinge& h = d.parts[i].hinge;
                if (h.parent < 0) continue;
                if (lock && !h.locked && h.angle < 0.15f) c.push_back(i);
                if (!lock && h.locked) c.push_back(i);
            }
            if (c.empty()) return false;
            int i = c[std::uniform_int_distribution<int>(0, (int)c.size() - 1)(rng)];
            d.parts[i].hinge.locked = lock;
            d.parts[i].hinge.angle = 0.0f;
            affected = { d.parts[i].id };
            return true;
        }

        bool ApplyAxisSnap(FoldState& s, const FoldModel& m, std::mt19937& rng, std::vector<int>& affected) {
            int i = RandomHingedPart(s.design, rng, false);
            if (i < 0 || s.design.parts[i].hinge.locked) return false;
            std::vector<Vector3> axes = CandidateAxes(s, i);
            Vector3 axis = axes[std::uniform_int_distribution<int>(0, (int)axes.size() - 1)(rng)];
            std::vector<Vector3> before = SubtreeClosed(s, i, 300);
            s.design.parts[i].hinge.axis = axis;
            KeepPose(s, i, m.Params(), before);
            affected = { s.design.parts[i].id };
            return true;
        }

        bool ApplyReseat(FoldState& s, const FoldModel& m, std::mt19937& rng, std::vector<int>& affected, bool flip) {
            int i = RandomHingedPart(s.design, rng, false);
            if (i < 0 || s.design.parts[i].hinge.locked) return false;
            Hinge& h = s.design.parts[i].hinge;
            if (flip) h.axis = Vector3Negate(h.axis);
            float maxA = MaxAngle(s.design, i, m.Params());
            h.angle = SearchAngle(maxA, 16, [&](float a) {
                s.design.parts[i].hinge.angle = a;
                return m.Evaluate(s);
            });
            affected = { s.design.parts[i].id };
            return true;
        }

        bool ApplyRule(Rule r, FoldState& s, const FoldModel& m, std::mt19937& rng, std::vector<int>& affected) {
            switch (r) {
                case RuleReparent: return ApplyReparent(s, m, rng, affected);
                case RuleSplit:    return ApplySplit(s, m, rng, affected);
                case RuleMerge:    return ApplyMerge(s, m, rng, affected);
                case RuleLock:     return ApplyLock(s, rng, affected, true);
                case RuleUnlock:   return ApplyLock(s, rng, affected, false);
                case RuleAxisSnap: return ApplyAxisSnap(s, m, rng, affected);
                case RuleFlip:     return ApplyReseat(s, m, rng, affected, true);
                case RuleReseat:   return ApplyReseat(s, m, rng, affected, false);
                default: return false;
            }
        }

        // ---- continuous descent ----------------------------------------------------------------------------------

        class Descent {
        public:
            Descent(const FoldModel& model) : m_Model(model) {
                float h = model.Spacing();
                const float steps[KindCount] = { 0.3f * h, 0.3f * h, 0.3f * h, 0.2f * h, 0.03f, 0.03f, 0.03f, 0.3f * h, 0.3f * h, 0.3f * h };
                const float rates[KindCount] = { 0.25f * h, 0.25f * h, 0.25f * h, 0.15f * h, 0.045f, 0.035f, 0.035f, 0.2f * h, 0.2f * h, 0.2f * h };
                for (int k = 0; k < KindCount; k++) { m_Step[k] = steps[k]; m_Rate[k] = rates[k]; }
            }

            std::vector<Param> Block(const Design& d, int hingeBlock, std::mt19937& rng, bool pivots) const {
                std::vector<Param> ps;
                for (int k = CenterX; k <= Radius; k++) ps.push_back({ (PKind)k, -1 });
                std::vector<int> hinges;
                for (int i = 0; i < (int)d.parts.size(); i++)
                    if (d.parts[i].hinge.parent >= 0 && !d.parts[i].hinge.locked) hinges.push_back(i);
                std::shuffle(hinges.begin(), hinges.end(), rng);
                if ((int)hinges.size() > hingeBlock) hinges.resize(hingeBlock);
                for (int i : hinges)
                    for (int k = Angle; k <= (pivots ? PivotZ : AxisV); k++) ps.push_back({ (PKind)k, i });
                return ps;
            }

            std::vector<float> Gradient(const FoldState& s, const std::vector<Param>& ps) const {
                std::vector<float> g(ps.size(), 0.0f);
                tbb::parallel_for(0, (int)ps.size(), [&](int i) {
                    float h = m_Step[ps[i].kind];
                    Design a = s.design, b = s.design;
                    ApplyParam(a, ps[i], h);
                    ApplyParam(b, ps[i], -h);
                    g[i] = (m_Model.Evaluate(a, s.samples) - m_Model.Evaluate(b, s.samples)) / (2.0f * h);
                });
                return g;
            }

            void Step(FoldState& s, const std::vector<Param>& ps, const std::vector<float>& g, float decay) {
                const float b1 = 0.9f, b2 = 0.999f;
                for (size_t i = 0; i < ps.size(); i++) {
                    int id = ps[i].part >= 0 ? s.design.parts[ps[i].part].id : 0;
                    AdamCell& c = m_Adam[((uint64_t)id << 8) | ps[i].kind];
                    c.t++;
                    c.m = b1 * c.m + (1 - b1) * g[i];
                    c.v = b2 * c.v + (1 - b2) * g[i] * g[i];
                    float mh = c.m / (1 - powf(b1, (float)c.t)), vh = c.v / (1 - powf(b2, (float)c.t));
                    ApplyParam(s.design, ps[i], -m_Rate[ps[i].kind] * decay * mh / (sqrtf(vh) + 1e-12f));
                }
                Repair(s, ps);
            }

            // Sign descent on the affected hinges, used to settle a rewrite before judging it (SRD's LocalOptimize).
            float Settle(FoldState& s, const std::vector<int>& ids, float current) const {
                std::vector<Param> ps;
                for (int id : ids) {
                    int i = s.design.IndexOf(id);
                    if (i >= 0 && s.design.parts[i].hinge.parent >= 0 && !s.design.parts[i].hinge.locked)
                        for (int k = Angle; k <= AxisV; k++) ps.push_back({ (PKind)k, i });
                }
                if (ps.empty()) return current;
                for (int step = 0; step < 2; step++) {
                    FoldState trial = s;
                    for (const Param& p : ps) {
                        float h = m_Step[p.kind];
                        Design a = s.design, b = s.design;
                        ApplyParam(a, p, h);
                        ApplyParam(b, p, -h);
                        float g = m_Model.Evaluate(a, s.samples) - m_Model.Evaluate(b, s.samples);
                        if (g != 0.0f) ApplyParam(trial.design, p, (g > 0 ? -1.0f : 1.0f) * m_Rate[p.kind]);
                    }
                    Repair(trial, ps);
                    float l = m_Model.Evaluate(trial);
                    if (l >= current) break;
                    s.design = std::move(trial.design);
                    current = l;
                }
                return current;
            }

            void Repair(FoldState& s, const std::vector<Param>& ps) const {
                m_Model.RepairState(s);
                std::vector<int> pivots;
                for (const Param& p : ps)
                    if (p.kind >= PivotX && p.kind <= PivotZ) pivots.push_back(p.part);
                std::sort(pivots.begin(), pivots.end());
                pivots.erase(std::unique(pivots.begin(), pivots.end()), pivots.end());
                for (int i : pivots) m_Model.ProjectPivot(s, i);
            }
        private:
            const FoldModel& m_Model;
            float m_Step[KindCount];
            float m_Rate[KindCount];
            std::unordered_map<uint64_t, AdamCell> m_Adam;
        };

        struct Candidate {
            FoldState state;
            Rule rule = RuleReseat;
            std::vector<int> affected;
            bool valid = false;
            float loss = 0.0f;
        };

        // Copies the hinges of the given parts (by id) from one design into another. False if a part is missing
        // or the result would contain a cycle.
        bool TransferHinges(Design& dst, const Design& src, const std::vector<int>& ids) {
            Design trial = dst;
            for (int id : ids) {
                int si = src.IndexOf(id), di = trial.IndexOf(id);
                if (si < 0 || di < 0) return false;
                Hinge h = src.parts[si].hinge;
                if (h.parent >= 0) {
                    h.parent = trial.IndexOf(src.parts[h.parent].id);
                    if (h.parent < 0) return false;
                }
                trial.parts[di].hinge = h;
            }
            if (HasCycle(trial)) return false;
            dst = std::move(trial);
            return true;
        }
    }

    // ---- FoldModel ---------------------------------------------------------------------------------------------

    FoldModel::FoldModel(std::shared_ptr<const ShapeAnalysis> shape, const FoldParams& params) : m_Shape(std::move(shape)) {
        SetParams(params);
    }

    void FoldModel::SetParams(const FoldParams& p) {
        m_Params = p;
        m_Lattice.clear();
        m_Detail.clear();
        if (!m_Shape || !m_Shape->ok) { m_Shape = nullptr; return; }
        const ShapeField& f = m_Shape->field;
        size_t count = m_Shape->interior.size();
        float target = std::clamp((float)p.samplesPerPart * 14.0f, 1500.0f, 9000.0f);
        int stride = std::max(1, (int)lroundf(cbrtf((float)count / target)));
        for (const Vector3& q : m_Shape->interior) {
            int i = (int)lroundf((q.x - f.origin.x) / f.voxel), j = (int)lroundf((q.y - f.origin.y) / f.voxel),
                k = (int)lroundf((q.z - f.origin.z) / f.voxel);
            if (i % stride == 0 && j % stride == 0 && k % stride == 0) m_Lattice.push_back(q);
        }
        m_Spacing = stride * f.voxel;
        size_t dstride = std::max<size_t>(1, m_Shape->detail.size() / 1500);
        float wsum = 0;
        for (size_t i = 0; i < m_Shape->detail.size(); i += dstride) {
            m_Detail.push_back(m_Shape->detail[i]);
            wsum += m_Shape->detail[i].weight;
        }
        m_DetailScale = wsum > 0 ? (float)m_Lattice.size() / wsum : 0.0f;
        m_ShellDirs = Utils::FibonacciSphere(std::max(64, p.shellSamples));
    }

    float FoldModel::RadiusMin() const { return EqualRadius() * fmaxf(0.05f, 1.0f - m_Params.volumeTolerance); }
    float FoldModel::RadiusMax() const { return EqualRadius() * (1.0f + m_Params.volumeTolerance); }

    FoldState FoldModel::MakeState(const Design& d) const {
        FoldState s;
        s.design = d;
        Assign(s);
        return s;
    }

    void FoldModel::Assign(FoldState& s) const {
        const Design& d = s.design;
        int n = (int)d.parts.size();
        s.samples.assign(n, PartSamples{});
        if (n == 0) return;
        auto nearest = [&](Vector3 p) {
            int best = 0;
            float bd = 1e30f;
            for (int j = 0; j < n; j++) {
                const SuperFrustum& f = d.parts[j].shape;
                float lower = Vector3Distance(p, f.position) - f.BoundingRadius();
                if (lower > bd) continue;
                float v = f.Distance(p);
                if (v < bd) { bd = v; best = j; }
            }
            return best;
        };
        std::vector<int> owner(m_Lattice.size()), downer(m_Detail.size());
        tbb::parallel_for(tbb::blocked_range<size_t>(0, m_Lattice.size()), [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i < r.end(); i++) owner[i] = nearest(m_Lattice[i]);
        });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, m_Detail.size()), [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i < r.end(); i++) downer[i] = nearest(m_Detail[i].position);
        });
        for (size_t i = 0; i < m_Lattice.size(); i++) s.samples[owner[i]].lattice.push_back(m_Lattice[i]);
        for (size_t i = 0; i < m_Detail.size(); i++) s.samples[downer[i]].detail.push_back(m_Detail[i]);
    }

    float FoldModel::CollisionLoss(const std::vector<PartSamples>& samples, const std::vector<Rigid>& xf, float* fraction,
                                   std::vector<Vector3>* lost, int stride) const {
        // per-thread scratch, reused between calls (the optimiser evaluates in parallel)
        thread_local std::vector<Vector3> pts;
        thread_local std::vector<int> owner;
        thread_local PointHash hash;
        pts.clear();
        owner.clear();
        for (size_t i = 0; i < samples.size(); i++)
            for (size_t k = 0; k < samples[i].lattice.size(); k += stride) {
                pts.push_back(xf[i].Apply(samples[i].lattice[k]));
                owner.push_back((int)i);
            }
        if (pts.empty()) { if (fraction) *fraction = 0; return 0; }
        const float h = 0.9f * m_Spacing, h2 = h * h, hard2 = 0.75f * 0.75f * m_Spacing * m_Spacing;
        hash.Build(pts, h);
        double loss = 0;
        size_t colliding = 0;
        for (size_t a = 0; a < pts.size(); a++) {
            bool hit = false;
            hash.Query(pts[a], [&](int b) {
                if (owner[b] == owner[a]) return;
                float d2 = Vector3DistanceSqr(pts[a], pts[b]);
                if (d2 >= h2) return;
                float k = 1.0f - sqrtf(d2) / h;
                loss += k * k;
                hit |= d2 < hard2;
            });
            if (hit) {
                colliding++;
                if (lost) lost->push_back(pts[a]);
            }
        }
        if (fraction) *fraction = (float)colliding / pts.size();
        return (float)(loss / pts.size());
    }

    float FoldModel::Evaluate(const Design& d, const std::vector<PartSamples>& samples, FoldMetrics* metrics,
                              FoldVisual* visual) const {
        int n = (int)d.parts.size();
        if (n == 0 || (int)samples.size() != n || !m_Shape) {
            if (metrics) *metrics = FoldMetrics{};
            return 0.0f;
        }
        const FoldParams& P = m_Params;
        std::vector<Rigid> xf = d.Transforms(1.0f);
        Vector3 c = d.ball.center;
        float R = fmaxf(d.ball.radius, 1e-4f);
        const float tauTrim = 0.5f * m_Spacing, tauShell = 0.5f * m_Spacing;

        // material outside the sphere is cut away (half a sample spacing of slack: samples sit on a voxel grid)
        double trim = 0, trimWeight = 0, trimHard = 0;
        const float slack = 0.5f * m_Spacing;
        for (int i = 0; i < n; i++) {
            for (const Vector3& p : samples[i].lattice) {
                Vector3 q = xf[i].Apply(p);
                float e = Vector3Distance(q, c) - R - slack;
                trim += Sigmoid(e / tauTrim) + 0.3f * fmaxf(e, 0.0f) / R;
                trimWeight += 1.0;
                if (e > 0) { trimHard += 1.0; if (visual) visual->lost.push_back(q); }
            }
            for (const DetailSample& ds : samples[i].detail) {
                float w = P.wDetail * ds.weight * m_DetailScale;
                if (w <= 0) continue;
                float e = Vector3Distance(xf[i].Apply(ds.position), c) - R - slack;
                trim += w * (Sigmoid(e / tauTrim) + 0.3f * fmaxf(e, 0.0f) / R);
                trimWeight += w;
                if (e > 0) trimHard += w;
            }
        }
        float trimLoss = trimWeight > 0 ? (float)(trim / trimWeight) : 0.0f;

        // material of different parts in the same place
        float collisionFraction = 0, pathFraction = 0;
        float collision = CollisionLoss(samples, xf, &collisionFraction, visual ? &visual->lost : nullptr);
        float path = 0;
        if (P.checkPath && P.wPath > 0) path = CollisionLoss(samples, d.Transforms(0.5f), &pathFraction, nullptr, 2);

        // the sphere shell must be covered exactly once
        std::vector<Vector3> centre(n);
        std::vector<float> reach(n);
        for (int j = 0; j < n; j++) {
            centre[j] = xf[j].Apply(d.parts[j].shape.position);
            reach[j] = d.parts[j].shape.BoundingRadius() + 3.0f * tauShell;
        }
        double uncovered = 0, doubled = 0;
        int coveredHard = 0, doubledHard = 0;
        float shellR = R - 0.5f * m_Spacing;
        for (const Vector3& u : m_ShellDirs) {
            Vector3 y = Vector3Add(c, Vector3Scale(u, shellR));
            float keep = 1.0f, sum = 0.0f;
            for (int j = 0; j < n; j++) {
                if (Vector3DistanceSqr(y, centre[j]) > reach[j] * reach[j]) continue;
                float o = Sigmoid(-d.parts[j].shape.Distance(xf[j].ApplyInverse(y)) / tauShell);
                keep *= 1.0f - o;
                sum += o;
            }
            uncovered += keep;
            doubled += fmaxf(sum - 1.0f, 0.0f);
            int state = keep < 0.5f ? (sum > 1.5f ? 2 : 1) : 0;
            coveredHard += state > 0;
            doubledHard += state == 2;
            if (visual) { visual->shell.push_back(y); visual->shellState.push_back((uint8_t)state); }
        }
        float K = (float)m_ShellDirs.size();

        // hinge pivots must sit inside both parts they join
        // ... and must stay inside the closed sphere, or the link between the parts is cut away
        double contact = 0, contactHard = 0, link = 0;
        int joints = 0;
        for (int i = 0; i < n; i++) {
            const Hinge& h = d.parts[i].hinge;
            if (h.parent < 0) continue;
            float dp = fmaxf(d.parts[h.parent].shape.Distance(h.pivot), 0.0f);
            float dc = fmaxf(d.parts[i].shape.Distance(h.pivot), 0.0f);
            float v = (dp + dc) / m_Spacing;
            contact += v * v;
            contactHard += v;
            float out = fmaxf(Vector3Distance(xf[h.parent].Apply(h.pivot), c) - (R - m_Spacing), 0.0f) / m_Spacing;
            link += out * out;
            joints++;
        }
        float contactLoss = joints ? (float)(contact / joints) : 0.0f;
        float linkLoss = joints ? (float)(link / joints) : 0.0f;

        int hinges = d.CountHinges();
        float size = (R - EqualRadius()) / (EqualRadius() * fmaxf(P.volumeTolerance, 0.02f));
        float loss = P.wVolume * size * size + P.wCoverage * (float)(uncovered / K) + P.wShellOverlap * (float)(doubled / K) + P.wTrim * trimLoss +
                     P.wCollision * collision + P.wPath * path + P.wContact * contactLoss + P.wLink * linkLoss + P.hingeCost * hinges;
        if (metrics) {
            metrics->valid = true;
            metrics->coverage = coveredHard / K;
            metrics->shellOverlap = doubledHard / K;
            metrics->trimmed = trimWeight > 0 ? (float)(trimHard / trimWeight) : 0.0f;
            metrics->collision = collisionFraction;
            metrics->pathCollision = pathFraction;
            metrics->contact = joints ? (float)(contactHard / joints) : 0.0f;
            metrics->loss = loss;
            metrics->hinges = hinges;
            metrics->radius = R;
            metrics->radiusRatio = R / fmaxf(EqualRadius(), 1e-6f);
        }
        return loss;
    }

    void FoldModel::RepairState(FoldState& s) const {
        Design& d = s.design;
        d.RepairHinges();
        for (int i = 0; i < (int)d.parts.size(); i++) {
            Hinge& h = d.parts[i].hinge;
            h.angle = std::clamp(h.angle, 0.0f, MaxAngle(d, i, m_Params));
        }
        d.ball.radius = std::clamp(d.ball.radius, RadiusMin(), RadiusMax());
    }

    bool FoldModel::ContactPoint(const FoldState& s, int a, int b, Vector3* point, int* count) const {
        const Design& d = s.design;
        const float reach = 1.5f * m_Spacing;
        Vector3 sum = { 0, 0, 0 };
        int n = 0;
        for (const Vector3& p : s.samples[a].lattice)
            if (d.parts[b].shape.Distance(p) < reach) { sum = Vector3Add(sum, p); n++; }
        for (const Vector3& p : s.samples[b].lattice)
            if (d.parts[a].shape.Distance(p) < reach) { sum = Vector3Add(sum, p); n++; }
        *count = n;
        if (n == 0) return false;
        *point = Vector3Scale(sum, 1.0f / n);
        return true;
    }

    void FoldModel::ProjectPivot(FoldState& s, int i) const {
        Design& d = s.design;
        Hinge& h = d.parts[i].hinge;
        if (h.parent < 0) return;
        const SuperFrustum& P = d.parts[h.parent].shape;
        const SuperFrustum& C = d.parts[i].shape;
        for (int it = 0; it < 10; it++) {
            float dp = P.Distance(h.pivot), dc = C.Distance(h.pivot);
            float worst = fmaxf(dp, dc);
            if (worst <= 0.0f) return;
            Vector3 g = (dp > dc ? P : C).Gradient(h.pivot, 0.25f * m_Spacing);
            float len = Vector3Length(g);
            if (len < 1e-6f) return;
            h.pivot = Vector3Subtract(h.pivot, Vector3Scale(g, (worst + 0.1f * m_Spacing) / len));
        }
    }

    // ---- tree construction -------------------------------------------------------------------------------------

    void BuildFoldTree(FoldState& s, const FoldModel& model) {
        Design& d = s.design;
        int n = (int)d.parts.size();
        if (n == 0) return;
        if ((int)s.samples.size() != n) model.Assign(s);
        for (Part& p : d.parts) { p.hinge.parent = -1; p.hinge.angle = 0; p.hinge.locked = false; }

        int root = 0;
        for (int i = 1; i < n; i++)
            if (s.samples[i].lattice.size() > s.samples[root].lattice.size()) root = i;
        // keep the root first: it is the default root everywhere else
        if (root != 0) {
            std::swap(d.parts[0], d.parts[root]);
            std::swap(s.samples[0], s.samples[root]);
            root = 0;
        }

        // contact graph
        std::vector<std::vector<int>> score(n, std::vector<int>(n, 0));
        std::vector<std::vector<Vector3>> point(n, std::vector<Vector3>(n));
        tbb::parallel_for(0, n, [&](int i) {
            for (int j = i + 1; j < n; j++) {
                float r = d.parts[i].shape.BoundingRadius() + d.parts[j].shape.BoundingRadius() + 2 * model.Spacing();
                if (Vector3DistanceSqr(d.parts[i].shape.position, d.parts[j].shape.position) > r * r) continue;
                Vector3 p;
                int c;
                if (model.ContactPoint(s, i, j, &p, &c)) { score[i][j] = score[j][i] = c; point[i][j] = point[j][i] = p; }
            }
        });

        // maximum spanning tree (Prim), bridging disconnected pieces by their closest samples
        std::vector<char> in(n, 0);
        in[root] = 1;
        std::vector<int> order = { root };
        for (int added = 1; added < n; added++) {
            int bi = -1, bj = -1, best = 0;
            for (int i = 0; i < n; i++)
                if (in[i])
                    for (int j = 0; j < n; j++)
                        if (!in[j] && score[i][j] > best) { best = score[i][j]; bi = i; bj = j; }
            Vector3 pivot;
            if (bj < 0) {
                float bd = 1e30f;
                for (int i = 0; i < n; i++)
                    if (in[i])
                        for (int j = 0; j < n; j++) {
                            if (in[j]) continue;
                            const auto& A = s.samples[i].lattice;
                            const auto& B = s.samples[j].lattice;
                            size_t sa = std::max<size_t>(1, A.size() / 60), sb = std::max<size_t>(1, B.size() / 60);
                            for (size_t x = 0; x < A.size(); x += sa)
                                for (size_t y = 0; y < B.size(); y += sb) {
                                    float dd = Vector3DistanceSqr(A[x], B[y]);
                                    if (dd < bd) { bd = dd; bi = i; bj = j; pivot = Vector3Lerp(A[x], B[y], 0.5f); }
                                }
                            if (A.empty() || B.empty()) {
                                float dd = Vector3DistanceSqr(d.parts[i].shape.position, d.parts[j].shape.position);
                                if (dd < bd) { bd = dd; bi = i; bj = j; pivot = Vector3Lerp(d.parts[i].shape.position, d.parts[j].shape.position, 0.5f); }
                            }
                        }
            } else {
                pivot = point[bi][bj];
            }
            if (bj < 0) break;
            in[bj] = 1;
            order.push_back(bj);
            Hinge& h = d.parts[bj].hinge;
            h.parent = bi;
            h.pivot = pivot;
            h.angle = 0.0f;
            Vector3 out = Vector3Subtract(d.parts[bj].shape.position, pivot);
            h.axis = Vector3Length(out) > 1e-6f ? Utils::AnyPerpendicular(Vector3Normalize(out)) : Vector3{ 1, 0, 0 };
        }

        d.ball.center = model.Centroid();
        d.ball.radius = model.EqualRadius();
        model.RepairState(s);

        // tuck every subtree into the sphere, parents first, avoiding parts already placed
        const FoldParams& P = model.Params();
        float R = d.ball.radius, sp = model.Spacing();
        std::vector<char> placed(n, 0);
        placed[root] = 1;
        for (size_t oi = 1; oi < order.size(); oi++) {
            int i = order[oi];
            std::vector<int> sub = d.Subtree(i);
            std::vector<char> inSub(n, 0);
            for (int k : sub) inSub[k] = 1;
            std::vector<Rigid> xf = d.Transforms(1.0f);
            std::vector<Vector3> others;
            for (int k = 0; k < n; k++)
                if (placed[k] && !inSub[k])
                    for (const Vector3& p : s.samples[k].lattice) others.push_back(xf[k].Apply(p));
            PointHash hash;
            hash.Build(others, sp);
            std::vector<std::pair<int, Vector3>> mine;
            for (int k : sub) {
                size_t stride = std::max<size_t>(1, s.samples[k].lattice.size() / 150);
                for (size_t q = 0; q < s.samples[k].lattice.size(); q += stride) mine.push_back({ k, s.samples[k].lattice[q] });
            }
            auto objective = [&]() {
                std::vector<Rigid> t = d.Transforms(1.0f);
                double acc = 0;
                for (const auto& [k, p] : mine) {
                    Vector3 q = t[k].Apply(p);
                    float e = fmaxf(Vector3Distance(q, d.ball.center) - 0.92f * R, 0.0f) / sp;
                    acc += e * e;
                    int hits = 0;
                    hash.Query(q, [&](int b) { hits += Vector3DistanceSqr(q, others[b]) < 0.8f * sp * sp; });
                    acc += 2.0 * hits;
                }
                return (float)acc;
            };
            Hinge& h = d.parts[i].hinge;
            Vector3 bestAxis = h.axis;
            float bestAngle = 0, bestValue = objective();
            for (const Vector3& axis : CandidateAxes(s, i)) {
                h.axis = axis;
                float a = SearchAngle(MaxAngle(d, i, P), 18, [&](float v) { h.angle = v; return objective(); });
                h.angle = a;
                float v = objective();
                if (v < bestValue) { bestValue = v; bestAxis = axis; bestAngle = a; }
            }
            h.axis = bestAxis;
            h.angle = bestAngle;
            placed[i] = 1;
        }
        model.RepairState(s);
    }

    void AimHinge(FoldState& s, const FoldModel& model, int i) {
        Design& d = s.design;
        if (i < 0 || i >= (int)d.parts.size() || d.parts[i].hinge.parent < 0) return;
        if ((int)s.samples.size() != (int)d.parts.size()) model.Assign(s);
        Hinge& h = d.parts[i].hinge;
        h.locked = false;
        Vector3 bestAxis = h.axis;
        float bestAngle = h.angle, bestValue = model.Evaluate(s);
        std::vector<Vector3> axes = CandidateAxes(s, i);
        axes.push_back(h.axis);
        for (const Vector3& axis : axes) {
            h.axis = axis;
            float a = SearchAngle(MaxAngle(d, i, model.Params()), 12, [&](float v) { h.angle = v; return model.Evaluate(s); });
            h.angle = a;
            float v = model.Evaluate(s);
            if (v < bestValue) { bestValue = v; bestAxis = axis; bestAngle = a; }
        }
        h.axis = bestAxis;
        h.angle = bestAngle;
    }

    // ---- stochastic rewrite descent ------------------------------------------------------------------------------

    FoldResult OptimizeFold(const Design& start, std::shared_ptr<const ShapeAnalysis> shape, const FoldParams& params,
                            JobControl& control) {
        FoldResult result;
        auto t0 = Clock::now();
        if (!shape || !shape->ok) { result.message = "Load a model first."; return result; }
        if (start.parts.empty()) { result.message = "There are no superfrustums to fold. Decompose or add some first."; return result; }
        FoldModel model(shape, params);
        if (!model.Valid()) { result.message = "The model could not be analysed."; return result; }
        FoldState state = model.MakeState(start);

        int roots = 0;
        for (const Part& p : state.design.parts) roots += p.hinge.parent < 0;
        if (roots > 1 && !params.keepTree) {
            control.SetStage("Building the hinge tree");
            BuildFoldTree(state, model);
        }
        model.RepairState(state);
        float loss = model.Evaluate(state);
        FoldState best = state;
        float bestLoss = loss;
        control.PushLoss(loss);
        control.Publish(state.design);

        std::mt19937 rng(params.seed);
        Descent descent(model);
        std::vector<Rule> rules = { RuleLock, RuleUnlock, RuleAxisSnap, RuleFlip, RuleReseat };
        if (!params.keepTree) {
            rules.push_back(RuleReparent);
            rules.push_back(RuleMerge);
            if (params.allowSplit) rules.push_back(RuleSplit);
        }
        int applied = 0;
        int iterations = std::max(1, params.iterations);
        for (int it = 0; it < iterations && !control.Cancelled(); it++) {
            float decay = 0.25f + 0.75f * 0.5f * (1.0f + cosf(PI * it / iterations));
            char stage[96];
            snprintf(stage, sizeof(stage), "Step %d / %d  (%d parts, %d hinges)", it + 1, iterations,
                     (int)state.design.parts.size(), state.design.CountHinges());
            control.SetStage(stage);

            // continuous step on a random block of hinges plus the sphere
            int hinges = state.design.CountHinges();
            int block = std::clamp(hinges / 3, std::min(3, hinges), 6);
            std::vector<Param> ps = descent.Block(state.design, block, rng, it % 4 == 0);
            std::vector<float> g = descent.Gradient(state, ps);
            descent.Step(state, ps, g, decay);
            loss = model.Evaluate(state);

            // grammar rewrites: sample K, settle each, apply the compatible improving ones
            if (params.restructure && it % std::max(1, params.rewriteEvery) == std::max(1, params.rewriteEvery) - 1) {
                int K = std::max(1, params.candidates);
                std::vector<Candidate> cands(K);
                std::vector<unsigned> seeds(K);
                for (int k = 0; k < K; k++) seeds[k] = (unsigned)rng();
                tbb::parallel_for(0, K, [&](int k) {
                    std::mt19937 local(seeds[k]);
                    Candidate& c = cands[k];
                    for (int attempt = 0; attempt < 6 && !c.valid; attempt++) {
                        c.state = state;
                        c.affected.clear();
                        c.rule = rules[std::uniform_int_distribution<int>(0, (int)rules.size() - 1)(local)];
                        c.valid = ApplyRule(c.rule, c.state, model, local, c.affected);
                    }
                    if (!c.valid) return;
                    model.RepairState(c.state);
                    c.loss = descent.Settle(c.state, c.affected, model.Evaluate(c.state));
                });
                std::vector<int> order;
                for (int k = 0; k < K; k++)
                    if (cands[k].valid && cands[k].loss < loss - 1e-6f) order.push_back(k);
                std::sort(order.begin(), order.end(), [&](int a, int b) { return cands[a].loss < cands[b].loss; });
                if (!order.empty()) {
                    const Candidate& top = cands[order[0]];
                    FoldState merged = top.state;
                    std::vector<int> used = top.affected;
                    int count = 1;
                    for (size_t o = 1; o < order.size(); o++) {
                        const Candidate& c = cands[order[o]];
                        if (Structural(c.rule)) continue;
                        bool clash = false;
                        for (int id : c.affected) clash |= std::find(used.begin(), used.end(), id) != used.end();
                        if (clash) continue;
                        if (!TransferHinges(merged.design, c.state.design, c.affected)) continue;
                        used.insert(used.end(), c.affected.begin(), c.affected.end());
                        count++;
                    }
                    model.RepairState(merged);
                    float mergedLoss = model.Evaluate(merged);
                    if (mergedLoss > top.loss) { merged = top.state; mergedLoss = top.loss; count = 1; }
                    if (mergedLoss < loss) {
                        state = std::move(merged);
                        loss = mergedLoss;
                        applied += count;
                    }
                }
            }

            if (loss < bestLoss) { bestLoss = loss; best = state; }
            control.PushLoss(loss);
            control.SetProgress((it + 1) / (float)iterations);
            if (it % 3 == 0 || it == iterations - 1) control.Publish(state.design);
        }

        result.design = best.design;
        model.Evaluate(best, &result.metrics);
        result.ok = true;
        result.cancelled = control.Cancelled();
        result.rewritesApplied = applied;
        result.milliseconds = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        char buf[200];
        snprintf(buf, sizeof(buf), "%sFold optimised in %.1f s: shell %.0f%% covered, %.1f%% detail trimmed, %.1f%% colliding, %d hinges, %d rewrites.",
                 result.cancelled ? "(stopped) " : "", result.milliseconds / 1000.0, result.metrics.coverage * 100.0f,
                 result.metrics.trimmed * 100.0f, result.metrics.collision * 100.0f, result.metrics.hinges, applied);
        result.message = buf;
        (void)RuleNames;
        return result;
    }
}
