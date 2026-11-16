#include "finish.hpp"
#include "data/clash.hpp"
#include "data/field.hpp"
#include "util/box.hpp"
#include "util/utils.hpp"
#include <raymath.h>
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <functional>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace Bakugan {
    namespace {
        // f(0) .. f(n - 1) on every core.
        void ParallelFor(int n, const std::function<void(int)>& f) {
            int workers = std::min(n, std::max(1, (int)std::thread::hardware_concurrency()));
            std::atomic<int> next{ 0 };
            auto work = [&] {
                for (int i = next++; i < n; i = next++) f(i);
            };
            std::vector<std::thread> threads;
            for (int k = 1; k < workers; k++) threads.emplace_back(work);
            work();
            for (std::thread& t : threads) t.join();
        }

        bool Turns(const Hinge& h) { return fabsf(h.end - h.start) > 1e-4f && h.t1 - h.t0 > 1e-5f; }

        // The hinges that turn, in the order they open.
        std::vector<int> MovingOrder(const Design& d) {
            std::vector<int> order;
            for (int h : OpeningOrder(d))
                if (Turns(d.hinges[h])) order.push_back(h);
            return order;
        }

        // Directions spread evenly over a sphere.
        std::vector<Vector3> SphereDirections(int n) {
            std::vector<Vector3> out(n);
            for (int i = 0; i < n; i++) {
                float y = 1.0f - 2.0f * (i + 0.5f) / n, r = sqrtf(fmaxf(0.0f, 1.0f - y * y)), phi = i * 2.39996323f;
                out[i] = { r * cosf(phi), y, r * sinf(phi) };
            }
            return out;
        }

        // A frame on the sphere's surface: z out of it, x and y along it.
        struct Frame {
            Vector3 o = { 0, 0, 0 }, x = { 1, 0, 0 }, y = { 0, 1, 0 }, z = { 0, 0, 1 };
            Vector3 At(float a, float b, float c) const {
                return Vector3Add(o, Vector3Add(Vector3Scale(x, a), Vector3Add(Vector3Scale(y, b), Vector3Scale(z, c))));
            }
            Vector3 Local(Vector3 p) const {
                Vector3 q = Vector3Subtract(p, o);
                return { Vector3DotProduct(q, x), Vector3DotProduct(q, y), Vector3DotProduct(q, z) };
            }
        };

        // A box given by its ranges in a frame, as a superfrustum (rounding: of its outline seen from above).
        SuperFrustum BoxIn(const Frame& f, float x0, float x1, float y0, float y1, float z0, float z1, float rounding = 0.0f) {
            SuperFrustum s;
            s.position = f.At(0.5f * (x0 + x1), 0.5f * (y0 + y1), 0.5f * (z0 + z1));
            Matrix m = MatrixIdentity();
            m.m0 = f.x.x; m.m1 = f.x.y; m.m2 = f.x.z;
            m.m4 = f.y.x; m.m5 = f.y.y; m.m6 = f.y.z;
            m.m8 = f.z.x; m.m9 = f.z.y; m.m10 = f.z.z;
            s.rotation = QuaternionNormalize(QuaternionFromMatrix(m));
            s.size = { x1 - x0, y1 - y0, z1 - z0 };
            s.rounding = rounding;
            return s;
        }

        // The frame at an angle along the great circle from direction a toward direction b (0: at a).
        Frame OnCircle(const SphereInfo& sp, Vector3 a, Vector3 b, float angle) {
            Vector3 t = Vector3Subtract(b, Vector3Scale(a, Vector3DotProduct(a, b)));
            t = Vector3Length(t) > 1e-6f ? Vector3Normalize(t) : Utils::AnyPerpendicular(a);
            Frame f;
            f.z = Vector3Add(Vector3Scale(a, cosf(angle)), Vector3Scale(t, sinf(angle)));
            f.x = Vector3Add(Vector3Scale(a, -sinf(angle)), Vector3Scale(t, cosf(angle)));
            f.y = Vector3CrossProduct(f.z, f.x);
            f.o = Vector3Add(sp.centre, Vector3Scale(f.z, sp.radius));
            return f;
        }

        const Shape* SphereShape(const Design& d) {
            for (const Shape& s : d.shapes)
                if (s.name == "Sphere" && s.op == Op::Union && s.sf.rounding > 0.99f && s.sf.dilation > 0.0f) return &s;
            return nullptr;
        }

        // A body's "Core": everything it was before anything was claimed for it, as one feature, so that what is
        // claimed for it sits beside that (and what is cut from the core doesn't cut the claims).
        int CoreOf(const Design& d, int bodyId) {
            for (int g : d.ChildGroups(bodyId))
                if (d.groups[g].name == "Core") return d.groups[g].id;
            return -1;
        }

        int WrapCore(Design& d, int bodyId) {
            int core = CoreOf(d, bodyId);
            if (core >= 0) return core;
            int body = d.GroupIndex(bodyId);
            float blend = d.groups[body].blend;
            core = d.groups[d.AddGroup("Core", bodyId)].id;
            d.groups[d.GroupIndex(core)].blend = blend;
            d.groups[d.GroupIndex(core)].color = d.groups[d.GroupIndex(bodyId)].color;
            for (Shape& s : d.shapes)
                if (s.group == bodyId) s.group = core;
            for (Group& g : d.groups)
                if (g.parent == bodyId && g.id != core) g.parent = core;
            return core;
        }

        // Whether any of a body is inside (or within `margin` of) a shape.
        bool Reaches(const BodyField& body, const SuperFrustum& s, float margin) {
            BoundingBox b = s.LocalBounds();
            Vector3 e = Vector3Subtract(b.max, b.min);
            const int n = 5;
            float slack = 0.5f * Vector3Length(e) / n + margin;
            for (int k = 0; k <= n; k++)
                for (int j = 0; j <= n; j++)
                    for (int i = 0; i <= n; i++) {
                        Vector3 p = s.ToWorld({ b.min.x + e.x * i / n, b.min.y + e.y * j / n, b.min.z + e.z * k / n });
                        if (body.Distance(p) < slack) return true;
                    }
            return false;
        }

        // Claims a region of the sphere for one body: `solids` (where they are inside the sphere) become its material,
        // and are cut from every other body; `gaps` are cut from every body (its own core too). The feature's id.
        int Claim(Design& d, const SphereInfo& sp, int holderBody, const std::vector<SuperFrustum>& solids, const std::vector<SuperFrustum>& gaps,
                  const std::string& name) {
            const Shape* sphere = SphereShape(d);
            SuperFrustum ball = sphere ? sphere->sf : SuperFrustum{};
            Color color = d.groups[d.GroupIndex(holderBody)].color;
            std::vector<int> bodies;
            std::vector<BodyField> fields;
            for (int g : d.Bodies()) {
                bodies.push_back(d.groups[g].id);
                fields.push_back(BodyField::Body(d, d.groups[g].id));
            }
            int core = WrapCore(d, holderBody);
            for (const SuperFrustum& g : gaps) d.shapes[d.AddShape(g, core, name + " gap", Op::Cut)].color = color;
            for (size_t k = 0; k < bodies.size(); k++) {
                if (bodies[k] == holderBody) continue;
                int into = CoreOf(d, bodies[k]);
                if (into < 0) into = bodies[k];
                for (const SuperFrustum& s : solids)
                    if (Reaches(fields[k], s, 0.2f)) d.AddShape(s, into, name + " space", Op::Cut);
                for (const SuperFrustum& g : gaps)
                    if (Reaches(fields[k], g, 0.2f)) d.AddShape(g, into, name + " gap", Op::Cut);
            }
            int feature = d.groups[d.AddGroup(name, holderBody)].id;
            d.groups[d.GroupIndex(feature)].color = color;
            for (const SuperFrustum& s : solids) d.shapes[d.AddShape(s, feature, name)].color = color;
            // ... inside the sphere only: less everything outside it
            int outside = d.groups[d.AddGroup("Flush", feature)].id;
            d.groups[d.GroupIndex(outside)].op = Op::Cut;
            SuperFrustum around;
            around.position = sp.centre;
            around.size = Vector3Scale({ 1, 1, 1 }, 2.6f * sp.radius);
            d.AddShape(around, outside, "Around");
            d.AddShape(ball, outside, "Sphere", Op::Cut);
            return feature;
        }

        // Points of a shape's box (a grid in its own frame) that are inside it.
        std::vector<Vector3> PointsIn(const SuperFrustum& s, int nx, int ny, int nz, float* volumeEach) {
            BoundingBox b = s.LocalBounds();
            Vector3 e = Vector3Subtract(b.max, b.min);
            if (volumeEach) *volumeEach = e.x * e.y * e.z / (nx * ny * nz);
            std::vector<Vector3> out;
            for (int k = 0; k < nz; k++)
                for (int j = 0; j < ny; j++)
                    for (int i = 0; i < nx; i++) {
                        Vector3 l = { b.min.x + e.x * (i + 0.5f) / nx, b.min.y + e.y * (j + 0.5f) / ny, b.min.z + e.z * (k + 0.5f) / nz };
                        if (s.LocalDistance(l) < 0.0f) out.push_back(s.ToWorld(l));
                    }
            return out;
        }

        std::vector<float> ClosedAngles(const Design& d) {
            std::vector<float> a(d.hinges.size());
            for (size_t h = 0; h < d.hinges.size(); h++) a[h] = d.hinges[h].end;
            return a;
        }

        // The hinge's angle a little way open: by `angle` (radians), or all of its turn if that is less.
        float AjarAngle(const Hinge& h, float angle) {
            float span = h.start - h.end;
            return h.end + (fabsf(span) > angle ? (span > 0.0f ? angle : -angle) : span);
        }

        // Which body (field) each direction's point just under the sphere's surface belongs to (-1: none).
        std::vector<int> Owners(const SphereInfo& sp, const BodySet& set, const std::vector<Vector3>& dirs) {
            std::vector<int> owner(dirs.size(), -1);
            ParallelFor((int)dirs.size(), [&](int k) {
                Vector3 p = Vector3Add(sp.centre, Vector3Scale(dirs[k], 0.985f * sp.radius));
                for (size_t i = 0; i < set.fields.size(); i++)
                    if (set.fields[i].Distance(p) < 0.0f) {
                        owner[k] = (int)i;
                        break;
                    }
            });
            return owner;
        }

        // How fast (per radian) a point turns about a hinge as it opens, and which way.
        Vector3 OpeningVelocity(const Hinge& h, Vector3 p) {
            Vector3 axis = Vector3Length(h.axis) > 1e-6f ? Vector3Normalize(h.axis) : Vector3{ 1, 0, 0 };
            return Vector3Scale(Vector3CrossProduct(axis, Vector3Subtract(p, h.pivot)), h.start >= h.end ? 1.0f : -1.0f);
        }
    }

    SphereInfo FindSphere(const Design& d) {
        SphereInfo s;
        if (const Shape* sh = SphereShape(d)) {
            s.ok = true;
            s.centre = sh->sf.position;
            s.radius = sh->sf.dilation + 0.5f * sh->sf.size.x;
        }
        return s;
    }

    bool HeldShut(const Design& d, int h, float tolerance, double* overlap) {
        if (overlap) *overlap = 0.0;
        if (h < 0 || h >= (int)d.hinges.size() || !Turns(d.hinges[h])) return false;
        BoundingBox bounds = d.Bounds();
        const float cell = std::clamp(Vector3Distance(bounds.min, bounds.max) / 56.0f, 0.3f, 2.0f);
        BodySet set = MakeBodySet(d, cell);
        std::vector<std::vector<char>> under = HingeSubtrees(d);
        // the farthest anything it turns is from its axis: opened until that has moved 0.6, 1.2 and 2.4 mm
        float farthest = 1.0f;
        for (size_t i = 0; i < set.fields.size(); i++)
            if (under[h][set.group[i]])
                for (int k = 0; k < 8; k++) farthest = fmaxf(farthest, Vector3Length(OpeningVelocity(d.hinges[h], BoxCorner(set.boxes[i], k))));
        Vector3 size = Vector3Subtract(bounds.max, bounds.min);
        const float reach = 0.15f * fmaxf(size.x, fmaxf(size.y, size.z));
        double most = 0.0;
        for (float moved : { 0.6f, 1.2f, 2.4f }) {
            std::vector<float> angles = ClosedAngles(d);
            angles[h] = AjarAngle(d.hinges[h], moved / farthest);
            std::vector<Rigid> xf = d.TransformsAt(angles);
            Clash clash;
            for (size_t i = 0; i < set.fields.size(); i++)
                for (size_t j = i + 1; j < set.fields.size(); j++)
                    if (under[h][set.group[i]] != under[h][set.group[j]])
                        SampleOverlap({ &set.fields[i], set.boxes[i], xf[set.group[i]] }, { &set.fields[j], set.boxes[j], xf[set.group[j]] }, cell,
                                      { 0.5f, 0.5f, 0.5f }, &clash, true);
            // what is caught far from the axis holds; next to the axis two edges only rub
            double volume = 0.0;
            const Rigid& parent = xf[d.GroupIndex(d.hinges[h].parent)];
            for (const OverlapSample& s : clash.samples)
                if (Vector3Length(OpeningVelocity(d.hinges[h], parent.ApplyInverse(s.at))) > reach) volume += s.volume;
            most = std::max(most, volume);
        }
        if (overlap) *overlap = most;
        return most > tolerance;
    }

    // ---- carving collisions away ------------------------------------------------------------------------------------------

    namespace {
        // Eigenvectors of a symmetric 3x3 matrix (Jacobi), as the columns of v.
        void Eigen(double a[3][3], double v[3][3]) {
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++) v[i][j] = i == j;
            for (int sweep = 0; sweep < 24; sweep++) {
                double off = fabs(a[0][1]) + fabs(a[0][2]) + fabs(a[1][2]);
                if (off < 1e-12) break;
                for (int p = 0; p < 3; p++)
                    for (int q = p + 1; q < 3; q++) {
                        if (fabs(a[p][q]) < 1e-15) continue;
                        double theta = 0.5 * atan2(2.0 * a[p][q], a[q][q] - a[p][p]), c = cos(theta), s = sin(theta);
                        for (int k = 0; k < 3; k++) {
                            double akp = a[k][p], akq = a[k][q];
                            a[k][p] = c * akp - s * akq;
                            a[k][q] = s * akp + c * akq;
                        }
                        for (int k = 0; k < 3; k++) {
                            double apk = a[p][k], aqk = a[q][k];
                            a[p][k] = c * apk - s * aqk;
                            a[q][k] = s * apk + c * aqk;
                        }
                        for (int k = 0; k < 3; k++) {
                            double vkp = v[k][p], vkq = v[k][q];
                            v[k][p] = c * vkp - s * vkq;
                            v[k][q] = s * vkp + c * vkq;
                        }
                    }
            }
        }

        struct Spot {
            Vector3 at;
            float volume;
        };

        // Rounded boxes that cover a set of points (each standing for `cell` of space), with `margin` to spare: the
        // points' own box along their principal axes, split in two (and again) while it is mostly empty.
        void Cover(std::vector<Spot> pts, float cell, float margin, int depth, std::vector<SuperFrustum>* out) {
            if (pts.empty()) return;
            Vector3 mean = { 0, 0, 0 };
            for (const Spot& s : pts) mean = Vector3Add(mean, s.at);
            mean = Vector3Scale(mean, 1.0f / pts.size());
            double cov[3][3] = {}, vec[3][3];
            for (const Spot& s : pts) {
                Vector3 q = Vector3Subtract(s.at, mean);
                const float c[3] = { q.x, q.y, q.z };
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++) cov[i][j] += c[i] * c[j];
            }
            Eigen(cov, vec);
            // the axes, longest spread first, right handed
            int order[3] = { 0, 1, 2 };
            std::sort(order, order + 3, [&](int a, int b) { return cov[a][a] > cov[b][b]; });
            Frame f;
            f.o = mean;
            f.x = Vector3Normalize({ (float)vec[0][order[0]], (float)vec[1][order[0]], (float)vec[2][order[0]] });
            f.y = { (float)vec[0][order[1]], (float)vec[1][order[1]], (float)vec[2][order[1]] };
            f.y = Vector3Normalize(Vector3Subtract(f.y, Vector3Scale(f.x, Vector3DotProduct(f.x, f.y))));
            if (Vector3Length(f.y) < 0.5f) f.y = Utils::AnyPerpendicular(f.x);
            f.z = Vector3CrossProduct(f.x, f.y);
            Vector3 lo = { FLT_MAX, FLT_MAX, FLT_MAX }, hi = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
            double volume = 0.0;
            for (const Spot& s : pts) {
                Vector3 l = f.Local(s.at);
                lo = Vector3Min(lo, l);
                hi = Vector3Max(hi, l);
                volume += s.volume;
            }
            Vector3 e = Vector3Subtract(hi, lo);
            double boxVolume = (double)(e.x + cell) * (e.y + cell) * (e.z + cell);
            if (pts.size() >= 12 && depth < 4 && boxVolume > 3.0 * volume) {
                // mostly empty (a bent or scattered patch): its two halves along the longest axis, each on its own
                std::sort(pts.begin(), pts.end(), [&](const Spot& a, const Spot& b) { return Vector3DotProduct(a.at, f.x) < Vector3DotProduct(b.at, f.x); });
                std::vector<Spot> second(pts.begin() + pts.size() / 2, pts.end());
                pts.resize(pts.size() / 2);
                Cover(std::move(pts), cell, margin, depth + 1, out);
                Cover(std::move(second), cell, margin, depth + 1, out);
                return;
            }
            SuperFrustum s = BoxIn(f, lo.x - margin, hi.x + margin, lo.y - margin, hi.y + margin, lo.z - margin, hi.z + margin, 0.4f);
            out->push_back(s);
        }

        // The points' connected patches (neighbours within a cell and a bit), each covered on its own.
        std::vector<SuperFrustum> CoverPatches(const std::vector<Spot>& all, float cell, float margin) {
            struct Key {
                int x, y, z;
                bool operator==(const Key& o) const { return x == o.x && y == o.y && z == o.z; }
            };
            struct Hash {
                size_t operator()(const Key& k) const { return (size_t)k.x * 73856093u ^ (size_t)k.y * 19349663u ^ (size_t)k.z * 83492791u; }
            };
            // The samples come from many poses, over and over through the same space: first one spot for every half
            // cell of space any of them is in (so a spot's volume is the space it stands for, counted once).
            const float fine = 0.5f * cell;
            std::unordered_map<Key, std::pair<Vector3, int>, Hash> seen;
            for (const Spot& s : all) {
                auto& at = seen[Key{ (int)floorf(s.at.x / fine), (int)floorf(s.at.y / fine), (int)floorf(s.at.z / fine) }];
                at.first = Vector3Add(at.first, s.at);
                at.second++;
            }
            std::vector<Spot> pts;
            pts.reserve(seen.size());
            for (const auto& [k, at] : seen) pts.push_back({ Vector3Scale(at.first, 1.0f / at.second), fine * fine * fine });
            const float voxel = 1.05f * cell;
            auto key = [&](Vector3 p) { return Key{ (int)floorf(p.x / voxel), (int)floorf(p.y / voxel), (int)floorf(p.z / voxel) }; };
            std::unordered_map<Key, std::vector<int>, Hash> grid;
            for (size_t i = 0; i < pts.size(); i++) grid[key(pts[i].at)].push_back((int)i);
            std::unordered_map<Key, int, Hash> patch;
            std::vector<std::vector<Spot>> patches;
            for (const auto& [start, members] : grid) {
                if (patch.count(start)) continue;
                int id = (int)patches.size();
                patches.emplace_back();
                std::vector<Key> stack = { start };
                patch[start] = id;
                while (!stack.empty()) {
                    Key k = stack.back();
                    stack.pop_back();
                    for (int i : grid[k]) patches[id].push_back(pts[i]);
                    for (int dz = -1; dz <= 1; dz++)
                        for (int dy = -1; dy <= 1; dy++)
                            for (int dx = -1; dx <= 1; dx++) {
                                Key n = { k.x + dx, k.y + dy, k.z + dz };
                                if (grid.count(n) && !patch.count(n)) {
                                    patch[n] = id;
                                    stack.push_back(n);
                                }
                            }
                }
            }
            std::vector<SuperFrustum> out;
            for (std::vector<Spot>& p : patches) Cover(std::move(p), cell, margin, 0, &out);
            return out;
        }
    }

    FinishResult CarveCollisions(const Design& in, const CarveParams& p, JobControl* c) {
        FinishResult r;
        r.design = in;
        Design d = in;
        const SphereInfo sp = FindSphere(in);
        BoundingBox bounds = in.Bounds();
        const float cell = std::clamp(Vector3Distance(bounds.min, bounds.max) / 56.0f, 0.3f, 2.0f);
        const float skin = sp.ok ? fmaxf(1.0f, 0.08f * sp.radius) : 1.0f;
        const float margin = p.clearance + 0.3f * cell;
        const int rounds = std::max(1, p.rounds);
        for (int round = 0;; round++) {
            if (c && c->Cancelled()) { r.cancelled = true; r.message = "Stopped."; return r; }
            if (c) c->SetProgress((float)round / (rounds + 1));
            BodySet set = MakeBodySet(d, cell);
            const int n = (int)set.fields.size();
            if (n < 2) { r.message = "Nothing to carve: fewer than two bodies."; return r; }
            std::vector<std::vector<char>> under = HingeSubtrees(d);
            std::vector<MotionCheck> checks = HingeChecks(d, p.checksPerHinge, fmodf(0.5f + 0.6180339887f * round, 1.0f));
            checks.insert(checks.begin(), MotionCheck{ 0.0f, -1, 0.0f });
            std::vector<Rigid> closed = d.Transforms(1.0f, true);
            // what overlaps, per pair of bodies, as points of each of the two (in its own frame)
            struct Found {
                std::vector<Spot> a, b;
                double shownA = 0.0, shownB = 0.0;       // the volume that is near the closed surface, in each
            };
            std::map<std::pair<int, int>, Found> found;
            std::vector<double> volume(checks.size(), 0.0);
            std::mutex mutex;
            ParallelFor((int)checks.size(), [&](int k) {
                if (c && c->Cancelled()) return;
                const MotionCheck& check = checks[k];
                std::vector<Rigid> xf = d.Transforms(check.t, true);
                for (int i = 0; i < n; i++)
                    for (int j = i + 1; j < n; j++) {
                        if (check.hinge >= 0 && !MovingApart(d, under, set.group[i], set.group[j], check.t)) continue;
                        Clash clash;
                        const Rigid &xa = xf[set.group[i]], &xb = xf[set.group[j]];
                        SampleOverlap({ &set.fields[i], set.boxes[i], xa }, { &set.fields[j], set.boxes[j], xb }, cell, { 0.5f, 0.5f, 0.5f }, &clash, true);
                        if (clash.samples.empty()) continue;
                        std::lock_guard<std::mutex> l(mutex);
                        volume[k] += clash.volume;
                        Found& f = found[{ i, j }];
                        for (const OverlapSample& s : clash.samples) {
                            Vector3 la = xa.ApplyInverse(s.at), lb = xb.ApplyInverse(s.at);
                            f.a.push_back({ la, s.volume });
                            f.b.push_back({ lb, s.volume });
                            if (sp.ok) {
                                // where that is when closed: near the sphere's surface it would show
                                if (sp.radius - Vector3Distance(closed[set.group[i]].Apply(la), sp.centre) < skin) f.shownA += s.volume;
                                if (sp.radius - Vector3Distance(closed[set.group[j]].Apply(lb), sp.centre) < skin) f.shownB += s.volume;
                            }
                        }
                    }
            });
            if (c && c->Cancelled()) { r.cancelled = true; r.message = "Stopped."; return r; }
            double worst = 0.0;
            for (double v : volume) worst = std::max(worst, v);
            if (round == 0) r.before = worst;
            r.after = worst;
            if (found.empty() || round >= rounds) break;
            for (auto& [pair, f] : found) {
                // carved from the one it shows less on when closed (the same: from the second, usually the smaller)
                bool fromA = f.shownA < f.shownB;
                int body = d.groups[set.group[fromA ? pair.first : pair.second]].id;
                for (const SuperFrustum& s : CoverPatches(fromA ? f.a : f.b, cell, margin)) {
                    d.AddShape(s, body, "Carved", Op::Cut);
                    r.cuts++;
                }
            }
        }
        if (c) c->SetProgress(1.0f);
        if (r.cuts == 0) {
            r.message = Utils::Format("Nothing collides on the way (%d poses in every hinge's turn): nothing carved.", p.checksPerHinge);
            return r;
        }
        r.ok = true;
        r.design = d;
        r.message = Utils::Format("Carved %d cut%s where bodies ran into each other: %.1f mm3 overlapped at the worst pose, %.2f mm3 now.", r.cuts,
                                  r.cuts == 1 ? "" : "s", r.before, r.after);
        return r;
    }

    // ---- ledges ---------------------------------------------------------------------------------------------------------------

    namespace {
        struct LedgeSizes {
            float over, width, thick, reach;
        };

        // A ledge of `holder` over the turned pieces, along the surface from direction `to` (the holder's) toward
        // direction `from` (a turned piece's): from `over` inside the holder's edge to `over` past the turned piece's.
        bool MakeLedge(const SphereInfo& sp, const BodySet& set, int holder, const std::vector<char>& turned, Vector3 from, Vector3 to,
                       const LedgeSizes& z, SuperFrustum* ledge, float* overhang) {
            const float R = sp.radius, arc = acosf(std::clamp(Vector3DotProduct(from, to), -1.0f, 1.0f));
            auto at = [&](float s) { return Vector3Add(sp.centre, Vector3Scale(OnCircle(sp, to, from, s * arc).z, 0.985f * R)); };
            auto inHolder = [&](float s) { return set.fields[holder].Distance(at(s)) < 0.0f; };
            auto inTurned = [&](float s) {
                Vector3 p = at(s);
                for (size_t i = 0; i < set.fields.size(); i++)
                    if (turned[i] && set.fields[i].Distance(p) < 0.0f) return true;
                return false;
            };
            if (!inHolder(0.0f) || !inTurned(1.0f)) return false;
            // where the holder ends, and where the turned piece begins
            const int steps = 24;
            float sh = 0.0f, st = 1.0f;
            for (int k = 1; k <= steps && inHolder((float)k / steps); k++) sh = (float)k / steps;
            for (float lo = sh, hi = fminf(1.0f, sh + 1.0f / steps), i = 0; i < 12; i++) {
                float mid = 0.5f * (lo + hi);
                (inHolder(mid) ? lo : hi) = mid;
                sh = lo;
            }
            for (int k = steps - 1; k >= 0 && (float)k / steps >= sh && inTurned((float)k / steps); k--) st = (float)k / steps;
            for (float hi = st, lo = fmaxf(sh, st - 1.0f / steps), i = 0; i < 12; i++) {
                float mid = 0.5f * (lo + hi);
                (inTurned(mid) ? hi : lo) = mid;
                st = hi;
            }
            float gap = (st - sh) * arc * R;
            if (gap > z.reach) return false;
            float length = gap + 2.0f * z.over;
            Frame f = OnCircle(sp, to, from, 0.5f * (sh + st) * arc);
            float sag = R - sqrtf(fmaxf(0.0f, R * R - 0.25f * length * length - 0.25f * z.width * z.width));
            *ledge = BoxIn(f, -0.5f * length, 0.5f * length, -0.5f * z.width, 0.5f * z.width, -(z.thick + sag), 0.04f * R, 0.25f);
            *overhang = z.over * z.width * z.thick;
            return true;
        }

        // The ledge's own material: points of it inside the sphere (in its holder's frame: the design pose).
        std::vector<Vector3> LedgePoints(const SphereInfo& sp, const SuperFrustum& ledge, float* volumeEach) {
            std::vector<Vector3> pts = PointsIn(ledge, 14, 10, 8, volumeEach), in;
            for (Vector3 p : pts)
                if (Vector3Distance(p, sp.centre) < sp.radius) in.push_back(p);
            return in;
        }

        // Closed, the hinge a little way open (until the ledge's place has moved a part of `thick`): the most of the
        // ledge the pieces it turns then run into.
        double LedgeCatches(const Design& d, const BodySet& set, const std::vector<std::vector<char>>& under, int h, int holderGroup,
                            const std::vector<Vector3>& pts, float volumeEach, Vector3 where, float thick) {
            std::vector<float> angles = ClosedAngles(d);
            std::vector<Rigid> closed = d.TransformsAt(angles);
            const float lever = fmaxf(Vector3Length(OpeningVelocity(d.hinges[h], where)), 1e-3f);
            int most = 0;
            for (float part : { 0.4f, 0.8f, 1.3f }) {
                angles[h] = AjarAngle(d.hinges[h], part * thick / lever);
                std::vector<Rigid> ajar = d.TransformsAt(angles);
                int hits = 0;
                for (Vector3 p : pts) {
                    Vector3 w = closed[holderGroup].Apply(p);
                    for (size_t i = 0; i < set.fields.size(); i++) {
                        if (!under[h][set.group[i]]) continue;
                        if (set.fields[i].Distance(ajar[set.group[i]].ApplyInverse(w)) < 0.0f) { hits++; break; }
                    }
                }
                most = std::max(most, hits);
            }
            return (double)most * volumeEach;
        }

        // Over the whole motion as the design has it: the largest share of the ledge anything else is ever inside of.
        float LedgeRubs(const Design& d, const BodySet& set, int holderGroup, const std::vector<Vector3>& pts) {
            std::vector<MotionCheck> checks = HingeChecks(d, 12, 0.5f), more = HingeChecks(d, 12, 0.0f);
            checks.insert(checks.end(), more.begin(), more.end());
            checks.push_back({ 0.0f, -1, 0.0f });
            std::vector<int> hits(checks.size(), 0);
            ParallelFor((int)checks.size(), [&](int k) {
                std::vector<Rigid> xf = d.Transforms(checks[k].t, true);
                for (Vector3 p : pts) {
                    Vector3 w = xf[holderGroup].Apply(p);
                    for (size_t i = 0; i < set.fields.size(); i++) {
                        if (set.group[i] == holderGroup) continue;
                        if (set.fields[i].Distance(xf[set.group[i]].ApplyInverse(w)) < -0.05f) { hits[k]++; break; }
                    }
                }
            });
            int worst = 0;
            for (int h : hits) worst = std::max(worst, h);
            return pts.empty() ? 1.0f : (float)worst / pts.size();
        }
    }

    FinishResult AddLedges(const Design& in, const LedgeParams& p, JobControl* c) {
        FinishResult r;
        r.design = in;
        const SphereInfo sp = FindSphere(in);
        if (!sp.ok) { r.message = "Ledges need a carved sphere (a design from the Generate tab)."; return r; }
        const std::vector<int> order = MovingOrder(in);
        if (order.empty()) { r.message = "No hinge turns: nothing to hold."; return r; }
        r.root = in.hinges[order[0]].id;
        r.dependents = (int)order.size() - 1;
        if (order.size() < 2) { r.message = "One hinge: it holds the design shut by itself."; return r; }
        const float R = sp.radius;
        LedgeSizes z;
        z.over = p.size > 0.0f ? p.size : fmaxf(1.2f, 0.12f * R);
        z.width = 2.5f * z.over;
        z.thick = 0.6f * z.over;
        z.reach = p.reach > 0.0f ? p.reach : 0.35f * R;
        const std::vector<Vector3> dirs = SphereDirections(4096);
        const float spacing = sqrtf(4.0f * PI / dirs.size()) * R;
        Design d = in;
        std::vector<int> earlier = { order[0] };
        for (size_t rank = 1; rank < order.size(); rank++) {
            if (c && c->Cancelled()) { r.cancelled = true; r.message = "Stopped."; return r; }
            if (c) c->SetProgress((float)rank / order.size());
            const int h = order[rank];
            const Hinge hg = d.hinges[h];
            BodySet set = MakeBodySet(d, 0.5f);
            std::vector<std::vector<char>> under = HingeSubtrees(d);
            std::vector<int> owner = Owners(sp, set, dirs);
            const int pg = d.GroupIndex(hg.parent);
            // the pieces the hinge turns, and those that could hold them: not turned by it, and parted from its parent
            // by a hinge that opens earlier
            std::vector<char> turned(set.fields.size(), 0), holder(set.fields.size(), 0);
            for (size_t i = 0; i < set.fields.size(); i++) {
                turned[i] = under[h][set.group[i]];
                if (turned[i]) continue;
                for (int e : earlier) holder[i] |= under[e][set.group[i]] != under[e][pg];
            }
            // where: a turned piece's surface that lifts off as the hinge opens, far from its axis, near a holder's
            struct Candidate {
                int from, to;
                float score;
            };
            std::vector<int> held, holders;
            std::vector<float> lever;
            float farthest = 0.0f;
            for (size_t k = 0; k < dirs.size(); k++) {
                if (owner[k] < 0) continue;
                if (holder[owner[k]]) holders.push_back((int)k);
                if (!turned[owner[k]]) continue;
                Vector3 v = OpeningVelocity(hg, Vector3Add(sp.centre, Vector3Scale(dirs[k], R)));
                float l = Vector3Length(v);
                if (Vector3DotProduct(v, dirs[k]) < 0.3f * l) continue;
                held.push_back((int)k);
                lever.push_back(l);
                farthest = fmaxf(farthest, l);
            }
            std::vector<Candidate> candidates;
            for (size_t i = 0; i < held.size(); i++) {
                if (lever[i] < 0.4f * farthest) continue;
                int nearest = -1;
                float best = FLT_MAX;
                for (int k : holders) {
                    float dist = Vector3Distance(dirs[held[i]], dirs[k]);
                    if (dist < best) { best = dist; nearest = k; }
                }
                float gap = fmaxf(0.0f, best * R - 1.5f * spacing);
                if (nearest < 0 || gap > z.reach) continue;
                candidates.push_back({ held[i], nearest, lever[i] / farthest - gap / z.reach });
            }
            std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
            bool done = false;
            std::vector<Vector3> tried;
            for (size_t k = 0; k < candidates.size() && tried.size() < 12 && !done; k++) {
                const Candidate& cand = candidates[k];
                bool close = false;
                for (Vector3 t : tried) close |= Vector3Distance(t, dirs[cand.from]) < 0.2f;
                if (close) continue;
                tried.push_back(dirs[cand.from]);
                SuperFrustum ledge;
                float overhang = 0.0f;
                if (!MakeLedge(sp, set, owner[cand.to], turned, dirs[cand.from], dirs[cand.to], z, &ledge, &overhang)) continue;
                const int holderGroup = set.group[owner[cand.to]];
                Design e = d;
                Claim(e, sp, e.groups[holderGroup].id, { ledge }, {}, "Ledge on " + hg.name);
                BodySet with = MakeBodySet(e, 0.5f);
                float each = 0.0f;
                std::vector<Vector3> pts = LedgePoints(sp, ledge, &each);
                // it has to catch the hinge while its holder is still there, and be in nothing's way after
                if (LedgeCatches(e, with, under, h, holderGroup, pts, each, ledge.position, z.thick) < 0.2 * overhang) continue;
                if (LedgeRubs(e, with, holderGroup, pts) > 0.03f) continue;
                d = e;
                done = true;
            }
            if (done) r.held++;
            else r.unheld.push_back(hg.name);
            earlier.push_back(h);
        }
        if (c) c->SetProgress(1.0f);
        r.ok = r.held > 0;
        if (r.ok) r.design = d;
        r.message = Utils::Format("Ledges: %d of the %d hinges after %s (the first to open) are held shut until an earlier one has opened", r.held,
                                  r.dependents, in.hinges[order[0]].name.c_str());
        if (!r.unheld.empty()) {
            r.message += "; nothing near enough could hold";
            for (size_t k = 0; k < r.unheld.size(); k++) r.message += (k ? ", " : " ") + r.unheld[k];
        }
        r.message += ".";
        return r;
    }

    // ---- the latch ------------------------------------------------------------------------------------------------------------

    // A rocker (bell crank) printed into the holding body, flush with the sphere:
    //
    //        press here                 catch
    //   ====================================---      <- the sphere's surface: the arm, and its catch over the lid's lip
    //                                  |  |  lid
    //                                  |  |          <- the leg
    //                                  '--'
    //                                   ||           <- a thin web down into the body: the spring, and the pivot
    //
    // The arm lies in the surface of the body that stays down at the seam, its catch reaching over a lip left on the
    // other body's edge (the lid's, which would lift). The leg goes down from the seam end of the arm to the web.
    // Pressing the arm's far end in turns the rocker about the web: its top, the catch with it, steps back from the
    // seam, off the lip, and the lid is free. The lid pushing up on the catch pushes straight down the leg, at the
    // pivot: it can't turn the rocker, so it holds.
    FinishResult AddLatch(const Design& in, const LatchParams& p, JobControl* c) {
        FinishResult r;
        r.design = in;
        const SphereInfo sp = FindSphere(in);
        if (!sp.ok) { r.message = "A latch needs a carved sphere (a design from the Generate tab)."; return r; }
        const std::vector<int> order = MovingOrder(in);
        int h = p.hinge >= 0 ? in.HingeIndex(p.hinge) : (order.empty() ? -1 : order[0]);
        if (h < 0 || !Turns(in.hinges[h])) { r.message = "No hinge to latch: none turns."; return r; }
        const Hinge hg = in.hinges[h];
        r.root = hg.id;
        const float R = sp.radius, k = std::clamp(p.scale, 0.5f, 2.0f), gap = std::clamp(p.clearance, 0.1f, 1.0f);
        // sizes (mm): the arm's length, the catch's reach over the lip, the width, the arm's thickness, the leg's
        // thickness and depth, the web's length, thickness and how far it goes into the body
        const float arm = fmaxf(6.0f, 0.45f * R) * k, reach = fmaxf(1.0f, 0.07f * R) * k, width = fmaxf(4.0f, 0.28f * R) * k,
                    thick = fmaxf(1.6f, 0.11f * R) * k, leg = fmaxf(1.6f, 0.10f * R) * k, deep = fmaxf(5.0f, 0.36f * R) * k,
                    webShort = fmaxf(2.0f, 0.13f * R) * k, web = 0.7f * fmaxf(1.0f, k), embed = 1.2f;
        const float length = arm + reach, top = 0.05f * R;

        BodySet set = MakeBodySet(in, 0.5f);
        std::vector<std::vector<char>> under = HingeSubtrees(in);
        const std::vector<Vector3> dirs = SphereDirections(4096);
        const float spacing = sqrtf(4.0f * PI / dirs.size()) * R;
        std::vector<int> owner = Owners(sp, set, dirs);
        // where: on the seam between a piece the hinge turns and one it doesn't, far from its axis
        struct Candidate {
            Vector3 down, up;            // directions: of the body that stays down there, and of the one that lifts
            int holder, other;           // their fields
            float lever;                 // how fast the two part there as the hinge opens, less where they slide past each other
        };
        std::vector<Candidate> candidates;
        std::vector<int> turnedPts, restPts;
        for (size_t i = 0; i < dirs.size(); i++) {
            if (owner[i] < 0) continue;
            (under[h][set.group[owner[i]]] ? turnedPts : restPts).push_back((int)i);
        }
        for (int i : turnedPts) {
            int nearest = -1;
            float best = 1.7f * spacing / R;
            for (int j : restPts) {
                float dist = Vector3Distance(dirs[i], dirs[j]);
                if (dist < best) { best = dist; nearest = j; }
            }
            if (nearest < 0) continue;
            Vector3 mid = Vector3Normalize(Vector3Add(dirs[i], dirs[nearest]));
            Vector3 v = OpeningVelocity(hg, Vector3Add(sp.centre, Vector3Scale(mid, R)));
            float l = Vector3Length(v), out = Vector3DotProduct(v, mid);
            if (fabsf(out) < 0.3f * l) continue;
            // the turned piece lifts there: the other holds it; it dips there (behind the axis): it holds the other
            if (out > 0.0f) candidates.push_back({ dirs[nearest], dirs[i], owner[nearest], owner[i], out * out / l });
            else candidates.push_back({ dirs[i], dirs[nearest], owner[i], owner[nearest], out * out / l });
        }
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.lever > b.lever; });
        if (candidates.empty()) { r.message = "No seam at the sphere's surface to put a latch on " + hg.name + " at."; return r; }

        std::string why = "nowhere along its seam are both bodies deep enough for a rocker and a lip";
        // One place, one way. Over: the rocker is in the body that stays down there, its catch at the surface over a lip
        // of the other. Under: the rocker is in the body that lifts, its catch under a lip left at the surface of the
        // one that stays (for where that one is too thin at the seam to carry a rocker, or the lid too thin for a lip
        // under a catch); its web is under the catch's tip, so that pressed, the catch steps back and down, off the lip.
        auto attempt = [&](Vector3 down, Vector3 up, int holderField, bool underLip, bool webInside, float webLength) {
            const BodyField& holder = set.fields[holderField];
            const int holderGroup = set.group[holderField];
            const bool holderTurns = under[h][holderGroup];
            auto lifted = [&](Vector3 q) {           // inside a body on the other side of the hinge
                for (size_t i = 0; i < set.fields.size(); i++)
                    if ((bool)under[h][set.group[i]] != holderTurns && set.fields[i].Distance(q) < 0.0f) return true;
                return false;
            };
            // the seam, exactly: where the holder ends along the surface between the two points
            const float arc = acosf(std::clamp(Vector3DotProduct(down, up), -1.0f, 1.0f));
            float lo = 0.0f, hi = 1.0f;
            for (int i = 0; i < 14; i++) {
                float mid = 0.5f * (lo + hi);
                Vector3 q = Vector3Add(sp.centre, Vector3Scale(OnCircle(sp, down, up, mid * arc).z, 0.985f * R));
                (holder.Distance(q) < 0.0f ? lo : hi) = mid;
            }
            const float seam = lo * arc;
            // the frame at the middle of the arm; in it the seam is at x = xs and the catch reaches on to length / 2
            const Frame f = OnCircle(sp, down, up, seam - 0.5f * (arm - reach) / R);
            const float xs = 0.5f * length - reach;
            const float sag = (0.25f * length * length + 0.25f * width * width) / (2.0f * R);
            const float zb = -(thick + sag);                      // the arm's underside
            const float zSurface = -(xs + 0.5f * reach) * (xs + 0.5f * reach) / (2.0f * R);     // the surface over the lip
            // how thick the other body is where the catch goes
            float lid = 0.0f;
            while (lid < thick + 3.0f && lifted(f.At(xs + 0.6f * reach, 0.0f, zSurface - lid - 0.1f))) lid += 0.1f;
            // where the web is: under the leg's middle; or, the catch under the lip, under the catch's tip (it then
            // steps back and down) or, where that is the other body's, just inside the seam (it then rises a little as
            // it steps back: the lip is left that much more room over it)
            const float xp = !underLip ? xs - 0.5f * leg : webInside ? xs - 0.5f * web - 0.2f : xs + reach - 0.5f * web - 0.1f;
            float play = 0.6f * gap;                              // between the catch and the lip, closed
            if (underLip && webInside) play += reach * (reach + xs - xp) / (deep + 0.5f * webLength - thick);
            // over: the lid keeps a lip of at least 0.9 mm under a catch of at least 0.7; under: the lip is what the
            // other body has at the surface there, up to 1.2 mm of it (and no less than 0.6)
            if (lid < (underLip ? 0.6f : 1.6f)) return false;
            const float catchThick = underLip ? fmaxf(1.0f, 0.6f * thick) : std::clamp(lid - 0.9f, 0.7f, thick);
            const float lip = underLip ? fminf(lid, 1.2f) : lid - catchThick;
            // the catch's top and underside
            const float zt = underLip ? zSurface - lip - play : top, zc = underLip ? zt - catchThick : zSurface - catchThick;
            // the body where the web goes into it: the rocker hangs from there (whatever else is in the rocker's place,
            // the lid's or another body's, is cut away for it)
            bool solid = true;
            for (float y : { 0.0f, 0.4f * width, -0.4f * width }) solid &= holder.Distance(f.At(xp, y, -deep - webLength - embed + 0.2f)) < 0.0f;
            if (!solid) return false;

            // how far it turns when pressed: the catch (above the pivot by `rise`) steps back by its reach and the gap
            const Vector3 pivot = f.At(xp, 0.0f, -deep - 0.5f * webLength);
            const float rise = 0.5f * (zc + fminf(zt, zSurface)) + deep + 0.5f * webLength;
            const float back = (reach + gap) * 1.15f, press = back / rise;
            const float lever = xp + 0.5f * length;                               // the arm's far end from the pivot
            const float travel = press * lever * 1.2f + 0.2f;                     // ... goes in this far
            const float w2 = 0.5f * width, g2 = w2 + gap;
            std::vector<SuperFrustum> solids, gaps;
            if (!underLip) {
                solids = {
                    BoxIn(f, -0.5f * length, xs, -w2, w2, zb, top, 0.15f),                                      // the arm
                    BoxIn(f, xs - 0.6f, 0.5f * length, -w2, w2, zc, top, 0.15f),                                // the catch
                    BoxIn(f, xs - leg, xs, -w2, w2, -deep, zb + 0.4f),                                          // the leg
                    BoxIn(f, xp - 0.5f * web, xp + 0.5f * web, -w2, w2, -deep - webLength - embed, -deep + 0.4f),   // the web
                };
                gaps = {
                    BoxIn(f, -0.5f * length - back - 0.2f, xs, -g2, g2, zb - travel, top, 0.15f),              // around and under the arm
                    BoxIn(f, xs, 0.5f * length + gap, -g2, g2, zc - play, top, 0.15f),                          // the catch's place over the lip
                    BoxIn(f, xs - leg - back - 0.2f, xs + gap, -g2, g2, -deep - gap, zb),                       // around the leg, room to swing back
                    BoxIn(f, xp - 0.5f * web - 0.7f, xp + 0.5f * web + 0.7f, -g2, g2, -deep - webLength, -deep),   // room for the web to bend
                };
            } else {
                solids = {
                    BoxIn(f, -0.5f * length, xs, -w2, w2, zb, top, 0.15f),                                      // the arm
                    BoxIn(f, xs - leg, xs + reach, -w2, w2, -deep, zt),                                         // the leg; its top is the catch
                    BoxIn(f, xp - 0.5f * web, xp + 0.5f * web, -w2, w2, -deep - webLength - embed, -deep + 0.4f),   // the web
                };
                gaps = {
                    BoxIn(f, -0.5f * length - back - 0.2f, xs, -g2, g2, zb - travel, top, 0.15f),              // around and under the arm
                    BoxIn(f, xs - leg - back - 0.2f, xs + reach + gap, -g2, g2, -deep - gap, zt + play),        // around the leg, under the lip
                    BoxIn(f, xp - 0.5f * web - 0.7f, xp + 0.5f * web + 0.7f, -g2, g2, -deep - webLength, -deep),   // room for the web to bend
                };
            }

            Design e = in;
            const int holderBody = in.groups[holderGroup].id;
            int feature = Claim(e, sp, holderBody, solids, gaps, "Latch");
            // checked with the rocker as a body of its own: e without it, and it alone
            Design bare = e;
            bare.RemoveGroup(bare.GroupIndex(feature));
            BodySet rest = MakeBodySet(bare, 0.5f);
            Design alone;
            {
                int body = alone.groups[alone.AddGroup("Rocker", -1)].id;
                for (const SuperFrustum& s : solids) alone.AddShape(s, body, "Rocker");
            }
            BodyField rocker = BodyField::Body(alone, alone.groups[0].id);
            auto inRocker = [&](Vector3 q) { return rocker.Distance(q) < 0.0f && Vector3Distance(q, sp.centre) < R; };
            std::vector<float> angles = ClosedAngles(bare);
            auto inLifted = [&](const std::vector<Rigid>& xf, Vector3 world) {
                for (size_t i = 0; i < rest.fields.size(); i++)
                    if ((bool)under[h][rest.group[i]] != holderTurns && rest.fields[i].Distance(xf[rest.group[i]].ApplyInverse(world)) < -0.03f)
                        return true;
                return false;
            };
            // 1. it holds: closed, the hinge a little way open (the two having parted by about the catch's thickness
            //    there), the catch and the lip run into each other
            int caught = 0, catchPoints = 0;
            const float hold = underLip ? play + 0.6f : catchThick;
            // (how far the two part along the surface's normal there, per radian: what lifts the lip against the catch)
            const float turning = fmaxf(fabsf(Vector3DotProduct(OpeningVelocity(hg, f.At(xs + 0.5f * reach, 0.0f, zc)), f.z)), 1e-3f);
            for (float part : { 0.4f, 0.8f, 1.2f }) {
                angles[h] = AjarAngle(hg, part * hold / turning);
                std::vector<Rigid> ajar = bare.TransformsAt(angles);
                int hits = 0;
                catchPoints = 0;
                for (int kz = 0; kz < 8; kz++)
                    for (int ky = 0; ky < 10; ky++)
                        for (int kx = 0; kx < 8; kx++) {
                            Vector3 q = f.At(xs + reach * (kx + 0.5f) / 8, -w2 + width * (ky + 0.5f) / 10, zc + (zt - zc) * (kz + 0.5f) / 8);
                            if (!inRocker(q)) continue;
                            catchPoints++;
                            hits += inLifted(ajar, ajar[holderGroup].Apply(q));
                        }
                caught = std::max(caught, hits);
            }
            if (catchPoints == 0 || caught < 0.2f * catchPoints) { why = "the lid doesn't run into the catch where one fits"; return false; }
            // 2. it lets go: pressed, the rocker is clear of its own body, and of the other all the way open
            const Rigid pressed = Rigid::AboutAxis(pivot, f.y, -press);
            std::vector<Vector3> points;
            for (int kz = 0; kz < 24; kz++)
                for (int ky = 0; ky < 8; ky++)
                    for (int kx = 0; kx < 28; kx++) {
                        Vector3 q = f.At(-0.5f * length + length * (kx + 0.5f) / 28, -w2 + width * (ky + 0.5f) / 8, -deep + (top + deep) * (kz + 0.5f) / 24);
                        if (inRocker(q)) points.push_back(q);
                    }
            int rubs = 0;
            for (Vector3 q : points) rubs += rest.fields[rest.fieldOf[holderGroup]].Distance(pressed.Apply(q)) < -0.03f;
            if (rubs > 0.02f * points.size()) { why = "the rocker has no room to turn where one fits"; return false; }
            int worst = 0;
            for (float part : { 0.25f, 0.5f, 0.75f }) {          // on its way there, closed: the catch clears the lip's underside
                const Rigid partly = Rigid::AboutAxis(pivot, f.y, -part * press);
                std::vector<float> shut = ClosedAngles(bare);
                std::vector<Rigid> xf = bare.TransformsAt(shut);
                int hits = 0;
                for (Vector3 q : points)
                    if (f.Local(q).z < zb) hits += inLifted(xf, xf[holderGroup].Apply(partly.Apply(q)));     // (the leg and what is on it)
                if (underLip) worst = std::max(worst, hits);
            }
            if (worst > 0.01f * points.size()) { why = "the catch has no room to step back under the lip where one fits"; return false; }
            for (int s = 0; s <= 16; s++) {
                angles[h] = hg.end + (hg.start - hg.end) * s / 16.0f;
                std::vector<Rigid> xf = bare.TransformsAt(angles);
                int hits = 0;
                for (Vector3 q : points) hits += inLifted(xf, xf[holderGroup].Apply(pressed.Apply(q)));
                worst = std::max(worst, hits);
            }
            if (worst > 0.01f * points.size()) { why = "pressed, the catch still stands in the lid's way where one fits"; return false; }

            r.ok = true;
            r.design = e;
            // what the web has to take: bent through `press` over its length, its skin stretches by about this much
            const float strain = 100.0f * web * press / (2.0f * webLength);
            r.message = Utils::Format("Latch on %s, in %s: press the far end of its arm in about %.1f mm and the catch steps %.1f mm back off the "
                                      "lip. Arm %.1f x %.1f mm and %.1f thick, catch %.1f mm %s a %.1f mm lip, web %.1f mm thick and %.1f long; %.2f mm around it. "
                                      "The web bends %.0f degrees (about %.1f%% strain at its skin: PETG or nylon take that better than PLA; "
                                      "a larger latch bends less). It doesn't snap shut by itself: press it while closing the lid.",
                                      hg.name.c_str(), in.groups[holderGroup].name.c_str(), press * lever, back, arm, width, thick, reach,
                                      underLip ? "under" : "over", lip, web, webLength, gap, press * RAD2DEG, strain);
            return true;
        };
        std::vector<Vector3> tried;
        for (size_t ci = 0; ci < candidates.size() && tried.size() < 24; ci++) {
            if (c && c->Cancelled()) { r.cancelled = true; r.message = "Stopped."; return r; }
            const Candidate& cand = candidates[ci];
            bool close = false;
            for (Vector3 t : tried) close |= Vector3Distance(t, cand.up) < 0.15f;
            if (close) continue;
            tried.push_back(cand.up);
            if (c) c->SetProgress((float)tried.size() / 24.0f);
            // the web as long as the body is deep enough for there: the longer it is, the less it is strained
            bool made = false;
            for (int way = 0; way < 3 && !made; way++)
                for (float longer : { 2.0f, 1.5f, 1.0f }) {
                    made = way == 0 ? attempt(cand.down, cand.up, cand.holder, false, false, longer * webShort)
                                    : attempt(cand.up, cand.down, cand.other, true, way == 2, longer * webShort);
                    if (made) break;
                }
            if (made) {
                if (c) c->SetProgress(1.0f);
                return r;
            }
        }
        r.message = "No latch on " + hg.name + ": " + why + ".";
        return r;
    }
}
