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
    //
    // Two bodies that run into each other on the way: one of them, as it is where they meet, is subtracted from the
    // other (a Clearance: data/design.hpp), grown by the clearance. "Where they meet" is every pose of the motion from
    // just before they first touch to just after they part, close enough together that the tool moves no more than
    // `step` between two of them: whatever it passes through in between is then within half a step of one of them,
    // and so inside the cut (grown by the clearance and half a step). The cut is the tool's whole sweep, with no gaps
    // between poses, and one clean surface rather than patches.
    // Which of the two gives way: the one that loses less, counting what would show on the closed sphere eight times.
    // Hinge hardware is never cut, nor a body's material next to its own knuckles: what runs into those cuts the
    // other body instead (a hingeOnly clearance).

    namespace {
        struct Moment {
            float t = 0.0f;
            Motion motion = Motion::Closing;
            std::vector<float> angles;
            std::vector<Rigid> xf;
        };

        Moment MomentAt(const Design& d, float t, Motion m) {
            Moment o;
            o.t = t;
            o.motion = m;
            o.angles.resize(d.hinges.size());
            for (size_t h = 0; h < d.hinges.size(); h++) o.angles[h] = d.HingeAngle((int)h, t, true, m);
            o.xf = d.TransformsAt(o.angles);
            return o;
        }

        // The motion, in order: the closing one (perHinge moments in every hinge's turn, evenly in its angle, and the
        // open and closed poses) and, if hinges open in their own time, the opening one.
        std::vector<std::vector<Moment>> MotionPaths(const Design& d, int perHinge, float phase) {
            std::vector<std::vector<Moment>> paths;
            std::vector<float> ts = { 0.0f, 1.0f };
            for (const MotionCheck& c : HingeChecks(d, perHinge, phase)) ts.push_back(c.t);
            std::sort(ts.begin(), ts.end());
            ts.erase(std::unique(ts.begin(), ts.end(), [](float a, float b) { return fabsf(a - b) < 1e-6f; }), ts.end());
            std::vector<Moment> closing;
            for (float t : ts) closing.push_back(MomentAt(d, t, Motion::Closing));
            paths.push_back(std::move(closing));
            if (d.HasOwnOpening()) {
                std::vector<Moment> opening;
                const int n = std::max(8, perHinge * (int)d.hinges.size());
                for (int k = 0; k <= n; k++) opening.push_back(MomentAt(d, (k + (k > 0 && k < n ? phase - 0.5f : 0.0f)) / n, Motion::Opening));
                paths.push_back(std::move(opening));
            }
            return paths;
        }

        // A clearance of `target` by `tool` over moments [a, b] of a path: the field samples the tool between them.
        Clearance SweepOver(const Design& d, const std::vector<Moment>& path, int a, int b, int target, int tool, float clearance, float step) {
            Clearance c;
            c.body = target;
            c.tool = tool;
            c.margin = clearance;
            c.step = step;
            c.upTo = d.nextId;
            for (const Hinge& h : d.hinges) c.hinges.push_back(h.id);
            for (int k = a; k <= b; k++) c.poses.push_back(path[k].angles);
            return c;
        }

        // What a clearance takes from its body: mm^3, and how much of that is within `skin` of the closed sphere's
        // surface. Sampled where the tool can reach.
        void Taken(const Design& with, const Design& without, const Clearance& c, const SphereInfo& sp, float skin, float spacing,
                   double* volume, double* shown) {
            *volume = 0.0;
            *shown = 0.0;
            BodyField before = BodyField::Body(without, c.body), after = BodyField::Body(with, c.body);
            // where the tool is at any of its poses, in the body's frame (its box's corners), and the margin around it
            BoundingBox box = before.bounds;
            {
                BodyField tool = BodyField::Body(with, c.tool);
                const int gt = with.GroupIndex(c.body), gb = with.GroupIndex(c.tool);
                Vector3 lo = { FLT_MAX, FLT_MAX, FLT_MAX }, hi = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
                for (size_t k = 0; k < c.poses.size(); k++) {
                    std::vector<Rigid> xf = with.TransformsAt(with.ClearanceAngles(c, (int)k));
                    Rigid back = xf[gb].Then(xf[gt].Inverse());          // the tool's frame to the body's
                    for (int i = 0; i < 8; i++) {
                        Vector3 q = back.Apply({ i & 1 ? tool.bounds.max.x : tool.bounds.min.x, i & 2 ? tool.bounds.max.y : tool.bounds.min.y,
                                                 i & 4 ? tool.bounds.max.z : tool.bounds.min.z });
                        lo = Vector3Min(lo, q);
                        hi = Vector3Max(hi, q);
                    }
                }
                box.min = Vector3Max(box.min, Vector3SubtractValue(lo, c.margin));
                box.max = Vector3Min(box.max, Vector3AddValue(hi, c.margin));
                if (box.max.x <= box.min.x || box.max.y <= box.min.y || box.max.z <= box.min.z) return;
            }
            const int gt = with.GroupIndex(c.body);
            const Rigid closed = with.TransformsAt(ClosedAngles(with))[gt];
            const int nx = std::max(1, (int)ceilf((box.max.x - box.min.x) / spacing)), ny = std::max(1, (int)ceilf((box.max.y - box.min.y) / spacing)),
                      nz = std::max(1, (int)ceilf((box.max.z - box.min.z) / spacing));
            std::vector<double> vol(nz, 0.0), show(nz, 0.0);
            const double each = (double)spacing * spacing * spacing;
            ParallelFor(nz, [&](int k) {
                for (int j = 0; j < ny; j++)
                    for (int i = 0; i < nx; i++) {
                        Vector3 q = { box.min.x + (i + 0.5f) * spacing, box.min.y + (j + 0.5f) * spacing, box.min.z + (k + 0.5f) * spacing };
                        float b = before.Distance(q);
                        if (b >= 0.0f) continue;
                        if (after.Distance(q) < 0.0f) continue;
                        vol[k] += each;
                        if (sp.ok && sp.radius - Vector3Distance(closed.Apply(q), sp.centre) < skin) show[k] += each;
                    }
            });
            for (int k = 0; k < nz; k++) { *volume += vol[k]; *shown += show[k]; }
        }

        bool HasKnuckles(const Design& d, int body) {
            for (const Hinge& h : d.hinges)
                if (h.hardware && (h.parent == body || h.child == body)) return true;
            return false;
        }

        // hingePairs: only bodies a hinge with hardware joins (AddHinges makes room for its hinges with this).
        FinishResult Carve(const Design& in, const CarveParams& p, JobControl* c, bool hingePairs) {
            FinishResult r;
            r.design = in;
            Design d = in;
            const SphereInfo sp = FindSphere(in);
            BoundingBox bounds = in.Bounds();
            const float cell = std::clamp(Vector3Distance(bounds.min, bounds.max) / 56.0f, 0.3f, 2.0f);
            const float skin = sp.ok ? fmaxf(1.0f, 0.08f * sp.radius) : 1.0f;
            const float clearance = fmaxf(0.05f, p.clearance);
            auto joined = [&](int a, int b) {
                for (const Hinge& h : d.hinges)
                    if (h.hardware && ((h.parent == a && h.child == b) || (h.parent == b && h.child == a))) return true;
                return false;
            };
            // start from the design without the clearances this would make again
            d.clearances.erase(std::remove_if(d.clearances.begin(), d.clearances.end(), [&](const Clearance& k) { return !hingePairs || joined(k.body, k.tool); }),
                               d.clearances.end());
            const MotionOverlap before = MeasureMotion(d, p.checksPerHinge, cell);
            r.before = before.worstVolume;
            const int rounds = std::max(1, p.rounds);
            int made = 0;
            const float step = std::clamp(0.5f * clearance, 0.03f, 0.1f);   // the sweep's samples, at most this far apart
            for (int round = 0; round < rounds; round++) {
                if (c && c->Cancelled()) { r.cancelled = true; r.message = "Stopped."; return r; }
                if (c) c->SetProgress(0.9f * round / rounds);
                BodySet set = MakeBodySet(d, cell);
                const int n = (int)set.fields.size();
                if (n < 2) { r.message = "Nothing to carve: fewer than two bodies."; return r; }
                std::vector<std::vector<char>> under = HingeSubtrees(d);
                // where in the motion each pair meets, and the points (each in its own frame) where it does
                struct Meeting {
                    std::vector<char> at;            // per moment of the path
                    std::vector<Vector3> inA, inB;
                };
                const std::vector<std::vector<Moment>> paths = MotionPaths(d, p.checksPerHinge, fmodf(0.5f + 0.6180339887f * round, 1.0f));
                std::map<std::pair<int, int>, std::vector<Meeting>> meet;      // per pair, per path
                std::mutex mutex;
                for (size_t pi = 0; pi < paths.size(); pi++) {
                    const std::vector<Moment>& path = paths[pi];
                    ParallelFor((int)path.size(), [&](int k) {
                        if (c && c->Cancelled()) return;
                        const Moment& m = path[k];
                        for (int i = 0; i < n; i++)
                            for (int j = i + 1; j < n; j++) {
                                const int a = d.groups[set.group[i]].id, b = d.groups[set.group[j]].id;
                                if (hingePairs && !joined(a, b)) continue;
                                Clash clash;
                                SampleOverlap({ &set.fields[i], set.boxes[i], m.xf[set.group[i]] }, { &set.fields[j], set.boxes[j], m.xf[set.group[j]] }, cell,
                                              { 0.5f, 0.5f, 0.5f }, &clash, true);
                                if (clash.samples.empty()) continue;
                                std::lock_guard<std::mutex> l(mutex);
                                std::vector<Meeting>& per = meet[{ i, j }];
                                if (per.empty()) per.resize(paths.size());
                                Meeting& mt = per[pi];
                                if (mt.at.empty()) mt.at.assign(path.size(), 0);
                                mt.at[k] = 1;
                                for (const OverlapSample& s : clash.samples) {
                                    mt.inA.push_back(m.xf[set.group[i]].ApplyInverse(s.at));
                                    mt.inB.push_back(m.xf[set.group[j]].ApplyInverse(s.at));
                                }
                            }
                    });
                }
                if (c && c->Cancelled()) { r.cancelled = true; r.message = "Stopped."; return r; }
                if (meet.empty()) break;
                for (auto& [pair, per] : meet) {
                    const int a = d.groups[set.group[pair.first]].id, b = d.groups[set.group[pair.second]].id;
                    for (size_t pi = 0; pi < per.size(); pi++) {
                        const Meeting& mt = per[pi];
                        if (mt.at.empty()) continue;
                        const std::vector<Moment>& path = paths[pi];
                        // each run of moments where they meet, from the one before to the one after
                        const int last = (int)path.size() - 1;
                        for (int k = 0; k <= last; k++) {
                            if (!mt.at[k]) continue;
                            int from = std::max(0, k - 1), to = k;
                            while (to < last && mt.at[to + 1]) to++;
                            k = to;
                            to = std::min(last, to + 1);
                            // both ways, then the one that takes less
                            Clearance cutA = SweepOver(d, path, from, to, a, b, clearance, step);      // a gives way
                            Clearance cutB = SweepOver(d, path, from, to, b, a, clearance, step);
                            double costA = 0.0, costB = 0.0, shown = 0.0, vol = 0.0;
                            {
                                Design t = d;
                                cutA.id = t.nextId++;
                                t.clearances.push_back(cutA);
                                Taken(t, d, cutA, sp, skin, 0.5f * cell, &vol, &shown);
                                costA = vol + 8.0 * shown;
                            }
                            {
                                Design t = d;
                                cutB.id = t.nextId++;
                                t.clearances.push_back(cutB);
                                Taken(t, d, cutB, sp, skin, 0.5f * cell, &vol, &shown);
                                costB = vol + 8.0 * shown;
                            }
                            const bool aGives = costA <= costB;
                            Clearance cut = aGives ? cutA : cutB, back = aGives ? cutB : cutA;
                            cut.id = d.nextId++;
                            d.clearances.push_back(cut);
                            made++;
                            // what the one giving way can't cut (its hardware, next to its knuckles) cuts the other
                            if (HasKnuckles(d, cut.body)) {
                                back.hingeOnly = true;
                                back.id = d.nextId;
                                Design t = d;
                                t.nextId++;
                                t.clearances.push_back(back);
                                Taken(t, d, back, sp, skin, 0.5f * cell, &vol, &shown);
                                if (vol > 0.0) {
                                    d = t;
                                    made++;
                                }
                            }
                        }
                    }
                }
            }
            if (c) c->SetProgress(0.95f);
            if (made == 0) {
                r.message = Utils::Format("Nothing collides on the way (%d poses in every hinge's turn): nothing carved.", p.checksPerHinge);
                return r;
            }
            // checked again, between the poses it looked at
            const MotionOverlap after = MeasureMotion(d, 2 * p.checksPerHinge, cell, 0.25f);
            r.after = after.worstVolume;
            r.cuts = made;
            if (c) c->SetProgress(1.0f);
            r.ok = true;
            r.design = d;
            r.message = Utils::Format("Carved %d clearance%s, each one body less the other's sweep and %.2f mm around it: %.1f mm3 overlapped at "
                                      "the worst pose, %.2f mm3 now.", made, made == 1 ? "" : "s", clearance + 0.5f * step, r.before, r.after);
            if (r.after > 0.01 && after.deepest < 0.01f) r.message += " (What is left only touches: under 0.01 mm deep.)";
            else if (r.after > 0.01) r.message += " What is left is between parts nothing cuts (hinge hardware, or next to two bodies' own knuckles).";
            return r;
        }
    }

    FinishResult CarveCollisions(const Design& in, const CarveParams& p, JobControl* c) { return Carve(in, p, c, false); }

    // ---- hinges ---------------------------------------------------------------------------------------------------------------

    // The smallest barrel that prints: a pin, the clearance around it, and a wall around that; three knuckles, each
    // long enough to hold (the parent the outer two, the child the one between).
    HingeSizes PlanHinge(float R, const HingeParams& p) {
        HingeSizes s;
        const float k = std::clamp(p.scale, 0.5f, 4.0f);
        s.clearance = std::clamp(p.clearance, 0.1f, 1.0f);
        s.pinRadius = fmaxf(0.6f, 0.8f * k);
        s.radius = s.pinRadius + s.clearance + fmaxf(0.6f, 0.8f * k);
        s.knuckles = 3;
        s.length = 3.0f * fmaxf(1.6f, 2.0f * k) + 2.0f * s.clearance;
        s.leafThickness = fmaxf(1.2f, 1.6f * k);
        s.leafLength = s.radius + (p.reach > 0.0f ? p.reach : fmaxf(1.0f, 1.2f * k));
        // flush: the barrel's ends are what comes nearest the surface
        const float in = R - s.radius - 0.1f;
        s.pivotRadius = R > 0.0f && p.flush ? sqrtf(fmaxf(0.0f, in * in - 0.25f * s.length * s.length)) : FLT_MAX;
        s.relief = R > 0.0f && p.flush ? 2.0f * s.radius : 0.0f;
        return s;
    }

    // Each hinge's hardware, sized by PlanHinge. Where along the axis: where both bodies are nearest it over the
    // barrel's whole length (the pivot slides along its axis, which changes nothing about the motion). Flush, the
    // axis also moves toward the sphere's centre until the barrel is inside the sphere: that does change the motion
    // near the hinge, so the motion is checked before and after. Each body's leaf (a short plate tying its knuckles
    // in) points where it finds the most of its own body and none of the other.
    FinishResult AddHinges(const Design& in, const HingeParams& p, JobControl* c) {
        FinishResult r;
        r.design = in;
        const SphereInfo sp = FindSphere(in);
        const HingeSizes sz = PlanHinge(sp.ok ? sp.radius : 0.0f, p);
        const bool flush = sp.ok && p.flush;
        if (flush && sz.pivotRadius <= 0.0f) { r.message = "No hinges: the sphere is too small for a barrel this size inside it."; return r; }
        // the bodies as they are without any hinge's hardware
        Design bare = in;
        for (Hinge& h : bare.hinges) h.hardware = false;
        const BodySet set = MakeBodySet(bare, 0.5f);
        BoundingBox bounds = in.Bounds();
        const float cell = std::clamp(Vector3Distance(bounds.min, bounds.max) / 56.0f, 0.3f, 2.0f);
        const MotionOverlap before = MeasureMotion(bare, 12, cell);
        Design d = in;
        std::vector<std::string> apart;
        int asked = 0;
        for (size_t h = 0; h < d.hinges.size(); h++) {
            Hinge& hg = d.hinges[h];
            if (p.hinge >= 0 && hg.id != p.hinge) continue;
            if (c && c->Cancelled()) { r.cancelled = true; r.message = "Stopped."; return r; }
            if (c) c->SetProgress(0.8f * h / d.hinges.size());
            const int pg = d.GroupIndex(hg.parent), cg = d.GroupIndex(hg.child);
            const int pf = pg >= 0 ? set.fieldOf[pg] : -1, cf = cg >= 0 ? set.fieldOf[cg] : -1;
            if (pf < 0 || cf < 0) continue;
            asked++;
            const BodyField &parent = set.fields[pf], &child = set.fields[cf];
            const Vector3 a = Vector3Normalize(hg.axis);
            const float half = 0.5f * sz.length;
            // the axis as a line: its point nearest the sphere's centre (or the pivot), and how far out that is
            Vector3 mid = hg.pivot, out = { 0, 0, 0 };
            float rho = 0.0f, u0 = 0.0f;
            if (flush) {
                u0 = -Vector3DotProduct(Vector3Subtract(sp.centre, hg.pivot), a);
                mid = Vector3Subtract(hg.pivot, Vector3Scale(a, u0));
                rho = Vector3Distance(mid, sp.centre);
                out = rho > 1e-4f ? Vector3Scale(Vector3Subtract(mid, sp.centre), 1.0f / rho) : Utils::AnyPerpendicular(a);
                mid = sp.centre;
            }
            // the barrel's middle at u along the axis (flush: on the line moved in as far as a barrel there needs)
            const float in2 = (sp.radius - sz.radius - 0.1f) * (sp.radius - sz.radius - 0.1f);
            auto at = [&](float u, float w) {
                if (!flush) return Vector3Add(mid, Vector3Scale(a, u + w));
                float most = sqrtf(fmaxf(0.0f, in2 - (fabsf(u) + half) * (fabsf(u) + half)));
                return Vector3Add(mid, Vector3Add(Vector3Scale(out, fminf(rho, most)), Vector3Scale(a, u + w)));
            };
            // how far the worse of the two bodies is from the barrel's axis there, over its length
            auto gap = [&](float u) {
                float worst = -FLT_MAX;
                for (float w : { -half, -0.5f * half, 0.0f, 0.5f * half, half }) {
                    Vector3 q = at(u, w);
                    worst = fmaxf(worst, fmaxf(parent.Distance(q), child.Distance(q)));
                }
                return worst;
            };
            const float reach = flush ? sqrtf(fmaxf(0.0f, in2)) - half : 2.0f * Vector3Distance(bounds.min, bounds.max);
            float best = u0, bestCost = FLT_MAX, bestGap = FLT_MAX;
            for (float u = -reach; u <= reach; u += 0.25f) {
                float g = gap(u);
                // on (or in) both is enough; of those places, the nearest to where the pivot is
                float cost = fmaxf(g, 0.0f) + 0.02f * fabsf(u - u0);
                if (cost < bestCost) { bestCost = cost; best = u; bestGap = g; }
            }
            if (bestCost == FLT_MAX) continue;
            // a leaf bridges a little; farther apart than that and the two aren't joined there
            if (bestGap > sz.leafLength - sz.radius) {
                apart.push_back(Utils::Format("%s (%.1f mm from %s)", hg.name.c_str(), bestGap,
                                              d.groups[parent.Distance(at(best, 0.0f)) > child.Distance(at(best, 0.0f)) ? pg : cg].name.c_str()));
                continue;
            }
            const Vector3 pivot = at(best, 0.0f);
            // how far the axis itself moved (sliding along it doesn't count)
            Vector3 shift = Vector3Subtract(pivot, hg.pivot);
            shift = Vector3Subtract(shift, Vector3Scale(a, Vector3DotProduct(shift, a)));
            r.moved = fmaxf(r.moved, Vector3Length(shift));
            hg.pivot = pivot;
            hg.hardware = true;
            hg.length = sz.length;
            hg.radius = sz.radius;
            hg.pinRadius = sz.pinRadius;
            hg.clearance = sz.clearance;
            hg.knuckles = sz.knuckles;
            hg.swapKnuckles = false;
            hg.removed = 0;
            hg.pin = p.pin;
            hg.headDiameter = 0.0f;
            hg.leafLength = sz.leafLength;
            hg.leafThickness = sz.leafThickness;
            hg.autoLeaves = false;
            // the leaves: for each body, the direction about the axis with the most of it and the least of the other
            const Vector3 ref = Utils::AnyPerpendicular(a);
            std::vector<Knuckle> knuckles = d.Knuckles((int)h);
            for (int side = 0; side < 2; side++) {
                const BodyField &own = side ? child : parent, &other = side ? parent : child;
                float bestAngle = 0.0f, bestScore = -FLT_MAX;
                for (int k = 0; k < 36; k++) {
                    float angle = 2.0f * PI * k / 36;
                    Vector3 leaf = Vector3RotateByQuaternion(ref, QuaternionFromAxisAngle(a, angle)), across = Vector3CrossProduct(a, leaf);
                    float score = 0.0f;
                    for (const Knuckle& kn : knuckles) {
                        if (kn.child != (side == 1)) continue;
                        for (float z : { kn.lo + 0.2f, 0.5f * (kn.lo + kn.hi), kn.hi - 0.2f })
                            for (int i = 1; i <= 12; i++) {
                                Vector3 q = Vector3Add(pivot, Vector3Add(Vector3Scale(a, z), Vector3Scale(leaf, sz.leafLength * ((i + 1) / 2) / 6.0f)));
                                q = Vector3Add(q, Vector3Scale(across, (i % 2 ? 0.5f : -0.5f) * sz.leafThickness));      // along both of its faces
                                if (flush && Vector3Distance(q, sp.centre) > sp.radius - 0.5f * sz.leafThickness) score -= 6.0f;
                                else if (other.Distance(q) < 0.0f) score -= 3.0f;
                                else if (own.Distance(q) < 0.0f) score += 1.0f;
                            }
                    }
                    if (score > bestScore) { bestScore = score; bestAngle = angle; }
                }
                hg.leafAngle[side] = bestAngle;
                // whatever of the other body is still where the leaf goes makes way for it
                Frame f;
                f.o = pivot;
                f.z = a;
                f.x = Vector3RotateByQuaternion(ref, QuaternionFromAxisAngle(a, bestAngle));
                f.y = Vector3CrossProduct(f.z, f.x);
                const float t = 0.5f * sz.leafThickness + sz.clearance;
                for (const Knuckle& kn : knuckles) {
                    if (kn.child != (side == 1)) continue;
                    SuperFrustum room = BoxIn(f, 0.0f, sz.leafLength + sz.clearance, -t, t, kn.lo - sz.clearance, kn.hi + sz.clearance);
                    if (Reaches(other, room, 0.0f)) d.AddShape(room, side ? hg.parent : hg.child, hg.name + " leaf room", Op::Cut);
                    // ... and so does any other body's (the two bodies' own fields only clear each other's knuckles)
                    for (size_t k = 0; k < set.fields.size(); k++)
                        if ((int)k != pf && (int)k != cf && Reaches(set.fields[k], room, 0.0f))
                            d.AddShape(room, d.groups[set.group[k]].id, hg.name + " leaf room", Op::Cut);
                }
            }
            // a body the hinge doesn't join that is where its barrel goes makes way for that too
            {
                Frame f;
                f.o = pivot;
                f.z = a;
                f.x = ref;
                f.y = Vector3CrossProduct(a, ref);
                const float w = sz.radius + sz.clearance;
                SuperFrustum room = BoxIn(f, -w, w, -w, w, -half - sz.clearance, half + sz.clearance, 0.3f);
                for (size_t k = 0; k < set.fields.size(); k++)
                    if ((int)k != pf && (int)k != cf && Reaches(set.fields[k], room, 0.0f))
                        d.AddShape(room, d.groups[set.group[k]].id, hg.name + " barrel room", Op::Cut);
            }
            // a separate pin under the surface: a channel from its head out of the sphere, to push it in by
            if (flush && p.pin == PinStyle::Separate) {
                float from = 0.0f, to = 0.0f, head = 0.0f;
                d.PinHead((int)h, &from, &to, &head);
                Frame f;
                f.z = a;
                f.x = ref;
                f.y = Vector3CrossProduct(a, ref);
                f.o = pivot;
                const float w = head + sz.clearance;
                SuperFrustum channel = BoxIn(f, -w, w, -w, w, to - 0.1f, to + 2.0f * sp.radius);
                for (size_t k = 0; k < set.fields.size(); k++)
                    if (Reaches(set.fields[k], channel, 0.2f)) d.AddShape(channel, d.groups[set.group[k]].id, hg.name + " pin channel", Op::Cut);
            }
            r.built++;
        }
        if (asked == 0) { r.message = "No hinges: no hinge joins two bodies with shapes."; return r; }
        std::string rest;
        for (size_t k = 0; k < apart.size(); k++) rest += (k ? ", " : " ") + apart[k];
        if (r.built == 0) {
            r.message = "No hinges built: the two bodies aren't both at the axis anywhere along it for" + rest + ".";
            return r;
        }
        if (c) c->SetProgress(0.5f);
        // under the surface, the two bodies of a hinge are in each other's way next to it as they turn: that room is cut
        if (flush) {
            CarveParams relief;
            relief.checksPerHinge = 24;
            relief.clearance = sz.clearance;
            relief.rounds = 3;
            FinishResult made = Carve(d, relief, nullptr, true);
            if (made.ok) { d = made.design; r.relieved = made.cuts; }
        }
        if (c && c->Cancelled()) { r.cancelled = true; r.message = "Stopped."; return r; }
        if (c) c->SetProgress(0.8f);
        // what it did to the design: the motion again, and how much of the hardware is in a body it doesn't belong to
        // when closed (or outside the sphere)
        const MotionOverlap after = MeasureMotion(d, 12, cell);
        double closedOverlap = 0.0;
        {
            BodySet built = MakeBodySet(d, 0.5f);
            std::vector<Rigid> xf = d.TransformsAt(ClosedAngles(d));
            for (size_t i = 0; i < built.fields.size(); i++)
                for (size_t j = i + 1; j < built.fields.size(); j++) {
                    Clash clash;
                    SampleOverlap({ &built.fields[i], built.boxes[i], xf[built.group[i]] }, { &built.fields[j], built.boxes[j], xf[built.group[j]] }, 0.5f * cell,
                                  { 0.5f, 0.5f, 0.5f }, &clash, false);
                    closedOverlap += clash.volume;
                }
        }
        if (c) c->SetProgress(1.0f);
        r.ok = true;
        r.design = d;
        r.message = Utils::Format("Hinges: %d of %d built, %s: a %.1f mm barrel %.1f mm long (3 knuckles) on a %.1f mm pin, %.2f mm clearance.", r.built, asked,
                                  p.pin == PinStyle::Fused ? "printed in place" : "for a separate pin", 2.0f * sz.radius, sz.length, 2.0f * sz.pinRadius, sz.clearance);
        if (r.moved > 0.05f)
            r.message += Utils::Format(" To be inside the sphere, axes moved up to %.1f mm toward its centre, which changes the motion a little.", r.moved);
        if (r.relieved > 0)
            r.message += Utils::Format(" Their bodies make each other room to turn (%d clearance%s).", r.relieved, r.relieved == 1 ? "" : "s");
        r.message += Utils::Format(" Overlapping on the way: %.1f mm3 at worst before, %.1f mm3 now", before.worstVolume, after.worstVolume);
        r.message += after.worstVolume > before.worstVolume + 0.5 ? " (Carve collisions away clears the rest)." : ".";
        if (closedOverlap > 0.5) r.message += Utils::Format(" Closed, %.1f mm3 of hardware is inside another body.", closedOverlap);
        if (!apart.empty()) r.message += " Not built, the two bodies aren't both at the axis:" + rest + ".";
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

// ---- floating bits ---------------------------------------------------------------------------------------------------
#include "data/build.hpp"

namespace Bakugan {
    FinishResult RemoveFloatingBits(const Design& in, const FloatingParams& p, JobControl* c) {
        FinishResult r;
        r.design = in;
        Design d = in;
        float voxel = p.voxel;
        if (voxel <= 0.0f) {                             // auto: the bodies' longest side over 250
            BoundingBox all = { { FLT_MAX, FLT_MAX, FLT_MAX }, { -FLT_MAX, -FLT_MAX, -FLT_MAX } };
            for (int g : in.Bodies()) {
                BodyField f = BodyField::Body(in, in.groups[g].id);
                if (!f.Empty()) all = { Vector3Min(all.min, f.bounds.min), Vector3Max(all.max, f.bounds.max) };
            }
            Vector3 e = all.min.x <= all.max.x ? Vector3Subtract(all.max, all.min) : Vector3{ 40, 40, 40 };
            voxel = std::clamp(fmaxf(e.x, fmaxf(e.y, e.z)) / 250.0f, 0.05f, 0.4f);
        }
        const double share = std::clamp((double)p.share, 0.0, 1.0);
        std::vector<int> bodies;
        for (int g : in.Bodies()) bodies.push_back(in.groups[g].id);
        std::vector<std::string> notes;
        float used = voxel;
        int tight = 0, specks = 0, failedAll = 0;
        std::vector<std::string> split;                  // bodies in several big pieces
        for (size_t k = 0; k < bodies.size(); k++) {
            if (c && c->Cancelled()) { r.cancelled = true; return r; }
            const int body = bodies[k];
            const std::string name = d.groups[d.GroupIndex(body)].name;
            if (c) {
                c->SetStage("Looking for floating bits: " + name);
                c->SetProgress((float)k / bodies.size());
            }
            BodyField f = BodyField::Body(d, body);
            if (f.Empty()) continue;
            std::vector<Island> islands = FindIslands([&](Vector3 q) { return f.Distance(q); }, f.bounds, voxel, share, c, &used);
            int here = 0, failed = 0, big = 0;
            for (const Island& is : islands) {
                specks += is.speck;
                big += !is.speck && !is.remove;
                if (!is.remove) continue;
                if (!is.covered) { failed++; continue; }
                int g = d.AddGroup("Floating bit", body);
                d.groups[g].op = Op::Cut;
                d.groups[g].color = d.groups[d.GroupIndex(body)].color;
                const int gid = d.groups[g].id;
                for (const BoundingBox& b : is.cover) {
                    SuperFrustum box;
                    box.position = Vector3Scale(Vector3Add(b.min, b.max), 0.5f);
                    box.size = Vector3Subtract(b.max, b.min);
                    box.rounding = 0.0f;
                    d.AddShape(box, gid, "Bit cut");
                }
                here++;
                r.removedVolume += is.volume;
                tight += is.tight > 0;
            }
            r.removed += here;
            failedAll += failed;
            if (big > 1) {
                std::string sizes;
                for (const Island& is : islands)
                    if (!is.speck && !is.remove) sizes += (sizes.empty() ? "" : ", ") + Utils::Format("%.0f", is.volume);
                split.push_back(name + " (" + sizes + " mm^3)");
            }
            if (here == 0) continue;
            // looked at again: what is left (a box can only take away what the voxels saw)
            BodyField after = BodyField::Body(d, body);
            std::vector<Island> again = FindIslands([&](Vector3 q) { return after.Distance(q); }, after.bounds, used, share, c);
            int left = 0;
            for (const Island& is : again) left += is.remove;
            r.left += left;
            if (!islands.empty() && !again.empty()) {
                double change = again[0].volume - islands[0].volume;
                if (fabs(change) > 0.002 * islands[0].volume)
                    notes.push_back(Utils::Format("%s's main piece changed by %.1f mm^3 (%.2f%%)", name.c_str(), change, 100.0 * change / islands[0].volume));
            }
        }
        if (c) c->SetProgress(1.0f);
        if (tight > 0)
            notes.push_back(Utils::Format("%d of the bits lay within a voxel of the rest of their body: cut up to it (at most half a voxel, "
                                          "%.2f mm, of the rest can have gone with them)", tight, 0.5f * used));
        if (failedAll > 0) notes.push_back(Utils::Format("%d bit%s too intricate to box in", failedAll, failedAll == 1 ? " was" : "s were"));
        if (!split.empty()) {
            std::string list;
            for (size_t k = 0; k < split.size(); k++) list += (k ? "; " : "") + split[k];
            notes.push_back("in big pieces, kept (raise the share to take the smaller away): " + list);
        }
        if (specks > 0) notes.push_back(Utils::Format("%d speck%s under 2 voxels across left alone", specks, specks == 1 ? "" : "s"));
        if (r.removed == 0) {
            r.message = Utils::Format("No floating bits smaller than %.0f%% of their body's largest piece (looked at in %.2f mm voxels).", 100.0 * share, used);
            for (const std::string& n : notes) r.message += " Note: " + n + ".";
            return r;
        }
        r.ok = true;
        r.design = d;
        r.message = Utils::Format("Removed %d floating bit%s (%.2f mm^3 in all; %.2f mm voxels): each a \"Floating bit\" cut feature in its body.", r.removed,
                                  r.removed == 1 ? "" : "s", r.removedVolume, used);
        if (r.left > 0) r.message += Utils::Format(" %d small bit%s still there afterwards.", r.left, r.left == 1 ? " is" : "s are");
        for (const std::string& n : notes) r.message += " Note: " + n + ".";
        return r;
    }
}
