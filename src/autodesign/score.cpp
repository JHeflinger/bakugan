#include "score.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>

// The score of a design, lower is better:
//
//   total = (1 - goal) * fit + goal * sphere + coverage penalty + costs + collision weight * collisions
//
// measured in this order (Scorer::Evaluate):
//   1. SampleDistances  the design's distance at every sample point: volume and surface samples of the mesh (open
//                       pose), points over the open design's box and over the closed shape's box
//   2. MeshFit          Volume loss: the mesh volume the design misses, the design volume off it, the surface gap
//   3. OpenMaterial     material outside the mesh and how far out it reaches; bodies overlapping in the open pose;
//                       the design's volume (what collisions are measured against)
//   4. ClosedShape      volume and centroid of the closed shape, so the sphere it should be; overlaps when closed
//   5. MotionClashes    bodies overlapping on the way between the poses (motion checks)
//   6. SphereFit        Volume loss: the closed shape against its sphere
//   7. Diagnose         (rewrite rounds only) a few spots of every kind of error, for the grammar
//   8. Silhouettes      Silhouettes loss: the outline comparisons replace fit and sphere
//   9. Costs            complexity, loose bodies, pivots off their bodies
// Every part is roughly 0 (perfect) .. 1. Collisions are kept apart in the total so the search can ramp them in.

namespace Bakugan::Auto {
    void Samples::Make(int quality, unsigned seed) {
        const int closedCount[3] = { 1000, 2000, 4000 };
        const int sphereCount[3] = { 1200, 2500, 5000 };
        const int shellCount[3] = { 200, 400, 800 };
        const int pairCount[3] = { 64, 128, 256 };
        int q = std::clamp(quality, 0, 2);
        Rng r(seed * 7919u + 17u);
        auto unit = [&]() { return Vector3{ Uniform(r), Uniform(r), Uniform(r) }; };
        for (int i = 0; i < closedCount[q]; i++) closedUnit.push_back(unit());
        for (int i = 0; i < closedCount[q]; i++) openUnit.push_back(unit());
        for (int i = 0; i < sphereCount[q]; i++) {
            Vector3 u = { Uniform(r, -1.35f, 1.35f), Uniform(r, -1.35f, 1.35f), Uniform(r, -1.35f, 1.35f) };
            sphereUnit.push_back(u);
            sphereIn.push_back(Vector3Length(u) <= 1.0f);
        }
        for (int i = 0; i < pairCount[q]; i++) pairUnit.push_back(unit());
        // a Fibonacci sphere: even coverage of the surface
        int n = shellCount[q];
        for (int i = 0; i < n; i++) {
            float y = 1.0f - 2.0f * (i + 0.5f) / n, rad = sqrtf(fmaxf(0.0f, 1.0f - y * y)), phi = i * 2.39996323f;
            shellDirs.push_back({ rad * cosf(phi), y, rad * sinf(phi) });
        }
    }

    void Scorer::Prepare(JobControl& c) {
        m_Tau = 0.008f * m_T.L;
        m_S.Make(m_P.quality, m_P.seed);
        if (m_P.loss == AutoLoss::Silhouettes) {
            c.SetStage("Drawing the mesh's outlines");
            m_Sil.Prepare(*m_T.mesh, m_T.box, m_T.L, m_P.views, m_P.viewResolution, m_Pool);
        }
    }

    struct Scorer::Eval {
        const Design& d;
        Probe pr;
        std::vector<Rigid> open, closed;
        const std::vector<Rigid>* openXf = nullptr;    // null when the open pose is the design pose
        BoundingBox openBox = {}, closedBox = {};      // around each pose
        Vector3 oext = {}, cext = {};
        bool useVolume = false;                         // measure the volume and surface samples of the mesh
        bool wantSpots = false;                         // collect diagnostics
        // the sample points, one after the other: mesh volume, mesh surface, open box, closed box
        int nV = 0, nS = 0, nO = 0, nC = 0, oBase = 0, cBase = 0;
        std::vector<float> dist, dist2, meshD;          // distance to the design, to its second body, to the mesh
        std::vector<int> who, who2;                     // nearest body, second nearest (group indices)
        double designVolume = 0.0, closedMass = 0.0;
        float offOpen = 0.0f, offClosed = 0.0f;         // share of the material outside the views' frame
        // found along the way
        std::vector<int> missing;                       // mesh volume samples the design misses
        std::vector<Spot> excess, holes, bumps;
        std::vector<int> clashes;                       // open and closed samples inside two bodies
        std::vector<Clash> motion;                      // ... and spots on the way
        explicit Eval(const Design& design) : d(design) {}
    };

    static bool IsIdentity(const std::vector<Rigid>& xf) {
        for (const Rigid& r : xf)
            if (fabsf(r.rotation.w) < 0.999999f || Vector3Length(r.position) > 1e-5f) return false;
        return true;
    }

    AutoScore Scorer::Evaluate(const Design& d, Diagnostics* diag) {
        m_Count++;
        AutoScore sc;
        Eval e(d);
        e.pr.Build(d, 0.002f * m_T.L);
        if (e.pr.fields.empty()) return sc;
        e.open = d.Transforms(0.0f, false);
        e.closed = d.Transforms(1.0f, false);
        const bool silhouettes = m_P.loss == AutoLoss::Silhouettes;
        // scored by silhouettes, the mesh's samples only find the spots a rewrite round aims at; the samples over
        // the open and closed designs are always needed (collisions, the sphere's size and centre)
        e.useVolume = !silhouettes || diag;
        e.wantSpots = diag != nullptr;

        SampleDistances(e);
        MeshFit(e, sc);
        OpenMaterial(e, sc);
        ClosedShape(e, sc);
        MotionClashes(e, sc);
        SphereFit(e, sc);
        if (diag) Diagnose(e, sc, *diag);
        if (silhouettes)
            m_Sil.Score(d, e.pr, e.open, e.closed, e.offOpen, e.offClosed, m_P.sphereBias, m_P.minViewCoverage, &sc);
        sc.penalty = Costs(d, e.pr);

        float w = std::clamp(m_P.sphereBias, 0.0f, 1.0f);
        sc.base = (1.0f - w) * sc.fit + w * sc.sphere + sc.coveragePenalty + sc.penalty;
        sc.total = sc.base + m_P.collisionWeight * sc.collision;
        sc.valid = true;
        return sc;
    }

    // 1. Every sample point's distance to the design (and, over the open and closed boxes, to its second nearest
    //    body, which is negative where two bodies overlap), in parallel.
    void Scorer::SampleDistances(Eval& e) {
        e.openXf = IsIdentity(e.open) ? nullptr : &e.open;
        BoundingBox none = { { FLT_MAX, FLT_MAX, FLT_MAX }, { -FLT_MAX, -FLT_MAX, -FLT_MAX } };
        e.openBox = e.closedBox = none;
        for (size_t i = 0; i < e.pr.fields.size(); i++) {
            BoundingBox o = TransformBox(e.pr.box[i], e.open[e.pr.group[i]]), c = TransformBox(e.pr.box[i], e.closed[e.pr.group[i]]);
            e.openBox = { Vector3Min(e.openBox.min, o.min), Vector3Max(e.openBox.max, o.max) };
            e.closedBox = { Vector3Min(e.closedBox.min, c.min), Vector3Max(e.closedBox.max, c.max) };
        }
        e.oext = Vector3Subtract(e.openBox.max, e.openBox.min);
        e.cext = Vector3Subtract(e.closedBox.max, e.closedBox.min);

        const float occCap = 6.0f * m_Tau, fitCap = 0.08f * m_T.L;
        e.nV = e.useVolume ? (int)m_T.vol.size() : 0;
        e.nS = e.useVolume ? (int)m_T.surf.size() : 0;
        e.nO = (int)m_S.openUnit.size();
        e.nC = (int)m_S.closedUnit.size();
        e.oBase = e.nV + e.nS;
        e.cBase = e.oBase + e.nO;
        int n = e.cBase + e.nC;
        e.dist.assign(n, 0.0f);
        e.dist2.assign(n, occCap);
        e.meshD.assign(e.nO, 0.0f);
        e.who.assign(n, -1);
        e.who2.assign(n, -1);
        m_Pool.ForEach(n, [&](int i) {
            if (i < e.nV) {
                e.dist[i] = e.pr.Distance(m_T.vol[i], e.openXf, occCap, &e.who[i]);
            } else if (i < e.oBase) {
                e.dist[i] = e.pr.Distance(m_T.surf[i - e.nV], e.openXf, fitCap, &e.who[i]);
            } else if (i < e.cBase) {
                Vector3 p = InBox(e.openBox.min, e.oext, m_S.openUnit[i - e.oBase]);
                e.pr.Distance2(p, e.openXf, occCap, &e.dist[i], &e.dist2[i], &e.who[i], &e.who2[i]);
                e.meshD[i - e.oBase] = m_T.field.Distance(p);
            } else {
                Vector3 p = InBox(e.closedBox.min, e.cext, m_S.closedUnit[i - e.cBase]);
                e.pr.Distance2(p, &e.closed, occCap, &e.dist[i], &e.dist2[i], &e.who[i], &e.who2[i]);
            }
        });
    }

    // 2. Open pose against the mesh, on the mesh's samples: the share of its volume filled (fitCover), the overlap
    //    over the union (fitIoU), and how far the mesh surface is from the design's (fitGap).
    void Scorer::MeshFit(Eval& e, AutoScore& sc) {
        const float fitCap = 0.08f * m_T.L;
        double inter = 0.0, uni = 0.0, nIn = 0.0;
        for (int i = 0; i < e.nV; i++) {
            float a = Occupancy(e.dist[i]);
            if (m_T.volIn[i]) {
                inter += a;
                uni += 1.0;
                nIn += 1.0;
                if (e.dist[i] > 0.0f) e.missing.push_back(i);
            } else {
                uni += a;
                if (e.dist[i] < 0.0f) e.excess.push_back({ m_T.vol[i], e.who[i] });
            }
        }
        sc.fitIoU = uni > 0.0 ? (float)(inter / uni) : 0.0f;
        sc.fitCover = nIn > 0 ? (float)(inter / nIn) : 0.0f;
        double gap = 0.0;
        for (int i = e.nV; i < e.oBase; i++) gap += fminf(fabsf(e.dist[i]), fitCap);
        sc.fitGap = e.nS > 0 ? (float)(gap / e.nS / fitCap) : 0.0f;
    }

    // 3. Open pose, on samples over the design's own box: material off the mesh (as a share of the design and of the
    //    mesh's volume, and how far out), bodies overlapping (half by the share of material inside two bodies, half by
    //    how deep), material outside the views' frame. Then the Volume fit: missing and extra volume (both relative
    //    to the mesh's volume, so a blob twice the mesh's size costs as much as covering none of it), gap and spill.
    void Scorer::OpenMaterial(Eval& e, AutoScore& sc) {
        const float occCap = 6.0f * m_Tau, L = m_T.L;
        const bool silhouettes = m_P.loss == AutoLoss::Silhouettes;
        double mass = 0.0, off = 0.0, far = 0.0, overlap = 0.0, depth = 0.0, unseen = 0.0;
        for (int k = 0; k < e.nO; k++) {
            int i = e.oBase + k;
            Vector3 p = InBox(e.openBox.min, e.oext, m_S.openUnit[k]);
            float a = Occupancy(e.dist[i]);
            mass += a;
            off += a * Occupancy(-e.meshD[k]);
            far += a * fmaxf(0.0f, e.meshD[k]);
            if (e.dist2[i] < occCap) overlap += Overlap(e.dist2[i]);
            depth += fminf(fmaxf(0.0f, -e.dist2[i]), occCap);
            if (e.dist[i] < 0.0f && e.meshD[k] > 0.0f) e.excess.push_back({ p, e.who[i] });
            if (e.dist2[i] < -m_Tau) e.clashes.push_back(i);
            if (silhouettes && m_Sil.OutsideFrame(p)) unseen += a;
        }
        if (mass > 1e-6) {
            e.offOpen = (float)(unseen / mass);
            double sampleVolume = (double)e.oext.x * e.oext.y * e.oext.z / e.nO;
            e.designVolume = mass * sampleVolume;
            sc.fitOutside = (float)(off / mass);
            sc.fitExcess = (float)(off * sampleVolume / fmax(m_T.field.Volume(), 1e-9));
            sc.fitSpill = (float)(far / mass / (0.1 * L));
            sc.clashOpen = (float)(overlap / mass);
            sc.collision += 0.5f * sc.clashOpen + 0.5f * (float)(depth / mass / occCap);
        }
        sc.fit = 0.35f * (1.0f - sc.fitCover) + 0.35f * fminf(sc.fitExcess, 2.0f) + 0.15f * sc.fitGap + 0.15f * fminf(sc.fitSpill, 2.0f);
    }

    // 4. Closed pose: its volume and centroid give the sphere it should become (of its own volume, or of the size
    //    asked for); bodies overlapping, as in the open pose.
    void Scorer::ClosedShape(Eval& e, AutoScore& sc) {
        const float occCap = 6.0f * m_Tau;
        const bool silhouettes = m_P.loss == AutoLoss::Silhouettes;
        double mass = 0.0, overlap = 0.0, depth = 0.0, unseen = 0.0;
        Vector3 centroid = { 0, 0, 0 };
        for (int k = 0; k < e.nC; k++) {
            int i = e.cBase + k;
            float a = Occupancy(e.dist[i]);
            Vector3 p = InBox(e.closedBox.min, e.cext, m_S.closedUnit[k]);
            mass += a;
            centroid = Vector3Add(centroid, Vector3Scale(p, a));
            if (e.dist2[i] < occCap) overlap += Overlap(e.dist2[i]);
            depth += fminf(fmaxf(0.0f, -e.dist2[i]), occCap);
            if (e.dist2[i] < -m_Tau) e.clashes.push_back(i);
            if (silhouettes && m_Sil.OutsideFrame(p)) unseen += a;
        }
        if (mass > 1e-6) {
            e.offClosed = (float)(unseen / mass);
            sc.clashClosed = (float)(overlap / mass);
            sc.collision += 0.5f * sc.clashClosed + 0.5f * (float)(depth / mass / occCap);
        }
        e.closedMass = mass;
        double boxVolume = (double)e.cext.x * e.cext.y * e.cext.z;
        double volume = e.nC > 0 ? mass / e.nC * boxVolume : 0.0;
        if (mass < 1e-3 || volume <= 0.0) {
            sc.sphere = 1.0f;
        } else {
            sc.centre = Vector3Scale(centroid, (float)(1.0 / mass));
            sc.radius = m_P.sphereDiameter > 0.0f ? m_P.sphereDiameter * 0.5f : (float)cbrt(3.0 * volume / (4.0 * PI));
        }
    }

    // 5. Bodies overlapping on the way: at motionChecks evenly spaced moments strictly between open and closed (of the
    //    closing motion, and of the opening one too when a hinge has its own opening timing), eased as the editor
    //    animates it. Two bodies can only overlap where their boxes do, so each moment only samples the meeting box of
    //    every pair of bodies whose world boxes meet (a fixed set of points). A moment scores like an end pose (half
    //    the share of the design's volume inside two bodies, half how deep, capped), and the moments' mean adds to the
    //    collisions: the whole motion between weighs as much as one end pose, however many moments are checked.
    void Scorer::MotionClashes(Eval& e, AutoScore& sc) {
        const int n = std::max(0, m_P.motionChecks), nf = (int)e.pr.fields.size();
        if (n == 0 || nf < 2 || e.designVolume <= 0.0) return;
        const float occCap = 6.0f * m_Tau;
        const Probe& pr = e.pr;

        struct Moment {
            float t;
            bool opening;
            std::vector<Rigid> xf;
            double overlap = 0.0, depth = 0.0;
        };
        std::vector<Moment> moments;
        const int motions = e.d.HasOwnOpening() ? 2 : 1;
        for (int m = 0; m < motions; m++)
            for (int k = 1; k <= n; k++) {
                float t = (float)k / (n + 1);
                moments.push_back({ t, m == 1, e.d.Transforms(t, true, m == 1 ? Motion::Opening : Motion::Closing) });
            }

        // the pairs of bodies whose boxes meet, per moment
        struct Pair {
            int moment, a, b;
            BoundingBox meet;
            double overlap = 0.0, depth = 0.0;
            std::vector<Vector3> deep;    // a few points well inside both
        };
        std::vector<Pair> pairs;
        std::vector<BoundingBox> world(nf);
        for (int m = 0; m < (int)moments.size(); m++) {
            for (int i = 0; i < nf; i++) world[i] = TransformBox(pr.box[i], moments[m].xf[pr.group[i]]);
            for (int i = 0; i < nf; i++)
                for (int j = i + 1; j < nf; j++) {
                    Vector3 lo = Vector3Max(world[i].min, world[j].min), hi = Vector3Min(world[i].max, world[j].max);
                    if (lo.x >= hi.x || lo.y >= hi.y || lo.z >= hi.z) continue;
                    Pair p;
                    p.moment = m;
                    p.a = i;
                    p.b = j;
                    p.meet = { lo, hi };
                    pairs.push_back(std::move(p));
                }
        }
        auto inside = [](const BoundingBox& b, Vector3 q) {
            return q.x >= b.min.x && q.x <= b.max.x && q.y >= b.min.y && q.y <= b.max.y && q.z >= b.min.z && q.z <= b.max.z;
        };
        m_Pool.For((int)pairs.size(), [&](int k) {
            Pair& t = pairs[k];
            const Rigid& xa = moments[t.moment].xf[pr.group[t.a]];
            const Rigid& xb = moments[t.moment].xf[pr.group[t.b]];
            Vector3 ext = Vector3Subtract(t.meet.max, t.meet.min);
            double overlap = 0.0, depth = 0.0;
            for (const Vector3& u : m_S.pairUnit) {
                Vector3 p = InBox(t.meet.min, ext, u);
                Vector3 qa = xa.ApplyInverse(p);
                if (!inside(pr.box[t.a], qa)) continue;        // outside a body's box: outside the body
                float da = pr.fields[t.a].Distance(qa);
                if (da >= occCap) continue;
                Vector3 qb = xb.ApplyInverse(p);
                if (!inside(pr.box[t.b], qb)) continue;
                float db = pr.fields[t.b].Distance(qb);
                float dd = fmaxf(da, db);                       // inside both by dd (negative)
                if (dd >= occCap) continue;
                overlap += Overlap(dd);
                depth += fminf(fmaxf(0.0f, -dd), occCap);
                if (dd < -m_Tau && t.deep.size() < 4) t.deep.push_back(p);
            }
            double vol = (double)ext.x * ext.y * ext.z / m_S.pairUnit.size();
            t.overlap = overlap * vol;
            t.depth = depth * vol;
        });
        for (const Pair& t : pairs) {
            Moment& m = moments[t.moment];
            m.overlap += t.overlap;
            m.depth += t.depth;
            if (e.wantSpots)
                for (const Vector3& p : t.deep) e.motion.push_back({ p, pr.group[t.a], pr.group[t.b], m.t, m.opening });
        }
        double sum = 0.0;
        for (const Moment& m : moments) {
            float share = (float)std::min(1.0, m.overlap / e.designVolume);
            sum += 0.5 * share + 0.5 * std::min(1.0, m.depth / e.designVolume / occCap);
            if (share > sc.clashMotion) {
                sc.clashMotion = share;
                sc.clashMotionT = m.t;
                sc.clashMotionOpening = m.opening;
            }
        }
        sc.collision += (float)(sum / moments.size());
    }

    // 6. Closed pose against its sphere (Volume loss, or when diagnosing): overlap over the union (sphereIoU), how far
    //    the sphere's surface is from the closed shape's (sphereGap), and how far closed material reaches outside the
    //    sphere (sphereSpill: a small part swung away from the ball barely dents the overlap, but should be pulled in).
    void Scorer::SphereFit(Eval& e, AutoScore& sc) {
        if (sc.radius <= 0.0f || !e.useVolume) return;
        const float occCap = 6.0f * m_Tau;
        const float R = fmaxf(sc.radius, 1e-3f), shellCap = 0.3f * R;
        int nP = (int)m_S.sphereUnit.size(), nH = (int)m_S.shellDirs.size(), n = nP + nH;
        std::vector<float> d(n);
        std::vector<int> who(n, -1);
        m_Pool.ForEach(n, [&](int i) {
            if (i < nP) d[i] = e.pr.Distance(Vector3Add(sc.centre, Vector3Scale(m_S.sphereUnit[i], R)), &e.closed, occCap, &who[i]);
            else d[i] = e.pr.Distance(Vector3Add(sc.centre, Vector3Scale(m_S.shellDirs[i - nP], R)), &e.closed, shellCap, &who[i]);
        });
        double si = 0.0, su = 0.0, sg = 0.0;
        for (int i = 0; i < nP; i++) {
            float a = Occupancy(d[i]);
            Spot at = { Vector3Add(sc.centre, Vector3Scale(m_S.sphereUnit[i], R)), who[i] };
            if (m_S.sphereIn[i]) {
                si += a;
                su += 1.0;
                if (d[i] > 0.0f) e.holes.push_back(at);
            } else {
                su += a;
                if (d[i] < 0.0f) e.bumps.push_back(at);
            }
        }
        for (int i = nP; i < n; i++) sg += fminf(fabsf(d[i]), shellCap);
        double spill = 0.0;
        for (int k = 0; k < e.nC; k++) {
            Vector3 p = InBox(e.closedBox.min, e.cext, m_S.closedUnit[k]);
            spill += Occupancy(e.dist[e.cBase + k]) * fmaxf(0.0f, Vector3Distance(p, sc.centre) - R) / R;
        }
        sc.sphereSpill = (float)(spill / e.closedMass);
        sc.sphereIoU = su > 0.0 ? (float)(si / su) : 0.0f;
        sc.sphereGap = nH > 0 ? (float)(sg / nH / shellCap) : 0.0f;
        sc.sphere = fminf(0.75f * (1.0f - sc.sphereIoU) + 0.25f * sc.sphereGap + 0.5f * sc.sphereSpill, 1.5f);
    }

    // 7. A few randomly picked spots of each kind of error (up to 24, and 16 overlaps on the way).
    void Scorer::Diagnose(Eval& e, const AutoScore&, Diagnostics& diag) {
        auto take = [&](const std::vector<Spot>& from, std::vector<Spot>* out) {
            out->clear();
            for (int k = 0; k < 24 && !from.empty(); k++) out->push_back(PickFrom(m_Rng, from));
        };
        take(e.holes, &diag.holes);
        take(e.bumps, &diag.bumps);
        diag.missing.clear();
        for (int k = 0; k < 24 && !e.missing.empty(); k++) {
            int i = PickFrom(m_Rng, e.missing);
            diag.missing.push_back({ m_T.vol[i], e.who[i] });
        }
        take(e.excess, &diag.excess);
        diag.clashes.clear();
        for (int k = 0; k < 24 && !e.clashes.empty(); k++) {
            int i = PickFrom(m_Rng, e.clashes);
            bool closed = i >= e.cBase;
            Vector3 p = closed ? InBox(e.closedBox.min, e.cext, m_S.closedUnit[i - e.cBase])
                               : InBox(e.openBox.min, e.oext, m_S.openUnit[i - e.oBase]);
            diag.clashes.push_back({ p, e.who[i], e.who2[i], closed ? 1.0f : 0.0f, false });
        }
        for (int k = 0; k < 16 && !e.motion.empty(); k++) diag.clashes.push_back(PickFrom(m_Rng, e.motion));
    }

    // 9. What keeps a design simple and buildable: every shape, feature and body costs a little (simplicity), every
    //    body beyond the first that hangs on no hinge costs 0.1, and a pivot must sit on (or in) both bodies it joins.
    float Scorer::Costs(const Design& d, const Probe& pr) const {
        const float L = m_T.L;
        int features = 0;
        for (const Group& g : d.groups) features += g.parent >= 0;
        int bodies = (int)d.Bodies().size(), roots = 0;
        for (int g : d.Bodies()) roots += d.ParentHinge(d.groups[g].id) < 0;
        float penalty = m_P.simplicity * 0.012f * ((float)d.shapes.size() + 0.5f * features + 0.5f * bodies);
        penalty += 0.1f * (float)std::max(0, roots - 1);
        const float slack = 0.015f * L;
        for (const Hinge& h : d.hinges) {
            int a = pr.Index(h.parent), b = pr.Index(h.child);
            float dp = a >= 0 ? pr.fields[a].Distance(h.pivot) : L, dc = b >= 0 ? pr.fields[b].Distance(h.pivot) : L;
            penalty += 3.0f * (fmaxf(0.0f, dp - slack) + fmaxf(0.0f, dc - slack)) / L;
        }
        return penalty;
    }
}
