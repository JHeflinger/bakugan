#include "grammar.hpp"
#include "util/utils.hpp"
#include <raymath.h>
#include <algorithm>
#include <array>
#include <functional>
#include <cfloat>
#include <cmath>

namespace Bakugan {
    namespace {
        float Uniform(std::mt19937& rng, float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); }

        Vector3 RandomDirection(std::mt19937& rng) {
            std::normal_distribution<float> g(0.0f, 1.0f);
            Vector3 d = { g(rng), g(rng), g(rng) };
            return Vector3Length(d) > 1e-6f ? Vector3Normalize(d) : Vector3{ 0, 1, 0 };
        }

        bool Has(const std::vector<int>& v, int x) { return std::find(v.begin(), v.end(), x) != v.end(); }

        // Add a cut: a small cap sliced off a piece where it meets the sphere's surface, by a box whose inner face is
        // just under the surface, and hinged to the rest of the piece along that face's edge, like a lid. The joint
        // doesn't open yet (open angle 0), so nothing looks or scores different until Adam opens it.
        bool AddCut(Carving& c, const RuleContext& ctx) {
            if ((int)c.pieces.size() >= ctx.p.maxPieces) return false;
            const float R = c.radius;
            int pi = std::uniform_int_distribution<int>(0, (int)c.pieces.size() - 1)(ctx.rng);
            std::vector<int> bodyOf;
            Design d = c.Build(&bodyOf);
            BodyField piece = BodyField::Body(d, bodyOf[pi]);

            // where: a point of the sphere's surface that's this piece's, and the cap under it is the piece's too
            float a = R * Uniform(ctx.rng, 0.35f, 0.7f), depth = a * Uniform(ctx.rng, 0.15f, 0.3f);
            // With dependencies, and other pieces to be next to: where another piece's surface is within the cap's reach,
            // so that one of the two can come to hold the other shut (if nowhere is, anywhere).
            std::vector<BodyField> others;
            if (ctx.p.dependencies)
                for (size_t k = 0; k < c.pieces.size(); k++)
                    if ((int)k != pi) others.push_back(BodyField::Body(d, bodyOf[k]));
            Vector3 dir = { 0, 0, 0 };
            bool found = false;
            for (int k = 0; k < 192 && !found; k++) {
                dir = RandomDirection(ctx.rng);
                Vector3 top = Vector3Add(c.centre, Vector3Scale(dir, R - 0.02f * R));
                found = piece.Distance(top) < 0.0f && piece.Distance(Vector3Add(c.centre, Vector3Scale(dir, R - 0.5f * depth))) < 0.0f;
                if (!found || others.empty() || k >= 128) continue;
                bool near = false;
                for (const BodyField& o : others) near |= o.Distance(top) < 0.6f * a;
                found = near;
            }
            if (!found) return false;

            Cutter cut;
            cut.id = c.nextId++;
            cut.sf.size = { a, a, a };
            cut.sf.rotation = QuaternionFromVector3ToVector3({ 0, 0, 1 }, dir);
            cut.sf.position = Vector3Add(c.centre, Vector3Scale(dir, R - depth + 0.5f * a));   // its inner face 'depth' under the surface
            cut.sf.rounding = 0.1f;
            c.cutters.push_back(cut);
            Piece cap;
            cap.id = c.nextId++;
            cap.inside = c.pieces[pi].inside;
            cap.outside = c.pieces[pi].outside;
            cap.inside.push_back(cut.id);
            c.pieces[pi].outside.push_back(cut.id);
            c.pieces.push_back(cap);
            const int rest = c.pieces[pi].id;

            // the hinge's line: along the cut's edge, where it comes out of the sphere's surface. First along the cut
            // face to its edge (the farthest the face still has both pieces against it), then straight out from there
            // to the surface: a lid hinged on its outer edge swings clear of its seat, one hinged at the bottom of its
            // wall would swing the wall into it.
            d = c.Build(&bodyOf);
            BodyField restField = BodyField::Body(d, bodyOf[pi]), capField = BodyField::Body(d, bodyOf.back());
            Vector3 side = Vector3Normalize(Vector3CrossProduct(dir, fabsf(dir.y) < 0.9f ? Vector3{ 0, 1, 0 } : Vector3{ 1, 0, 0 }));
            float turn = Uniform(ctx.rng, 0.0f, 2.0f * PI);
            side = Vector3RotateByQuaternion(side, QuaternionFromAxisAngle(dir, turn));
            Vector3 face = Vector3Add(c.centre, Vector3Scale(dir, R - depth)), pivot = face;
            float out = 0.0f;
            for (int k = 1; k <= 64; k++) {
                Vector3 q = Vector3Add(face, Vector3Scale(side, k * R / 64.0f));
                Vector3 above = Vector3Add(q, Vector3Scale(dir, 0.01f * R)), below = Vector3Subtract(q, Vector3Scale(dir, 0.01f * R));
                if (capField.Distance(above) >= 0.0f || restField.Distance(below) >= 0.0f) break;
                pivot = q;
                out = k * R / 64.0f;
            }
            if (out < R) {
                Vector3 onSurface = Vector3Add(c.centre, Vector3Add(Vector3Scale(side, out), Vector3Scale(dir, 0.995f * sqrtf(R * R - out * out))));
                const float near = 0.01f * ctx.loss.L();
                if (capField.Distance(onSurface) < near && restField.Distance(onSurface) < near) pivot = onSurface;
            }
            // with hinges to be built under the surface: as deep as their barrel needs, from the start
            if (ctx.p.buildHinges && ctx.p.hinges.flush) {
                const float most = PlanHinge(R, ctx.p.hinges).pivotRadius, at = Vector3Distance(pivot, c.centre);
                if (at > most && at > 1e-4f) pivot = Vector3Add(c.centre, Vector3Scale(Vector3Subtract(pivot, c.centre), most / at));
                // ... and, the lid being thinnest at its rim, in from the rim until the lid is within a knuckle's reach
                // (as long as the rest still is)
                const float slack = ctx.loss.HingeSlack();
                for (int k = 0; k < 24 && capField.Distance(pivot) > slack; k++) {
                    Vector3 q = Vector3Subtract(pivot, Vector3Scale(side, R / 64.0f));
                    float r = Vector3Distance(q, c.centre);
                    if (r > most) q = Vector3Add(c.centre, Vector3Scale(Vector3Subtract(q, c.centre), most / r));
                    if (restField.Distance(q) > 2.0f * slack) break;
                    pivot = q;
                }
            }

            // the joints the piece had go with the part their pivot is in (or nearer)
            int anchor = rest;                              // the part on the piece's own joint
            for (Joint& j : c.joints) {
                bool toCap = capField.Distance(j.pivot) < restField.Distance(j.pivot);
                if (j.child == rest && toCap) { j.child = cap.id; anchor = cap.id; }
                else if (j.parent == rest && toCap) j.parent = cap.id;
            }
            Joint j;
            j.id = c.nextId++;
            j.parent = anchor;
            j.child = anchor == rest ? cap.id : rest;
            j.pivot = pivot;
            j.axis = Vector3Normalize(Vector3CrossProduct(dir, side));
            j.open = 0.0f;
            j.t0 = Uniform(ctx.rng, 0.0f, 0.5f);
            j.t1 = j.t0 + 0.5f;
            for (const Joint& o : c.joints) j.order = std::max(j.order, o.order + 1);   // it opens last, until the order is looked at again
            c.joints.push_back(j);
            Tidy(c);
            return true;
        }

        // Split: a random piece (any with material, equally likely) cut in two halves of equal volume by a plane (a box much bigger than the piece, one face
        // on the plane), the halves hinged together at angle 0. Nothing about the design changes: the two halves fill
        // exactly what the piece did and move as one (the hinge never turns until Adam opens it), so the loss before and
        // after is the same (to sampling: the halves' fields are the piece's, cut), and every parameter that was there
        // keeps its Adam memory (memory is per parameter id; the new cutter and joint start fresh). The new hinge is put
        // where it costs nothing: on the cut face, its barrel's length along the face, inside both halves.
        bool Split(Carving& c, const RuleContext& ctx) {
            if ((int)c.pieces.size() >= ctx.p.maxPieces) return false;
            const float R = c.radius;
            std::vector<int> bodyOf;
            Design d = c.Build(&bodyOf);

            // 1. a random piece, of those with material in them (volumes by sampling each piece's box)
            auto inside = [&](const BodyField& f, BoundingBox box, int n, std::vector<Vector3>* points) {
                Vector3 e = Vector3Subtract(box.max, box.min);
                int in = 0;
                for (int k = 0; k < n; k++) {
                    Vector3 q = { box.min.x + e.x * Uniform(ctx.rng, 0, 1), box.min.y + e.y * Uniform(ctx.rng, 0, 1), box.min.z + e.z * Uniform(ctx.rng, 0, 1) };
                    if (f.Distance(q) < 0.0f) {
                        in++;
                        if (points) points->push_back(q);
                    }
                }
                return (double)in / n * e.x * e.y * e.z;
            };
            std::vector<int> some;
            for (size_t k = 0; k < c.pieces.size(); k++)
                if (inside(BodyField::Body(d, bodyOf[k]), c.PieceBounds(c.pieces[k]), 1500, nullptr) > 0.0) some.push_back((int)k);
            std::shuffle(some.begin(), some.end(), ctx.rng);   // (one too slight to sample well: the next drawn instead)
            int pi = -1;
            std::vector<Vector3> pts;
            for (int k : some) {
                pts.clear();
                for (int tries = 0; tries < 6 && pts.size() < 4000; tries++) inside(BodyField::Body(d, bodyOf[k]), c.PieceBounds(c.pieces[k]), 4000, &pts);
                if (pts.size() >= 200) { pi = k; break; }
            }
            if (pi < 0) return false;
            const BodyField piece = BodyField::Body(d, bodyOf[pi]);

            // 2. the plane, at the median of the piece's points along a direction (half its volume either side), and the
            // hinge on it. Of 24 random directions, the one whose cut face has the best place for a hinge: where its
            // barrel, along the face, lies in both halves (on the plane, either half's field is the piece's, so this is
            // judged on the piece alone), and fits under the surface; a piece that isn't big has few such places, and a
            // hinge outside them is charged by the hinge term at once. Among those that fit, the one least edge on to
            // every view: a view along the plane sees the new seam as a soft line a pixel wide (each piece's silhouette
            // fades over a pixel at its own surface, the cut face too), the one thing about a split the loss can see.
            const HingeSizes hs = PlanHinge(R, ctx.p.hinges);
            const float halfBarrel = 0.5f * hs.length, slack = ctx.loss.HingeSlack();
            const bool flush = ctx.p.buildHinges && ctx.p.hinges.flush;
            const std::vector<Vector3> views = ctx.loss.ViewDirections();
            struct Plan {
                Vector3 dir, pivot, axis, onPlane, side;
                float offset = 0.0f, cost = FLT_MAX, edge = -1.0f;
            };
            Plan plan;
            std::vector<float> along(pts.size());
            for (int t = 0; t < 24; t++) {
                Plan q;
                q.dir = RandomDirection(ctx.rng);
                q.edge = 1.0f;
                for (const Vector3& v : views) q.edge = fminf(q.edge, fabsf(Vector3DotProduct(q.dir, v)));
                for (size_t k = 0; k < pts.size(); k++) along[k] = Vector3DotProduct(pts[k], q.dir);
                std::nth_element(along.begin(), along.begin() + along.size() / 2, along.end());
                q.offset = along[along.size() / 2] - Vector3DotProduct(c.centre, q.dir);
                if (fabsf(q.offset) >= R) continue;
                q.onPlane = Vector3Add(c.centre, Vector3Scale(q.dir, q.offset));
                q.side = Vector3Normalize(Vector3CrossProduct(q.dir, fabsf(q.dir.y) < 0.9f ? Vector3{ 0, 1, 0 } : Vector3{ 1, 0, 0 }));
                const Vector3 side2 = Vector3CrossProduct(q.dir, q.side);
                const float disc = sqrtf(R * R - q.offset * q.offset);
                int found = 0;
                for (int k = 0; k < 600 && found < 24; k++) {
                    float a = Uniform(ctx.rng, 0.0f, 2.0f * PI), r = disc * sqrtf(Uniform(ctx.rng, 0.0f, 1.0f));
                    Vector3 f = Vector3Add(q.onPlane, Vector3Add(Vector3Scale(q.side, r * cosf(a)), Vector3Scale(side2, r * sinf(a))));
                    const float depthHere = piece.Distance(f);
                    if (depthHere >= -0.01f * R) continue;
                    found++;
                    for (int m = 0; m < 12; m++) {
                        float an = PI * m / 12.0f;
                        Vector3 ax = Vector3Add(Vector3Scale(q.side, cosf(an)), Vector3Scale(side2, sinf(an)));
                        float cost = flush ? fmaxf(0.0f, Vector3Distance(f, c.centre) - hs.pivotRadius) : 0.0f;
                        for (float w : { -halfBarrel, -0.5f * halfBarrel, 0.0f, 0.5f * halfBarrel, halfBarrel })
                            cost += 2.0f * fmaxf(0.0f, piece.Distance(Vector3Add(f, Vector3Scale(ax, w))) - slack);
                        cost += 1e-6f * depthHere;         // (of those that fit: deepest in the piece, most room)
                        if (cost < q.cost) { q.cost = cost; q.pivot = f; q.axis = ax; }
                    }
                }
                if (q.cost == FLT_MAX) continue;
                // the piece's own hinges go with the half their pivot is in: their barrels should stay on that side
                for (const Joint& j : c.joints) {
                    if (j.parent != c.pieces[pi].id && j.child != c.pieces[pi].id) continue;
                    const Vector3 ja = Vector3Normalize(j.axis);
                    const float sp = Vector3DotProduct(Vector3Subtract(j.pivot, q.onPlane), q.dir) >= 0.0f ? 1.0f : -1.0f;
                    for (float w : { -halfBarrel, 0.0f, halfBarrel }) {
                        const float s = sp * Vector3DotProduct(Vector3Subtract(Vector3Add(j.pivot, Vector3Scale(ja, w)), q.onPlane), q.dir);
                        q.cost += fmaxf(0.0f, -s - slack);
                    }
                }
                // fits (to a hundredth of a radius, in all): then the views decide
                const bool fits = q.cost < 0.01f * R, planFits = plan.cost < 0.01f * R;
                if (fits != planFits ? fits : (fits ? q.edge > plan.edge : q.cost < plan.cost)) plan = q;
            }
            if (plan.cost == FLT_MAX) return false;
            const Vector3 dir = plan.dir, onPlane = plan.onPlane;
            const float offset = plan.offset;

            // 3. the cutter: a box past the sphere on the far side of the plane, every way (no more than a cutter may be:
            // Set would clamp it at Adam's first step, which would move the plane)
            Cutter cut;
            cut.id = c.nextId++;
            const float depth = 1.05f * (R - offset) + 0.01f * R, wide = 2.1f * sqrtf(R * R - offset * offset) + 0.02f * R;
            if (fmaxf(depth, wide) > 1.99f * ctx.loss.L()) return false;
            cut.sf.size = { wide, wide, depth };
            cut.sf.rotation = QuaternionFromVector3ToVector3({ 0, 0, 1 }, dir);
            cut.sf.position = Vector3Add(onPlane, Vector3Scale(dir, 0.5f * depth));
            cut.sf.rounding = 0.0f;
            c.cutters.push_back(cut);
            Piece half;                                     // the half on the box's side
            half.id = c.nextId++;
            half.inside = c.pieces[pi].inside;
            half.outside = c.pieces[pi].outside;
            half.inside.push_back(cut.id);
            c.pieces[pi].outside.push_back(cut.id);
            c.pieces.push_back(half);
            const int rest = c.pieces[pi].id;
            d = c.Build(&bodyOf);
            const BodyField restField = BodyField::Body(d, bodyOf[pi]), halfField = BodyField::Body(d, bodyOf.back());

            // 4. the hinge: the place found for it on the cut face (both halves along its barrel; it does nothing yet)
            const Vector3 pivot = plan.pivot, axis = plan.axis;

            // the joints the piece had go with the half their pivot is in (or nearer), as for a cut
            int anchor = rest;
            for (Joint& j : c.joints) {
                bool toHalf = halfField.Distance(j.pivot) < restField.Distance(j.pivot);
                if (j.child == rest && toHalf) { j.child = half.id; anchor = half.id; }
                else if (j.parent == rest && toHalf) j.parent = half.id;
            }
            Joint j;
            j.id = c.nextId++;
            j.parent = anchor;
            j.child = anchor == rest ? half.id : rest;
            j.pivot = pivot;
            j.axis = axis;
            j.open = 0.0f;                                  // it does nothing until Adam opens it
            j.t0 = Uniform(ctx.rng, 0.0f, 0.5f);
            j.t1 = j.t0 + 0.5f;
            for (const Joint& o : c.joints) j.order = std::max(j.order, o.order + 1);   // it opens last (its turn moves nothing)
            c.joints.push_back(j);
            Tidy(c);
            return true;
        }

        // One hinge (pivot, axis, angle) that moves points X as near as it can to Y: least squares, from a starting hinge
        // (Nelder-Mead over the axis' two angles, the turn, and the pivot). The root mean square of what's left.
        struct HingeFit {
            Vector3 pivot, axis;
            float angle = 0.0f, rms = 0.0f;
        };
        HingeFit FitHinge(const std::vector<Vector3>& X, const std::vector<Vector3>& Y, HingeFit start) {
            auto unpack = [](const std::array<float, 6>& v) {
                HingeFit h;
                h.axis = { cosf(v[1]) * cosf(v[0]), cosf(v[1]) * sinf(v[0]), sinf(v[1]) };
                h.angle = v[2];
                h.pivot = { v[3], v[4], v[5] };
                return h;
            };
            auto cost = [&](const std::array<float, 6>& v) {
                HingeFit h = unpack(v);
                const Rigid r = Rigid::AboutAxis(h.pivot, h.axis, h.angle);
                double sum = 0.0;
                for (size_t i = 0; i < X.size(); i++) sum += Vector3DistanceSqr(r.Apply(X[i]), Y[i]);
                return sum / std::max<size_t>(1, X.size());
            };
            Vector3 a = Vector3Normalize(start.axis);
            std::array<float, 6> v0 = { atan2f(a.y, a.x), asinf(std::clamp(a.z, -1.0f, 1.0f)), start.angle, start.pivot.x, start.pivot.y, start.pivot.z };
            // the simplex: steps of a tenth of a radian, and of the points' spread
            float spread = 1.0f;
            for (const Vector3& x : X) spread = fmaxf(spread, Vector3Distance(x, X[0]));
            const float step[6] = { 0.1f, 0.1f, 0.1f, 0.1f * spread, 0.1f * spread, 0.1f * spread };
            std::array<std::array<float, 6>, 7> pts;
            std::array<double, 7> val;
            for (int i = 0; i < 7; i++) {
                pts[i] = v0;
                if (i > 0) pts[i][i - 1] += step[i - 1];
                val[i] = cost(pts[i]);
            }
            for (int it = 0; it < 600; it++) {
                std::array<int, 7> order;
                for (int i = 0; i < 7; i++) order[i] = i;
                std::sort(order.begin(), order.end(), [&](int x, int y) { return val[x] < val[y]; });
                const int best = order[0], worst = order[6], second = order[5];
                std::array<float, 6> centroid{};
                for (int i = 0; i < 6; i++)
                    for (int d = 0; d < 6; d++) centroid[d] += pts[order[i]][d] / 6.0f;
                auto along = [&](float t) {
                    std::array<float, 6> q;
                    for (int d = 0; d < 6; d++) q[d] = centroid[d] + t * (pts[worst][d] - centroid[d]);
                    return q;
                };
                std::array<float, 6> r = along(-1.0f);
                double fr = cost(r);
                if (fr < val[best]) {
                    std::array<float, 6> e = along(-2.0f);
                    double fe = cost(e);
                    if (fe < fr) { pts[worst] = e; val[worst] = fe; }
                    else { pts[worst] = r; val[worst] = fr; }
                } else if (fr < val[second]) {
                    pts[worst] = r;
                    val[worst] = fr;
                } else {
                    std::array<float, 6> c2 = along(0.5f);
                    double fc = cost(c2);
                    if (fc < val[worst]) { pts[worst] = c2; val[worst] = fc; }
                    else
                        for (int i = 0; i < 7; i++) {
                            if (i == best) continue;
                            for (int d = 0; d < 6; d++) pts[i][d] = pts[best][d] + 0.5f * (pts[i][d] - pts[best][d]);
                            val[i] = cost(pts[i]);
                        }
                }
                if (val[order[6]] - val[order[0]] < 1e-10) break;
            }
            int best = 0;
            for (int i = 1; i < 7; i++)
                if (val[i] < val[best]) best = i;
            HingeFit h = unpack(pts[best]);
            h.rms = (float)sqrt(val[best]);
            if (h.angle > PI) h.angle -= 2.0f * PI;
            if (h.angle < -PI) h.angle += 2.0f * PI;
            return h;
        }

        // Remove a piece: one too thin, small or slight to matter goes, its material with it (the cutters only it used go
        // too). The pieces that hung from it are hung from its parent instead, each by one hinge that moves it, in the
        // open pose, as near as one hinge can to where the two hinges took it. Two turns about different axes compose to
        // a turn and a slide along its axis, which no hinge makes; so that hinge is fitted (least squares) to points
        // spread over the child and everything hanging from it, starting from the composed turn about the axis line
        // that comes closest. The child's own hinge id is kept for it (its Adam memory with it).
        //
        // Which piece: every piece a hinge turns (not the root, which holds the rest still) is scored by how slight it
        // is (sampled volume, as a share of the sphere, times its mean depth inside, in radii: small or thin both score
        // low) plus how far its removal would move the rest (the fitted hinges' root mean square misses, in radii, times
        // the share of the sphere that moves). One whose children would move more than a fifth of a radius isn't
        // considered. Of the rest, the k-th slightest is drawn with weight 2^-k. The design loses the piece's material,
        // and nothing else moves but by those misses: the smallest change taking a piece away can be.
        bool RemovePiece(Carving& c, const RuleContext& ctx) {
            if (c.pieces.size() < 2) return false;
            std::vector<int> bodyOf;
            Design d = c.Build(&bodyOf);
            const double sphere = 4.0 / 3.0 * PI * c.radius * c.radius * c.radius;
            const Vector3 lo = Vector3SubtractValue(c.centre, c.radius), hi = Vector3AddValue(c.centre, c.radius);
            auto indexOf = [&](int id) {
                for (size_t k = 0; k < c.pieces.size(); k++)
                    if (c.pieces[k].id == id) return (int)k;
                return -1;
            };
            // each piece's volume share and points spread over it (corners and centre of its bounds, within the sphere's)
            std::vector<double> volume(c.pieces.size()), thick(c.pieces.size());
            std::vector<std::vector<Vector3>> spots(c.pieces.size());
            for (size_t k = 0; k < c.pieces.size(); k++) {
                const BodyField f = BodyField::Body(d, bodyOf[k]);
                BoundingBox box = c.PieceBounds(c.pieces[k]);
                box = { Vector3Max(box.min, lo), Vector3Min(box.max, hi) };
                const Vector3 e = Vector3Subtract(box.max, box.min);
                if (e.x <= 0 || e.y <= 0 || e.z <= 0) continue;  // (nothing there: volume 0)
                int in = 0;
                double depth = 0.0;
                const int n = 1200;
                for (int s = 0; s < n; s++) {
                    Vector3 q = { box.min.x + e.x * Uniform(ctx.rng, 0, 1), box.min.y + e.y * Uniform(ctx.rng, 0, 1), box.min.z + e.z * Uniform(ctx.rng, 0, 1) };
                    float dist = f.Distance(q);
                    if (dist < 0.0f) {
                        in++;
                        depth -= dist;
                        if (spots[k].size() < 64) spots[k].push_back(q);
                    }
                }
                volume[k] = (double)in / n * e.x * e.y * e.z / sphere;
                thick[k] = in > 0 ? depth / in / c.radius : 0.0;
            }
            // a piece's subtree: it and everything hanging from it
            std::function<void(int, std::vector<int>&)> subtree = [&](int id, std::vector<int>& out) {
                out.push_back(id);
                for (const Joint& j : c.joints)
                    if (j.parent == id) subtree(j.child, out);
            };
            // the hinge each child of a removed piece would get, fitted
            auto rehang = [&](const Joint& above, const Joint& j, double* moved) {
                const Rigid composed = Rigid::AboutAxis(j.pivot, j.axis, j.open).Then(Rigid::AboutAxis(above.pivot, above.axis, above.open));
                HingeFit h;
                Vector3 axis;
                float angle;
                QuaternionToAxisAngle(QuaternionNormalize(composed.rotation), &axis, &angle);
                if (angle > PI) angle -= 2.0f * PI;
                h.pivot = j.pivot;
                h.axis = j.axis;
                h.angle = 0.0f;
                if (fabsf(angle) > 1e-5f && Vector3Length(axis) > 1e-6f) {
                    // the composed turn, about the axis line closest to the same place: (I - R) q = t across the axis
                    axis = Vector3Normalize(axis);
                    const Vector3 t = composed.position;
                    const Vector3 e1 = Utils::AnyPerpendicular(axis), e2 = Vector3CrossProduct(axis, e1);
                    const float tx = Vector3DotProduct(t, e1), ty = Vector3DotProduct(t, e2);
                    const float cs = cosf(angle), sn = sinf(angle), det = (1.0f - cs) * (1.0f - cs) + sn * sn;
                    const float qx = ((1.0f - cs) * tx - sn * ty) / det, qy = (sn * tx + (1.0f - cs) * ty) / det;
                    h.pivot = Vector3Add(Vector3Scale(e1, qx), Vector3Scale(e2, qy));
                    h.axis = axis;
                    h.angle = angle;
                }
                // fitted to the points of the child and all below it
                std::vector<int> below;
                subtree(j.child, below);
                std::vector<Vector3> X, Y;
                double share = 0.0;
                for (int id : below) {
                    int k = indexOf(id);
                    if (k < 0) continue;
                    share += volume[k];
                    for (const Vector3& x : spots[k]) {
                        X.push_back(x);
                        Y.push_back(composed.Apply(x));
                    }
                }
                if (!X.empty()) {
                    HingeFit start = h;
                    const Rigid r0 = Rigid::AboutAxis(h.pivot, h.axis, h.angle);
                    double s0 = 0.0;
                    for (size_t i = 0; i < X.size(); i++) s0 += Vector3DistanceSqr(r0.Apply(X[i]), Y[i]);
                    start.rms = (float)sqrt(s0 / X.size());
                    // (the composed turn moves every point by the same slide, and keeps the child's turn exactly: a
                    // fit replaces it only where it misses by clearly less)
                    HingeFit fit = FitHinge(X, Y, start);
                    h = fit.rms < 0.8f * start.rms ? fit : start;
                }
                // along its axis: the pivot nearest where the child's own hinge was (the motion is the same)
                const Vector3 a = Vector3Normalize(h.axis);
                h.pivot = Vector3Add(h.pivot, Vector3Scale(a, Vector3DotProduct(Vector3Subtract(j.pivot, h.pivot), a)));
                h.axis = a;
                if (moved) *moved += h.rms / c.radius * share;
                return h;
            };

            struct Candidate {
                int piece;
                double score;
            };
            std::vector<Candidate> cand;
            for (size_t k = 0; k < c.pieces.size(); k++) {
                const int id = c.pieces[k].id;
                const Joint* above = nullptr;
                for (const Joint& j : c.joints)
                    if (j.child == id) above = &j;
                if (!above) continue;                    // (an empty piece is the first to go: volume 0)
                double moved = 0.0;
                bool tooFar = false;
                for (const Joint& j : c.joints) {
                    if (j.parent != id) continue;
                    double m = 0.0;
                    HingeFit h = rehang(*above, j, &m);
                    moved += m;
                    tooFar |= h.rms > 0.2f * c.radius;
                }
                if (tooFar) continue;
                cand.push_back({ (int)k, volume[k] * thick[k] + moved });
            }
            if (cand.empty()) return false;
            std::sort(cand.begin(), cand.end(), [](const Candidate& a, const Candidate& b) { return a.score < b.score; });
            double total = 0.0;
            for (size_t k = 0; k < cand.size(); k++) total += ldexp(1.0, -(int)k);
            double pick = Uniform(ctx.rng, 0.0f, 1.0f) * total;
            size_t which = 0;
            for (; which + 1 < cand.size(); which++) {
                pick -= ldexp(1.0, -(int)which);
                if (pick <= 0.0) break;
            }
            const int pi = cand[which].piece, gone = c.pieces[pi].id;
            int up = -1;
            for (size_t k = 0; k < c.joints.size(); k++)
                if (c.joints[k].child == gone) up = (int)k;
            if (up < 0) return false;
            const Joint above = c.joints[up];
            for (Joint& j : c.joints) {
                if (j.parent != gone) continue;
                HingeFit h = rehang(above, j, nullptr);
                j.parent = above.parent;
                j.pivot = h.pivot;
                j.axis = h.axis;
                j.open = h.angle;
                j.order = std::min(j.order, above.order);     // it turns when the first of the two did
            }
            // the piece, its hinge, and the cutters no piece uses any more
            c.joints.erase(c.joints.begin() + up);
            c.pieces.erase(c.pieces.begin() + pi);
            c.cutters.erase(std::remove_if(c.cutters.begin(), c.cutters.end(), [&](const Cutter& k) { return !k.hollow && c.PiecesUsing(k.id).empty(); }),
                            c.cutters.end());
            // the opening order without gaps (ties: the older first)
            std::vector<size_t> rank(c.joints.size());
            for (size_t k = 0; k < rank.size(); k++) rank[k] = k;
            std::stable_sort(rank.begin(), rank.end(), [&](size_t a, size_t b) { return c.joints[a].order < c.joints[b].order; });
            for (size_t r = 0; r < rank.size(); r++) c.joints[rank[r]].order = (int)r;
            Tidy(c);
            return true;
        }

        // Add a hollow: material taken away where the open design shows outside the mesh. At one of those pixels
        // (Loss::ExtraSpots), a rounded bar through its piece along the view that saw it, as long as the piece is deep
        // there, so the pixel clears at once. If the bar reaches the sphere's surface, the surface term weighs that; Adam
        // then shapes and moves it (inward, where it can).
        bool AddHollow(Carving& c, const RuleContext& ctx) {
            const std::vector<Loss::Spot>& spots = ctx.loss.ExtraSpots();
            if (spots.empty() || c.Hollows() >= ctx.p.maxHollows) return false;
            const Loss::Spot& spot = spots[std::uniform_int_distribution<size_t>(0, spots.size() - 1)(ctx.rng)];
            if (spot.piece < 0 || spot.piece >= (int)c.pieces.size()) return false;
            std::vector<int> bodyOf;
            Design d = c.Build(&bodyOf);
            BodyField piece = BodyField::Body(d, bodyOf[spot.piece]);
            // the piece's depth along the view, both ways from the spot
            const float R = c.radius, step = 0.01f * R;
            float ends[2] = { 0.0f, 0.0f };
            for (int side = 0; side < 2; side++) {
                Vector3 dir = Vector3Scale(spot.dir, side == 0 ? 1.0f : -1.0f);
                float t = 0.0f;
                while (t < 2.2f * R && piece.Distance(Vector3Add(spot.at, Vector3Scale(dir, t + step))) < 0.0f) t += step;
                ends[side] = t;
            }
            Cutter h;
            h.id = c.nextId++;
            h.hollow = true;
            float width = R * Uniform(ctx.rng, 0.12f, 0.3f), length = ends[0] + ends[1] + 2.0f * step;
            h.sf.size = { width, width, length };
            h.sf.rotation = QuaternionFromVector3ToVector3({ 0, 0, 1 }, spot.dir);
            h.sf.position = Vector3Add(spot.at, Vector3Scale(spot.dir, 0.5f * (ends[0] - ends[1])));
            h.sf.rounding = 0.5f;
            c.cutters.push_back(h);
            return true;
        }

        // Remove a hollow (any).
        bool RemoveHollow(Carving& c, const RuleContext& ctx) {
            std::vector<int> hollows;
            for (size_t k = 0; k < c.cutters.size(); k++)
                if (c.cutters[k].hollow) hollows.push_back((int)k);
            if (hollows.empty()) return false;
            c.cutters.erase(c.cutters.begin() + hollows[std::uniform_int_distribution<size_t>(0, hollows.size() - 1)(ctx.rng)]);
            return true;
        }

        // Remove a cut: two pieces one cutter split (and nothing since), joined by a joint (likelier the less it opens),
        // become one piece again, the cutter and the joint gone. The joints of the one that hung from the other go to it.
        bool RemoveCut(Carving& c, const RuleContext& ctx) {
            struct Candidate {
                int joint, cutter;
            };
            std::vector<Candidate> found;
            std::vector<double> weight;
            for (size_t ji = 0; ji < c.joints.size(); ji++) {
                const Joint& j = c.joints[ji];
                int a = c.PieceIndex(j.parent), b = c.PieceIndex(j.child);
                if (a < 0 || b < 0) continue;
                const Piece &A = c.pieces[a], &B = c.pieces[b];
                for (const Cutter& k : c.cutters) {
                    if (k.hollow) continue;
                    bool split = (Has(A.outside, k.id) && Has(B.inside, k.id)) || (Has(A.inside, k.id) && Has(B.outside, k.id));
                    if (!split || c.PiecesUsing(k.id).size() != 2) continue;
                    auto without = [&](std::vector<int> v) {
                        v.erase(std::remove(v.begin(), v.end(), k.id), v.end());
                        std::sort(v.begin(), v.end());
                        return v;
                    };
                    if (without(A.inside) != without(B.inside) || without(A.outside) != without(B.outside)) continue;
                    found.push_back({ (int)ji, k.id });
                    weight.push_back(1.0 / (fabs(j.open) + 0.1));
                }
            }
            if (found.empty()) return false;
            Candidate pick = found[std::discrete_distribution<size_t>(weight.begin(), weight.end())(ctx.rng)];
            Joint j = c.joints[pick.joint];
            Piece& keep = c.pieces[c.PieceIndex(j.parent)];
            auto drop = [&](std::vector<int>& v) { v.erase(std::remove(v.begin(), v.end(), pick.cutter), v.end()); };
            drop(keep.inside);
            drop(keep.outside);
            const int keepId = keep.id;
            for (Joint& g : c.joints)
                if (g.parent == j.child) g.parent = keepId;
            c.joints.erase(c.joints.begin() + pick.joint);
            c.pieces.erase(c.pieces.begin() + c.PieceIndex(j.child));
            c.cutters.erase(c.cutters.begin() + c.CutterIndex(pick.cutter));
            Tidy(c);                                        // (an ordered carving's joints close ranks)
            return true;
        }
    }

    const std::vector<Rule>& Rules() {
        static const std::vector<Rule> rules = {
            { "add a cut",       1.0f, AddCut,       false },
            { "remove a cut",    0.5f, RemoveCut,    true },
            { "add a hollow",    0.5f, AddHollow,    false },
            { "remove a hollow", 0.25f, RemoveHollow, true },
            { "split",           0.5f, Split,        false },   // (new ones go last: weights are kept in this order)
            { "remove a piece",  0.25f, RemovePiece, true },
        };
        return rules;
    }

    const Rule* ApplyRandomRule(Carving& c, const RuleContext& ctx, const std::vector<float>& weights) {
        const std::vector<Rule>& rules = Rules();
        auto weight = [&](size_t i) { return std::max(0.0f, i < weights.size() ? weights[i] : rules[i].weight); };
        float total = 0.0f;
        for (size_t i = 0; i < rules.size(); i++) total += weight(i);
        if (total <= 0.0f) return nullptr;                  // every rule switched off
        float pick = Uniform(ctx.rng, 0.0f, total);
        for (size_t i = 0; i < rules.size(); i++) {
            if (weight(i) <= 0.0f) continue;
            if (pick <= weight(i) || i + 1 == rules.size()) return rules[i].apply(c, ctx) ? &rules[i] : nullptr;
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
