#include "grammar.hpp"
#include <raymath.h>
#include <algorithm>
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

        // Split: the biggest piece cut in two halves of equal volume by a plane (a box much bigger than the piece, one face
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

            // 1. the biggest piece: volumes by sampling each piece's box
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
            int pi = -1;
            double most = 0.0;
            for (size_t k = 0; k < c.pieces.size(); k++) {
                double v = inside(BodyField::Body(d, bodyOf[k]), c.PieceBounds(c.pieces[k]), 1500, nullptr);
                if (v > most) { most = v; pi = (int)k; }
            }
            if (pi < 0) return false;
            const BodyField piece = BodyField::Body(d, bodyOf[pi]);
            const BoundingBox bounds = c.PieceBounds(c.pieces[pi]);
            std::vector<Vector3> pts;
            for (int tries = 0; tries < 6 && pts.size() < 4000; tries++) inside(piece, bounds, 4000, &pts);
            if (pts.size() < 200) return false;

            // 2. the plane: a random direction, at the median of the piece's points along it (half its volume either side).
            // Of a few drawn, the one least edge on to every view: a view along the plane sees the new seam as a soft line
            // a pixel wide (each piece's silhouette fades over a pixel at its own surface, the cut face too), the one
            // thing about a split the loss can see. Face on, it sees nothing.
            Vector3 dir = RandomDirection(ctx.rng);
            {
                const std::vector<Vector3> views = ctx.loss.ViewDirections();
                float best = -1.0f;
                for (int k = 0; k < 24; k++) {
                    Vector3 cand = k == 0 ? dir : RandomDirection(ctx.rng);
                    float worst = 1.0f;
                    for (const Vector3& v : views) worst = fminf(worst, fabsf(Vector3DotProduct(cand, v)));
                    if (worst > best) { best = worst; dir = cand; }
                }
            }
            std::vector<float> along(pts.size());
            for (size_t k = 0; k < pts.size(); k++) along[k] = Vector3DotProduct(pts[k], dir);
            std::nth_element(along.begin(), along.begin() + along.size() / 2, along.end());
            const float median = along[along.size() / 2];
            const float offset = median - Vector3DotProduct(c.centre, dir);   // the plane, from the sphere's centre
            if (fabsf(offset) >= R) return false;
            const Vector3 onPlane = Vector3Add(c.centre, Vector3Scale(dir, offset));      // nearest the centre

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

            // 4. the hinge: on the cut face, where both halves are along its barrel (no cost from the hinge term)
            const Vector3 side = Vector3Normalize(Vector3CrossProduct(dir, fabsf(dir.y) < 0.9f ? Vector3{ 0, 1, 0 } : Vector3{ 1, 0, 0 }));
            const Vector3 side2 = Vector3CrossProduct(dir, side);
            const float disc = sqrtf(R * R - offset * offset);
            const HingeSizes hs = PlanHinge(R, ctx.p.hinges);
            const float halfBarrel = 0.5f * hs.length, slack = ctx.loss.HingeSlack();
            const bool flush = ctx.p.buildHinges && ctx.p.hinges.flush;
            std::vector<Vector3> face;                      // points of the cut face (in the piece, on the plane)
            for (int k = 0; k < 2000 && face.size() < 64; k++) {
                float a = Uniform(ctx.rng, 0.0f, 2.0f * PI), r = disc * sqrtf(Uniform(ctx.rng, 0.0f, 1.0f));
                Vector3 q = Vector3Add(onPlane, Vector3Add(Vector3Scale(side, r * cosf(a)), Vector3Scale(side2, r * sinf(a))));
                if (piece.Distance(q) < -0.01f * R) face.push_back(q);
            }
            if (face.empty()) {                            // (sampling missed a thin piece: undo)
                c.pieces.pop_back();
                c.pieces[pi].outside.pop_back();
                c.cutters.pop_back();
                return false;
            }
            float bestCost = FLT_MAX;
            Vector3 pivot = face[0], axis = side;
            for (const Vector3& q : face)
                for (int k = 0; k < 12; k++) {
                    float a = PI * k / 12.0f;
                    Vector3 ax = Vector3Add(Vector3Scale(side, cosf(a)), Vector3Scale(side2, sinf(a)));
                    float cost = flush ? fmaxf(0.0f, Vector3Distance(q, c.centre) - hs.pivotRadius) : 0.0f;
                    for (float w : { -halfBarrel, 0.0f, halfBarrel }) {
                        Vector3 p = Vector3Add(q, Vector3Scale(ax, w));
                        cost += fmaxf(0.0f, restField.Distance(p) - slack) + fmaxf(0.0f, halfField.Distance(p) - slack);
                    }
                    // among those that cost nothing: the one deepest in the piece (most room for Adam to move it)
                    cost += 1e-6f * piece.Distance(q);
                    if (cost < bestCost) { bestCost = cost; pivot = q; axis = ax; }
                }

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
            { "split",           0.5f, Split,        false },   // (last: weights are kept in this order)
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
