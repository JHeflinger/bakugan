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
            Vector3 dir = { 0, 0, 0 };
            bool found = false;
            for (int k = 0; k < 64 && !found; k++) {
                dir = RandomDirection(ctx.rng);
                found = piece.Distance(Vector3Add(c.centre, Vector3Scale(dir, R - 0.02f * R))) < 0.0f &&
                        piece.Distance(Vector3Add(c.centre, Vector3Scale(dir, R - 0.5f * depth))) < 0.0f;
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

            // the hinge's line: along the cut face, at its edge (the farthest the face still has both pieces against it)
            d = c.Build(&bodyOf);
            BodyField restField = BodyField::Body(d, bodyOf[pi]), capField = BodyField::Body(d, bodyOf.back());
            Vector3 side = Vector3Normalize(Vector3CrossProduct(dir, fabsf(dir.y) < 0.9f ? Vector3{ 0, 1, 0 } : Vector3{ 1, 0, 0 }));
            float turn = Uniform(ctx.rng, 0.0f, 2.0f * PI);
            side = Vector3RotateByQuaternion(side, QuaternionFromAxisAngle(dir, turn));
            Vector3 face = Vector3Add(c.centre, Vector3Scale(dir, R - depth)), pivot = face;
            for (int k = 1; k <= 64; k++) {
                Vector3 q = Vector3Add(face, Vector3Scale(side, k * R / 64.0f));
                Vector3 above = Vector3Add(q, Vector3Scale(dir, 0.01f * R)), below = Vector3Subtract(q, Vector3Scale(dir, 0.01f * R));
                if (capField.Distance(above) >= 0.0f || restField.Distance(below) >= 0.0f) break;
                pivot = q;
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
            c.joints.push_back(j);
            return true;
        }

        // ---- cuts aimed by transport, and splits ------------------------------------------------------------------------

        // Splits piece pi with cutter `cut` (the new piece: the part of it inside the cutter) and hinges the two parts
        // along the edge of the cut face, closed (open angle 0): nothing looks or scores different until Adam opens it,
        // as with "add a cut". face: a point of the cut face, in the piece; out: the face's normal, into the new piece.
        // The joint's axis is set so that opening it (a positive angle) lifts the new piece out of the rest, like a lid.
        // With grains (transport's material, from the last evaluation), the hinge goes where along the face's edge
        // opening it most lowers transport (twelve tried: the slope of sum |x - to|^2 over the material it moves, measured
        // by opening each a little); with `needsGain`, the rule doesn't apply when no side lowers it. Without, one side
        // at random.
        bool SplitAndHinge(Carving& c, int pi, Cutter cut, Vector3 face, Vector3 out, const RuleContext& ctx,
                           const std::vector<Loss::Grain>* grains, bool needsGain) {
            const Carving before = c;
            const float R = c.radius;
            cut.id = c.nextId++;
            c.cutters.push_back(cut);
            Piece cap;
            cap.id = c.nextId++;
            cap.inside = c.pieces[pi].inside;
            cap.outside = c.pieces[pi].outside;
            cap.inside.push_back(cut.id);
            c.pieces[pi].outside.push_back(cut.id);
            c.pieces.push_back(cap);
            const int rest = c.pieces[pi].id, capIndex = (int)c.pieces.size() - 1;
            std::vector<int> bodyOf;
            Design d = c.Build(&bodyOf);
            BodyField restField = BodyField::Body(d, bodyOf[pi]), capField = BodyField::Body(d, bodyOf[capIndex]);

            // both parts have material (points through the sphere in each)
            int inCap = 0, inRest = 0;
            for (int k = 0; k < 600; k++) {
                Vector3 q = Vector3Add(c.centre, Vector3Scale(RandomDirection(ctx.rng), R * cbrtf(Uniform(ctx.rng, 0.0f, 1.0f))));
                inCap += capField.Distance(q) < 0.0f;
                inRest += restField.Distance(q) < 0.0f;
            }
            if (inCap < 4 || inRest < 4) { c = before; return false; }

            // the joints the piece had go with the part their pivot is in (or nearer)
            int anchor = rest;
            for (Joint& j : c.joints) {
                bool toCap = capField.Distance(j.pivot) < restField.Distance(j.pivot);
                if (j.child == rest && toCap) { j.child = cap.id; anchor = cap.id; }
                else if (j.parent == rest && toCap) j.parent = cap.id;
            }
            const int child = anchor == rest ? cap.id : rest;
            const BodyField& childField = child == cap.id ? capField : restField;
            std::vector<const Loss::Grain*> moving;                  // the material the new joint turns
            if (grains)
                for (const Loss::Grain& g : *grains)
                    if (g.piece == pi && childField.Distance(g.local) < 0.0f) moving.push_back(&g);

            Vector3 side0 = Vector3Normalize(Vector3CrossProduct(out, fabsf(out.y) < 0.9f ? Vector3{ 0, 1, 0 } : Vector3{ 1, 0, 0 }));
            const bool aim = grains && !moving.empty();
            const int sides = aim ? 12 : 1;
            const float turn0 = Uniform(ctx.rng, 0.0f, 2.0f * PI);
            Joint best;
            float bestSlope = FLT_MAX;
            bool found = false;
            for (int si = 0; si < sides; si++) {
                Vector3 side = Vector3RotateByQuaternion(side0, QuaternionFromAxisAngle(out, turn0 + si * 2.0f * PI / sides));
                // the hinge's line: along the cut face, at its edge (the farthest the face still has both parts against it)
                Vector3 pivot = face;
                for (int k = 1; k <= 64; k++) {
                    Vector3 q = Vector3Add(face, Vector3Scale(side, k * 2.0f * R / 64.0f));
                    Vector3 above = Vector3Add(q, Vector3Scale(out, 0.01f * R)), below = Vector3Subtract(q, Vector3Scale(out, 0.01f * R));
                    if (capField.Distance(above) >= 0.0f || restField.Distance(below) >= 0.0f) break;
                    pivot = q;
                }
                Joint j;
                j.id = c.nextId;
                j.parent = anchor;
                j.child = child;
                j.pivot = pivot;
                j.axis = Vector3Normalize(Vector3CrossProduct(out, side));
                j.open = 0.0f;
                j.t0 = Uniform(ctx.rng, 0.0f, 0.5f);
                j.t1 = j.t0 + 0.5f;
                // opened a little: which way the new piece goes (relative to the rest), and the material it moves
                const float delta = 0.02f;
                auto opened = [&](const Joint& jj, std::vector<Rigid>* xf, std::vector<int>* bo) {
                    Carving t = c;
                    t.joints.push_back(jj);
                    t.joints.back().open = delta;
                    Design dt = t.Build(bo);
                    *xf = dt.Transforms(0.0f, false);
                    for (size_t k = 0; k < bo->size(); k++) (*bo)[k] = dt.GroupIndex((*bo)[k]);
                };
                auto closedXf = [&](std::vector<Rigid>* xf, std::vector<int>* bo) {
                    Design dt = c.Build(bo);
                    *xf = dt.Transforms(0.0f, false);
                    for (size_t k = 0; k < bo->size(); k++) (*bo)[k] = dt.GroupIndex((*bo)[k]);
                };
                std::vector<Rigid> xf0, xf1;
                std::vector<int> bo0, bo1;
                closedXf(&xf0, &bo0);
                opened(j, &xf1, &bo1);
                Vector3 probe = Vector3Add(face, Vector3Scale(out, 0.05f * R));       // a point of the new piece
                auto lift = [&](const std::vector<Rigid>& xf, const std::vector<int>& bo) {
                    Vector3 w = xf[bo[capIndex]].Apply(probe);
                    return Vector3DotProduct(Vector3Subtract(xf[bo[pi]].ApplyInverse(w), probe), out);
                };
                if (lift(xf1, bo1) - lift(xf0, bo0) < 0.0f) {          // it went into the rest: the other way round
                    j.axis = Vector3Negate(j.axis);
                    opened(j, &xf1, &bo1);
                }
                float slope = 0.0f;
                if (aim) {
                    const int ci = child == cap.id ? capIndex : pi;
                    double sum = 0.0;
                    for (const Loss::Grain* g : moving) {
                        Vector3 x0 = xf0[bo0[ci]].Apply(g->local), x1 = xf1[bo1[ci]].Apply(g->local);
                        sum += 2.0 * Vector3DotProduct(Vector3Subtract(x0, g->to), Vector3Subtract(x1, x0)) / delta;
                    }
                    slope = (float)(sum / moving.size());
                }
                if (!found || slope < bestSlope) { best = j; bestSlope = slope; found = true; }
            }
            if (!found || (needsGain && (!aim || bestSlope >= 0.0f))) { c = before; return false; }
            best.id = c.nextId++;
            c.joints.push_back(best);
            return true;
        }

        // Aim a cut: like "add a cut" (a cap off a piece, hinged like a lid, closed), but where transport says material
        // is far from where it belongs. A grain is drawn by how far it has to travel (squared); the cap is cut toward
        // the sphere's surface over it, deep enough to take it; and the hinge goes where along the cap's edge opening it
        // sends that material toward where it's needed. It doesn't apply unless opening the cap would lower transport.
        // Needs transport on (its grains).
        bool AimCut(Carving& c, const RuleContext& ctx) {
            const std::vector<Loss::Grain>& grains = ctx.loss.Grains();
            if (grains.empty() || (int)c.pieces.size() >= ctx.p.maxPieces) return false;
            std::vector<double> weight;
            for (const Loss::Grain& g : grains)
                weight.push_back(g.piece >= 0 && g.piece < (int)c.pieces.size() ? Vector3DistanceSqr(g.at, g.to) : 0.0);
            double total = 0.0;
            for (double w : weight) total += w;
            if (total <= 0.0) return false;
            const Loss::Grain& g = grains[std::discrete_distribution<size_t>(weight.begin(), weight.end())(ctx.rng)];
            const float R = c.radius;
            Vector3 off = Vector3Subtract(g.local, c.centre);
            float r = Vector3Length(off);
            if (r < 0.25f * R) return false;                       // too deep for a cap
            Vector3 dir = Vector3Scale(off, 1.0f / r);
            float a = R * Uniform(ctx.rng, 0.35f, 0.7f);
            float depth = std::clamp(R - r + 0.05f * R, 0.08f * R, 0.85f * a);
            std::vector<int> bodyOf;
            Design d = c.Build(&bodyOf);
            BodyField piece = BodyField::Body(d, bodyOf[g.piece]);
            if (piece.Distance(Vector3Add(c.centre, Vector3Scale(dir, R - 0.02f * R))) >= 0.0f ||
                piece.Distance(Vector3Add(c.centre, Vector3Scale(dir, R - 0.5f * depth))) >= 0.0f)
                return false;
            Cutter cut;
            cut.sf.size = { a, a, a };
            cut.sf.rotation = QuaternionFromVector3ToVector3({ 0, 0, 1 }, dir);
            cut.sf.position = Vector3Add(c.centre, Vector3Scale(dir, R - depth + 0.5f * a));   // its inner face 'depth' under the surface
            cut.sf.rounding = 0.1f;
            return SplitAndHinge(c, g.piece, cut, Vector3Add(c.centre, Vector3Scale(dir, R - depth)), dir, ctx, &grains, true);
        }

        // Split a piece: cut in two by a plane through it, at a random point of it, facing a random way (a box bigger
        // than the sphere, its face on the plane), the two halves hinged along the edge of the cut, closed. Big parts
        // (a wing, a tail) in one rewrite, where "add a cut" makes small caps. With transport on, the hinge goes where
        // opening it helps transport most.
        bool SplitPiece(Carving& c, const RuleContext& ctx) {
            if ((int)c.pieces.size() >= ctx.p.maxPieces) return false;
            int pi = std::uniform_int_distribution<int>(0, (int)c.pieces.size() - 1)(ctx.rng);
            std::vector<int> bodyOf;
            Design d = c.Build(&bodyOf);
            BodyField piece = BodyField::Body(d, bodyOf[pi]);
            const float R = c.radius;
            Vector3 point = c.centre;
            bool inside = false;
            for (int k = 0; k < 200 && !inside; k++) {
                point = Vector3Add(c.centre, Vector3Scale(RandomDirection(ctx.rng), R * 0.9f * cbrtf(Uniform(ctx.rng, 0.0f, 1.0f))));
                inside = piece.Distance(point) < -0.03f * R;
            }
            if (!inside) return false;
            Vector3 n = RandomDirection(ctx.rng);
            Cutter cut;
            const float big = 2.6f * R;
            cut.sf.size = { big, big, big };
            cut.sf.rotation = QuaternionFromVector3ToVector3({ 0, 0, 1 }, n);
            cut.sf.position = Vector3Add(point, Vector3Scale(n, 0.5f * big));
            cut.sf.rounding = 0.0f;
            const std::vector<Loss::Grain>& grains = ctx.loss.Grains();
            bool fresh = !grains.empty();
            for (const Loss::Grain& g : grains) fresh &= g.piece < (int)c.pieces.size();
            return SplitAndHinge(c, pi, cut, point, n, ctx, fresh ? &grains : nullptr, false);
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
            return true;
        }
    }

    const std::vector<Rule>& Rules() {
        static const std::vector<Rule> rules = {
            { "add a cut",       1.0f, AddCut,       false },
            { "remove a cut",    0.5f, RemoveCut,    true },
            { "add a hollow",    0.5f, AddHollow,    false },
            { "remove a hollow", 0.25f, RemoveHollow, true },
            { "aim a cut",       1.0f, AimCut,       false },
            { "split a piece",   0.5f, SplitPiece,   false },
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
