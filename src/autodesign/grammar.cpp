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
            if (c.coreLocked && c.InCore(c.pieces[pi])) return false;
            std::vector<int> bodyOf;
            Design d = c.Build(&bodyOf);
            BodyField piece = BodyField::Body(d, bodyOf[pi]);

            // where: a point of the sphere's surface that's this piece's, and the cap under it is the piece's too
            float a = R * Uniform(ctx.rng, 0.35f, 0.7f), depth = a * Uniform(ctx.rng, 0.15f, 0.3f);
            // a new cap starts thick enough to keep (thinWeight), as far as the cap's size allows
            if (ctx.p.thinWeight > 0.0f) depth = fminf(fmaxf(depth, 1.2f * ctx.p.minThickness), 0.6f * a);
            // ... and big enough to carry its hinge (Loss::Solid turns a smaller one down)
            // (what is left of it 0.4 of the least thickness in from its faces has to be a quarter of PieceVolume)
            {
                const float in = 0.8f * ctx.p.minThickness;
                depth = fmaxf(depth, in + 0.4f * ctx.p.minThickness);
                a = fminf(fmaxf(a, in + sqrtf(0.5f * ctx.loss.PieceVolume() / (depth - in))), R);
            }
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
            const Piece cap = c.pieces[c.Split(pi, cut.id)];
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

        // ---- reaching for the mesh ------------------------------------------------------------------------------------

        Vector3 FieldGradient(const BodyField& f, Vector3 x, float h) {
            auto at = [&](Vector3 d) { return f.Distance(Vector3Add(x, d)); };
            return { (at({ h, 0, 0 }) - at({ -h, 0, 0 })) / (2.0f * h), (at({ 0, h, 0 }) - at({ 0, -h, 0 })) / (2.0f * h),
                     (at({ 0, 0, h }) - at({ 0, 0, -h })) / (2.0f * h) };
        }

        // The rotation whose local x, y, z are these three (unit, right handed).
        Quaternion FromBasis(Vector3 x, Vector3 y, Vector3 z) {
            Quaternion q1 = QuaternionFromVector3ToVector3({ 0, 0, 1 }, z);
            Vector3 x1 = Vector3RotateByQuaternion({ 1, 0, 0 }, q1);
            float angle = atan2f(Vector3DotProduct(Vector3CrossProduct(x1, x), z), Vector3DotProduct(x1, x));
            Quaternion q2 = QuaternionFromAxisAngle(z, angle);
            Quaternion a = QuaternionMultiply(q2, q1), b = QuaternionMultiply(q1, q2);
            auto fits = [&](Quaternion q) {
                return Vector3DotProduct(Vector3RotateByQuaternion({ 1, 0, 0 }, q), x) + Vector3DotProduct(Vector3RotateByQuaternion({ 0, 1, 0 }, q), y);
            };
            return fits(a) >= fits(b) ? a : b;
        }

        // Reach for the mesh: a part of the mesh the open design is far from (Loss::FarTarget) gets a limb. From the
        // piece nearest to it, at that piece's point nearest to it, a strip is cut back along the piece's surface (a
        // box from that point along a chord of the surface, everything above its floor), and hinged at that point,
        // across the strip. The hinge starts opened to the angle that swings the strip's far end nearest the missing
        // part: unlike the other rules this one changes the open pose at once, and is kept if that scores better after
        // the lookahead. Cut from the sphere it makes an arm; cut from an arm's end it makes the next segment of a
        // chain that unrolls outward.
        bool ReachChain(Carving& c, const RuleContext& ctx, int segments) {
            if ((int)c.pieces.size() + segments > ctx.p.maxPieces) return false;
            int pi = 0;
            Vector3 goal;
            float missing = 0.0f;
            if (!ctx.loss.FarTarget(c, ctx.rng, &pi, &goal, &missing)) return false;
            if (c.coreLocked && c.InCore(c.pieces[pi])) return false;
            const float R = c.radius, step = R / 16.0f;
            std::vector<int> bodyOf;
            Design d = c.Build(&bodyOf);
            BodyField piece = BodyField::Body(d, bodyOf[pi]);
            // the piece's point nearest the goal
            Vector3 p = goal;
            for (int k = 0; k < 32; k++) {
                float dist = piece.Distance(p);
                if (fabsf(dist) < 0.002f * R) break;
                Vector3 g = FieldGradient(piece, p, 0.01f * R);
                float gg = Vector3DotProduct(g, g);
                if (gg < 1e-12f) break;
                Vector3 move = Vector3Scale(g, -dist / gg);
                if (Vector3Length(move) > 0.5f * R) move = Vector3Scale(Vector3Normalize(move), 0.5f * R);
                p = Vector3Add(p, move);
            }
            if (fabsf(piece.Distance(p)) > 0.03f * R || Vector3Distance(goal, p) < 1e-3f * R) return false;
            const Vector3 n = Vector3Normalize(Vector3Subtract(goal, p));
            // the surface under a point: down along -n from above it; false when there is none within reach
            auto surface = [&](Vector3 x, Vector3* on) {
                for (float z = 0.3f * R; z > -1.2f * R; z -= step * 0.5f) {
                    Vector3 q = Vector3Add(x, Vector3Scale(n, z));
                    if (piece.Distance(q) < 0.0f) { *on = q; return true; }
                }
                return false;
            };
            const Carving before = c;
            const Placement placed = ctx.loss.Placed();
            const float asItIs = ctx.loss.Evaluate(before, placed).total;
            for (int attempt = 0; attempt < 4; attempt++) {
                c = before;
                // as much material as is missing there (somewhat less or more), as far as a strip of the piece can be:
                // its floor this deep at the hinge, deeper for a bigger one (its width follows below)
                const float wanted = missing * Uniform(ctx.rng, 0.5f, 1.2f);
                const float depth = std::clamp(sqrtf(wanted / (2.0f * R)), 1.6f * ctx.p.minThickness, 0.8f * R);
                // which way back along the surface: where the piece goes on longest
                Vector3 back = { 0, 0, 0 }, end = p;
                int longest = 0;
                for (int k = 0; k < 10; k++) {
                    Vector3 b = RandomDirection(ctx.rng);
                    b = Vector3Subtract(b, Vector3Scale(n, Vector3DotProduct(b, n)));
                    if (Vector3Length(b) < 0.2f) continue;
                    b = Vector3Normalize(b);
                    int reach = 0;
                    Vector3 last = p;
                    for (int s = 1; s <= 40; s++) {
                        Vector3 on;
                        if (!surface(Vector3Subtract(p, Vector3Scale(b, s * step)), &on)) break;
                        if (piece.Distance(Vector3Subtract(on, Vector3Scale(n, 0.5f * depth))) >= 0.0f) break;   // too thin there
                        reach = s;
                        last = on;
                    }
                    if (reach > longest) { longest = reach; back = b; end = last; }
                }
                if (longest * step < 0.5f * R) continue;
                // the strip: part of that way (the rest of the piece keeps some), along the chord to where it ends
                {
                    int take = std::max(8, (int)(longest * Uniform(ctx.rng, 0.55f, 0.9f)));
                    Vector3 on;
                    if (take < longest && surface(Vector3Subtract(p, Vector3Scale(back, take * step)), &on)) end = on;
                }
                const float length = Vector3Distance(p, end);
                if (length < 0.4f * R) continue;
                const Vector3 bx = Vector3Scale(Vector3Subtract(p, end), 1.0f / length);          // along the strip, toward the hinge
                Vector3 nz = Vector3Subtract(n, Vector3Scale(bx, Vector3DotProduct(n, bx)));
                if (Vector3Length(nz) < 0.2f) continue;
                nz = Vector3Normalize(nz);                                                         // out of the strip's floor
                const Vector3 across = Vector3CrossProduct(nz, bx);
                const float width = std::clamp(wanted / (length * depth), fmaxf(0.35f * R, 1.5f * ctx.loss.PieceVolume() / (length * depth)), 1.4f * R);
                const float high = 2.4f * R;
                Cutter cut;
                cut.id = c.nextId++;
                cut.sf.size = { length, width, high + depth };
                cut.sf.rotation = FromBasis(bx, across, nz);
                cut.sf.position = Vector3Add(p, Vector3Add(Vector3Scale(bx, -0.5f * length), Vector3Scale(nz, 0.5f * (high - depth))));
                cut.sf.rounding = 0.05f;
                c.cutters.push_back(cut);
                const int stripIndex = c.Split(pi, cut.id);
                const Piece strip = c.pieces[stripIndex];
                const int rest = c.pieces[pi].id;
                std::vector<int> bo;
                Design ds = c.Build(&bo);
                BodyField restField = BodyField::Body(ds, bo[pi]), stripField = BodyField::Body(ds, bo[stripIndex]);
                // the joints the piece had go with the part their pivot is in; the strip has to be the part that hangs
                bool hangs = true;
                for (Joint& j : c.joints) {
                    bool toStrip = stripField.Distance(j.pivot) < restField.Distance(j.pivot);
                    if (j.child == rest && toStrip) hangs = false;
                    else if (j.parent == rest && toStrip) j.parent = strip.id;
                }
                if (!hangs) continue;
                if (segments > 1) {                                 // (a joint on the strip: it would have to go to a segment)
                    bool onStrip = false;
                    for (const Joint& o : c.joints) onStrip |= o.parent == strip.id;
                    if (onStrip) continue;
                }
                Joint j;
                j.id = c.nextId++;
                j.parent = rest;
                j.child = strip.id;
                j.pivot = Vector3Subtract(p, Vector3Scale(nz, 0.03f * R));
                j.axis = across;
                j.t0 = Uniform(ctx.rng, 0.0f, 0.5f);
                j.t1 = j.t0 + 0.5f;
                for (const Joint& o : c.joints) j.order = std::max(j.order, o.order + 1);
                // A chain: the strip cut across into segments, each hinged to the one before it at the cut, on the outer
                // surface. The segments are wedges, wider outside: turned outward about their outer edges, they
                // straighten without running into each other, so the chain can reach further than the sphere is wide.
                std::vector<int> segIndex = { stripIndex };
                std::vector<Joint> chain = { j };
                std::vector<Vector3> along = { p };                 // the outer surface at each hinge, then the far end
                bool made = true;
                for (int k = 1; k < segments; k++) {
                    Vector3 on;
                    if (!surface(Vector3Subtract(p, Vector3Scale(bx, length * k / segments)), &on)) { made = false; break; }
                    Cutter split;
                    split.id = c.nextId++;
                    const float big = 2.6f * R;
                    split.sf.size = { big, big, big };
                    split.sf.rotation = QuaternionFromVector3ToVector3({ 0, 0, 1 }, Vector3Negate(bx));
                    split.sf.position = Vector3Add(on, Vector3Scale(bx, -0.5f * big));
                    c.cutters.push_back(split);
                    const int li = segIndex.back();
                    const Piece next = c.pieces[c.Split(li, split.id)];
                    Joint jk;
                    jk.id = c.nextId++;
                    jk.parent = c.pieces[li].id;
                    jk.child = next.id;
                    jk.pivot = Vector3Subtract(on, Vector3Scale(nz, 0.03f * R));
                    jk.axis = across;
                    jk.t0 = j.t0;
                    jk.t1 = j.t1;
                    jk.order = chain.back().order + 1;
                    segIndex.push_back((int)c.pieces.size() - 1);
                    chain.push_back(jk);
                    along.push_back(on);
                }
                if (!made) continue;
                along.push_back(end);
                const int lastIndex = segIndex.back();
                const Vector3 tip = Vector3Subtract(end, Vector3Scale(nz, 0.5f * depth));     // in the last segment
                // how much each hinge after the first turns to straighten the chain: the angle between the chords of
                // the two segments it joins
                std::vector<float> bends(chain.size(), 0.0f);
                for (size_t k = 1; k < chain.size(); k++) {
                    Vector3 a0 = Vector3Subtract(along[k], along[k - 1]), a1 = Vector3Subtract(along[k + 1], along[k]);
                    bends[k] = atan2f(Vector3DotProduct(Vector3CrossProduct(a1, a0), across), Vector3DotProduct(a1, a0));
                }
                auto withOpen = [&](float first, float bend) {
                    Carving t = c;
                    for (size_t k = 0; k < chain.size(); k++) {
                        Joint jj = chain[k];
                        jj.open = k == 0 ? first : bend * bends[k];
                        t.joints.push_back(jj);
                    }
                    return t;
                };
                auto tipOf = [&](const Carving& t) {                // where the chain's tip is, opened, in the piece's frame
                    std::vector<int> bo;
                    Design dt = t.Build(&bo);
                    std::vector<Rigid> xf = dt.Transforms(0.0f, true);
                    return xf[dt.GroupIndex(bo[pi])].ApplyInverse(xf[dt.GroupIndex(bo[lastIndex])].Apply(tip));
                };
                // which way straightens it (the sign of the hinges' angles): the way the tip ends up farther away
                float sign = 1.0f;
                if (chain.size() > 1 && Vector3Distance(tipOf(withOpen(0.0f, -1.0f)), p) > Vector3Distance(tipOf(withOpen(0.0f, 1.0f)), p)) sign = -1.0f;
                // the first hinge opened to an angle that swings the tip near the goal (the chain straightened all or
                // half the way): of the four that bring it nearest, the one the design scores best with (its fit, and
                // what it runs into on the way), if that is better than the design as it is
                struct Try {
                    float gap, first, bend;
                };
                std::vector<Try> tries;
                for (float bend : chain.size() > 1 ? std::vector<float>{ sign, 0.5f * sign } : std::vector<float>{ 0.0f })
                    for (int k = 2; k <= 17; k++)
                        for (float turn : { 1.0f, -1.0f }) {
                            float first = turn * k * 10.0f * DEG2RAD;
                            float gap = Vector3Distance(tipOf(withOpen(first, bend)), goal);
                            if (gap < Vector3Distance(tip, goal) - 0.1f * R) tries.push_back({ gap, first, bend });
                        }
                std::sort(tries.begin(), tries.end(), [](const Try& x, const Try& y) { return x.gap < y.gap; });
                Carving best;
                float bestTotal = asItIs;
                for (size_t k = 0; k < tries.size() && k < 4; k++) {
                    Carving t = withOpen(tries[k].first, tries[k].bend);
                    Tidy(t);
                    if (ctx.loss.SnapJoints(t) != 0 || !ctx.loss.Solid(t)) continue;
                    float total = ctx.loss.Evaluate(t, placed).total;
                    if (total < bestTotal) { bestTotal = total; best = t; }
                }
                if (bestTotal < asItIs) { c = best; return true; }
            }
            c = before;
            return false;
        }

        bool Reach(Carving& c, const RuleContext& ctx) { return ReachChain(c, ctx, 1); }

        // Peel a limb: as "reach for the mesh", but the strip is a chain of two to four segments hinged end to end,
        // straightened as it opens: a limb (a wing, a tail, a leg) longer than the sphere is wide.
        bool Peel(Carving& c, const RuleContext& ctx) { return ReachChain(c, ctx, std::uniform_int_distribution<int>(2, 4)(ctx.rng)); }

        // Thicken a panel: part of the core joined to a piece of the shell, so that the panel, opened, carries more
        // material where the shape needs depth (a head, a body seen from the side). The second stage of a search from
        // a shell and a core: the shell's pieces make the outline, then the core's material is handed to them. A box
        // from inside the core out through the panel (toward the middle of the panel) cuts that part off the core
        // piece behind the panel, and it becomes a region of the panel's piece: they share the face where the core
        // meets the shell, so they're one solid (Loss::Solid turns down any that isn't).
        bool Thicken(Carving& c, const RuleContext& ctx) {
            if (c.coreLocked) return false;
            int coreId = -1;
            for (const Cutter& k : c.cutters)
                if (k.core) coreId = k.id;
            if (coreId < 0) return false;
            std::vector<int> shells;
            for (size_t k = 0; k < c.pieces.size(); k++)
                if (!Has(c.pieces[k].inside, coreId)) shells.push_back((int)k);
            if (shells.empty()) return false;
            const int si = shells[std::uniform_int_distribution<size_t>(0, shells.size() - 1)(ctx.rng)];
            const float R = c.radius;
            std::vector<int> bodyOf;
            Design d = c.Build(&bodyOf);
            std::vector<BodyField> fields;
            for (size_t k = 0; k < c.pieces.size(); k++) fields.push_back(BodyField::Body(d, bodyOf[k]));
            // the middle of the panel, seen from the centre
            Vector3 sum = { 0, 0, 0 };
            int in = 0;
            for (int k = 0; k < 3000; k++) {
                Vector3 q = Vector3Add(c.centre, Vector3Scale(RandomDirection(ctx.rng), R * cbrtf(Uniform(ctx.rng, 0.0f, 1.0f))));
                if (fields[si].Distance(q) < 0.0f) { sum = Vector3Add(sum, Vector3Subtract(q, c.centre)); in++; }
            }
            if (in < 10 || Vector3Length(sum) < 1e-6f) return false;
            Vector3 dir = Vector3Normalize(sum);
            dir = Vector3Normalize(Vector3Add(dir, Vector3Scale(RandomDirection(ctx.rng), 0.25f)));   // (somewhat either side of it)
            // the core piece behind it along that way, and how far out the core goes there
            int ki = -1;
            float coreEnd = 0.0f;
            for (float t = 0.0f; t < R; t += 0.01f * R) {
                Vector3 q = Vector3Add(c.centre, Vector3Scale(dir, t));
                int who = -1;
                for (size_t k = 0; k < c.pieces.size(); k++)
                    if (Has(c.pieces[k].inside, coreId) && c.pieces[k].extra.empty() && fields[k].Distance(q) < 0.0f) who = (int)k;
                if (who >= 0) { ki = who; coreEnd = t; }
                else if (ki >= 0) break;
            }
            if (ki < 0 || coreEnd < 0.1f * R) return false;
            const Carving before = c;
            Cutter cut;
            cut.id = c.nextId++;
            const float big = 2.6f * R, from = coreEnd * Uniform(ctx.rng, 0.0f, 0.7f), width = coreEnd * Uniform(ctx.rng, 0.6f, 1.6f);
            cut.sf.size = { width, width, big };
            cut.sf.rotation = QuaternionFromVector3ToVector3({ 0, 0, 1 }, dir);
            cut.sf.position = Vector3Add(c.centre, Vector3Scale(dir, from + 0.5f * big));
            cut.sf.rounding = 0.3f;
            c.cutters.push_back(cut);
            const int part = c.Split(ki, cut.id);
            // both what goes and what stays of the core piece have material
            d = c.Build(&bodyOf);
            BodyField partField = BodyField::Body(d, bodyOf[part]), restField = BodyField::Body(d, bodyOf[ki]);
            int inPart = 0, inRest = 0;
            for (int k = 0; k < 1500; k++) {
                Vector3 q = Vector3Add(c.centre, Vector3Scale(RandomDirection(ctx.rng), R * cbrtf(Uniform(ctx.rng, 0.0f, 1.0f))));
                inPart += partField.Distance(q) < 0.0f;
                inRest += restField.Distance(q) < 0.0f;
            }
            if (inPart < 6 || inRest < 6) { c = before; return false; }
            // the part becomes a region of the panel's piece
            const Piece moved = c.pieces[part];
            c.pieces[si].extra.push_back({ moved.inside, moved.outside });
            for (const Region& r : moved.extra) c.pieces[si].extra.push_back(r);
            c.pieces.erase(c.pieces.begin() + part);
            Tidy(c);
            return true;
        }

        // Split a piece: cut in two by a plane through it, at a random point of it, facing a random way (a box bigger
        // than the sphere, its face on the plane), the two halves hinged along the edge of the cut face, closed (open
        // angle 0: nothing looks or scores different until Adam opens it). Big parts (a wing, a tail) in one rewrite,
        // where "add a cut" makes caps; under Loss::Solid, which turns small pieces down, most pieces come from here.
        bool SplitPiece(Carving& c, const RuleContext& ctx) {
            if ((int)c.pieces.size() >= ctx.p.maxPieces) return false;
            const int pi = std::uniform_int_distribution<int>(0, (int)c.pieces.size() - 1)(ctx.rng);
            if (c.coreLocked && c.InCore(c.pieces[pi])) return false;
            std::vector<int> bodyOf;
            Design d = c.Build(&bodyOf);
            BodyField piece = BodyField::Body(d, bodyOf[pi]);
            const float R = c.radius;
            Vector3 face = c.centre;
            bool inside = false;
            for (int k = 0; k < 200 && !inside; k++) {
                face = Vector3Add(c.centre, Vector3Scale(RandomDirection(ctx.rng), R * 0.9f * cbrtf(Uniform(ctx.rng, 0.0f, 1.0f))));
                inside = piece.Distance(face) < -0.03f * R;
            }
            if (!inside) return false;
            const Vector3 out = RandomDirection(ctx.rng);      // the face's normal, into the new piece
            Cutter cut;
            const float big = 2.6f * R;
            cut.id = c.nextId++;
            cut.sf.size = { big, big, big };
            cut.sf.rotation = QuaternionFromVector3ToVector3({ 0, 0, 1 }, out);
            cut.sf.position = Vector3Add(face, Vector3Scale(out, 0.5f * big));
            cut.sf.rounding = 0.0f;
            c.cutters.push_back(cut);
            const int partIndex = c.Split(pi, cut.id);
            const Piece part = c.pieces[partIndex];
            const int rest = c.pieces[pi].id;
            d = c.Build(&bodyOf);
            BodyField restField = BodyField::Body(d, bodyOf[pi]), partField = BodyField::Body(d, bodyOf[partIndex]);

            // the joints the piece had go with the part their pivot is in (or nearer)
            int anchor = rest;
            for (Joint& j : c.joints) {
                bool toPart = partField.Distance(j.pivot) < restField.Distance(j.pivot);
                if (j.child == rest && toPart) { j.child = part.id; anchor = part.id; }
                else if (j.parent == rest && toPart) j.parent = part.id;
            }
            // the hinge's line: along the cut face, at its edge (the farthest the face still has both parts against it)
            Vector3 side = Vector3Normalize(Vector3CrossProduct(out, fabsf(out.y) < 0.9f ? Vector3{ 0, 1, 0 } : Vector3{ 1, 0, 0 }));
            side = Vector3RotateByQuaternion(side, QuaternionFromAxisAngle(out, Uniform(ctx.rng, 0.0f, 2.0f * PI)));
            Vector3 pivot = face;
            for (int k = 1; k <= 64; k++) {
                Vector3 q = Vector3Add(face, Vector3Scale(side, k * 2.0f * R / 64.0f));
                Vector3 above = Vector3Add(q, Vector3Scale(out, 0.01f * R)), below = Vector3Subtract(q, Vector3Scale(out, 0.01f * R));
                if (partField.Distance(above) >= 0.0f || restField.Distance(below) >= 0.0f) break;
                pivot = q;
            }
            Joint j;
            j.id = c.nextId++;
            j.parent = anchor;
            j.child = anchor == rest ? part.id : rest;
            j.pivot = pivot;
            j.axis = Vector3Normalize(Vector3CrossProduct(out, side));
            j.open = 0.0f;
            j.t0 = Uniform(ctx.rng, 0.0f, 0.5f);
            j.t1 = j.t0 + 0.5f;
            for (const Joint& o : c.joints) j.order = std::max(j.order, o.order + 1);   // it opens last, until the order is looked at again
            // the axis turned so that opening it (a positive angle) lifts the new part off the rest, like a lid: a
            // point of the part just over the face, with the joint opened a little, should move away from the face
            {
                Carving t = c;
                t.joints.push_back(j);
                t.joints.back().open = 0.02f;
                std::vector<int> bo;
                Design dt = t.Build(&bo);
                std::vector<Rigid> xf = dt.Transforms(0.0f, false);
                const Vector3 probe = Vector3Add(face, Vector3Scale(out, 0.05f * R));
                Vector3 w = xf[dt.GroupIndex(bo[partIndex])].Apply(probe);
                if (Vector3DotProduct(Vector3Subtract(xf[dt.GroupIndex(bo[pi])].ApplyInverse(w), probe), out) < 0.0f) j.axis = Vector3Negate(j.axis);
            }
            c.joints.push_back(j);
            Tidy(c);
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
                if (!A.extra.empty() || !B.extra.empty()) continue;      // (regions joined to them: not one cutter's split)
                for (const Cutter& k : c.cutters) {
                    if (k.core) continue;
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
            { "reach for the mesh", 1.0f, Reach,     false },
            { "peel a limb",     1.0f, Peel,         false },
            { "add a cut",       0.5f, AddCut,       false },
            { "remove a cut",    0.5f, RemoveCut,    true },
            { "split a piece",   0.5f, SplitPiece,   false },
            { "thicken a panel", 1.0f, Thicken,      false },
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
