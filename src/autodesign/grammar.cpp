#include "grammar.hpp"
#include "edits.hpp"
#include <algorithm>
#include <cfloat>

namespace Bakugan::Auto {
    namespace {
        // What a rule needs before it can be drawn.
        enum Need : unsigned {
            None = 0,
            Cuts = 1 << 0,           // cuts are allowed
            Features = 1 << 1,       // blended features are allowed
            Hinges = 1 << 2,         // the design has a hinge
            Excess = 1 << 3,         // the score found material off the mesh
            Clashes = 1 << 4,        // the score found bodies overlapping (and collisions count)
            Timing = 1 << 5,         // hinge timing is searched
        };

        // Keys naming what a rewrite owns or uses: a shape, group or hinge by id, or the hinge tree's shape (one
        // change to it per round).
        long long Key(char kind, int id) { return ((long long)kind << 32) | (unsigned)id; }
        const long long TreeKey = ((long long)'T' << 32);
    }

    // The grammar. Weights are how often each rule is drawn among those that can apply: the paper draws them about
    // evenly; additions are a little more likely here, since they carry most early progress. Needs: what must hold
    // for it to be drawn. Patient: its lookahead is longer (see RuleInfo).
    struct Grammar::RuleInfo {
        Rule rule;
        const char* name;
        float weight;
        unsigned needs;
        bool patient;                // a long lookahead: the change itself is (nearly) neutral and only pays once the
                                     // shapes it made have moved apart or settled, which takes more than a step or two
        bool (Grammar::*draw)(const DrawContext&, Rewrite&);
        bool (Grammar::*apply)(Design&, const Rewrite&, Focus&) const;
    };

    const Grammar::RuleInfo Grammar::Rules[] = {
        // columns: rule, name, weight, needs, patient, draw, apply
        // shapes and material
        { Rule::AddShape,        "add a shape",             2.0f, None,            false, &Grammar::DrawAddShape,       &Grammar::ApplyAddShape },
        { Rule::AddCut,          "add a cut",               1.0f, Cuts,            false, &Grammar::DrawAddCut,         &Grammar::ApplyAddCut },
        { Rule::RemoveShape,     "remove a shape",          1.5f, None,            false, &Grammar::DrawPickShape,      &Grammar::ApplyRemoveShape },
        { Rule::ToggleOp,        "flip add / cut",          0.6f, Cuts,            false, &Grammar::DrawPickShape,      &Grammar::ApplyToggleOp },
        { Rule::MoveShape,       "move a shape to a body",  0.8f, None,            false, &Grammar::DrawMoveShape,      &Grammar::ApplyMoveShape },
        { Rule::SplitShape,      "split a shape",           0.6f, None,            true,  &Grammar::DrawSplitShape,     &Grammar::ApplySplitShape },
        { Rule::MergeShapes,     "merge two shapes",        0.6f, None,            true,  &Grammar::DrawMergeShapes,    &Grammar::ApplyMergeShapes },
        // bodies
        { Rule::AddBody,         "add a hinged body",       1.0f, None,            false, &Grammar::DrawAddBody,        &Grammar::ApplyAddBody },
        { Rule::SplitBody,       "split a body",            1.0f, None,            false, &Grammar::DrawSplitBody,      &Grammar::ApplySplitBody },
        { Rule::RemoveBody,      "remove a body",           0.5f, None,            false, &Grammar::DrawRemoveBody,     &Grammar::ApplyRemoveBody },
        { Rule::MergeBody,       "merge a body",            0.8f, Hinges,          false, &Grammar::DrawMergeBody,      &Grammar::ApplyMergeBody },
        // blended features
        { Rule::AddFeature,      "add a feature",           0.5f, Features,        false, &Grammar::DrawAddFeature,     &Grammar::ApplyAddFeature },
        { Rule::DissolveFeature, "dissolve a feature",      0.4f, None,            false, &Grammar::DrawPickFeature,    &Grammar::ApplyDissolveFeature },
        { Rule::ToggleFeature,   "flip a feature",          0.3f, Cuts | Features, false, &Grammar::DrawPickFeature,    &Grammar::ApplyToggleFeature },
        // hinges
        { Rule::AddHinge,        "add a hinge",             1.0f, None,            false, &Grammar::DrawAddHinge,       &Grammar::ApplyAddHinge },
        { Rule::RemoveHinge,     "remove a hinge",          0.3f, Hinges,          false, &Grammar::DrawRemoveHinge,    &Grammar::ApplyRemoveHinge },
        { Rule::Rehang,          "rehang a hinge",          0.6f, Hinges,          false, &Grammar::DrawRehang,         &Grammar::ApplyRehang },
        { Rule::TurnHinge,       "turn a hinge",            1.2f, Hinges,          false, &Grammar::DrawTurnHinge,      &Grammar::ApplyTurnHinge },
        // repairs, where the score found errors
        { Rule::RemoveStray,     "remove stray material",   1.0f, Excess,          false, &Grammar::DrawRemoveStray,    &Grammar::ApplyRemoveStray },
        { Rule::CarveClash,      "carve an overlap",        1.2f, Cuts | Clashes,  false, &Grammar::DrawCarveClash,     &Grammar::ApplyAddCut },
        // hinge timing
        { Rule::Retime,          "retime a hinge",          0.8f, Hinges | Timing, false, &Grammar::DrawRetime,         &Grammar::ApplyTiming },
        { Rule::OwnOpening,      "own opening timing",      0.3f, Hinges | Timing, false, &Grammar::DrawOwnOpening,     &Grammar::ApplyTiming },
    };
    static_assert(sizeof(Grammar::Rules) / sizeof(Grammar::Rules[0]) == (size_t)Rule::Count, "one row per rule, in Rule order");

    bool Conflict(const Rewrite& a, const Rewrite& b) {
        auto hits = [](const std::vector<long long>& x, const std::vector<long long>& y) {
            for (long long k : x)
                if (std::find(y.begin(), y.end(), k) != y.end()) return true;
            return false;
        };
        return hits(a.owns, b.owns) || hits(a.owns, b.uses) || hits(b.owns, a.uses);
    }

    DrawContext::DrawContext(const Design& design, const Diagnostics& dg, const AutoScore& s) : d(design), diag(dg), score(s) {
        bodies = d.Bodies();
        for (const Group& g : d.groups)
            if (g.parent >= 0) features.push_back(g.id);
    }

    // ---- drawing and applying ---------------------------------------------------------------------------------------

    bool Grammar::Allowed(const RuleInfo& info, const DrawContext& c) const {
        unsigned n = info.needs;
        if ((n & Cuts) && !m_Space.cuts) return false;
        if ((n & Features) && !m_Space.features) return false;
        if ((n & Hinges) && c.d.hinges.empty()) return false;
        if ((n & Excess) && c.diag.excess.empty()) return false;
        if ((n & Clashes) && (c.diag.clashes.empty() || m_P.collisionWeight <= 0.0f)) return false;
        if ((n & Timing) && !m_Space.timing) return false;
        return true;
    }

    bool Grammar::Draw(const DrawContext& c, Rewrite* out) {
        float weight[(int)Rule::Count], sum = 0.0f;
        for (int i = 0; i < (int)Rule::Count; i++) weight[i] = Allowed(Rules[i], c) ? Rules[i].weight : 0.0f;
        for (float w : weight) sum += w;
        float pick = Uniform(m_Rng) * sum;
        int drawn = 0;
        while (drawn < (int)Rule::Count - 1 && pick >= weight[drawn]) pick -= weight[drawn++];
        const RuleInfo& rule = Rules[drawn];
        Rewrite r;
        r.rule = rule.rule;
        if (!(this->*rule.draw)(c, r)) return false;
        *out = std::move(r);
        return true;
    }

    bool Grammar::Apply(Design& d, const Rewrite& r, Focus* f) const {
        if (!(this->*Rules[(int)r.rule].apply)(d, r, *f)) return false;
        if (r.hasTiming) {
            int h = d.HingeIndex(f->hinge >= 0 ? f->hinge : r.hinge);
            if (h >= 0) {
                Hinge& hg = d.hinges[h];
                hg.t0 = r.t0;
                hg.t1 = r.t1;
                hg.ownOpening = r.ownOpening;
                hg.o0 = r.o0;
                hg.o1 = r.o1;
                ClampTiming(hg);
                if (f->hinge < 0) f->hinge = hg.id;
            }
        }
        return true;
    }

    void Grammar::Remember(Rewrite& r, const Design& d, const Focus& f) const {
        int s = d.ShapeIndex(f.shape);
        if (s >= 0) {
            r.sf = d.shapes[s].sf;
            r.hasShape = true;
        }
        int s2 = d.ShapeIndex(f.shape2);
        if (s2 >= 0) r.sf2 = d.shapes[s2].sf;
        int h = d.HingeIndex(f.hinge);
        if (h >= 0) {
            const Hinge& hg = d.hinges[h];
            r.pivot = hg.pivot;
            r.axis = hg.axis;
            r.angle = hg.end - hg.start;
            r.hasHinge = true;
            r.hasTiming = true;
            r.t0 = hg.t0;
            r.t1 = hg.t1;
            r.ownOpening = hg.ownOpening;
            r.o0 = hg.o0;
            r.o1 = hg.o1;
        }
    }

    const char* Grammar::Name(Rule r) { return r < Rule::Count ? Rules[(int)r].name : "?"; }
    bool Grammar::Patient(Rule r) { return r < Rule::Count && Rules[(int)r].patient; }

    // ---- shared checks ------------------------------------------------------------------------------------------------

    bool Grammar::IsBody(const Design& d, int id) const {
        int g = d.GroupIndex(id);
        return g >= 0 && d.groups[g].parent < 0;
    }

    bool Grammar::LastMaterial(const Design& d, int s) const { return d.AddsMaterial(s) && MaterialShapes(d, d.ShapeBody(s)) <= 1; }

    void Grammar::Tuned(Design& d, const Rewrite& r, int s, Focus& f) const {
        if (r.hasShape) d.shapes[s].sf = r.sf;
        f.shape = d.shapes[s].id;
    }

    // ---- shapes and material ------------------------------------------------------------------------------------------

    // A new shape where material is missing (fitted to the gap), sometimes into one of the body's blended features.
    bool Grammar::DrawAddShape(const DrawContext& c, Rewrite& r) {
        int g;
        if (!FitRegion(c, false, &r.sf, &g)) return false;
        int body = c.d.groups[g].id;
        r.group = body;
        std::vector<int> feats;
        for (int id : c.features) {
            const Group& gr = c.d.groups[c.d.GroupIndex(id)];
            if (gr.op == Op::Union && c.d.BodyOf(id) == body) feats.push_back(id);
        }
        if (!feats.empty() && Chance(m_Rng, 0.3f)) r.group = PickFrom(m_Rng, feats);
        r.hasShape = true;
        r.uses = { Key('G', body), Key('G', r.group) };
        return true;
    }

    bool Grammar::ApplyAddShape(Design& d, const Rewrite& r, Focus& f) const {
        if (d.GroupIndex(r.group) < 0 || !m_Space.RoomForShape(d)) return false;
        f.shape = d.shapes[d.AddShape(r.sf, r.group, "Shape")].id;
        return true;
    }

    // A new cut where the design sticks out of the mesh or the sphere.
    bool Grammar::DrawAddCut(const DrawContext& c, Rewrite& r) {
        int g;
        if (!FitRegion(c, true, &r.sf, &g)) return false;
        r.group = c.d.groups[g].id;
        r.hasShape = true;
        r.uses = { Key('G', r.group) };
        return true;
    }

    bool Grammar::ApplyAddCut(Design& d, const Rewrite& r, Focus& f) const {
        if (!IsBody(d, r.group) || !m_Space.RoomForShape(d)) return false;
        f.shape = d.shapes[d.AddShape(r.sf, r.group, "Cut", Op::Cut)].id;
        return true;
    }

    bool Grammar::DrawPickShape(const DrawContext& c, Rewrite& r) {
        if (c.d.shapes.empty()) return false;
        int s = Pick(m_Rng, (int)c.d.shapes.size());
        r.shape = c.d.shapes[s].id;
        r.owns = { Key('S', r.shape) };
        r.uses = { Key('G', c.d.ShapeBody(s)) };
        return true;
    }

    bool Grammar::ApplyRemoveShape(Design& d, const Rewrite& r, Focus&) const {
        int s = d.ShapeIndex(r.shape);
        if (s < 0 || d.shapes.size() <= 1 || LastMaterial(d, s)) return false;
        d.RemoveShape(s);
        return true;
    }

    bool Grammar::ApplyToggleOp(Design& d, const Rewrite& r, Focus& f) const {
        int s = d.ShapeIndex(r.shape);
        if (s < 0 || LastMaterial(d, s)) return false;
        Shape& sh = d.shapes[s];
        sh.op = sh.op == Op::Union ? Op::Cut : Op::Union;
        sh.color = sh.op == Op::Cut ? Color{ 225, 90, 90, 255 } : d.groups[d.GroupIndex(sh.group)].color;
        Tuned(d, r, s, f);
        return true;
    }

    bool Grammar::DrawMoveShape(const DrawContext& c, Rewrite& r) {
        if (c.bodies.size() < 2 || c.d.shapes.empty()) return false;
        int s = Pick(m_Rng, (int)c.d.shapes.size());
        int from = c.d.ShapeBody(s);
        r.shape = c.d.shapes[s].id;
        r.other = RandomBody(c);
        if (r.other == from) return false;
        r.owns = { Key('S', r.shape) };
        r.uses = { Key('G', from), Key('G', r.other) };
        return true;
    }

    bool Grammar::ApplyMoveShape(Design& d, const Rewrite& r, Focus& f) const {
        int s = d.ShapeIndex(r.shape);
        if (s < 0 || !IsBody(d, r.other) || d.ShapeBody(s) == r.other || LastMaterial(d, s)) return false;
        d.MoveShape(s, r.other);
        if (d.shapes[s].op == Op::Union) d.shapes[s].color = d.groups[d.GroupIndex(r.other)].color;
        Tuned(d, r, s, f);
        return true;
    }

    // One shape cut in two across one of its axes, 30..70% along it: together the halves fill what the shape did, and
    // the lookahead then tunes both. Usually it is aimed: the shape nearest a spot where the score found material
    // missing or extra (open pose) is cut across the axis pointing most towards the spot, and the half on the spot's
    // side reaches out towards a missing spot or pulls back from an extra one (its cut face stays where it is, so the
    // halves stay joined): a limb that grows or shortens from a joint. Otherwise any shape, across an axis drawn by its
    // length, the halves left as they are. The reverse of MergeShapes.
    bool Grammar::DrawSplitShape(const DrawContext& c, Rewrite& r) {
        const Design& d = c.d;
        if (d.shapes.empty()) return false;
        int s = -1, axis = 0;
        bool towardsMinus = false;                       // the new half is the one towards -axis
        float stretch = 1.0f;                            // what the new half's length is multiplied by
        bool missing = Chance(m_Rng, 0.5f);
        const std::vector<Spot>& spots = missing ? c.diag.missing : c.diag.excess;
        if (!spots.empty() && Chance(m_Rng, 0.7f)) {
            const Spot& sp = PickFrom(m_Rng, spots);
            if (sp.body < 0 || sp.body >= (int)d.groups.size()) return false;
            Vector3 q = d.Transforms(0.0f, false)[sp.body].ApplyInverse(sp.p);     // in the design pose
            float best = FLT_MAX;
            for (int k : d.GroupShapes(d.groups[sp.body].id)) {
                float dist = fabsf(d.shapes[k].sf.Distance(q));
                if (dist < best) { best = dist; s = k; }
            }
            if (s < 0) return false;
            const SuperFrustum& f = d.shapes[s].sf;
            Vector3 local = Vector3RotateByQuaternion(Vector3Subtract(q, f.position), QuaternionInvert(f.rotation));
            float most = -1.0f;
            for (int a = 0; a < 3; a++) {
                float w = fabsf(Coord(local, a)) / fmaxf(Coord(f.size, a), 1e-6f);   // how far out along it, by its length
                if (w > most) { most = w; axis = a; }
            }
            towardsMinus = Coord(local, axis) < 0.0f;
            stretch = missing ? Uniform(m_Rng, 1.05f, 1.35f) : Uniform(m_Rng, 0.65f, 0.95f);
        } else {
            s = Pick(m_Rng, (int)d.shapes.size());
            const SuperFrustum& f = d.shapes[s].sf;
            float pick = Uniform(m_Rng) * (f.size.x + f.size.y + f.size.z);
            axis = pick < f.size.x ? 0 : (pick < f.size.x + f.size.y ? 1 : 2);
            towardsMinus = Chance(m_Rng, 0.5f);
        }
        const SuperFrustum& f = d.shapes[s].sf;
        if (f.onion > 1e-3f) return false;               // a shell split in two is no longer a shell
        SuperFrustum minus, plus;
        if (!SplitShape(f, axis, Uniform(m_Rng, 0.3f, 0.7f), &minus, &plus)) return false;
        r.sf2 = towardsMinus ? plus : minus;             // the shape itself becomes this half ...
        r.sf = towardsMinus ? minus : plus;              // ... and this one is new
        if (stretch != 1.0f) {
            float len = Coord(r.sf.size, axis), grow = len * (stretch - 1.0f);
            Vector3 dir = { 0, 0, 0 };
            Coord(dir, axis) = towardsMinus ? -1.0f : 1.0f;
            Coord(r.sf.size, axis) = len + grow;
            r.sf.position = Vector3Add(r.sf.position, Vector3Scale(Vector3RotateByQuaternion(dir, f.rotation), 0.5f * grow));
        }
        r.shape = d.shapes[s].id;
        r.hasShape = true;
        r.owns = { Key('S', r.shape) };
        r.uses = { Key('G', d.shapes[s].group) };
        return true;
    }

    bool Grammar::ApplySplitShape(Design& d, const Rewrite& r, Focus& f) const {
        int s = d.ShapeIndex(r.shape);
        if (s < 0 || !m_Space.RoomForShape(d)) return false;
        Shape& sh = d.shapes[s];
        sh.sf = r.sf2;
        int half = d.AddShape(r.sf, sh.group, sh.name, sh.op);
        f.shape = d.shapes[half].id;                     // the lookahead tunes both halves
        f.shape2 = r.shape;
        return true;
    }

    // Two shapes of the same group doing the same thing (both adding or both cutting), touching or nearly so, become
    // one oriented box fitted to points inside either: a simpler design where one shape does the work of two. The
    // reverse of SplitShape.
    bool Grammar::DrawMergeShapes(const DrawContext& c, Rewrite& r) {
        const Design& d = c.d;
        if (d.shapes.size() < 2) return false;
        int a = Pick(m_Rng, (int)d.shapes.size());
        const Shape& sa = d.shapes[a];
        auto reach = [](const SuperFrustum& f) { return Vector3Length(f.LocalBounds().max); };
        // its partners: same group and op, solid, their bounding spheres no more than 5% of the mesh's size apart
        std::vector<std::pair<float, int>> near;
        for (int b = 0; b < (int)d.shapes.size(); b++) {
            const Shape& sb = d.shapes[b];
            if (b == a || sb.group != sa.group || sb.op != sa.op || sb.sf.onion > 1e-3f) continue;
            float gap = Vector3Distance(sa.sf.position, sb.sf.position) - reach(sa.sf) - reach(sb.sf);
            if (gap < 0.05f * m_T.L) near.push_back({ gap, b });
        }
        if (near.empty() || sa.sf.onion > 1e-3f) return false;
        std::sort(near.begin(), near.end());
        int b = Chance(m_Rng, 0.7f) ? near[0].second : PickFrom(m_Rng, near).second;
        const SuperFrustum &fa = sa.sf, &fb = d.shapes[b].sf;
        // points in a ball around both, kept where either shape is
        Vector3 mid = Vector3Lerp(fa.position, fb.position, 0.5f);
        float radius = fmaxf(Vector3Distance(mid, fa.position) + reach(fa), Vector3Distance(mid, fb.position) + reach(fb));
        std::vector<Vector3> keep;
        for (int i = 0; i < 800; i++) {
            Vector3 p = Vector3Add(mid, Vector3Scale(RandomInBall(m_Rng), radius));
            if (fa.Distance(p) < 0.0f || fb.Distance(p) < 0.0f) keep.push_back(p);
        }
        if (keep.size() < 20) return false;
        r.sf = BoxAround(keep, 0.01f, 0.015f * m_T.L);
        bool aBigger = fa.size.x * fa.size.y * fa.size.z >= fb.size.x * fb.size.y * fb.size.z;
        r.sf.rounding = aBigger ? fa.rounding : fb.rounding;    // the look of the bigger one
        r.shape = sa.id;
        r.other = d.shapes[b].id;
        r.hasShape = true;
        r.owns = { Key('S', r.shape), Key('S', r.other) };
        r.uses = { Key('G', sa.group) };
        return true;
    }

    bool Grammar::ApplyMergeShapes(Design& d, const Rewrite& r, Focus& f) const {
        int a = d.ShapeIndex(r.shape), b = d.ShapeIndex(r.other);
        if (a < 0 || b < 0 || d.shapes[a].group != d.shapes[b].group || d.shapes[a].op != d.shapes[b].op) return false;
        d.shapes[a].sf = r.sf;
        d.RemoveShape(b);
        f.shape = r.shape;
        return true;
    }

    // ---- bodies -----------------------------------------------------------------------------------------------------

    // A new piece where material is missing, hinged to the nearest body.
    bool Grammar::DrawAddBody(const DrawContext& c, Rewrite& r) {
        int g;
        if (!FitRegion(c, false, &r.sf, &g)) return false;
        r.group = c.d.groups[g].id;
        Design tmp = c.d;
        int body = tmp.groups[tmp.AddGroup("Body", -1)].id;
        tmp.AddShape(r.sf, body, "Shape");
        r.pivot = PivotBetween(tmp, r.group, body, BodyHint(tmp, r.group), r.sf.position);
        ChooseHinge(r.pivot, r.sf.position, &r.axis, &r.angle);
        r.hasShape = r.hasHinge = true;
        r.uses = { Key('G', r.group) };
        return true;
    }

    bool Grammar::ApplyAddBody(Design& d, const Rewrite& r, Focus& f) const {
        if (!IsBody(d, r.group) || !m_Space.RoomForShape(d) || !m_Space.RoomForBody(d)) return false;
        int body = d.groups[d.AddGroup("Body", -1)].id;
        f.shape = d.shapes[d.AddShape(r.sf, body, "Shape")].id;
        if (!d.CanHinge(r.group, body)) return false;
        int h = d.AddHinge(r.group, body);
        SetupHinge(d, h, r.pivot, r.axis, r.angle, true, m_T.L);
        f.hinge = d.hinges[h].id;
        return true;
    }

    // One shape of a body becomes a body of its own, hinged to what is left.
    bool Grammar::DrawSplitBody(const DrawContext& c, Rewrite& r) {
        if (c.bodies.empty()) return false;
        int b = RandomBody(c);
        std::vector<int> mat;
        for (int s : c.d.DirectShapes(b))
            if (c.d.AddsMaterial(s)) mat.push_back(s);
        if (mat.size() < 2) return false;
        int s = PickFrom(m_Rng, mat);
        r.shape = c.d.shapes[s].id;
        Design tmp = c.d;
        int body = tmp.groups[tmp.AddGroup("Body", -1)].id;
        tmp.MoveShape(s, body);
        Vector3 at = c.d.shapes[s].sf.position;
        r.pivot = PivotBetween(tmp, b, body, BodyHint(tmp, b), at);
        ChooseHinge(r.pivot, at, &r.axis, &r.angle);
        r.hasHinge = true;
        r.owns = { Key('S', r.shape) };
        r.uses = { Key('G', b) };
        return true;
    }

    bool Grammar::ApplySplitBody(Design& d, const Rewrite& r, Focus& f) const {
        int s = d.ShapeIndex(r.shape);
        if (s < 0 || !m_Space.RoomForBody(d)) return false;
        int from = d.ShapeBody(s);
        // a body's own material shape, and not its last
        if (d.shapes[s].group != from || !d.AddsMaterial(s) || MaterialShapes(d, from) <= 1) return false;
        int nb = d.AddGroup("Body", -1);
        int body = d.groups[nb].id;
        d.MoveShape(s, body);
        d.shapes[s].color = d.groups[nb].color;
        if (!d.CanHinge(from, body)) return false;
        int h = d.AddHinge(from, body);
        SetupHinge(d, h, r.pivot, r.axis, r.angle, true, m_T.L);
        f.hinge = d.hinges[h].id;
        return true;
    }

    bool Grammar::DrawRemoveBody(const DrawContext& c, Rewrite& r) {
        if (c.bodies.size() < 2) return false;
        r.group = RandomBody(c);
        r.owns = { Key('G', r.group), TreeKey };
        return true;
    }

    bool Grammar::ApplyRemoveBody(Design& d, const Rewrite& r, Focus&) const {
        if (!IsBody(d, r.group) || d.Bodies().size() <= 1) return false;
        RemoveBody(d, r.group);
        return true;
    }

    bool Grammar::DrawMergeBody(const DrawContext& c, Rewrite& r) {
        const Hinge& h = PickFrom(m_Rng, c.d.hinges);
        r.group = h.child;
        r.owns = { Key('G', h.child), TreeKey };
        r.uses = { Key('G', h.parent) };
        return true;
    }

    bool Grammar::ApplyMergeBody(Design& d, const Rewrite& r, Focus&) const { return IsBody(d, r.group) && MergeBody(d, r.group); }

    // ---- blended features ---------------------------------------------------------------------------------------------

    // One or two shapes of a group become a blended feature.
    bool Grammar::DrawAddFeature(const DrawContext& c, Rewrite& r) {
        int g = PickFrom(m_Rng, c.d.groups).id;
        std::vector<int> direct = c.d.DirectShapes(g);
        if (direct.size() < 2) return false;
        std::shuffle(direct.begin(), direct.end(), m_Rng);
        r.shapes = { c.d.shapes[direct[0]].id };
        if (Chance(m_Rng, 0.6f)) r.shapes.push_back(c.d.shapes[direct[1]].id);
        r.value = Uniform(m_Rng, 0.01f, 0.04f) * m_T.L;
        for (int id : r.shapes) r.owns.push_back(Key('S', id));
        r.uses = { Key('G', g) };
        return true;
    }

    bool Grammar::ApplyAddFeature(Design& d, const Rewrite& r, Focus&) const {
        int parent = -1;
        for (int id : r.shapes) {
            int s = d.ShapeIndex(id);
            if (s < 0 || (parent >= 0 && d.shapes[s].group != parent)) return false;
            parent = d.shapes[s].group;
        }
        if (parent < 0) return false;
        int g = d.GroupIndex(d.GroupItems(r.shapes, {}));
        if (g < 0 || d.groups[g].parent < 0) return false;
        d.groups[g].blend = r.value;
        return true;
    }

    bool Grammar::DrawPickFeature(const DrawContext& c, Rewrite& r) {
        if (c.features.empty()) return false;
        r.group = PickFrom(m_Rng, c.features);
        r.owns = { Key('G', r.group) };
        r.uses = { Key('G', c.d.BodyOf(r.group)) };
        return true;
    }

    bool Grammar::ApplyDissolveFeature(Design& d, const Rewrite& r, Focus&) const {
        int g = d.GroupIndex(r.group);
        if (g < 0 || d.groups[g].parent < 0) return false;
        d.Ungroup(g);
        return true;
    }

    bool Grammar::ApplyToggleFeature(Design& d, const Rewrite& r, Focus&) const {
        int g = d.GroupIndex(r.group);
        if (g < 0 || d.groups[g].parent < 0) return false;
        d.groups[g].op = d.groups[g].op == Op::Union ? Op::Cut : Op::Union;
        return MaterialShapes(d, d.BodyOf(r.group)) > 0;
    }

    // ---- hinges -------------------------------------------------------------------------------------------------------

    // Hangs a loose body on the nearest body it can hang from.
    bool Grammar::DrawAddHinge(const DrawContext& c, Rewrite& r) {
        std::vector<int> roots = Roots(c.d);
        if (roots.size() < 2) return false;
        r.group = PickFrom(m_Rng, roots);
        Vector3 at = BodyHint(c.d, r.group);
        int g = NearestBody(c.d, at, r.group);
        if (g < 0) return false;
        r.other = c.d.groups[g].id;
        if (!c.d.CanHinge(r.other, r.group)) {
            r.other = RandomBody(c);
            if (!c.d.CanHinge(r.other, r.group)) return false;
        }
        r.pivot = PivotBetween(c.d, r.other, r.group, BodyHint(c.d, r.other), at);
        ChooseHinge(r.pivot, at, &r.axis, &r.angle);
        r.hasHinge = true;
        r.owns = { TreeKey, Key('G', r.group) };
        r.uses = { Key('G', r.other) };
        return true;
    }

    bool Grammar::ApplyAddHinge(Design& d, const Rewrite& r, Focus& f) const {
        if (!IsBody(d, r.group) || !IsBody(d, r.other) || d.ParentHinge(r.group) >= 0 || !d.CanHinge(r.other, r.group)) return false;
        int h = d.AddHinge(r.other, r.group);
        SetupHinge(d, h, r.pivot, r.axis, r.angle, true, m_T.L);
        f.hinge = d.hinges[h].id;
        return true;
    }

    bool Grammar::DrawRemoveHinge(const DrawContext& c, Rewrite& r) {
        r.hinge = PickFrom(m_Rng, c.d.hinges).id;
        r.owns = { TreeKey, Key('H', r.hinge) };
        return true;
    }

    bool Grammar::ApplyRemoveHinge(Design& d, const Rewrite& r, Focus&) const {
        int h = d.HingeIndex(r.hinge);
        if (h < 0) return false;
        d.RemoveHinge(h);
        return true;
    }

    // Hangs a body from another one.
    bool Grammar::DrawRehang(const DrawContext& c, Rewrite& r) {
        const Hinge& h = PickFrom(m_Rng, c.d.hinges);
        r.hinge = h.id;
        r.other = RandomBody(c);
        if (r.other == h.parent || !c.d.CanHinge(r.other, h.child, c.d.HingeIndex(h.id))) return false;
        r.pivot = PivotBetween(c.d, r.other, h.child, BodyHint(c.d, r.other), BodyHint(c.d, h.child));
        r.owns = { TreeKey, Key('H', r.hinge) };
        r.uses = { Key('G', r.other) };
        return true;
    }

    bool Grammar::ApplyRehang(Design& d, const Rewrite& r, Focus& f) const {
        int h = d.HingeIndex(r.hinge);
        if (h < 0 || !IsBody(d, r.other) || !d.CanHinge(r.other, d.hinges[h].child, h)) return false;
        d.hinges[h].parent = r.other;
        d.hinges[h].pivot = r.pivot;
        if (r.hasHinge) SetupHinge(d, h, r.pivot, r.axis, r.angle, false, m_T.L);
        f.hinge = r.hinge;
        return true;
    }

    // A large turn the descent can't make in small steps: anywhere, or a quarter, half or three quarter turn on.
    bool Grammar::DrawTurnHinge(const DrawContext& c, Rewrite& r) {
        const Hinge& h = PickFrom(m_Rng, c.d.hinges);
        r.hinge = h.id;
        r.angle = Chance(m_Rng, 0.5f) ? Uniform(m_Rng, -PI, PI) : WrapAngle(h.end - h.start + (float)(1 + Pick(m_Rng, 3)) * PI * 0.5f);
        r.owns = { Key('H', r.hinge) };
        return true;
    }

    bool Grammar::ApplyTurnHinge(Design& d, const Rewrite& r, Focus& f) const {
        int h = d.HingeIndex(r.hinge);
        if (h < 0) return false;
        if (r.hasHinge) SetupHinge(d, h, r.pivot, r.axis, r.angle, false, m_T.L);
        else d.hinges[h].end = d.hinges[h].start + r.angle;
        f.hinge = r.hinge;
        return true;
    }

    // ---- repairs ----------------------------------------------------------------------------------------------------

    // The shape that put material off the mesh at a spot the score found: removed, or shrunk towards the mesh.
    bool Grammar::DrawRemoveStray(const DrawContext& c, Rewrite& r) {
        const Spot& sp = PickFrom(m_Rng, c.diag.excess);
        if (sp.body < 0 || sp.body >= (int)c.d.groups.size() || c.d.groups[sp.body].parent >= 0) return false;
        int body = c.d.groups[sp.body].id;
        Vector3 q = c.d.Transforms(0.0f, false)[sp.body].ApplyInverse(sp.p);
        int best = -1;
        float bd = FLT_MAX;
        for (int s : c.d.GroupShapes(body)) {
            if (!c.d.AddsMaterial(s)) continue;
            float dd = c.d.shapes[s].sf.Distance(q);
            if (dd < bd) { bd = dd; best = s; }
        }
        if (best < 0) return false;
        r.shape = c.d.shapes[best].id;
        if (Chance(m_Rng, 0.5f)) {
            r.sf = c.d.shapes[best].sf;
            Vector3 toward = Vector3Subtract(m_T.centre, r.sf.position);
            r.sf.size = Vector3Scale(r.sf.size, Uniform(m_Rng, 0.5f, 0.85f));
            r.sf.position = Vector3Add(r.sf.position, Vector3Scale(toward, Uniform(m_Rng, 0.0f, 0.3f)));
            r.hasShape = true;
        }
        r.owns = { Key('S', r.shape) };
        r.uses = { Key('G', body) };
        return true;
    }

    bool Grammar::ApplyRemoveStray(Design& d, const Rewrite& r, Focus& f) const {
        int s = d.ShapeIndex(r.shape);
        if (s < 0) return false;
        if (r.hasShape) {
            Tuned(d, r, s, f);
            return true;
        }
        if (LastMaterial(d, s) || !d.AddsMaterial(s)) return false;
        d.RemoveShape(s);
        return true;
    }

    // A cut in one of two overlapping bodies, fitted to where they overlap (in the pose they overlap in).
    bool Grammar::DrawCarveClash(const DrawContext& c, Rewrite& r) {
        const Clash& cl = PickFrom(m_Rng, c.diag.clashes);
        int ga = cl.a, gb = cl.b;
        if (ga < 0 || gb < 0 || ga >= (int)c.d.groups.size() || gb >= (int)c.d.groups.size()) return false;
        if (Chance(m_Rng, 0.5f)) std::swap(ga, gb);
        std::vector<Rigid> xf = c.d.Transforms(cl.t, true, cl.opening ? Motion::Opening : Motion::Closing);
        BodyField fa = BodyField::Body(c.d, c.d.groups[ga].id), fb = BodyField::Body(c.d, c.d.groups[gb].id);
        if (fa.Empty() || fb.Empty()) return false;
        float rho = m_T.L * Uniform(m_Rng, 0.04f, 0.15f);
        std::vector<Vector3> keep;
        for (int i = 0; i < 300; i++) {
            Vector3 p = Vector3Add(cl.p, Vector3Scale(RandomInBall(m_Rng), rho));
            Vector3 qa = xf[ga].ApplyInverse(p);
            if (fa.Distance(qa) < 0.0f && fb.Distance(xf[gb].ApplyInverse(p)) < 0.0f) keep.push_back(qa);
        }
        if (keep.size() < 10) return false;
        r.sf = BoxAround(keep, 0.02f, 0.015f * m_T.L);
        r.sf.size = Vector3Scale(r.sf.size, Uniform(m_Rng, 1.05f, 1.3f));
        r.sf.rounding = Uniform(m_Rng);
        r.hasShape = true;
        r.group = c.d.groups[ga].id;
        r.uses = { Key('G', r.group) };
        return true;
    }

    // ---- hinge timing -------------------------------------------------------------------------------------------------

    // A new window for when a hinge turns: anywhere, or just before or after another hinge's (folding one part out of
    // the way of another first is a jump the descent can't make in small steps).
    bool Grammar::DrawRetime(const DrawContext& c, Rewrite& r) {
        const Hinge& h = PickFrom(m_Rng, c.d.hinges);
        r.hinge = h.id;
        r.hasTiming = true;
        r.ownOpening = h.ownOpening;
        r.t0 = h.t0;
        r.t1 = h.t1;
        r.o0 = h.o0;
        r.o1 = h.o1;
        bool opening = h.ownOpening && Chance(m_Rng, 0.5f);
        float w = Uniform(m_Rng, 0.2f, 0.8f), a = Uniform(m_Rng, 0.0f, 1.0f - w);
        if (c.d.hinges.size() >= 2 && Chance(m_Rng, 0.5f)) {
            const Hinge& o = PickFrom(m_Rng, c.d.hinges);
            float oa = opening ? o.o0 : o.t0, ob = opening ? o.o1 : o.t1;
            a = Chance(m_Rng, 0.5f) ? ob : oa - w;                 // after it, or before it
            a = std::clamp(a, 0.0f, 1.0f - w);
        }
        (opening ? r.o0 : r.t0) = a;
        (opening ? r.o1 : r.t1) = a + w;
        r.owns = { Key('H', r.hinge) };
        return true;
    }

    // A hinge opens on its own timing (a window of the opening motion), or goes back to retracing its closing.
    bool Grammar::DrawOwnOpening(const DrawContext& c, Rewrite& r) {
        const Hinge& h = PickFrom(m_Rng, c.d.hinges);
        r.hinge = h.id;
        r.hasTiming = true;
        r.t0 = h.t0;
        r.t1 = h.t1;
        r.ownOpening = !h.ownOpening;
        float w = Uniform(m_Rng, 0.2f, 0.8f), a = Uniform(m_Rng, 0.0f, 1.0f - w);
        r.o0 = a;
        r.o1 = a + w;
        r.owns = { Key('H', r.hinge) };
        return true;
    }

    bool Grammar::ApplyTiming(Design& d, const Rewrite& r, Focus& f) const {
        if (d.HingeIndex(r.hinge) < 0) return false;
        f.hinge = r.hinge;                  // Apply sets the timing it carries
        return true;
    }
}
