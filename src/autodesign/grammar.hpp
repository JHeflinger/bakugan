#pragma once
// The discrete half of the design space: the grammar's rewrite rules, each a structural change to a design.
//
// Every rule has two halves:
//   Draw   makes all of a rewrite's random choices for one design (where, what, how much), so that it can be applied
//          later, alone or together with other rewrites, without drawing again
//   Apply  makes the change on a design, which may already carry other rewrites: false when it no longer applies
//
// Adding a rule: an entry in Rule, a row in the rule table in grammar.cpp (name, weight, what it needs, whether its
// lookahead is patient), and its Draw... / Apply... pair.
#include "params.hpp"
#include "probe.hpp"

namespace Bakugan::Auto {
    enum class Rule : unsigned char {
        AddShape, AddCut, RemoveShape, ToggleOp, MoveShape,           // shapes and material
        SplitShape, MergeShapes,                                       // one shape into two, two into one
        AddBody, SplitBody, RemoveBody, MergeBody,                     // bodies
        AddFeature, DissolveFeature, ToggleFeature,                    // blended features
        AddHinge, RemoveHinge, Rehang, TurnHinge,                      // hinges
        RemoveStray, CarveClash,                                       // repairs aimed where the score found errors
        Retime, OwnOpening,                                            // hinge timing
        Count
    };

    // One rewrite with all its random choices made. Elements are named by id (stable while a design is edited),
    // never by index.
    struct Rewrite {
        Rule rule = Rule::AddShape;
        int shape = -1;              // the shape it acts on
        int group = -1;              // the body or feature it acts on, or that receives a shape
        int other = -1;              // a second body (a hinge's parent, a shape's new body)
        int hinge = -1;              // the hinge it acts on
        std::vector<int> shapes;     // shapes it groups (AddFeature)
        bool hasShape = false;       // `sf` is a new shape, or `shape`'s new values
        SuperFrustum sf;
        SuperFrustum sf2;            // SplitShape: what `shape` itself becomes (`sf` is the new half)
        bool hasHinge = false;       // a new hinge's (or `hinge`'s new) pivot, axis and turn
        Vector3 pivot = { 0, 0, 0 }, axis = { 0, 1, 0 };
        float angle = 0.0f;          // closing turn (end - start)
        bool hasTiming = false;      // ... and its timing (see Hinge: t0..t1, and o0..o1 with ownOpening)
        float t0 = 0.0f, t1 = 1.0f;
        bool ownOpening = false;
        float o0 = 0.0f, o1 = 1.0f;
        float value = 0.0f;          // a new feature's blend
        std::vector<long long> owns; // elements it changes or removes ...
        std::vector<long long> uses; // ... and elements it only needs to still be there
        float gain = 0.0f;           // how much it improved the objective in its lookahead
    };

    // Two rewrites conflict when one changes something the other changes or needs: only one of them may be applied
    // in the same round.
    bool Conflict(const Rewrite& a, const Rewrite& b);

    // What a rewrite is drawn from: the design, where its last score found it wrong, and that score (its sphere).
    struct DrawContext {
        const Design& d;
        const Diagnostics& diag;
        const AutoScore& score;
        std::vector<int> bodies;     // group indices
        std::vector<int> features;   // group ids of the groups inside bodies
        DrawContext(const Design& design, const Diagnostics& dg, const AutoScore& s);
    };

    class Grammar {
    public:
        Grammar(const AutoParams& p, const Target& t, const DesignSpace& s, Rng& rng) : m_P(p), m_T(t), m_Space(s), m_Rng(rng) {}

        // Draws one rewrite: a rule by its weight among those that can apply, then its choices. False when the
        // drawn rule turned out not to apply to this design.
        bool Draw(const DrawContext& c, Rewrite* out);
        // Applies it (and the timing it carries), naming what it created or changed in *f for the lookahead.
        bool Apply(Design& d, const Rewrite& r, Focus* f) const;
        // Writes what the lookahead settled on back into the rewrite, so applying it again (combined with other
        // rewrites) gives the tuned shape and hinge rather than the first guess.
        void Remember(Rewrite& r, const Design& d, const Focus& f) const;

        // Hangs `child` from `parent` on a new hinge at the pivot, turning as ChooseHinge picks. Returns its id.
        int AttachHinge(Design& d, int parent, int child, Vector3 pivot, Vector3 childPoint);
        static const char* Name(Rule r);
        static bool Patient(Rule r);     // its lookahead keeps trying after a step that doesn't help (see grammar.cpp)

        struct RuleInfo;                 // a row of the rule table (grammar.cpp)
        static const RuleInfo Rules[];
    private:
        bool Allowed(const RuleInfo& info, const DrawContext& c) const;

        // the rules
        bool DrawAddShape(const DrawContext& c, Rewrite& r);
        bool DrawAddCut(const DrawContext& c, Rewrite& r);
        bool DrawPickShape(const DrawContext& c, Rewrite& r);         // RemoveShape, ToggleOp
        bool DrawMoveShape(const DrawContext& c, Rewrite& r);
        bool DrawSplitShape(const DrawContext& c, Rewrite& r);
        bool DrawMergeShapes(const DrawContext& c, Rewrite& r);
        bool DrawAddBody(const DrawContext& c, Rewrite& r);
        bool DrawSplitBody(const DrawContext& c, Rewrite& r);
        bool DrawRemoveBody(const DrawContext& c, Rewrite& r);
        bool DrawMergeBody(const DrawContext& c, Rewrite& r);
        bool DrawAddFeature(const DrawContext& c, Rewrite& r);
        bool DrawPickFeature(const DrawContext& c, Rewrite& r);       // DissolveFeature, ToggleFeature
        bool DrawAddHinge(const DrawContext& c, Rewrite& r);
        bool DrawRemoveHinge(const DrawContext& c, Rewrite& r);
        bool DrawRehang(const DrawContext& c, Rewrite& r);
        bool DrawTurnHinge(const DrawContext& c, Rewrite& r);
        bool DrawRemoveStray(const DrawContext& c, Rewrite& r);
        bool DrawCarveClash(const DrawContext& c, Rewrite& r);
        bool DrawRetime(const DrawContext& c, Rewrite& r);
        bool DrawOwnOpening(const DrawContext& c, Rewrite& r);

        bool ApplyAddShape(Design& d, const Rewrite& r, Focus& f) const;
        bool ApplyAddCut(Design& d, const Rewrite& r, Focus& f) const;        // AddCut, CarveClash
        bool ApplyRemoveShape(Design& d, const Rewrite& r, Focus& f) const;
        bool ApplyToggleOp(Design& d, const Rewrite& r, Focus& f) const;
        bool ApplyMoveShape(Design& d, const Rewrite& r, Focus& f) const;
        bool ApplySplitShape(Design& d, const Rewrite& r, Focus& f) const;
        bool ApplyMergeShapes(Design& d, const Rewrite& r, Focus& f) const;
        bool ApplyAddBody(Design& d, const Rewrite& r, Focus& f) const;
        bool ApplySplitBody(Design& d, const Rewrite& r, Focus& f) const;
        bool ApplyRemoveBody(Design& d, const Rewrite& r, Focus& f) const;
        bool ApplyMergeBody(Design& d, const Rewrite& r, Focus& f) const;
        bool ApplyAddFeature(Design& d, const Rewrite& r, Focus& f) const;
        bool ApplyDissolveFeature(Design& d, const Rewrite& r, Focus& f) const;
        bool ApplyToggleFeature(Design& d, const Rewrite& r, Focus& f) const;
        bool ApplyAddHinge(Design& d, const Rewrite& r, Focus& f) const;
        bool ApplyRemoveHinge(Design& d, const Rewrite& r, Focus& f) const;
        bool ApplyRehang(Design& d, const Rewrite& r, Focus& f) const;
        bool ApplyTurnHinge(Design& d, const Rewrite& r, Focus& f) const;
        bool ApplyRemoveStray(Design& d, const Rewrite& r, Focus& f) const;
        bool ApplyTiming(Design& d, const Rewrite& r, Focus& f) const;       // Retime, OwnOpening

        // where new material goes (placement.cpp)
        SuperFrustum NewShape(Vector3 at, bool cut);
        bool PlaceSpot(const DrawContext& c, bool forCut, Vector3* at, int* body, bool* sphere);
        bool FitRegion(const DrawContext& c, bool cut, SuperFrustum* f, int* body);
        void ChooseHinge(Vector3 pivot, Vector3 childPoint, Vector3* axis, float* angle);
        int RandomBody(const DrawContext& c) { return c.d.groups[c.bodies[Pick(m_Rng, (int)c.bodies.size())]].id; }
        bool IsBody(const Design& d, int id) const;
        bool LastMaterial(const Design& d, int shapeIndex) const;     // the last shape adding material to its body
        void Tuned(Design& d, const Rewrite& r, int shapeIndex, Focus& f) const;   // the lookahead's values
    private:
        const AutoParams& m_P;
        const Target& m_T;
        const DesignSpace& m_Space;
        Rng& m_Rng;
    };
}
