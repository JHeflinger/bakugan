# autodesign: generating a design from a mesh

The editor (and the test suite) only call one function, declared in `autodesign.hpp`:

```cpp
AutoResult AutoDesign(const TriMesh& mesh, const Design& start, const AutoParams& p, JobControl& c, std::shared_ptr<AutoLive> live);
```

Everything else lives in `namespace Bakugan::Auto`.

## How a run goes

`Search::Run` in `search.cpp` is the whole algorithm, step by step. It's Stochastic Rewrite Descent (Kodnongbua, Zhang, Sharp, Schulz, *Design for Descent*, SIGGRAPH Asia 2025):

1. **Measure the mesh** (`Target::Build`): sample points, an inside test, a signed distance field. Then the fixed samples every design is scored on, and the mesh's outlines (`Scorer::Prepare`).
2. **Start** (`Search::Start`): split the mesh into hinged pieces (`Search::Seed`, `start.cpp`), or take the editor's design (`Search::FromEditor`). It's made valid and scored, and becomes the best design so far.
3. **Rounds**, until the budget of scorings runs out:
   - a. **Descent** (`DescentPhase` → `DescentStep`): SPSA steps on one random element's continuous parameters, with the structure fixed.
   - b. **Rewrites** (`RewriteRound`): grammar rewrites are drawn and tried, each with a short lookahead. Every one that helps is applied at once, best first, and none of them conflict.
   - c. If the search has wandered far from the best design, it goes back to it.

   Collisions are ramped in over the first 60% of the budget.
4. **Result**: the best design scored on the way.

## The modules

| file | what it is |
|---|---|
| `search.hpp/.cpp` | the algorithm: the loop, descent steps, lookahead, rewrite rounds |
| `start.cpp` | starting designs: k-means seed with a hinge tree, or the editor's design |
| `target.hpp/.cpp` | the input mesh, measured once: samples, inside test, signed distance field |
| `space.hpp/.cpp` | the design space's limits, and `Sanitize` / `Project` / `Finalize` |
| `score.hpp/.cpp` | the score: every term, in the order it is measured (see the comment at the top of `score.cpp`) |
| `silhouettes.hpp/.cpp` | the Silhouettes loss: outlines of the mesh, the design and the sphere |
| `probe.hpp` | a design's bodies as distance fields, and the diagnostics a score hands to the grammar |
| `params.hpp/.cpp` | the continuous half of the design space: every number a descent step may move |
| `grammar.hpp/.cpp` | the discrete half: the rule table, and each rule's Draw / Apply pair |
| `placement.cpp` | where the grammar puts new shapes, cuts and hinges (aimed at the score's diagnostics) |
| `edits.hpp/.cpp` | design edits the grammar is built from (remove or merge a body, set up a hinge, ...) |
| `progress.hpp/.cpp` | what the UI sees while a run goes: the best design, its outlines, history, recording |
| `common.hpp/.cpp` | random numbers, geometry helpers, the thread pool |
| `suite*.cpp/hpp` | the test suite: every mesh in a folder, into one video |

## Extending it

**A new rewrite rule:**

1. Add an entry to `Rule` (`grammar.hpp`).
2. Add a row to the rule table in `grammar.cpp`: its name, its weight, and what it needs (cuts, features, hinges, ...).
3. Write `DrawYourRule`, which makes every random choice and records what the rewrite owns and uses, so conflicts are found.
4. Write `ApplyYourRule`, which makes the change and returns false when it no longer applies. It names what it created in the `Focus`, so the lookahead tunes that.

A `static_assert` checks that the table has one row per rule.

Mark a rule *patient* in the table when its change is nearly neutral by itself and only pays off once the shapes it made have moved, as with SplitShape and MergeShapes. Its lookahead then takes twice the steps and doesn't stop at the first step that fails.

To see how each rule fares, set the environment variable `BAKUGAN_RULE_STATS=1`. At the end of a run, each rule's tried and helped counts are printed to stderr.

**A new continuous parameter:**

1. Add it to `Dof` and the `Dofs` table in `params.cpp` (its part and unit).
2. List it in `Parameters::Of`.
3. Say how a step moves it in `Parameters::Step`.
4. If it needs a range, clamp it in `DesignSpace::Project`.

**A new score term:**

1. Add a stage method to `Scorer` and call it from `Scorer::Evaluate` in its place.
2. Add its numbers to `AutoScore` (`autodesign.hpp`) if the UI or the log should show them.
3. Add it to `base` or `collision` in `Evaluate`. Collisions are ramped in, so only overlap-like terms belong there.

**Reproducibility:** a run is deterministic for a given seed and settings. One random generator, owned by the search, is drawn from in a fixed order. The score's sample points come from their own generator, so changing the score doesn't change the search's draws.

## Testing without the editor

`tools/autodesign/build.sh` builds `autotest`, which runs the generator on an .obj and writes the design. It also builds `suitetest`, which runs the test suite. Neither needs the GUI.
