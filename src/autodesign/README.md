# autodesign

This module generates a hinged superfrustum design from a mesh. When the design is open, its silhouettes should match the mesh's. When it is closed, they should match a sphere's. Its bodies shouldn't pass through each other at any point while it opens and closes.

The editor and the test suite call one function, `AutoDesign`, declared in `autodesign.hpp`.

## Files

| file | what it is |
|---|---|
| `autodesign.cpp` | the optimizer, top to bottom: start design, Adam steps, rewrite rounds, result |
| `adam.hpp` | Adam, with memory kept per parameter key, so parameters can come and go |
| `params.hpp/.cpp` | the design's differentiable parameters: their units, ranges and keys |
| `loss.hpp/.cpp` | the loss (silhouettes against targets, and collisions) and its gradient; where collisions are checked |
| `grammar.hpp/.cpp` | the rewrite rules (for now, one rule that adds a superfrustum) |
| `suite*.cpp/hpp` | the test suite: every mesh in a folder, into one video |

## The loop (`AutoDesign`)

1. **Targets.** The mesh's silhouettes are drawn from `views` directions. The target sphere sits at the mesh's centre of volume and has the same volume.
2. **Start.** The mesh's box is halved along its longest side. Each half gets a box in its own body, and the two bodies are hinged to fold onto each other.
3. **Steps.** Until the step budget runs out:
   - Every step is one Adam step on every parameter.
   - Every `stepsPerRound` steps there is a rewrite round. Rewrites are drawn from the grammar. Each rewritten design, and the unchanged design, gets `lookahead` Adam steps of its own. Whichever has the lowest loss afterwards carries on, with its Adam memory.
4. **Result.** The best design seen during the run.

## Collisions

Bodies are checked for overlaps in the open pose, the closed pose, and `motionChecks` poses between them (`CheckTimes` in `loss.cpp`).

- **Where the poses between go.** The timeline is cut wherever a hinge starts or stops turning. Stretches where no hinge turns get no checks.
- **How they're spread.** A share `1 - motionCheckBias` of the checks is spread evenly over the stretches where at least one hinge turns. The rest goes by how many hinges turn at once.
- **Why they follow the hinges.** The checks sit at evenly spaced quantiles of that distribution, so they move along smoothly when Adam changes a hinge's timing (`t0`, `t1`, now parameters). The optimizer can't hide a collision by scheduling every hinge's turn where nothing is checked.

For each pose, every two bodies whose boxes meet are sampled where they meet, using the same unit points each time. The term is the depth of the overlap integrated over its volume, relative to the sphere's volume times 2% of the mesh's size. The open pose and the closed pose count fully; the poses between share one weight.

## The gradient

The loss depends on the design only through the bodies' distances at certain points: where the rays of edge pixels come closest to a body, and the samples inside overlaps.

Silhouettes are soft. A pixel's coverage comes from how close its ray gets to the body: 0 at a pixel outside, 1 at a pixel inside. So only pixels on a silhouette's edge depend on the parameters.

For each of those pixels:

- The loss's slope with respect to the pixel is known from the comparison with the target.
- The pixel's slope with respect to its ray's nearest distance is the edge's slope.
- That distance changes, to first order, only through the body's field at the ray's nearest point (the envelope theorem).

The last part is measured by nudging one parameter either way and re-evaluating the body's distance at those points. That takes about a thousand field evaluations per parameter, instead of drawing the whole design again. Collision samples work the same way: each sample's slope goes to the body whose distance sets the overlap depth there.

The times of the poses between are held fixed while the gradient is taken. They are placed from the design, not optimized directly.

`tools/autodesign/gradcheck` compares this gradient against brute-force differences of the whole loss. It agrees to within about 30% for most parameters. The rest differ because of occlusion between bodies and the rays' discrete steps, and because rotation and axis components are renormalized after every change, which the brute-force check doesn't account for.

## Extending it

**A rule** is a function `bool (Design&, const RuleContext&)` plus a row in the table in `grammar.cpp`. Change the design and return true, or return false when the rule can't apply.

New parameters are picked up automatically. Adam gives them fresh memory, and drops the memory of parameters that disappear.

**A differentiable number** needs a row in its kind's table in `params.cpp` (its unit and range), and a case in `Ref`. The gradient needs nothing more: it nudges whatever `Ref` points at.
