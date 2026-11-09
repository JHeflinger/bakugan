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
| `semantic.cpp` | the semantic part of the loss: the views drawn as pictures and compared by DINOv3 |
| `dino.hpp/.cpp` | DINOv3 through ONNX Runtime, loaded at run time (its C API header is in `src/vendor/onnxruntime`) |
| `grammar.hpp/.cpp` | the rewrite rules: add or remove a body, a union or a cut; split a body; merge two |
| `suite*.cpp/hpp` | the test suite: every mesh in a folder, into one video |

## The loop (`AutoDesign`)

1. **Targets.** The mesh's silhouettes are drawn from `views` directions. The target sphere has the mesh's volume and starts at the mesh's centre of volume.
   - With `moveSphere` on, its centre is three more Adam parameters, so the closed design can become a sphere anywhere.
   - The centre's gradient comes from the closed pose's share of the loss, measured again with the disc shifted a quarter pixel each way.
   - It's kept where the views can see the whole sphere.
   - Scoring a design without optimizing it (0 steps) first fits the sphere to it.
2. **Start.** The mesh's box is halved along its longest side. Each half gets a box in its own body, and the two bodies are hinged to fold onto each other.
3. **Steps.** Until the step budget runs out:
   - Every step is one Adam step on every parameter.
   - Every `stepsPerRound` steps there is a rewrite round. Rewrites are drawn from the grammar. Each rewritten design, and the unchanged design, gets `lookahead` Adam steps of its own. Whichever has the lowest loss afterwards carries on, with its Adam memory.
4. **Result.** The best design seen during the run.

## The rules

Rules that add something:

- **Add a body.** A very small box, in a new body hinged to the nearest body, at a random point on the mesh.
- **Add a union** / **add a cut.** A superfrustum that adds or removes material in a body, aimed at a spot where the last loss found the design wrong (`Loss::ExtraSpots` / `OverlapSpots`):
  - At a spot where the open design shows outside the mesh, it's a thin shape straight through the body along the view that saw it.
  - At a spot where two bodies overlap, it's a small shape there.
- **Split a body.** A body cut in half across the middle of its box, as two bodies: each is the whole body (copied) plus one big cut that takes away the other half. They're joined on the cut by a hinge that doesn't turn yet (open angle = closed angle = 0), pivoted at the point on the cut nearest both halves. The body's own hinge and the hinges hanging from it go to whichever half their pivot is nearer. Nothing changes in any pose until Adam turns the new hinge (or moves the cuts).
  - The axis is drawn by the box's sides (longer ones likelier). If that puts the cut edge-on to a view in the open or closed pose, it leans up to ~25° until it isn't. Edge on, rays running along the seam inside the body only graze each half (a half's field there is no deeper than its distance to the cut face), and the silhouettes would show a hairline crack the joined body didn't have.

Rules that take something away (`simplifies` in the table):

- **Remove a body.** Any body on a hinge. Its children hang from its parent instead, on their own hinges. The root body stays, and so do at least two bodies.
- **Remove a union.** A material shape from a body that has another.
- **Remove a cut.**
- **Merge bodies.** A body folded into its parent, likelier the less its hinge turns. Its contents become one new feature of the parent, and the bodies hanging from it hang from the parent. Where the hinge held it in the open pose is baked into its shapes (and into everything hanging from it), so the open pose stays exactly as it was and every other pose changes only by the turn the hinge used to make. If the parent has cuts or a blend of its own, its own contents go into a feature first, so they don't reach into the merged-in one.

How often each rule is drawn is its weight (`AutoParams::ruleWeights`, in `AutoRules()` order; empty means the table's defaults). The Grammar section of the Generate tab sets them, and a running job picks up changes at its next rewrite round (through `AutoLive::ruleWeights`). A weight of 0 switches a rule off.

A rewrite is judged against leaving the design alone, after both have had the same lookahead. One that adds has to beat it by `rewriteMargin`. One that takes away is kept if it's no more than `rewriteMargin` worse, because a simpler design that does about as well is better.

Adam optimizes cuts like any other shape. Nothing in the parameters or the gradient depends on whether a shape adds or cuts, since a body's field includes its cuts.

## The semantic loss

Off by default (`semanticWeight` 0). On, each view is also drawn as a grey picture and compared with the target's picture by DINOv3. The open pose is compared with the mesh, and the closed pose with the sphere, weighted by the goal like the silhouettes.

**Setting it up.**
1. Export the model once: `tools/autodesign/export_dinov3.py` (it needs PyTorch and `transformers`, and the DINOv3 weights from Hugging Face, which are gated: accept the license on the model page and log in first). It writes one ONNX graph that takes a view and its target and gives back their loss, plus the loss's gradient for every pixel of both. The views are drawn at the size it was exported for: 112 pixels by default (7 × 7 patches), and `--size 224` gives 14 × 14 at about four times the cost.
2. Get ONNX Runtime 1.16 or newer as a shared library: `onnxruntime.dll`, `libonnxruntime.so` or `libonnxruntime.dylib`, from its GitHub releases or from a pip install of `onnxruntime` (in its `capi` folder). The editor loads it only when the semantic loss is on, so nothing about the build changes. The GPU builds of ONNX Runtime can run it with CUDA (`semanticGpu`).
3. In the Generate tab's Semantic loss section, set a weight and pick the two files.

If either file can't be loaded, the run stops and says why, rather than carry on without the loss you asked for.

**What's compared.** `semanticLook` sets how every picture is drawn, the design's and the targets' alike:
- **Silhouette:** white on black.
- **Depth:** nearer is brighter.
- **Shaded:** a light grey material under a key light from the viewer's upper left and a dimmer fill from the right, on dark grey.

`patchWeight` picks the embeddings: 0 compares each view's global (CLS) embedding with the target's, 1 compares the patch embeddings place by place (the two pictures share a camera), and anything between mixes the two. The loss per view is 1 − cosine similarity.

**How the pictures are drawn.**
- **The design.** Every body a pixel's ray reaches is a layer. Its coverage comes from how near the ray comes, softened over a pixel as for silhouettes. Its shade comes from where the ray enters it (or, past its edge, where it comes closest). Layers composite front to back. Normals are measured across a pixel, which blurs creases (where unions and cuts meet) just enough that their shading moves smoothly as they do.
- **The mesh.** Depth buffered at 3 × 3 samples a pixel, with smooth vertex normals (vertices welded by position).
- **The sphere.** Drawn analytically, the same way as the design.

**Its gradient.** DINOv3 gives the loss's slope for every pixel. A pixel's value depends on each layer's coverage and shade:
- *Coverage* changes only on edges, exactly as for silhouettes: a slope against the body's distance at the ray's nearest point.
- *Shade* changes as the surface moves under the ray. Per parameter, the body is nudged, the shaded point slides along its ray back onto the surface (one Newton step), and it's shaded again. That costs about seven field evaluations per point per parameter, so each step follows `semanticSamples` shaded points drawn by how much they matter, weighted to keep the sum unbiased. Edges are always all followed.
- *The sphere's centre* gets its slope from DINOv3's gradient for the target picture, times how the sphere's picture changes as it moves a quarter pixel each way.

Against brute force differences of the whole loss (`gradcheck` with `SEM=…`; DINOv3 with random weights, since the real ones can't be downloaded here), the gradient as a whole points the same way: cosine 0.997 drawn as silhouettes, 0.995 as depth, and 0.89 shaded (0.91 with the test's two bodies apart), over its 52 parameters. Shading changes faster with the parameters than coverage does, so brute force there needs small steps (`H=0.25`) to agree with itself. The one thing it doesn't follow is a shaded picture switching from one body's surface to another's where two bodies pass through each other. The collision term keeps that rare.

**Cost.** DINOv3 runs once per view per pose per evaluation (12 runs with 6 views), forward and backward. ViT-S/16 at 112 × 112 took about 130 ms a view on the 2-core machine this was written on, i.e. about 1.6 s a step. Expect a fraction of that on a desktop CPU, and much less with CUDA. Fewer views, or a lower step budget, are the other knobs. `FitSphere` (scoring only) still places the sphere by silhouettes alone.

## Collisions

Bodies are checked for overlaps in the open pose, the closed pose, and `motionChecks` poses between them (`CheckTimes` in `loss.cpp`).

- **Where the poses between go.** The timeline is cut wherever a hinge starts or stops turning. Stretches where no hinge turns get no checks.
- **How they're spread.** A share `1 - motionCheckBias` of the checks is spread evenly over the stretches where at least one hinge turns. The rest goes by how many hinges turn at once.
- **Why they follow the hinges.** The checks sit at evenly spaced quantiles of that distribution, so they move along smoothly when Adam changes a hinge's timing (`t0`, `t1`, now parameters). The optimizer can't hide a collision by scheduling every hinge's turn where nothing is checked.

For each pose, every two bodies whose boxes meet are sampled where they meet, using the same unit points each time. The term is the depth of the overlap integrated over its volume, relative to the sphere's volume times 2% of the mesh's size. The open pose and the closed pose count fully; the poses between share one weight.

## Hinges

A hinge's pivot should sit where its two bodies meet. For each hinge, the term adds how far the pivot is outside each of its two bodies, beyond 1% of the mesh's size, over the mesh's size. Both the pivot and the bodies are in the design pose, the frame pivots are given in. It's weighted by `hingeWeight`.

Its gradient is measured directly, by nudging each parameter that moves a pivot or changes one of its bodies. New bodies start with their pivot on both bodies: the start design's two boxes both reach the hinge in the middle, and the add-a-superfrustum rule's thin box reaches from the pivot out to its point on the mesh.

## The gradient

(The semantic loss's gradient is described above.) The rest of the loss depends on the design only through the bodies' distances at certain points: where the rays of edge pixels come closest to a body, and the samples inside overlaps.

Silhouettes are soft. A pixel's coverage comes from how close its ray gets to the body: 0 at a pixel outside, 1 at a pixel inside. So only pixels on a silhouette's edge depend on the parameters. The ray is sphere traced; on an edge pixel the steps only bracket its nearest point, so a short golden-section search closes in on it. Otherwise the edge would shift with where the steps happen to land, which depends on how the body's field is put together (a split body would score differently from the same shape whole).

For each of those pixels:

- The loss's slope with respect to the pixel is known from the comparison with the target.
- The pixel's slope with respect to its ray's nearest distance is the edge's slope.
- That distance changes, to first order, only through the body's field at the ray's nearest point (the envelope theorem).

The last part is measured by nudging one parameter either way and re-evaluating the body's distance at those points. That takes about a thousand field evaluations per parameter, instead of drawing the whole design again. Collision samples work the same way: each sample's slope goes to the body whose distance sets the overlap depth there.

The times of the poses between are held fixed while the gradient is taken. They are placed from the design, not optimized directly.

`tools/autodesign/gradcheck` compares this gradient against brute-force differences of the whole loss. It agrees to within about 30% for most parameters. The rest differ because of occlusion between bodies and the rays' discrete steps, and because rotation and axis components are renormalized after every change, which the brute-force check doesn't account for.

## Extending it

**A rule** is a function `bool (Design&, const RuleContext&)` plus a row in the table in `grammar.cpp` (name, default weight, function, and whether it simplifies). Change the design and return true, or return false when the rule can't apply. It shows up in the UI's Grammar section by itself.

New parameters are picked up automatically. Adam gives them fresh memory, and drops the memory of parameters that disappear.

**A differentiable number** needs a row in its kind's table in `params.cpp` (its unit and range), and a case in `Ref`. The gradient needs nothing more: it nudges whatever `Ref` points at.
