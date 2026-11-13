# autodesign

This module carves a sphere into hinged pieces so that, opened, it looks like a mesh. Closed, the pieces are the sphere again, always, whatever the optimizer does. Open, their silhouettes (or DINOv3's view of them) should match the mesh's, and the pieces shouldn't pass through each other at any point while it opens and closes.

The editor and the test suite call one function, `AutoDesign`, declared in `autodesign.hpp`.

## Files

| file | what it is |
|---|---|
| `autodesign.cpp` | the optimizer, top to bottom: the sphere, Adam steps, rewrite rounds, result |
| `carving.hpp/.cpp` | what's designed: the sphere, its cutters, pieces and joints; building it into a `Design`; its parameters |
| `adam.hpp` | Adam, with memory kept per parameter key, so parameters can come and go |
| `loss.hpp/.cpp` | the loss (the open pose against the placed mesh, collisions, pivots) and its gradient; where collisions are checked |
| `transport.cpp` | the transport part of the loss: optimal transport between the sphere's material, opened, and the placed mesh |
| `shape.cpp` | the 3D shape part of the loss: the Chamfer distance and signed distances against the placed mesh |
| `semantic.cpp` | the semantic part of the loss: the views drawn as pictures and compared by DINOv3 |
| `dino.hpp/.cpp` | DINOv3 through ONNX Runtime, loaded at run time (its C API header is in `src/vendor/onnxruntime`) |
| `grammar.hpp/.cpp` | the rewrite rules: add or remove a cut (and its hinge), add or remove a hollow |
| `suite*.cpp/hpp` | the test suite: every mesh in a folder, into one video |

## The carving

A carving (`carving.hpp`) is a sphere and a set of **cutters**, superfrustums. Each cutter split one piece in two when it was added: the part inside the cutter and the part outside it. So every **piece** is the sphere, inside some cutters and outside others. Together the pieces fill the sphere exactly, and no two share any material, for any shape or position of the cutters: closed, the design is the sphere.

**Hollows** are cutters too, but they only take material away: from every piece, without making one. Closed, the pieces then fill the sphere less its hollows. The loss's surface term keeps hollows inside the sphere, so from outside it stays a whole sphere (some leeway: what hollows may cost is `surfaceWeight`).

Pieces are joined by **joints**, one per cut. A joint turns its child piece about an axis through a pivot, from 0 (closed: in the sphere) to its open angle, during its share `t0..t1` of the motion. The joints form a tree, and the piece the first cut left is its root.

`Carving::Build` makes it an ordinary `Design`, which the loss scores and the editor shows and exports:
- a body per piece, holding the sphere (an exact one: a speck rounded out to the radius);
- a cut for every cutter the piece is outside of, and for every hollow;
- a cut feature "outside this cutter" for every cutter it's inside of: a box around the sphere, less the cutter (the piece is the sphere less everything outside the cutter);
- a hinge per joint, with closed angle 0.

## The loop (`AutoDesign`)

1. **Target.** The mesh, drawn finely from `views` directions. The sphere sits at the mesh's centre of volume and has its volume (or `sphereDiameter`).
   - The mesh may be moved (`moveTarget`): its offset is three more Adam parameters. It's kept where the views see all of it.
   - Its scale is set, not optimized (`scaleTarget`, "Match volumes"): the size at which it has the sphere's volume, so the sphere's material fills it exactly. With the sphere sized from the mesh that's 1×; with `sphereDiameter` set, the mesh is resized to suit it. (Free, the scale shrank to about 0.6× on every mesh tried: a small mesh is easy to cover, and the sphere then held about 4.6× the material the mesh had room for.) Off, the mesh is compared at its own size.
   - Their gradient comes from the targets drawn again (resampled from the fine drawing) with each nudged two units either way, against the design as drawn.
2. **Start.** The sphere, whole: one piece.
3. **Steps.** Until the step budget runs out:
   - Every step is one Adam step on every parameter: the cutters', the joints' and the placement's.
   - Every `stepsPerRound` steps there is a rewrite round. Rewrites are drawn from the grammar. Each rewritten carving, and the unchanged one, gets `lookahead` Adam steps of its own. Whichever has the lowest loss afterwards carries on, with its Adam memory.
4. **Result.** The best carving seen during the run, built into a design (`AutoResult::design`; the carving itself is in `AutoResult::carving`). The editor shows the reference mesh where it was placed for it.

With 0 steps it only scores the editor's design as it is: its open pose against the mesh where it is.

## The rules

- **Add a cut** (adds). A small cap sliced off a piece where it meets the sphere's surface: a box whose inner face is a little under the surface. The cap is hinged to the rest of the piece along the edge of that face, like a lid. The joint's open angle starts at 0, so nothing looks different (`rulecheck`: the open pose's field is unchanged to 0.0000 mm) until Adam opens it. The joints the piece had go with whichever part their pivot is in. No more than `maxPieces` pieces.
- **Add a hollow** (adds). At a pixel where the open design shows outside the placed mesh (`Loss::ExtraSpots`, from the last evaluation), a rounded bar through its piece along the view that saw it, as long as the piece is deep there, so the pixel clears at once. Where the bar reaches the sphere's surface the surface term weighs it, and Adam reshapes and moves it from there. No more than `maxHollows`. A hollow inside the sphere only changes the silhouettes where an opened piece shows the sphere's inside (a cut face); the outline of the sphere's own surface only shrinks by carving through it, which the surface term charges for. In the runs so far (robot, creature) hollows were kept now and then, carving under 2% of the surface, without making the fit better on average than carving with cuts alone, which is why they're drawn half as often by default.
- **Remove a hollow** (simplifies). Any hollow.
- **Remove a cut** (simplifies). Two pieces that one cutter split, and that no later cut has split since, joined by a joint (likelier the less it opens), become one piece again; the cutter and the joint go. The joints of the piece that hung from the other go to it.

How often each rule is drawn is its weight (`AutoParams::ruleWeights`, in `AutoRules()` order; empty means the table's defaults). The Grammar section of the Generate tab sets them, and a running job picks up changes at its next rewrite round (through `AutoLive::ruleWeights`). A weight of 0 switches a rule off.

A rewrite is judged against leaving the carving alone, after both have had the same lookahead. One that adds has to beat it by `rewriteMargin`. One that takes away is kept if it's no more than `rewriteMargin` worse, because a simpler design that does about as well is better.

## The loss

Only the open pose and the motion are scored. Closed, the pieces are the sphere.

- **Silhouettes.** Per view, with T the placed mesh's coverage of a pixel and D the design's: covered = Σ min(T, D) / Σ T, extra = Σ max(0, D − T) / Σ T, and the view's loss is (1 − covered) + extra. `fit` is the mean over the views. A view covered less than `minViewCoverage` costs 10 per whole share short.
- **Semantics** (below), when `semanticWeight` > 0.
- **Collisions**, weighted by `collisionWeight`.
- **Hinges**: pivots away from their pieces, weighted by `hingeWeight`.
- **Surface**: hollows reaching the sphere's surface, weighted by `surfaceWeight`. Over 1024 points spread just under the surface, the mean of d / (d + 1% of the mesh's size), d how deep the point is inside a hollow: about the share of the surface carved away, rising smoothly as a hollow reaches it (so it has a gradient). Its gradient is measured directly, by nudging the hollows' parameters.

total = `silhouetteWeight` · (fit + coverage penalty) + `semanticWeight` · semantic + `collisionWeight` · collision + `hingeWeight` · hinges + `surfaceWeight` · surface.

## The 3D terms

All off by default. They compare the open design with the placed mesh in 3D, not as outlines from a few views.

**Transport** (`transportWeight`, `transportPoints`; `transport.cpp`). The carving can't grow or shrink, only rearrange the sphere's material, so the question is where that material goes. Silhouettes, the SDF and the Chamfer distance only see mistakes near the mesh: a piece opened to the wrong place, far from where it's needed, gets no pull from them. Optimal transport (the earth mover's distance) pairs every bit of material with a bit of the mesh it should fill, however far away, and pulls it there.
- *The material*: fixed points spread evenly through the sphere, in its frame. Each belongs to the piece it's inside (a point in a hollow: to none) and moves rigidly with that piece's open pose, so there's nothing to sample again.
- *The mesh*: fixed points filling it (winding number over ½), moved with the placement.
- *The masses*, in shares of the sphere's volume: a material point weighs its share of the sphere, a mesh point its share of the placed mesh's volume (over the sphere's). The transport is unbalanced: mass left unmatched costs ρ = (the mesh's size)² a unit, so moving material is always cheaper than dropping it and only a real difference in amount goes unmatched. A mesh with less room than the sphere's material can't take all of it, and material taken by hollows leaves part of the mesh unfilled (`transportUnfilled`): both cost.
- *The cost*: entropic unbalanced optimal transport with squared distances (Sinkhorn in the log domain, ε = (5% of the mesh's size)², warm started from the last evaluation), over (10% of the mesh's size)². With ρ that large, plain Sinkhorn settles the balance between the two sides (a constant shared by both potentials) only ε/ρ of the way a round, so after every half-round the best such shift is made exactly (translation invariant Sinkhorn, Séjourné et al. 2022). Without it the mass terms' slopes came out half what they should.
- *Its gradient*, with the plan held fixed. A material point's slope is 2r(x − T), r the mass it sends and T where; the joints move it (`Anchor.grad`). A mesh point's is 2c(y − S) likewise, which gives the placement's slope exactly (through the scale its mass changes too). A cutter hands material from one piece to another along its face, and a hollow takes it away: mass added at x changes the cost by φ(x) = ρ(1 − e^(−f(x)/ρ)) − ε(e^(−f(x)/ρ) − B), f(x) the dual potential there, and near a piece's surface its share of a point is taken as ½ − D/(2τ), τ = 2% of the mesh's size (`Anchor.massSlope`).
- Against brute force (`ONLY3D=1 OTW=1 gradcheck` on `assets/zard.obj`): cosine 0.992 over all 64 parameters, 0.92 over the carving's own. The joints' and the placement's (scale included) agree to a few parts in a thousand. The cutters' are rougher, because brute force there is itself a step function (points change piece one at a time).
- *Cost*: about 5× the silhouettes' a step at 1024 points (34 ms against 7 ms); it grows with the square of `transportPoints`.

**SDF** (`sdfWeight`) and **Chamfer** (`chamferWeight`) (`shape.cpp`, the mesh's points: `shapeSamples`).
- *SDF*: points in and around the mesh, whose signed distance T to it is known. The design's D should match it within 5% of the mesh's size of its surface; deeper inside, D only has to be that far inside too, and farther out, that far outside. So it tells inside from outside and says which way to move.
- *Chamfer*: points on the mesh's surface each cost the design's |D| there, and points on the design's surface (fixed points in each piece's box pulled onto it by Newton steps, where no other piece covers them) each cost the distance to the mesh (a box tree over its triangles). Carved pieces can't follow fine surface detail, so it's best light.
- Both depend on the design only through the pieces' distances at world points. A design-surface point's slope is −(∂u/∂x · n) against its piece's distance there: raising the distance by e moves the surface back by e along its normal n. Against brute force: cosine 0.97 (SDF), 0.95 (Chamfer).

`autotest` takes them as `OTW`, `OTN`, `SDFW`, `CHAMW`, `SHAPEN`, and measures every result with all three, whatever the run used.

## The semantic loss

Off by default (`semanticWeight` 0). On, each view of the open design is also drawn as a grey picture and compared with the placed mesh's picture by DINOv3.

**Setting it up.**
1. Export the model once: `tools/autodesign/export_dinov3.py` (it needs PyTorch and `transformers`, and the DINOv3 weights from Hugging Face, which are gated: accept the license on the model page and log in first). It writes one ONNX graph that takes a view and its target and gives back their loss, plus the loss's gradient for every pixel of both. The views are drawn at the size it was exported for: 112 pixels by default (7 × 7 patches), and `--size 224` gives 14 × 14 at about four times the cost.
2. Get ONNX Runtime 1.16 or newer as a shared library: `onnxruntime.dll`, `libonnxruntime.so` or `libonnxruntime.dylib`, from its GitHub releases or from a pip install of `onnxruntime` (in its `capi` folder). The editor loads it only when the semantic loss is on, so nothing about the build changes. The GPU builds of ONNX Runtime can run it with CUDA (`semanticGpu`).
3. In the Generate tab's Semantic loss section, set a weight and pick the two files.

If either file can't be loaded, the run stops and says why, rather than carry on without the loss you asked for.

**What's compared.** `semanticLook` sets how both pictures are drawn:
- **Silhouette:** white on black.
- **Depth:** nearer is brighter.
- **Shaded:** a light grey material under a key light from the viewer's upper left and a dimmer fill from the right, on dark grey.

`patchWeight` picks the embeddings: 0 compares each view's global (CLS) embedding with the target's, 1 compares the patch embeddings place by place (the two pictures share a camera), and anything between mixes the two. The loss per view is 1 − cosine similarity.

**How the pictures are drawn.**
- **The design.** Every piece a pixel's ray reaches is a layer. Its coverage comes from how near the ray comes, softened over a pixel as for silhouettes. Its shade comes from where the ray enters it (or, past its edge, where it comes closest). Layers composite front to back. Normals are measured across a pixel, which blurs creases (where the sphere and cuts meet) just enough that their shading moves smoothly as they do.
- **The mesh.** Depth buffered at 3 × 3 samples a pixel where it is, with smooth vertex normals (vertices welded by position), then resampled where the placement puts it. Shading and silhouettes don't change as it moves or scales; depth is worked out again from each sample's depth.

**Its gradient.** DINOv3 gives the loss's slope for every pixel of both pictures.
- *Coverage* changes only on edges, exactly as for silhouettes: a slope against the piece's distance at the ray's nearest point.
- *Shade* changes as the surface moves under the ray. Per parameter, the piece is nudged, the shaded point slides along its ray back onto the surface (one Newton step), and it's shaded again. That costs about seven field evaluations per point per parameter, so each step follows `semanticSamples` shaded points drawn by how much they matter, weighted to keep the sum unbiased. Edges are always all followed.
- *The placement* gets its slope from DINOv3's gradient for the mesh's picture, times how that picture changes as it's nudged.

**Cost.** DINOv3 runs once per view per evaluation (6 runs with 6 views), forward and backward. ViT-S/16 at 112 × 112 took about 130 ms a view on the 2-core machine this was written on. Expect a fraction of that on a desktop CPU, and much less with CUDA.

## Collisions

Pieces are checked for overlaps in the open pose, and `motionChecks` poses between open and closed (`CheckTimes` in `loss.cpp`). Closed needs no check.

- **Where the poses between go.** The timeline is cut wherever a joint starts or stops turning. Stretches where no joint turns get no checks.
- **How they're spread.** A share `1 - motionCheckBias` of the checks is spread evenly over the stretches where at least one joint turns. The rest goes by how many joints turn at once.
- **Why they follow the joints.** The checks sit at evenly spaced quantiles of that distribution, so they move along smoothly when Adam changes a joint's timing (`t0`, `t1`). The optimizer can't hide a collision by scheduling every turn where nothing is checked.

For each pose, every two pieces whose boxes meet are sampled where they meet, using the same unit points each time. A piece's box is the sphere's, cut down by the boxes of the cutters it's inside of. The term is the depth of the overlap integrated over its volume, relative to the sphere's volume times 2% of the mesh's size. The open pose counts fully; the poses between share one weight.

## Hinges

A joint's pivot should sit where its two pieces meet. For each joint, the term adds how far the pivot is outside each of its two pieces, beyond 1% of the mesh's size, over the mesh's size. Both are in the closed pose, the frame pivots are given in. A new cut's pivot starts on the edge of its cut face, on both pieces.

## The gradient

The silhouettes and collisions depend on the carving only through the pieces' distances at certain points: where the rays of edge pixels come closest to a piece, and the samples inside overlaps.

Silhouettes are soft. A pixel's coverage comes from how close its ray gets to the piece: 0 at a pixel outside, 1 at a pixel inside. So only pixels on a silhouette's edge depend on the parameters. The ray is sphere traced; on an edge pixel the steps only bracket its nearest point, so a short golden-section search closes in on it.

For each of those pixels:
- the loss's slope with respect to the pixel is known from the comparison with the target;
- the pixel's slope with respect to its ray's nearest distance is the edge's slope;
- that distance changes, to first order, only through the piece's field and pose at the ray's nearest point (the envelope theorem).

The last part is measured by nudging one parameter either way, building the carving again, and re-evaluating the distances at those points: for a cutter, the fields of the pieces it shapes (a hollow's: every piece's); for a joint, the poses. Collision samples work the same way: each sample's slope goes to the piece whose distance sets the overlap depth there. The times of the poses between are held fixed while the gradient is taken.

`tools/autodesign/gradcheck` compares this gradient against brute-force differences of the whole loss, for a carving of three pieces and a hollow. Over the cutters' and joints' 64 parameters it points the same way: cosine 0.995. The placement's four are themselves differences of the targets, so they agree with brute force at the same step. Run with `MINCOVER=0` to leave out the coverage penalty's kink, which brute force with a large step jumps across.

## Extending it

**A rule** is a function `bool (Carving&, const RuleContext&)` plus a row in the table in `grammar.cpp` (name, default weight, function, and whether it simplifies). Change the carving and return true, or return false when the rule can't apply. It shows up in the UI's Grammar section by itself. Any rule that only adds or removes cutters, splitting or merging pieces as above, keeps the closed pose the sphere.

New parameters are picked up automatically. Adam gives them fresh memory, and drops the memory of parameters that disappear.

**A differentiable number** needs a row in its kind's table in `carving.cpp` (its unit and range), and a case in `Ref`. The gradient needs nothing more: it nudges whatever `Ref` points at and builds the carving again.
