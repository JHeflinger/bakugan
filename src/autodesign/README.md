# autodesign

This module carves a sphere into hinged pieces so that, opened, it looks like a mesh. Closed, the pieces are the sphere again, always, whatever the optimizer does. Open, their silhouettes (or DINOv3's view of them) should match the mesh's, and the pieces shouldn't pass through each other at any point while it opens and closes.

The editor and the test suite call one function, `AutoDesign`, declared in `autodesign.hpp`.

## Files

| file | what it is |
|---|---|
| `autodesign.cpp` | the optimizer, top to bottom: the sphere, Adam steps, rewrite rounds, result |
| `carving.hpp/.cpp` | what's designed: the sphere, its cutters, pieces and joints; building it into a `Design`; its parameters |
| `adam.hpp` | Adam, with memory kept per parameter key, so parameters can come and go |
| `loss.hpp/.cpp` | the loss (the open pose against the placed mesh, collisions, holding, pivots) and its gradient; the opening order |
| `thickness.cpp` | the thickness part of the loss: no piece thinner than can be printed |
| `snap.cpp` | what keeps a carving buildable at every step: hinges on their seams, every piece one solid big enough for its hinges |
| `tracer.hpp` | the silhouettes drawn somewhere else than on the CPU's cores (the editor's GPU: `ui/gputrace.cpp`) |
| `../data/clash.hpp/.cpp` | overlaps between bodies on their exact fields, and where in the motion to look for them |
| `../data/finish.hpp/.cpp` | the finishing passes: hinges, carving collisions away, ledges, the press latch; and the check that the result prints |
| `semantic.cpp` | the semantic part of the loss: the views drawn as pictures and compared by DINOv3 |
| `dino.hpp/.cpp` | DINOv3 through ONNX Runtime, loaded at run time (its C API header is in `src/vendor/onnxruntime`) |
| `grammar.hpp/.cpp` | the rewrite rules: add a cut or split a piece (with its hinge), remove a cut |
| `suite*.cpp/hpp` | the test suite: every mesh in a folder, into one video |

## The carving

A carving (`carving.hpp`) is a sphere and a set of **cutters**, superfrustums. Each cutter split one piece in two when it was added: the part inside the cutter and the part outside it. So every **piece** is the sphere, inside some cutters and outside others. Together the pieces fill the sphere exactly, and no two share any material, for any shape or position of the cutters: closed, the design is the sphere.

Pieces are joined by **joints**, one per cut. A joint turns its child piece about an axis through a pivot, from 0 (closed: in the sphere) to its open angle, during its share `t0..t1` of the motion. The joints form a tree, and the piece the first cut left is its root.

`Carving::Build` makes it an ordinary `Design`, which the loss scores and the editor shows and exports:
- a body per piece, holding the sphere (an exact one: a speck rounded out to the radius);
- a cut for every cutter the piece is outside of;
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
   - After each step every hinge is put back on the seam between its two pieces, and the carving has to stay buildable (below): a cutter whose move would break that stays where it is for that step.
   - What overlaps cost rises steadily over the run, from `collisionWeight` to `collisionRamp` times it: early on the search may try things, by the end the pieces have to be clear of each other. Carvings from different times are compared as they'd score at the end of the ramp.
   - Every `stepsPerRound` steps there is a rewrite round. First the opening order is looked at again (below). Then rewrites are drawn from the grammar. Each rewritten carving, and the unchanged one, gets `lookahead` Adam steps of its own. Whichever has the lowest loss afterwards carries on, with its Adam memory.
4. **Result.** The best carving seen during the run, built into a design (`AutoResult::design`; the carving itself is in `AutoResult::carving`). The editor shows the reference mesh where it was placed for it.
5. **Finishing** (below): the hinges' hardware and carving by default, ledges and a latch when asked. They change the design only, not the carving. With hinges built, the result is checked for printing (`AutoResult::printable`). Then the finished design's motion is checked once more, at 24 poses in every hinge's turn on a finer grid than the search could afford (`AutoResult::checked`).

With 0 steps it only scores the editor's design as it is: its open pose against the mesh where it is.

## The rules

- **Add a cut** (adds). A small cap sliced off a piece where it meets the sphere's surface: a box whose inner face is a little under the surface. The cap is hinged to the rest of the piece along the edge of that face, like a lid. The joint's open angle starts at 0, so nothing looks different (`rulecheck`: the open pose's field is unchanged to 0.0000 mm) until Adam opens it. The joints the piece had go with whichever part their pivot is in. No more than `maxPieces` pieces.
- **Reach for the mesh** (adds). Aimed, where the others are random. `Loss::FarTarget` draws a point inside the placed mesh (its visual hull over the views) that the open design is far from, likelier the farther. From the piece nearest to it, at that piece's point nearest to it, a strip is cut back along the piece's surface (a box along a chord of the surface, everything above its floor), sized by how much of the mesh is missing around the point, and hinged at that point across the strip. Unlike the other rules it changes the open pose at once: of the three opening angles that swing the strip's far end nearest the point, the one the design scores best with is taken, and only if that beats the design as it is. Cut from the sphere it makes a limb; cut from a limb's end, the next segment of a chain.
- **Split a piece** (adds). A plane through a random point of a piece, facing a random way, cuts it in two; the halves are hinged along the edge of the cut face, closed, the axis turned so that opening lifts the new half off the rest. Big parts in one rewrite. Under the rules for staying buildable (below) most small caps are turned down, and most pieces come from here.
- **Remove a cut** (simplifies). Two pieces that one cutter split, and that no later cut has split since, joined by a joint (likelier the less it opens), become one piece again; the cutter and the joint go. The joints of the piece that hung from the other go to it.

How often each rule is drawn is its weight (`AutoParams::ruleWeights`, in `AutoRules()` order; empty means the table's defaults). The Grammar section of the Generate tab sets them, and a running job picks up changes at its next rewrite round (through `AutoLive::ruleWeights`). A weight of 0 switches a rule off.

A rewrite is judged against leaving the carving alone, after both have had the same lookahead. One that adds has to beat it by `rewriteMargin`. One that takes away is kept if it's no more than `rewriteMargin` worse, because a simpler design that does about as well is better.

## The loss

Only the open pose and the motion are scored. Closed, the pieces are the sphere.

- **Silhouettes.** Per view, with T the placed mesh's coverage of a pixel and D the design's: covered = Σ min(T, D) / Σ T, extra = Σ max(0, D − T) / Σ T, and the view's loss is (1 − covered) + extra. `fit` is the mean over the views. A view covered less than `minViewCoverage` costs 10 per whole share short.
- **Reach**, weighted by `reachWeight`. The fit only changes where the design's outline is, so a part of the mesh nothing is near pulls on nothing, and a blob over the body scores well enough to stay. Every missed pixel therefore also counts by how far it is from the nearest covered pixel (beyond one pixel, over 10% of the mesh's size), and that slope goes to the piece covering that nearest pixel: it is pulled toward what is missing, however far away.
- **Semantics** (below), when `semanticWeight` > 0.
- **Collisions**, weighted by `collisionWeight` (times the ramp).
- **Holding** (with `dependencies`): hinges nothing could hold shut, weighted by `dependencyWeight`.
- **Hinges**: pivots away from their pieces, weighted by `hingeWeight`.
- **Thickness**: pieces thinner than `minThickness` (the largest ball that fits in them), weighted by `thinWeight`.

total = `silhouetteWeight` · (fit + coverage penalty) + `reachWeight` · reach + `semanticWeight` · semantic + `collisionWeight` · collision + `hingeWeight` · hinges + `thinWeight` · thin + `dependencyWeight` · holding.

## Staying buildable

The loss alone doesn't keep a carving printable. Left to it, a cutter that moves through a piece leaves part of it cut off from the rest: material that still moves with the piece in the open pose (and fills the mesh's silhouette nicely) but that nothing joins to it, and pieces shrink to slivers too small for the hinge they hang from. So every Adam step and every rewrite has to leave the carving buildable (`Loss::SnapJoints`, `Loss::Solid`, both in `snap.cpp`):

- **Hinges on seams.** Each joint's pivot is projected back onto the seam its two pieces share, its axis along it (and, with hinges built under the surface, deep enough for the barrel).
- **One solid.** Each piece, taken 0.4 · `minThickness` in from its surface, is still one part (on a grid a quarter of `minThickness`; `MeasureSolidity` in `data/finish.hpp`): no more than half a percent of the sphere may be in parts not joined to the rest of their piece. Taking the piece in first means two parts joined only by a thin neck count as apart, since the cuts made later for hinges and collisions would part them.
- **Hinges on the solid.** Each pivot is at the largest part of both of its pieces.
- **Big enough.** Each piece has at least twice the volume of the box around a hinge's barrel, for every hinge on it (`Loss::PieceVolume`; a new cap starts at least that big).

A rewrite that breaks any of these isn't tried. An Adam step that does is taken again cutter by cutter: each cutter moves only if the carving is still buildable with it moved, so the one cutter that would break it stays put for that step and everything else carries on. The result's message says in how many steps that happened.

## What was measured, and removed

Each part was switched off in turn over four meshes (`zard`, `cow`, `human`, `teapot`, 80 mm long) and two seeds, 1500 steps, and the fit (share of the silhouettes missed + share extra) compared with all of it on (0.51). Differences under about 0.02 are noise at eight runs.

| part | fit without it | verdict |
|---|---|---|
| hollows (cutters that only remove material) | 0.50 | no gain, and they cut pieces apart: removed |
| "aim a cut" (needed transport) | 0.50 | did nothing with transport off, a quarter of all rewrite draws: removed |
| "split a piece" | 0.49 | no gain then, but that was before the rules for staying buildable: with them, runs without it ended with 2 pieces and 62% cover instead of 72%. Kept (without its transport-aimed hinge placement) |
| optimal transport (on) | 0.50 | no gain, 4× slower: removed |
| 3D SDF / Chamfer (on) | 0.49 | no gain: removed |
| "remove a cut" | 0.51 | without it 6 pieces instead of 4, 3× slower, more overlap: kept |
| thickness | 0.50 | without it the thinnest piece is 1.6 mm instead of 3.3: kept |
| coverage penalty | 0.50 | without it cover drops from 72% to 63%: kept |
| the mesh may move | 0.51 | without it cover drops from 72% to 69%: kept |
| collision ramp | 0.47 | without it 4× the overlap left on the way: kept |
| opening order | 0.49 | no clear effect on fit; kept (dependencies need it) |

With what was kept, and the rules for staying buildable, the same eight runs fit a little better than before (0.49; 71% of the silhouettes covered, 20% extra) and all eight print: every piece one solid, every hinge built. Before, none did (every design had loose crumbs, and two had pieces nothing held).

The semantic loss, dependencies, ledges and the latch are off by default and weren't measured (the semantic loss needs a model that isn't in the repository).

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

Pieces are checked for overlaps in the open pose, and in `hingeChecks` poses of every hinge's turn (`HingeChecks` in `data/clash.cpp`). Closed needs no check.

- **Per hinge, not per second.** A hinge's checks are spread evenly over its *angle*. A turn that takes all of the motion and one that takes a twentieth of it are looked at equally closely, so turning faster hides nothing and gains nothing.
- **Between the last ones.** Every Adam step shifts the checks (and the sampling grid) by the golden ratio, so each step looks between the poses the steps before it looked at. An overlap can't sit between two checks for long. Rewrites that are compared with each other look in the same places.
- **Who is checked against whom.** In a pose of a hinge's turn, every two pieces that hinge (or another turning then) moves against each other. The rest stand as they did at the end of an earlier turn, which was checked then.
- **How much.** Two pieces are sampled on a grid over the box where they can meet (`SampleOverlap`). A cell the fields show to be clear of either piece is skipped, one inside both counts whole, and one a surface passes through is split in eight and sampled again: an overlap thicker than half a cell can't fall between samples.
- **What it costs.** A point inside two pieces costs its depth, plus up to 2% of the mesh's size more, reached at a quarter of that depth. Depth alone makes thin overlaps nearly free (their cost goes with the square of their thickness); this way a shallow one already costs five times its depth. Over the sphere's volume times 2% of the mesh's size. The open pose counts fully; each hinge's poses share one weight.

### The opening order

With `orderHinges` (the default) the hinges open one after another, each in its own equal share of the motion; timing is then no parameter (`Carving::ordered`, `Joint::order`, and `Tidy` sets `t0`/`t1` from them). `Loss::BestOrder` finds the order greedily: from closed, the hinge whose turn overlaps least with everything as it then stands opens next, then the next freest with that one open, and so on. A hinge that needs another out of its way so comes after it. The order is looked at again at every rewrite round and at the end, and taken when the carving scores better with it.

Without it, each hinge's `t0` and `t1` are Adam's, and hinges may turn at once.

## Dependencies

With `dependencies`, every hinge but the first to open should be held shut by a piece that moves away before its turn. Then one hinge keeps the whole design closed, and letting it go lets the rest unfurl in order (the hinges are spring loaded).

- **Who can hold a hinge.** A piece that doesn't turn with the hinge, and that an earlier hinge parts from the hinge's parent (so it has gone by the hinge's turn; the parent itself never goes).
- **Where.** On the turned pieces' surface where it lifts off as the hinge opens, at least 0.4 of the farthest point's lever from the axis (a hold next to the axis holds little).
- **The term** (`Loss::Dependencies`). Per hinge after the first, how far the nearest such piece is along the sphere's surface, over the radius (1 when there is none). Its gradient is a stand-in: along the surface between the two nearest points, whatever lies outside both pieces pulls the nearer one toward it.
- **Helping it along.** New cuts are tried first where another piece's surface is within reach, and among turns that are equally free the opening order prefers one that can be held.
- **The ledges** (`AddLedges`, when the search is done). For each hinge after the first, a small tab is cut from the sphere itself: it becomes part of the holding piece and lies over the piece the hinge turns. Material changes hands, none is added, so the closed surface stays flush. A ledge is kept only if the hinge runs into it when it starts to open with the holder still in place, and nothing else runs into it during the motion. The result says how many hinges got one; the rest are named.

## Finishing

The same passes the Generate tab's Finishing section runs on any design (`data/finish.hpp`). After a search, hinges and carving are on by default, and with hinges built the result has to print:

- **Join what nothing holds** (`FuseUnheld`). A piece whose hinge couldn't be built (its two pieces don't meet along the axis), or whose hinge doesn't turn, becomes part of the piece it hung from.
- **Cut loose crumbs away** (`RemoveLoose`). Whatever of a body isn't joined to the part its hinge is on would fall out of the print: it is cut away.
- **Check it prints** (`CheckPrintable`, `tools/autodesign/printcheck` for saved designs). Every body is one solid, and every body but the root hangs from a hinge whose knuckles are part of both bodies' solids. The cuts that make room around a hinge can leave its knuckles on a stub nothing joins to the piece; then the finishing is done again with leaves twice, then three times as long (they reach further into the piece, and no cut reaches hardware). A hinge that still doesn't hold is given up and its piece joined to the one it hung from. The message ends with the verdict.

The passes themselves:

- **Carve collisions away** (`CarveCollisions`). The motion is checked at `checksPerHinge` poses of every hinge's turn. Whatever is inside two bodies is covered with rounded boxes (the overlap's own box along its principal axes, split while it is mostly empty), `clearance` larger all round, and cut from the body it shows less on when closed. Then it checks again between those poses and carves what is left, up to `rounds` times. It works on any design; nothing is added anywhere, so nothing sticks out of the sphere.
- **Add hinges** (`AddHinges`): each hinge's hardware; see Hinges.
- **Add ledges**: as above, for a carved sphere.
- **Add latch** (`AddLatch`). A press latch on the hinge that opens first (or the selected one): a small rocker printed into the surface, flush with it. A thin web joins its leg to the body below; pressing the arm's far end in turns the rocker about the web, and its catch steps back off a lip on the other body. Where the body that stays down is deep enough at the seam, the rocker is in it and its catch lies *over* a lip on the lid's edge. Where it isn't (a thin edge), the rocker is in the lid and its catch hooks *under* a lip left at the surface of the other body; its web is then under the catch's tip where that is the lid's own material, and just inside the seam where it isn't (the lip gets a little more room over the catch there, because the catch rises slightly as it steps back). Either way it is cut from the sphere, with `clearance` around it. A place is taken only if the lid runs into the catch when it starts to open, and, with the rocker pressed, clears it all the way open. The web is made as long as the body is deep there (the longer, the less it is strained); the result says how far it bends. The rocker needs about 0.6 of the sphere's radius of depth at the seam in one of the two bodies: where neither has it, there is no latch, and the result says so.

## The silhouettes on the GPU

`AutoParams::gpu` is a `SilhouetteTracer` (`tracer.hpp`). With one, each evaluation hands every piece's field over as GLSL and gets back, per pixel of every view, the piece that covers it most, how near its ray comes and where: what `Loss::Trace` finds on the CPU, marched the same way step for step. The comparison with the mesh and the gradient (which re-evaluates fields at edge pixels and overlap samples) stay on the CPU, and so do the DINOv3 pictures.

The editor's tracer (`ui/gputrace.cpp`) draws with the editor's own OpenGL context. That belongs to the UI thread, so the search hands each request over and waits; the UI thread answers once a frame and stays at it for most of the frame while a search keeps asking. The window redraws about a dozen times a second meanwhile. All views go into one float target, one rectangle per piece; a piece's numbers are uniforms, so shaders are compiled only when a rewrite changes a piece's structure. If the tracer can't draw (no float targets, a shader that won't build, the UI thread busy for two seconds), that evaluation is drawn on the CPU.

How much it gains depends on how much of a step is drawing: with 5 pieces, about 40% of a step at 64 pixels a view and 70% at 256 (measured with a software renderer, so the share is all that can be said here).

## Hinges

A hinge has to sit where its two pieces meet, over the whole length its barrel takes up, or nothing can be built on it. For each joint the term takes three points on the axis (the pivot, and half a barrel's length either way: `PlanHinge` in `data/finish.hpp`) and adds how far each of the two pieces is from each, beyond `HingeSlack` (half of what a knuckle's leaf can bridge: 0.6 mm at the default size and reach); the mean of the three, over the mesh's size. Both pieces are in the closed pose, the frame pivots are given in.

It used to look at the pivot alone, with 1% of the mesh's size to spare. A pivot is a point: the axis through it was free to turn away from the seam, and a pivot could float that far off both pieces, so hinges came out joined at one spot or not at all. Raising its weight didn't change what it measured.

This term is also what limits the search most. With it ignored (`hingeWeight` 0) the same run finds twice the pieces and far better cover, on axes tens of millimetres from the pieces they turn. `hinges.reach` is the honest way to give the search more room: the pieces may then be that far from the axis, and the hinge is built with leaves that long.

With `buildHinges` the search plans for the hardware `AddHinges` builds when it is done:

- **Under the surface** (`hinges.flush`): the term also adds how far the pivot is from being deep enough for the barrel to be inside the sphere, and a new lid's pivot starts that deep, in from its rim until the lid is within reach.
- **Room to turn.** An axis under the surface has its two pieces in each other's way next to it as they turn, whatever their shapes. `AddHinges` cuts that room (`HingeSizes::relief`), so overlaps between a hinge's two pieces within that distance of its axis aren't counted as collisions.

### The hardware (`AddHinges`)

The smallest barrel that prints: a 1.6 mm pin, the clearance around it and a 0.8 mm wall (3.8 mm across), three knuckles 2 mm long each; the parent has the outer two and, printed in place, the pin. `hinges.scale` scales it. For each hinge:

- **Where along the axis**: where both bodies are nearest the axis over the barrel's whole length. Sliding the pivot along its axis changes nothing about the motion.
- **Under the surface** (`flush`, a carved sphere): the axis moves toward the sphere's centre until the barrel is inside the sphere. On a design that wasn't searched with `buildHinges` this moves the axis by the barrel's radius or more, and the lid is often not there any more: the hinge is then not built, and named. Without `flush` the barrel is on the axis as it is, and sticks out of the sphere when the axis is at its surface.
- **Leaves**: each body's knuckles are tied in by a short plate, pointed where it finds the most of its own body and none of the other, inside the sphere.
- **Room**: the other body is cut away around each leaf, any third body around the barrel, and (flush) the two bodies where they would rub next to the axis as they turn.
- **A separate pin** (`pin`): both bodies are bored for a pin pushed in afterwards; under the surface, a channel is cut from the pin's end out of the sphere to push it in by.

A hinge is built only if both bodies are within a leaf's reach of the axis there. The result gives the sizes, what was moved and cut, and the motion's overlap before and after. Carving collisions away afterwards never cuts hardware: what a knuckle or leaf runs into gives way, and next to a hinge each body gives way along the other's knuckles, so no body is cut loose from its own.

## The gradient

The silhouettes and collisions depend on the carving only through the pieces' distances at certain points: where the rays of edge pixels come closest to a piece, and the samples inside overlaps.

Silhouettes are soft. A pixel's coverage comes from how close its ray gets to the piece: 0 at a pixel outside, 1 at a pixel inside. So only pixels on a silhouette's edge depend on the parameters. The ray is sphere traced; on an edge pixel the steps only bracket its nearest point, so a short golden-section search closes in on it.

For each of those pixels:
- the loss's slope with respect to the pixel is known from the comparison with the target;
- the pixel's slope with respect to its ray's nearest distance is the edge's slope;
- that distance changes, to first order, only through the piece's field and pose at the ray's nearest point (the envelope theorem).

The last part is measured by nudging one parameter either way, building the carving again, and re-evaluating the distances at those points: for a cutter, the fields of the pieces it shapes; for a joint, the poses. Collision samples work the same way: each sample's slope goes to the piece whose distance sets the overlap depth there. The times of the poses between are held fixed while the gradient is taken.

`tools/autodesign/gradcheck` compares this gradient against brute-force differences of the whole loss, for a carving of three pieces. Over the cutters' and joints' parameters it points the same way (cosine 0.995 when last measured). The placement's four are themselves differences of the targets, so they agree with brute force at the same step. Run with `MINCOVER=0` to leave out the coverage penalty's kink, which brute force with a large step jumps across.

## Extending it

**A rule** is a function `bool (Carving&, const RuleContext&)` plus a row in the table in `grammar.cpp` (name, default weight, function, and whether it simplifies). Change the carving and return true, or return false when the rule can't apply. It shows up in the UI's Grammar section by itself. Any rule that only adds or removes cutters, splitting or merging pieces as above, keeps the closed pose the sphere.

New parameters are picked up automatically. Adam gives them fresh memory, and drops the memory of parameters that disappear.

**A differentiable number** needs a row in its kind's table in `carving.cpp` (its unit and range), and a case in `Ref`. The gradient needs nothing more: it nudges whatever `Ref` points at and builds the carving again.
