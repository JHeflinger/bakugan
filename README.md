# foldball

Turns a mesh into a hinged shape that folds into a ball. It works backwards from the ball: a sphere is cut by planes
into bodies joined by hinges, and the cuts, the hinges and how far each one opens are optimized so that, unfolded, it
looks like the mesh. Closed it is the sphere by construction (it rolls), whatever the optimizer does. The
optimization is Adam over every continuous parameter, inside a stochastic rewrite descent over the design's structure
(split, merge, join, carve, uncarve). C, raylib for the window, nfd for file dialogs; nothing else.

## Building

With [tiny](https://github.com/JHeflinger/tiny) (the `.tinyconf` pulls raylib and nfd from
[minis](https://github.com/JHeflinger/minis)):

    tiny            # -> build/bin.exe

On Linux raylib's module links `glfw`, `GL`, `GLU`, `pthread`, `m`, and nfd's GTK 3 (Arch: `glfw mesa glu gtk3`). The optimizer core
(`src/core`, `src/opt`, `src/post`) doesn't use raylib at all; only `src/app` does.

## Using it

    build/bin.exe                       # the GUI
    build/bin.exe gui mesh.obj          # the GUI with a mesh (or a .fold design) open
    build/bin.exe help                  # the command line

Headless (for a server, tmux, ...):

    foldball optimize mesh.obj --steps 4000 -c settings.ini -s loss.surface_weight=2 --post
    foldball optimize --start run/best.fold --steps 2000      # carry on from a saved design
    foldball post design.fold -o design_built.fold            # hinges, carving, ledges, latch
    foldball export design_built.fold -o parts/               # STL parts (mm)
    foldball preview design.fold -o design.png                # a picture, open to closed
    foldball config -o settings.ini                           # every setting, with what it does
    foldball suite meshes/ -c settings.ini --steps 2000       # every .obj in a folder, into one video

While `optimize` runs, type a letter and Enter: `p` pauses and resumes, `s` prints the status, `w` writes the best
design so far now, `q` stops (keeping the best so far). Ctrl-C is the same as `q` (twice quits at once); `kill -USR1`
writes the best design too. The run's folder (`foldball_out/<mesh>_<n>/`) keeps `status.txt`, `best.fold` and
`best.png` up to date, so another shell can look in at any time. At the end it writes `<mesh>.fold`, `<mesh>.png`,
`report.txt`, `history.csv` and (with `--post`) the built design and its STL parts.

In the GUI: Load mesh, Start (Pause / Resume / Stop), watch the design (drag to orbit, the pose slider or
"animate" moves it between closed and open; checkboxes show or hide the target, the hinges, the closed sphere and
the floor grid), the loss over the steps, and the pictures the loss compares (red the
design, green the target, white both). Load design starts from a saved design; Save design and Save / Load settings
keep them. Post process and Export STL finish it. Every setting is in the panel (hover for what it does).

## The test suite

Point it at a folder of .obj files and a settings file (`foldball suite <folder> -c settings.ini`, or in the GUI's Test
suite section: a folder, and the current settings or an .ini). Every mesh is optimized in turn with those settings and
recorded: the best design so far at `suite.frames` evenly spaced steps (open, the mesh faint behind it, turning slowly,
with the step, loss and coverage), then the result folding shut and opening again (`suite.fold_frames`; with
`suite.post` it is post processed first). At the end every mesh's recording goes side by side into one square-ish
video, `suite.mp4` (`suite.tile` pixels a mesh, `suite.fps`; ffmpeg must be on the PATH: Arch `pacman -S ffmpeg`),
with `suite.png` (the last frame), `summary.csv` (a row per mesh: loss, coverage, parts, hinges, time) and per mesh
`<stem>/<stem>.fold` and `<stem>.png`, all in `<output.dir>/suite_<n>/` with the settings used.

While it runs: `p` pauses, `n` ends the current mesh's run (keeping its best so far) and goes on to the next, `q`
stops (the meshes done so far still make the video), `s` prints where it stands; the GUI has the same buttons, shows
each mesh's best design in the view and the latest video frame in the panel. Without ffmpeg the frames are kept
(`<stem>/frames.rgb`, raw RGB) and it says so.

## How it works

**The design** (`src/core/design.h`). A sphere of radius 1 (the export size is `size_mm`) cut by planes: every body
is a union of cells (intersections of half spaces) intersected with the sphere, so closed, the bodies partition the
sphere exactly. Everything is an exact analytic distance field (the sphere, half spaces, capsules): there is no grid,
so nothing is limited by a resolution but the meshes made for drawing and printing. Material can be taken from the
sphere by *hollows* (capsules), and every bit of the sphere's surface a hollow takes costs (`surface_weight`): the
ball should stay a ball, to roll.

**Bent cuts** (`model.max_bend`, 0 by default: flat cuts only). A cut can bend into a piece of a sphere of radius
1/|k| through the plane's own point n d (its centre n (d + 1/k): k > 0 bulges toward -n, k < 0 toward +n), and k is
one more parameter Adam moves (`optimizer.lr_bend`), within max_bend / R (1: as round as the ball). Its field is the
exact signed distance to that sphere, written as (2a - k|u|^2) / (1 + sqrt(1 - 2ak + k^2|u|^2)) with a = n.q - d and
u = q - n d, so it is exactly n.q - d at k = 0 and has no 1/k to blow up near it; the gradient in k at flat,
-(|u|^2 - a^2) / 2, is what lets a cut start to bend. Every cut starts flat, so nothing jumps when one is made. Two
spheres meet in a circle, so a bent cut's seam is still a circle (in the plane n.q = (k(r^2 + d^2) + 2d) / (2(1 + kd)))
and its hinges sit exactly on it, the axis along it. A bend is held back where the cut would stop crossing the hinges'
sphere (it would close into a bubble inside the ball, which no hinge could reach).

A hinge sits on the seam of the plane it was cut along: where that plane meets the sphere, inset by the knuckle
radius (`knuckle_radius_mm`), its axis along the seam. Its parameters are where on the seam (an angle, `phi`) and how
far it opens; its knuckles are tangent to the sphere's surface, touching both bodies, by construction. The closed
pose is the design as it is; opening turns every hinge by its angle, one after another in the opening order. Files:
`.fold` (text, with any carving grids from post processing as raw floats after it).

**The loss** (`src/opt/eval.c`). Only the open pose is scored (closed is the sphere): it is drawn from several views
as soft silhouettes and depth and compared with the target drawn the same way. A ray's coverage is
sigmoid(-dmin / k), dmin the nearest it comes to the union of the opened bodies. Where the open pose sits and how big
it is relative to the mesh doesn't matter, so that is learnt too (a scale and an offset). How the views are compared
is `loss.fit_look`:

- `0` silhouettes: per pixel |D - T| (coverage), over the target's coverage: outlines only;
- `1` (the default) silhouettes, plus where both cover a pixel how far apart their front surfaces are along the view
  (robust: d² / (d² + scale²)), weighted by `depth_weight`;
- `2` depth rasters instead: each view is a picture of nearness, coverage times 1 at the front of the frame down to
  0.25 at its back (so covered never looks like empty), and the pictures are compared pixel by pixel (|vD - vT| over
  the target's sum). Outline and depth are one comparison: a piece in the right place but too far back costs as a
  piece partly in the wrong place does. Its gradient goes through the edges' coverage (as silhouettes) and through
  where each ray enters the open pose (its depth moves by -1 / (grad f . ray) as the field there rises).

Then:

- hinges: points along each barrel must lie within both of their bodies (where on the seam it sits decides this:
  a body can be cut away from part of its seam);
- collisions: bodies overlapping, open and through every hinge's turn (closed they can't). Each turn is checked at
  a fixed number of poses (`checks_per_hinge`), so making a turn faster dodges nothing. The weight grows over the run
  (`collision_ramp`). The opening order is re-sorted, least colliding first, when that does better;
- holding: every hinge but the root (the first to open) should have a body next to its piece, closed, that opens
  before it, so a ledge can hold it shut until that body has moved away;
- one piece: a body's material apart from its largest connected part costs (and is pushed away);
- surface: hollows breaking the sphere's surface cost.

The gradient is closed form, not finite differences of the loss: every term hands out *slopes* (a point of a body in
some pose, and d loss / d field there; for a silhouette pixel, the ray's nearest point, by the envelope theorem), and
one pass turns them into every parameter's gradient: the cut planes and hollows through whichever of them decides the
field there, the hinges through the bodies' transforms, the placement through the pictures.

**The grammar** (`src/opt/grammar.c`). Every `steps_per_round` steps a few rewrites are tried; each try (and the
design left alone) gets `lookahead` Adam steps, and the best is kept (a rewrite when it's no worse than leaving the
design alone by `rewrite_margin`).

- *split*: a body cut by a plane into two pieces, each in one piece (directions along the axes, the body's principal
  axes and at random; offsets from near a tip to halving), with a hinge on the new seam where both halves are right
  at its barrel. Of the candidates, the cut and opening angle the loss likes best (searched outright: a piece has to
  swing well out before it covers anything, so the pull on a hinge at angle 0 says little).
- *merge*: a hinge that opens less than `merge_angle_deg` goes, its child body becomes part of its parent.
- *join*: a body under `join_share` of the sphere becomes part of its parent (the sphere stays whole); the hinges
  that hung from it hang from its parent, each re-angled to take its piece as near as it can to where it was.
- *carve*: a hollow starts, at nothing, somewhere in a body; Adam grows it where it helps.
- *uncarve*: a hollow goes.

**Semantic deformation** (`src/opt/semantic.c`, `[semantic]`). Every `every` steps the target (the input's pose) is
deformed as rigidly as possible (ARAP, on a simplified cage of the input, the full mesh carried along) toward the
design's open pose (placed in the mesh's space), in rounds of soft constraints; the deformed target is kept when the loss is lower with it and it
is judged the same shape. Semantic equivalence is geometric here, with no learned model: locally rigid (mean
stretch under `max_strain`), the same volume (within `max_volume_change`), and not run through itself. The
deformation always starts from the input as loaded, so it can't drift.

**Feature restoration** (`src/post/features.c`, `post.features`, off by default). The optimizer only makes the
parts' outlines match the mesh; this puts the mesh's own surface back onto them. Per part, on a grid of
`feature_resolution` points across the sphere (kept per part as an offset to its field, so nothing else changes), the
mesh's exact signed distance is measured where each point sits in the open pose (the semantic deformation's target, if
one was kept). Where the mesh's surface lies within `feature_reach_mm` of the part's, the part takes it; farther, the
part is left alone (blended smoothly between). Taking material away is always allowed; adding it only where it
collides with nothing: not within the clearance of another part, closed or at any pose of a hinge turn that moves the
two apart, and never past the sphere. Where a feature doesn't fit (and on the sphere's surface for what is taken away):
- `feature_flatten` (0..1): how much it is flattened (0 full height, 1 flat);
- `feature_sink` (0..1): what still sticks out is pressed in: the surface within `feature_sink_mm` of it sinks with
  it, so the feature sits whole in a recess, flush (1), or is squared off flush (0);
- whatever still doesn't fit is cut off.

So closed it is still a sphere, carrying the mesh's features as a texture. It runs first in post processing (before
the hinges, carving, ledges and latch, which then work around it); on the 15-part test design it took about 50 s at the
default 192 points (0.3 mm for a 60 mm sphere).

**Post processing** (`src/post/post.c`): fragments removed; hinges built on their seams (three knuckles alternating
between the bodies, tangent to the sphere and clipped to it, a pin fused into the parent's, a bore in the child's,
room cleared around each in the other); what still collides along the motion carved away smoothly from the smaller
body with a clearance (a swept distance grid per body, `carve_resolution` across the sphere); ledges added where a
hinge isn't held yet (on a body that opens first, reaching over the hinge's piece into a recess, checked to hold it
and to let go); and a press latch on the root hinge, flush with the sphere (a flexible tab with a hook under a lip of
the other body). Nothing added ever reaches past the sphere. STL export meshes every body by dual contouring
(`src/post/mesher.c`): vertices where the planes the surface crossings lie on meet, so cut edges and corners come out
sharp at any resolution.

## Tests

`tests/` has small programs against the core (not built by tiny): `gradtest` checks the closed form gradient against
finite differences, `opttest` runs an optimization and writes a picture, `coretest` checks that closed is exactly the
sphere, that hinges sit on their seams, and the files:

    gcc -std=gnu11 -O2 -Isrc tests/gradtest.c src/core/*.c src/opt/*.c src/post/*.c -lm -lpthread -o gradtest
    ./gradtest mesh.obj [collision depth hinge surface]

## What to expect

The ball is guaranteed; how much the unfolded shape looks like the mesh depends on the shape. Long thin parts (a
figure's arms and legs) need several splits and large turns, and a sphere has only so much surface to peel them from,
so expect the likeness of the overall shape and proportions rather than fine detail. A run of 3000 steps on a T-posed
figure takes a few minutes. Raise `surface_weight` to keep the ball smoother, lower it to let hollows shape the open
pose more. Post processing reports what it couldn't do (a hinge with no place for a ledge, no latch position) rather
than failing. Designs from earlier versions (`FOLDBALL 1`) can't be loaded: the representation changed.

## Code

    src/core   math, meshes (OBJ), distance grids (the target, carving), the design, settings (.ini), threads
    src/opt    the loss and gradient, Adam, the grammar, semantic deformation, the optimization loop
    src/post   meshing, STL, pictures (PNG), post processing
    src/app    the command line, the GUI (raylib), the test suite (suite.c: no raylib)
