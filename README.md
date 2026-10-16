# Bakugan Everything

A visual design tool that turns any closed mesh into a Bakugan: a figure that folds into a sphere on spring-loaded,
one-way hinges and pops back open.

It uses the same stack and conventions as before: raylib, Dear ImGui (rlImGui), ImGuizmo, and OpenVDB with TBB, built
with [tiny](https://github.com/JHeflinger/tiny).

The window has three areas: the inspector on the left, the **Figure** view (open pose) in the middle, and
the **Ball** view (fold preview) on the right.

## Building

```
tiny            # downloads the minis modules, builds build/bin.exe
tiny -r         # build and run
tiny -rv        # once after adding a module (tiny caches compiled vendors)
```

System packages: OpenVDB, TBB, Imath, Boost headers (for OpenVDB), GLFW and OpenGL, as for the original project. The
`.tinyconf` adds `-O2` because the optimisers are numerically heavy.

There is also a headless regression test of the whole data pipeline:

```
g++ -std=c++17 -O2 -Isrc -Ibuild/modules/raylib/include tests/pipeline_test.cpp src/data/*.cpp src/util/*.cpp \
    build/modules/raylib/lib/linux_amdx64_libraylib.a -lopenvdb -ltbb -lImath -lglfw -lGL -lpthread -ldl -lm \
    -o build/pipeline_test
build/pipeline_test full critter      # also: quick, and torus | ball
```

## Using it

Load a model in any of these ways:

- **Open model...** opens a native file dialog (tinyfiledialogs; on Linux it needs `zenity`, `kdialog` or `yad`).
- Type a path into the box beside it and press Enter or **Load**.
- Pick a demo (two creatures and the raylib primitives) from the library.
- Drop an `.obj/.gltf/.glb/.iqm/.vox/.m3d` file onto either view.

Drag and drop may not work on Wayland compositors such as Hyprland. raylib's bundled GLFW is X11-only, so the app runs
under XWayland, and drops from Wayland file managers don't always reach it. Use the dialog there.

OBJ files go through our own reader, which tolerates files without normals or texture coordinates (raylib's OBJ loader
crashes on some of those). The model is normalised to 2 units, and its volume sets the target sphere: the radius of the sphere
with the same volume.

### Automatic tab

1. **Superfrustum decomposition.** Splits the shape into superfrustum primitives.
2. **Fold into a sphere.** Builds a hinge tree and optimises it with grammar rewrites. *Size leeway* is how far the
   ball's radius may stray from the equal-volume radius.
3. **Restore detail.** Builds the final pieces and exports them.

The **Pipeline** choice decides how the parts and the fold are found:

- **Separate** fits the superfrustums to the mesh first, then optimises how those fixed parts fold.
- **Joint** starts from a coarse decomposition and then optimises the superfrustums and the fold together, so part
  layouts that fit the sphere better can win.

Every finished optimisation is added to **Compare runs**, with parts/hinges, shell coverage, detail kept, whether the
motion is collision free, superfrustum fit (IoU) and time.

Two options shape what counts as a valid Bakugan:

- **Single latch**: one latch on the root holds one hinge. Every other hinge is caught by a neighbouring hinge and
  springs open once that one has opened far enough, so a single release pops the whole ball.
- **Collision-free motion**: the finished pieces may not pass through each other anywhere between figure and ball.

**Run everything** does all three in a row. Every stage runs on a worker thread. The views show its progress live, and
**Stop** keeps the best result so far.

### Manual design tab

- **Superfrustums**: add, duplicate and delete parts, and edit them with gizmos: W move, E rotate, R scale.
  - Q switches the gizmo between the part, its hinge and the sphere.
  - The inspector sets every superfrustum parameter: size, rounding, taper, bulge, dilation and onion.
  - **Fit selected / Fit all** gradient-fits your primitives to the mesh.
- **Hinge tree**: parent each part, mark joints rigid, and set the pivot, axis, fold angle and mechanical stop.
  - **Snap pivot** moves the pivot to where the part touches its parent.
  - **Auto-aim** searches the axis and angle that best tuck the part into the sphere.
  - **Flip** reverses the fold direction; **Make root** re-roots the tree.
- **In the Ball view**, drag the ring around the selected hinge to fold it directly.
  - Space or **Pop open** releases the ball: the springs snap every hinge open, with a bounce.
  - **Shell gaps** shows uncovered (red) and double-covered (amber) sphere surface.
  - **Lost material** shows what gets trimmed or collides.
- **Release (latch chain)**: choose which hinge releases the selected one (or the latch, for a hinge on the root) and
  how far that hinge must open first. **Default chain** releases every hinge from the one it hangs from.
- **Fold analysis (live)** re-scores the design on every edit, including whether the motion is collision free and
  where it is blocked. In the Ball view, **Releases** draws the latch (red) and the release arrows (magenta), and
  **Contacts** shows pieces touching at the current fold fraction while you scrub the slider.
- **Refine with optimiser** runs the automatic fold optimiser starting from your design. Tick *Keep my hinge tree* to
  only tune it.

Everything is undoable (Cmd/Ctrl+Z, Shift+Z / Y).

The **Files** panel does the following:
- Saves and loads designs as text, by path or with **Save as... / Open design...** (Cmd/Ctrl+S saves).
- Exports `<prefix>_open.obj` (the figure), `<prefix>_closed.obj` (the ball) and `<prefix>_hinges.txt`, which lists
  every pivot, axis, fold angle and joint type.

## How it works

### SuperFrustums (`data/superfrustum.*`)

The primitive follows SuperFit (Ganeshan et al.), with the same 8 parameters: size (3), rounding, taper, bulge,
dilation and onion. It is written from scratch here:

- A rounded-rectangle cross section gives a 2D distance `u`.
- `(u, z)` is measured against a trapezoid. The slanted edge gives taper, and the inner edge gives the onion shell.
- An arc mapping of the `xz` plane bends the axis (bulge).
- Dilation offsets the result.
- The parts' poses are stored separately.

A small core with a large dilation is a sphere, which the decomposition uses to seed blobs.

### Decomposition (`data/decompose.*`), ResFit-style

1. **Voxelise.** The mesh becomes a dense level set (OpenVDB `meshToSignedDistanceField`).
2. **Seed (morphological shape decomposition).** Repeatedly take the thickest residual core: an exact Euclidean
   distance transform, erosion to 55% of the maximum depth, its connected component, then dilation back.
   - Each region is initialised by PCA, choosing the axis by elongation or flatness. Taper comes from the end cross
     sections and rounding from the fill ratio.
   - A region that one primitive explains badly (IoU < 0.66, such as a ring or an L shape) is split along its main
     axis and seeded recursively.
   - When only thin structures remain (plates, wings), they are seeded from connected components. Fringes left around
     fitted parts are skipped.
3. **Fit.** Every primitive is fitted at once by Adam on a soft-occupancy loss: a probabilistic union, plus an overlap
   penalty, with finite-difference gradients in parallel over the primitives.
4. **Prune.** Remove primitives whose removal lowers loss + parsimony. The residual seeds the next round.

### The fold model (`data/design.*`, `data/fold.*`)

- **Parts form a tree.** Each non-root part hangs on a one-way revolute hinge (pivot, axis, closed angle, stop),
  defined in the open pose.
- **The root stays still.** Folding by a fraction `t` rotates every hinge by `t·angle`.
- **The objective** is evaluated on samples of the input's material, each owned by its nearest superfrustum, plus
  curvature-weighted surface samples. Its terms are:
  - **Shell coverage**: every point of the sphere skin should be inside exactly one part's superfrustum when closed.
  - **Trimmed detail**: material that ends up outside the sphere, with surface detail weighted up.
  - **Collision**: material of two parts in the same place, closed and half way through the fold (a smooth kernel on
    a spatial hash).
  - **Pivot contact**: each pivot must lie in both parts it joins.
  - **Joints inside the sphere**: so no link of a chain gets cut away.
  - **Size prior**: towards the equal-volume radius, within the leeway band.
  - **Hinge cost**: each working hinge costs a little (simplicity).

### Shape grammar and Stochastic Rewrite Descent

The search follows "Design for Descent" (Kodnongbua, Zhang, Sharp, Schulz). Each continuous step is an Adam update
(finite differences, in parallel) on the sphere plus a random block of hinges: angles, axes, and every 4th step the
pivots.

Every few steps, K rewrites are proposed. Each one is settled with a local step (SRD's LocalOptimize) and kept only if
it lowers the objective. The compatible winners (disjoint parts, no cycles) are applied together, falling back to the
single best one if combining them is worse.

The rules follow the paper's design guidelines:

| Rule | Effect | Guideline |
| --- | --- | --- |
| Reparent | Moves a limb to another touching part; the new pivot is the contact, the angle is re-chosen so the limb lands where it was | reversible, jump-continuous |
| Split / Merge | Cuts a long part in two with a new 0° hinge (the union is unchanged), or fuses coaxial parts whose hinge is ~0° | reversible, jump-continuous |
| Lock / Unlock | Turns a ~0° hinge into a rigid joint and back | reversible, continuous |
| Axis snap | Switches the hinge axis to a principal axis of either part, keeping the pose | local control |
| Flip / Reseat | Searches the best angle in the other or the same fold direction | repairability |

After every step, a repair pass clamps the angles to their stops, the radius to the leeway band, and projects pivots
back into both parts.

### Single latch (`data/design.*`)

Each working hinge has a **trigger**: either the latch (exactly one "master" hinge, which must hang off the root,
where the magnet lever is) or a neighbouring hinge. Neighbours are the hinge it hangs from, its siblings, or its
children. A hinge's catch lets go once its trigger has opened a **release** share of its travel. The triggers form a
tree rooted at the latch, so one release opens everything.

From that tree the design derives a release timeline (`Design::Plan`):

- Each spring turns at a steady rate, so bigger folds take longer and ease into their stop.
- A hinge starts when its trigger reaches the release point.
- The fold slider and the pop-open animation follow this timeline, and closing is the same path run in reverse.

The optimiser treats the wiring as part of the design. The grammar gains **rewire** (hand a hinge's catch to another
neighbour) and **move latch** rules, and the release points are continuous parameters.

### Collision-free motion

This follows the trajectory-wide collision objective of Megaro et al.'s compliant-mechanism design tool (SIGGRAPH
2017), and the staged fold sequences and fold stops of Felton et al. (Science 2014).

The motion is checked with the **finished pieces**, not the superfrustums:

- **Pieces.** Each part's surviving material (inside the sphere, and not carved away by a higher-priority part when
  closed, exactly as the restoration does) plus samples of its shell plates.
- **Along the timeline.** Pieces are moved along the release timeline. Pairs that come closer than the sample spacing
  count as contact, except two kinds:
  - pairs that already touch in the open pose: the cut between neighbouring pieces;
  - the knuckle of the hinge joining two parts.
- **In the figure.** Shell plates must not sit inside another part's material in the open pose.

During optimisation, a few moments are sampled, re-jittered every step, plus the open pose and the last approach
into the ball. The kept result is judged at 32 moments. If the best design still isn't collision free, a **repair
phase** keeps optimising with the motion term boosted, changing only hinges and the release chain, until it is.
"Collision free" means at most 0.4% of the samples touch at any moment, which allows for sampling noise.

### Joint shape and fold optimisation

This follows Design for Descent, which optimises grammar structure and continuous parameters together rather than in
separate stages.

- **The objective** becomes fold + `wFit` × the decomposition's soft-occupancy fit loss.
- **Continuous steps** also move superfrustum parameters: pose, size, rounding, taper, bulge and dilation. Each
  candidate re-assigns the material to the nearest superfrustum through a cached distance matrix.
- **Extra grammar rules:**

  | Rule | What it does | Undone by |
  | --- | --- | --- |
  | Add part | Seeds a primitive by PCA on input material no superfrustum explains, glued rigidly to its nearest part | Remove part |
  | Remove part | Drops a small part; its material goes to its neighbours | Add part |

  Together with split and merge, these let it try different superfrustum configurations while it folds.

On the critter demo, with the motion constraint on, the joint pipeline finished with 77% of the shell covered by
superfrustums against 44% for the separate one. The restored sphere was 92% closed against 64%. Detail kept and
collision-free motion were about the same, at a slightly lower superfrustum fit (89% vs 90% IoU).

### Detail restoration (`data/restore.*`)

Each part's final piece is computed as a field in its open-pose frame:

```
piece = ( (input material ∩ the part's cell)  ∪  spars  ∪  (its share of the sphere shell) )  ∩  sphere
```

All of it is evaluated in the closed pose:

- **Cells** are nearest-superfrustum regions of the input's level set.
- **Spars** link the part's own hinge to its centre and to its children's hinges.
- **Shell shares** go to the closest closed superfrustum. Shell plates may reach a little past it to close gaps.

Pieces are built in priority order (the root first, then bigger parts). Each later piece is carved where an earlier
piece already sits in the closed pose. The superfrustums act only as bounding volumes: every surface that isn't on the
outer sphere is the original level set, so the detail is restored. OpenVDB `volumeToMesh` meshes each piece.

## Limitations

- **Thin structures.** Parts thinner than about two voxels (wings, ears) can be missed by the decomposition; raise
  *Resolution* or add a superfrustum by hand.
- **Volume budget.** A sphere with the same volume as the figure can't hold every limb without loss. The optimiser
  trades trimmed detail against collisions and shell coverage. Widen the size leeway or change the weights to move
  that balance.
- **Contact checks are sampled.** Motion and contact checks work at the sample spacing (about 1/30 of the model). Very
  thin clearances below that are not resolved.
- **Hinge hardware.** Hinges are modelled as ideal revolute joints. Pins, springs and clearances still have to be
  designed for printing.
