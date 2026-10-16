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

Load a model from the library (two demo creatures and the raylib primitives), or drop an `.obj/.gltf/.glb/.iqm/.vox/.m3d`
onto either view. The model is normalised to 2 units, and its volume sets the target sphere: the radius of the sphere
with the same volume.

### Automatic tab

1. **Superfrustum decomposition.** Splits the shape into superfrustum primitives.
2. **Fold into a sphere.** Builds a hinge tree and optimises it with grammar rewrites. *Size leeway* is how far the
   ball's radius may stray from the equal-volume radius.
3. **Restore detail.** Builds the final pieces and exports them.

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
- **Fold analysis (live)** re-scores the design on every edit.
- **Refine with optimiser** runs the automatic fold optimiser starting from your design. Tick *Keep my hinge tree* to
  only tune it.

Everything is undoable (Cmd/Ctrl+Z, Shift+Z / Y).

The **Files** panel does the following:
- Saves and loads designs as text (Cmd/Ctrl+S saves; dropping a `.txt` loads one).
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
- **Hinge hardware.** Hinges are modelled as ideal revolute joints. Pins, springs and clearances still have to be
  designed for printing.
