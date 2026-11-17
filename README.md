# Bakugan Hinges

Design hinged objects out of superfrustums, check that they close without parts running into each other, and export
the parts as STL files for 3D printing. Built with raylib + Dear ImGui (rlImGui), ImGuizmo, nativefiledialog (GTK3 on Linux), OpenVDB
and oneTBB, using [tiny](https://github.com/JHeflinger/tiny).

Units are millimetres. Y is up in the app; STL export converts to Z up for slicers.

## Concepts

- **Shape**: one superfrustum (rounded-rectangle section swept along its local Z, with taper, bulge, dilation and onion
  shell). A shape either *adds* material or *cuts* it from the rest of its group.
- **Body**: a top-level group. It is one rigid object: it turns as a whole on its hinge and prints as one part.
- **Feature**: a group inside a body. Its shapes are unioned (optionally with a smooth *blend*) and the result is added
  to, or cut from, the enclosing group, so several superfrustums can act as one design element (a knob, a lip, a slot).
- **Hinge**: joins a parent body (stays) and a child body (turns) about an axis through a pivot. Angles are relative to
  the pose you modelled (0): an **open** angle and a **closed** angle, up to a full 360 degree turn apart. *Moves during*
  sequences hinges within the closing motion.
- **Door hinge hardware**: knuckles alternating between the two bodies, leaves tying them into their bodies, clearance
  cut around the other body's knuckles, and either a **separate pin with a head** (bodies print apart, push the pin in)
  or a **fused pin** (print in place, the joined bodies print together).

## Foldaball CLI
Foldaball has functionality as both a UI design space or as a command-line application. You can run the following commands:
```
foldaball optimize bunny.obj -c settings.ini -o results/bunny     # one mesh (the Generate button)
foldaball suite meshes/ -c settings.ini -o results/suite           # every .obj in a folder
foldaball score results/bunny/bunny.txt bunny.obj                  # score a saved design, no search
foldaball config > settings.ini                                    # every setting, with its value and a comment
```
### CLI Config
Settings for Foldaballs generated via the CLI can be configured through an INI file. Config file can be selected with the `-c` flag.
```ini
[optimizer]
steps = 6000
seed = 7

[goal]
views = 8
dependencies = true
```
`foldaball config` prints every setting with its current value, a comment and its
allowed range. To check the validity of a config file, run `foldaball config -c myfile.ini`. 

### Results

`optimize` writes to `-o DIR` (default `./foldaball_out`), each named after the mesh (`<name>`):

| file | what |
|---|---|
| `<name>.txt` | the design: open it in the editor |
| `<name>.png` | the result open, a third and two thirds closed, and closed |
| `<name>.report.txt` | the message and score in words |
| `<name>.result.json` | the same, for scripts |
| `<name>.history.csv` | loss at every step |
| `<name>.views.png` | the silhouettes scored last, against the placed mesh's |
| `<name>.config.ini` | the settings that were used: running with this file repeats the run |
| `<name>.partial.txt` | the best so far, saved every `checkpoint_seconds`; removed when the run finishes |
| `<name>_<pose>_<part>.stl` | with `output.stl = true` |

`suite` writes to `-o DIR` (default `<folder>/autodesign_suite`): per mesh `<name>.txt`, `.png`, `.report.txt`, plus
`summary.csv` (one row per mesh, written as each finishes), `suite.config.ini` and, with `suite.video = true`, `suite.mp4`.
**ffmpeg is only needed for the video**: `--no-video` (or `suite.video = false`) skips it, and so the check for it.

## UI Application

If the command is run with no arguments the visual design space will open rather than the cli.

### Views

- **Design** (left): the shapes as modelled. Gizmo for shapes, whole groups / multi-selections, and hinges
  (pivot / axis). Collision spots are marked in red with the share of the motion where they happen.
- **Output** (right): the built parts (CSG + hinge hardware, meshed with OpenVDB) moving from open to closed. The
  timeline shows collisions in red; colliding parts flash red with the contact points. With a hinge selected, scrub to
  open or closed and drag the ring to set that angle.

**Reference mesh** (Model tab, the *Mesh...* toolbar button, or drop a file on a view): import a rough shape to fit
superfrustums to by hand. *Solid* draws it opaque, so a superfrustum shows only where it pokes out of the mesh: you see
at a glance where your shapes overshoot. *X-ray* also shows the buried parts of the shapes through it, *Ghost* is
see-through. Scale, offset, rotation and colour are adjustable; shapes stay clickable through the mesh. It is never
exported or collision-checked.

The closing motion is re-checked after every edit: each body's surface is tested against every other body's signed
distance field at evenly spaced moments (Hinges tab: *Checks*, *Tolerance*).

### Generate

The Generate tab carves a sphere into hinged pieces that, opened, look like the reference mesh (`src/autodesign/README.md`
has the details). For something that can be printed and played with:

- **Collisions**: every hinge's turn is checked at the same number of poses however fast it turns, and the hinges open
  one at a time, the freest first. *... by the end* is how much more overlaps cost at the end of the search.
- **Hinges hold each other shut**: every hinge but the first to open is held by a piece that moves away before its
  turn (small ledges cut from the sphere's surface). One hinge then keeps the design closed.
- **Hinges**: *Build the hinges when done* gives every hinge the smallest barrel that prints (three knuckles on a
  printed-in-place pin), under the sphere's surface so the closed sphere stays round, and has the search keep each
  axis where one fits. *Add hinges* does the same on the current design (the selected hinge, or all).
- **Finishing**: *Carve collisions away* cuts out whatever still overlaps on the way; *Latch* puts a press latch, flush
  with the sphere, on the first hinge. Both run when a search is done, or on the current design with the buttons.
- **Draw the silhouettes on the GPU** (Optimizer): the search's views are raymarched by the graphics card. The editor
  redraws only a dozen times a second while it runs.

### Keys

| | |
|---|---|
| Click / Ctrl or Shift+click | select / add to selection |
| Q | pick shapes, features or bodies |
| W / E / R | move / rotate / scale |
| X | toggle cut on the selected shapes |
| Ctrl+G / Ctrl+Shift+G | group / ungroup |
| Shift+A, Shift+D, Del | add box, duplicate, delete |
| Space | play close / open |
| F | frame the view |
| Ctrl+Z / Ctrl+Shift+Z, Ctrl+Y | undo / redo |
| Ctrl+S | save the design |
| RMB or Alt+LMB drag / MMB or Shift+RMB drag / wheel | orbit / pan / zoom |
| 1 3 7 (Ctrl: opposite), 5 | axis views, orthographic |

### Export

Print tab: writes `<prefix>_<part>.stl` per body and per separate pin, optionally `<prefix>_assembly.stl` with every
part, and `<prefix>_notes.txt` (part sizes, hinge table, print notes). Parts are exported in the open, closed or
modelled pose; *Keep bodies apart* trims bodies that touch in that pose so parts printed together don't fuse. Keep the
print voxel well below the hinge clearance.

## Building

```
./scripts/get-tiny.sh
./scripts/linux/install-deps.sh     # or scripts/macos/setup.sh, scripts/windows/install-deps.sh (MSYS2 MINGW64)
./.tools/tiny -f
```

Designs are plain text (`bakugan-hinges 1`): `group`, `shape` and `hinge` lines, names last on each line.
