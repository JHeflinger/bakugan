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

## Views

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

## Keys

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

## Export

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
