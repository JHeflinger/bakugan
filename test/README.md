# Rig Editor

A two-panel 3D editor built with raylib and Dear ImGui (via rlImGui).

- **Left panel (Rig):** drop a model, then build and edit skeletons made of ball joints.
- **Right panel (Reference):** drop a model to view it. It shows the left panel's armature live and lets you **pose** it. Dragging a joint uses inverse kinematics: its parent joints bend to follow, bones never stretch, and ball-joint limits are respected. Dragging the root moves the whole armature. **IK chain** limits how many parents bend, **Rotate (E)** rotates a single joint, and **Reset pose** clears the pose. The bones themselves are only edited on the left. **Fit armature to this model** (under View...) scales the armature to this model's size.

## Build & run

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/rig_editor
```

CMake fetches raylib 6.0, Dear ImGui 1.92.9b, rlImGui and ImGuizmo on the first configure.

## Usage

The left panel holds one skeleton. It always has a single `root`, and every other bone is
a child of an existing joint, so the skeleton is always one connected piece.

| Action | How |
| --- | --- |
| Load a model | Drag an `.obj .gltf .glb .iqm .vox .m3d` file from Finder onto a panel, or drag an item from the sidebar library |
| Select a joint | Click it in the left panel, or in the sidebar hierarchy |
| Add a child bone | Select a joint, then **Add child bone** (or **C**). The new bone continues in the parent bone's direction. **Ctrl/Cmd+click** places it where you click (inside the mesh when you click the model) |
| Move a joint | **Move (W)** gizmo: drag an arrow or plane handle |
| Rotate a ball joint | **Rotate (E)** gizmo. The **Swing** and **Twist** limits constrain it (yellow cone) |
| Gizmo axes | **Local axes** switches between the joint's own frame and world axes |
| Move subtree | Off: children stay in place and the bones stretch. On: the children move with the joint |
| Split a bone | Select a joint, set the number of pieces, then **Split bone into pieces** (or right-click → **Split bone in 2**). Joints are added evenly along the bone from its parent; nothing moves |
| See bones inside the model | **X-ray** (on by default in the left panel) plus the opacity slider next to it |
| Snap the view to an axis | Click a ball on the axis widget (top-right of the panel); click it again to view from the other side. Drag the widget to orbit. Keys **1 / 3 / 7** = front / right / top (Ctrl/Cmd: opposite side). Snapping switches to orthographic and orbiting switches back; **5** toggles it yourself |
| Delete | **Del / Backspace** or right-click in the hierarchy. Removes the bone and its children (the root can't be deleted) |
| Start over | **Clear bones** (keeps only the root) or **Humanoid template** |
| Undo / redo | **Cmd/Ctrl+Z** / **Cmd/Ctrl+Shift+Z** (or Ctrl+Y), or the sidebar buttons. Covers skeleton edits on the left and posing on the right; a whole drag is one step |
| Camera | RMB or Alt+drag: orbit · MMB or Shift+RMB: pan · wheel: zoom · **F**: frame |
