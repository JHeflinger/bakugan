# Rig Editor

A small C++ starting point for a model-rigging editor built with raylib and Dear ImGui.

## Current features

- Two independent 3D viewports: **Rig Viewport** and **Reference Viewport**.
- OS file-manager drag-and-drop into either viewport.
- Loads OBJ, GLTF/GLB, IQM, and M3D models using raylib's model loader.
- Multiple models can be dropped into each viewport.
- Left viewport has a skeleton-editing context menu.
- Add root bones at the mouse position.
- Add child bones under the currently selected bone.
- Click joints to select them.
- Delete selected bones.
- Rename bones and edit local position in the Skeleton panel.
- Independent orbit/zoom cameras in the two viewports.
- Bones are stored as a parent/child hierarchy with local positions, ready for adding rotation, bind-pose data, skinning, and serialization later.

## Controls

**Both viewports**

- Middle mouse drag: orbit
- Mouse wheel: zoom
- Drag a supported model file from your desktop/file manager onto the viewport

**Rig Viewport**

- Left click a joint: select bone
- Right click: skeleton context menu
- `Add Root Bone`: creates a root bone on the Z=0 construction plane
- `Add Child Bone`: creates a child of the selected bone on the construction plane

## Build

This project uses CMake FetchContent. A network connection is required the first time CMake configures the project because it downloads the dependencies.

```bash
cmake -S . -B build
cmake --build build -j
./build/rig_editor
```

On Windows, the executable will normally be under `build/Debug/rig_editor.exe` or `build/Release/rig_editor.exe` depending on the generator/configuration.

## Dependency versions

The scaffold pins:

- raylib 6.0
- Dear ImGui 1.92.7
- rlImGui `Raylib_6_0`

The rlImGui project publishes a matching Raylib 6.0 / ImGui 1.92.7 release.

## Next logical step

The skeleton currently represents the *rig design layer*, not actual mesh deformation. To turn it into a real rigging tool, the next layer would map the authored bones onto the dropped model's existing skeleton (where one exists), or generate a skinning skeleton and bind-pose matrices for models without one.
