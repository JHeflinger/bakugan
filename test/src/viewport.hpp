#pragma once
#include "skeleton.hpp"
#include "imgui.h"
#include <string>
#include <vector>

enum class GizmoOp { Translate, Rotate };

// One 3D panel: an orbit camera rendering into a texture shown in an ImGui window.
// Only a viewport with rigEditable set has a skeleton the user can build and pose.
struct Viewport {
    std::string title;
    bool rigEditable = false;

    // render target (sized to the panel every frame)
    RenderTexture2D target = {};
    int texW = 0, texH = 0;                       // in framebuffer pixels
    float pixelScale = 1.0f;                      // framebuffer pixels per UI point
    Rectangle screenRect = { 0, 0, 0, 0 };        // where the image was drawn last frame

    // orbit camera
    Camera3D camera = {};
    float yaw = 35.0f, pitch = 20.0f, distance = 5.5f;
    Vector3 focus = { 0, 1, 0 };

    // model
    Model model = {};
    bool hasModel = false;
    std::string modelName;
    float modelScale = 1.0f;
    Vector3 modelOffset = { 0, 0, 0 };
    BoundingBox bounds = { { -1, 0, -1 }, { 1, 2, 1 } };   // world-space, after normalisation

    // display options
    bool showModel = true, wireframe = false, showGrid = true, xraySkeleton = true;
    float modelAlpha = 1.0f;

    // rig (only used when rigEditable): a single hierarchy that always has a root (joint 0)
    Skeleton skeleton;
    int selJoint = 0;                             // -1 = nothing selected
    GizmoOp gizmoOp = GizmoOp::Translate;
    bool gizmoLocal = false;                      // gizmo axes: joint's local frame vs world
    bool moveSubtree = false;                     // moving a joint carries its children along

    Viewport(std::string t, bool editable);
    void Unload();

    void SetModel(Model m, const std::string& name, Shader lit);
    void ClearModel();
    void FrameAll();
    Matrix ModelTransform() const;

    // Draws toolbar + image, the bone gizmo, and handles mouse/keyboard input.
    // screenRect is updated so callers can put a drag-drop target over the image.
    void DrawPanel();
    void Render(Shader lit, int viewPosLoc);

    void ResetSkeleton();                         // just a root, centred in the model
    int AddChildBone(int parent);                 // extends along the parent bone; returns the new joint
    int AddChildBoneAt(int parent, Vector3 worldPos);
    void DeleteSelectedJoint();                   // the root can't be deleted
    void MoveJoint(int joint, Vector3 worldPos);
    float JointRadius() const;

private:
    bool mouseCaptured = false;                   // a drag started in this viewport (orbit/pan)

    void UpdateCamera();
    Ray MouseRay(Vector2 local) const;
    bool RayModel(Ray ray, RayCollision* out) const;
    Vector3 PlacementPoint(Ray ray);
    int PickJoint(Ray ray) const;
    bool DrawGizmo(ImVec2 pos, float w, float h);
    void HandleInput(bool hovered, bool active, Vector2 local, bool gizmoBusy);
    void DrawToolbar();
    void DrawSkeleton();
};
