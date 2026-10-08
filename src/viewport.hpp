#pragma once
#include "skeleton.hpp"
#include "imgui.h"
#include <string>
#include <unordered_map>
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
    float yaw = 35.0f, pitch = 20.0f, distance = 5.5f;    // degrees; pitch in [-90, 90]
    Vector3 focus = { 0, 1, 0 };
    bool ortho = false;

    // model
    Model model = {};
    bool hasModel = false;
    std::string modelName;
    float modelScale = 1.0f;
    Vector3 modelOffset = { 0, 0, 0 };
    BoundingBox bounds = { { -1, 0, -1 }, { 1, 2, 1 } };   // world-space, after normalisation

    // display options
    bool showModel = true, wireframe = false, showGrid = true, xraySkeleton = true;
    bool xrayModel = false;                       // draw the model see-through
    float xrayAlpha = 0.35f;

    // rig (only used when rigEditable): a single hierarchy that always has a root (joint 0)
    Skeleton skeleton;
    int selJoint = 0;                             // -1 = nothing selected
    GizmoOp gizmoOp = GizmoOp::Translate;
    bool gizmoLocal = false;                      // gizmo axes: joint's local frame vs world
    bool moveSubtree = false;                     // moving a joint carries its children along
    int splitSegments = 2;

    // live, read-only view of another viewport's armature (the right panel mirrors the left)
    const Viewport* mirrorOf = nullptr;
    bool showMirror = true;
    bool fitMirror = true;                        // scale/place it by the two models' bounds
    Matrix MirrorPlacement() const;               // mirrored armature space -> this viewport's

    // Pose of the mirrored armature (right panel). Bone lengths/structure always come live from
    // the source; these override joint rotations (by joint id) and move the root.
    std::unordered_map<int, Quaternion> poseRot;
    Vector3 poseRootOffset = { 0, 0, 0 };
    int ikChain = 0;                              // ancestors IK may rotate (0 = up to the root)
    Skeleton PosedSkeleton() const;
    void ResetPose();

    Viewport(std::string t, bool editable);
    void Unload();

    void SetModel(Model m, const std::string& name, Shader lit);
    void ClearModel();
    void FrameAll();
    // axis 0/1/2 = X/Y/Z: look from that side (switches to ortho, like Blender)
    void SnapView(int axis, bool negative, bool flipIfAligned = false);
    Matrix ModelTransform() const;

    // Draws toolbar + image, the bone gizmo, and handles mouse/keyboard input.
    // screenRect is updated so callers can put a drag-drop target over the image.
    void DrawPanel();
    void Render(Shader lit, int viewPosLoc);

    void ResetSkeleton();                         // just a root, centred in the model
    int AddChildBone(int parent);                 // extends along the parent bone; returns the new joint
    int AddChildBoneAt(int parent, Vector3 worldPos);
    void DeleteSelectedJoint();                   // the root can't be deleted
    void SplitSelectedBone();                     // splits parent->selected into splitSegments bones
    void MoveJoint(int joint, Vector3 worldPos);
    float JointRadius() const;

private:
    bool mouseCaptured = false;                   // a drag started in this viewport (orbit/pan)
    bool navDragging = false;                     // orbiting by dragging the axis widget
    bool autoOrtho = false;                       // ortho was switched on by a view snap
    bool animating = false;
    float targetYaw = 0, targetPitch = 0;

    void UpdateCamera();
    void CameraBasis(Vector3* forward, Vector3* right, Vector3* up) const;
    void Orbit(float dx, float dy);
    bool DrawNavGizmo(ImVec2 pos, float w, bool hovered);
    Ray MouseRay(Vector2 local) const;
    bool RayModel(Ray ray, RayCollision* out) const;
    Vector3 PlacementPoint(Ray ray);
    int PickJoint(Ray ray, const Skeleton& s, Matrix place) const;
    void ApplyPoseGizmo(const Skeleton& posed, Matrix m);
    bool DrawGizmo(ImVec2 pos, float w, float h);
    void HandleInput(bool hovered, bool active, Vector2 local, bool gizmoBusy);
    void DrawToolbar();
    void DrawSkeleton(const Skeleton& s, int sel, Matrix place, bool showLimits);
};
