#pragma once
#include "data/skeleton.hpp"
#include <unordered_map>
#include <imgui.h>

namespace Bakugan {
    class Viewport {
    public:
        Matrix MirrorPlacement() const;
        Skeleton PosedSkeleton() const;
        void ResetPose();
        Viewport(std::string t, bool editable);
        void Unload();
        void SetModel(Model m, const std::string& name, Shader lit);
        void ClearModel();
        void FrameAll();
        void SnapView(int axis, bool negative, bool flipIfAligned = false);
        Matrix ModelTransform() const;
        void DrawPanel();
        void Render(Shader lit, int viewPosLoc);
        void ResetSkeleton();
        int AddChildBone(int parent);
        int AddChildBoneAt(int parent, Vector3 worldPos);
        void DeleteSelectedJoint();
        void SplitSelectedBone();
        void MoveJoint(int joint, Vector3 worldPos);
    private:
        float JointRadius() const;
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
    public:
    public:
        Vector3 RootOffset() const { return m_PoseRootOffset; }
        void SetRootOffset(Vector3 offset) { m_PoseRootOffset = offset; }
        std::unordered_map<int, Quaternion> PoseRotations() const { return m_PoseRotations; }
        std::unordered_map<int, Quaternion>& PoseRotationsRef() { return m_PoseRotations; }
        const Viewport* Mirror() const { return m_Mirror; }
        void SetMirror(Viewport* ptr) { m_Mirror = ptr; }
        int SplitSegments() const { return m_SplitSegments; }
        void SetSplitSegments(int s) { m_SplitSegments = s; }
        int SelectedJoint() const { return m_SelectedJoint; }
        void SelectJoint(int j) { m_SelectedJoint = j; }
        Skeleton& SkeletonRef() { return m_Skeleton; }
        Skeleton GetSkeleton() const { return m_Skeleton; }
        BoundingBox Bounds() const { return m_Bounds; }
        Rectangle ScreenRectangle() const { return m_ScreenRectangle; }
        bool Editable() const { return m_Editable; }
        std::string Title() const { return m_Title; }
    private:
        std::string m_Title;
        bool m_Editable = false;
        RenderTexture2D m_Target = {};
        int m_TexWidth = 0;
        int m_TexHeight = 0;
        float m_PixelScale = 1.0f;
        Rectangle m_ScreenRectangle = { 0, 0, 0, 0 };
        Camera3D m_Camera = {};
        float m_Yaw = 35.0f;
        float m_Pitch = 20.0f;
        float m_Distance = 5.5f;
        Vector3 m_Focus = { 0, 1, 0 };
        bool m_Ortho = false;
        Model m_Model = {};
        bool m_HasModel = false;
        std::string m_ModelName;
        float m_ModelScale = 1.0f;
        Vector3 m_ModelOffset = { 0, 0, 0 };
        BoundingBox m_Bounds = { { -1, 0, -1 }, { 1, 2, 1 } };
        bool m_ShowModel = true;
        bool m_Wireframe = false;
        bool m_ShowGrid = true;
        bool m_XraySkeleton = true;
        bool m_XrayModel = false;
        float m_XrayAlpha = 0.35f;
        Skeleton m_Skeleton;
        int m_SelectedJoint = 0;
        GizmoOp m_GizmoOperation = GizmoOp::Translate;
        bool m_LocalGizmo = false;
        bool m_MoveSubtree = false;
        int m_SplitSegments = 2;
        const Viewport* m_Mirror = nullptr;
        bool m_ShowMirror = true;
        bool m_FitMirror = true;
        std::unordered_map<int, Quaternion> m_PoseRotations;
        Vector3 m_PoseRootOffset = { 0, 0, 0 };
        int m_IKChain = 0;
        bool m_MouseCaptured = false;
        bool m_NavDragging = false;
        bool m_AutoOrtho = false;
        bool m_Animating = false;
        float m_TargetYaw = 0;
        float m_TargetPitch = 0;
    };
}
