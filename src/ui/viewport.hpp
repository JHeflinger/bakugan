#pragma once
#include "util/structs.hpp"
#include <functional>
#include <string>
#include <imgui.h>

namespace Bakugan {
    // A 3D panel: render target, orbit camera, Blender-style axis widget, and mouse/keyboard navigation.
    // What is drawn inside is up to the owner (Begin3D/End3D), as are the toolbar and gizmos (callbacks).
    class Viewport {
    public:
        explicit Viewport(std::string title);
        void Unload();
        // toolbar: ImGui widgets above the view. overlay: draws on top of the image (gizmos); returns true when it
        // owns the mouse.
        void DrawPanel(const std::function<void()>& toolbar, const std::function<bool(ImVec2, float, float)>& overlay,
                       const char* help);
        bool Begin3D(Color background, bool grid);
        void End3D();
        void Frame(BoundingBox box);
        void SnapView(int axis, bool negative, bool flipIfAligned = false);
        Ray MouseRay() const;
        bool Clicked() const { return m_Clicked; }
        bool Hovered() const { return m_Hovered; }
        void GizmoMatrices(float* view, float* proj) const;
        Vector3 CameraPosition() const { return m_Camera.position; }
        const Camera3D& GetCamera() const { return m_Camera; }
        bool Ortho() const { return m_Ortho; }
        void SetOrtho(bool o) { m_Ortho = o; m_AutoOrtho = false; }
        Rectangle ScreenRectangle() const { return m_ScreenRectangle; }
        const std::string& Title() const { return m_Title; }
    private:
        void UpdateCamera();
        void CameraBasis(Vector3* forward, Vector3* right, Vector3* up) const;
        void Orbit(float dx, float dy);
        bool DrawNavGizmo(ImVec2 pos, float w, bool hovered);
        void HandleInput(bool hovered, bool active);
    private:
        std::string m_Title;
        RenderTexture2D m_Target = {};
        int m_TexWidth = 0;
        int m_TexHeight = 0;
        float m_PixelScale = 1.0f;
        Rectangle m_ScreenRectangle = { 0, 0, 0, 0 };
        Vector2 m_LocalMouse = { 0, 0 };
        Camera3D m_Camera = {};
        float m_Yaw = 35.0f;
        float m_Pitch = 20.0f;
        float m_Distance = 180.0f;   // millimetres
        Vector3 m_Focus = { 0, 20, 0 };
        bool m_Ortho = false;
        bool m_AutoOrtho = false;
        bool m_Animating = false;
        float m_TargetYaw = 0;
        float m_TargetPitch = 0;
        bool m_MouseCaptured = false;
        bool m_NavDragging = false;
        bool m_Clicked = false;
        bool m_Hovered = false;
        Vector2 m_PressPos = { 0, 0 };
    };
}
