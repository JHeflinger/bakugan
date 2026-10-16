#include "viewport.hpp"
#include <ImGuizmo.h>
#include <algorithm>
#include <raymath.h>
#include <cstdint>
#include <cstring>
#include <rlgl.h>
#include <cmath>

namespace Bakugan {
    Viewport::Viewport(std::string title) : m_Title(std::move(title)) {}

    void Viewport::Unload() {
        if (m_TexWidth > 0) UnloadRenderTexture(m_Target);
        m_TexWidth = m_TexHeight = 0;
    }

    void Viewport::Frame(BoundingBox b) {
        m_Focus = Vector3Scale(Vector3Add(b.min, b.max), 0.5f);
        m_Distance = Vector3Distance(b.min, b.max) * 1.45f + 0.4f;
    }

    void Viewport::UpdateCamera() {
        float y = m_Yaw * DEG2RAD, p = m_Pitch * DEG2RAD;
        m_Camera.target = m_Focus;
        m_Camera.position = Vector3Add(m_Focus, { m_Distance * cosf(p) * sinf(y), m_Distance * sinf(p),
                                                  m_Distance * cosf(p) * cosf(y) });
        m_Camera.up = { -sinf(p) * sinf(y), cosf(p), -sinf(p) * cosf(y) };
        m_Camera.projection = m_Ortho ? CAMERA_ORTHOGRAPHIC : CAMERA_PERSPECTIVE;
        m_Camera.fovy = m_Ortho ? m_Distance * 2.0f * tanf(22.5f * DEG2RAD) : 45.0f;
    }

    void Viewport::CameraBasis(Vector3* forward, Vector3* right, Vector3* up) const {
        *forward = Vector3Normalize(Vector3Subtract(m_Camera.target, m_Camera.position));
        *right = Vector3Normalize(Vector3CrossProduct(*forward, m_Camera.up));
        *up = Vector3CrossProduct(*right, *forward);
    }

    void Viewport::Orbit(float dx, float dy) {
        m_Yaw -= dx * 0.4f;
        m_Pitch = Clamp(m_Pitch + dy * 0.4f, -90.0f, 90.0f);
        m_Animating = false;
        if (m_AutoOrtho) { m_Ortho = false; m_AutoOrtho = false; }
    }

    static float WrapDegrees(float a) {
        a = fmodf(a + 180.0f, 360.0f);
        return (a < 0 ? a + 360.0f : a) - 180.0f;
    }

    void Viewport::SnapView(int axis, bool negative, bool flipIfAligned) {
        float ty = 0, tp = 0;
        if (axis == 0) ty = negative ? -90.0f : 90.0f;
        if (axis == 1) tp = negative ? -90.0f : 90.0f;
        if (axis == 2) ty = negative ? 180.0f : 0.0f;
        float curYaw = m_Animating ? m_TargetYaw : m_Yaw, curPitch = m_Animating ? m_TargetPitch : m_Pitch;
        bool aligned = fabsf(curPitch - tp) < 0.5f && (axis == 1 || fabsf(WrapDegrees(curYaw - ty)) < 0.5f);
        if (aligned && flipIfAligned) { SnapView(axis, !negative, false); return; }   // view from the other side
        m_TargetYaw = curYaw + WrapDegrees(ty - curYaw);
        m_TargetPitch = tp;
        m_Animating = true;
        if (!m_Ortho) { m_Ortho = true; m_AutoOrtho = true; }
    }

    Ray Viewport::MouseRay() const {
        return GetScreenToWorldRayEx(m_LocalMouse, m_Camera, m_TexWidth, m_TexHeight);
    }

    static void ToFloat16(const Matrix& m, float* out) {
        float16 f = MatrixToFloatV(m);
        memcpy(out, f.v, sizeof(f.v));
    }

    void Viewport::GizmoMatrices(float* view, float* proj) const {
        ToFloat16(GetCameraMatrix(m_Camera), view);
        float w = m_ScreenRectangle.width, h = fmaxf(m_ScreenRectangle.height, 1.0f);
        float top = m_Camera.fovy * 0.5f, aspect = w / h;
        ToFloat16(m_Ortho ? MatrixOrtho(-top * aspect, top * aspect, -top, top, RL_CULL_DISTANCE_NEAR, RL_CULL_DISTANCE_FAR)
                          : MatrixPerspective(m_Camera.fovy * DEG2RAD, aspect, RL_CULL_DISTANCE_NEAR, RL_CULL_DISTANCE_FAR), proj);
    }

    void Viewport::HandleInput(bool hovered, bool active) {
        ImGuiIO& io = ImGui::GetIO();
        if (hovered && io.MouseWheel != 0.0f)
            m_Distance = Clamp(m_Distance * powf(0.88f, io.MouseWheel), 0.2f, 200.0f);
        if (active) {
            bool orbit = ImGui::IsMouseDown(ImGuiMouseButton_Right) && !io.KeyShift;
            orbit |= ImGui::IsMouseDown(ImGuiMouseButton_Left) && io.KeyAlt;
            bool pan = ImGui::IsMouseDown(ImGuiMouseButton_Middle);
            pan |= ImGui::IsMouseDown(ImGuiMouseButton_Right) && io.KeyShift;
            if (orbit) {
                if (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f) Orbit(io.MouseDelta.x, io.MouseDelta.y);
            } else if (pan) {
                Vector3 fwd, right, up;
                CameraBasis(&fwd, &right, &up);
                float k = m_Distance * 0.0015f;
                m_Focus = Vector3Add(m_Focus, Vector3Scale(right, -io.MouseDelta.x * k));
                m_Focus = Vector3Add(m_Focus, Vector3Scale(up, io.MouseDelta.y * k));
            }
        }
        if (hovered && !io.WantTextInput) {
            bool opposite = io.KeyCtrl || io.KeySuper;
            if (ImGui::IsKeyPressed(ImGuiKey_1) || ImGui::IsKeyPressed(ImGuiKey_Keypad1)) SnapView(2, opposite);
            if (ImGui::IsKeyPressed(ImGuiKey_3) || ImGui::IsKeyPressed(ImGuiKey_Keypad3)) SnapView(0, opposite);
            if (ImGui::IsKeyPressed(ImGuiKey_7) || ImGui::IsKeyPressed(ImGuiKey_Keypad7)) SnapView(1, opposite);
            if (ImGui::IsKeyPressed(ImGuiKey_5) || ImGui::IsKeyPressed(ImGuiKey_Keypad5)) { m_Ortho = !m_Ortho; m_AutoOrtho = false; }
        }
    }

    void Viewport::DrawPanel(const std::function<void()>& toolbar, const std::function<bool(ImVec2, float, float)>& overlay,
                             const char* help) {
        if (m_Animating) {
            float k = 1.0f - expf(-GetFrameTime() * 14.0f);
            m_Yaw += (m_TargetYaw - m_Yaw) * k;
            m_Pitch += (m_TargetPitch - m_Pitch) * k;
            if (fabsf(m_TargetYaw - m_Yaw) < 0.05f && fabsf(m_TargetPitch - m_Pitch) < 0.05f) {
                m_Yaw = m_TargetYaw;
                m_Pitch = m_TargetPitch;
                m_Animating = false;
            }
        }
        UpdateCamera();
        if (toolbar) toolbar();
        ImVec2 avail = ImGui::GetContentRegionAvail();
        int w = (int)fmaxf(avail.x, 1.0f), h = (int)fmaxf(avail.y, 1.0f);
        m_PixelScale = GetWindowScaleDPI().x;
        int pw = (int)(w * m_PixelScale), ph = (int)(h * m_PixelScale);
        if (pw != m_TexWidth || ph != m_TexHeight) {
            if (m_TexWidth > 0) UnloadRenderTexture(m_Target);
            m_Target = LoadRenderTexture(pw, ph);
            SetTextureFilter(m_Target.texture, TEXTURE_FILTER_BILINEAR);
            m_TexWidth = pw;
            m_TexHeight = ph;
        }
        ImVec2 pos = ImGui::GetCursorScreenPos();
        m_ScreenRectangle = { pos.x, pos.y, (float)w, (float)h };
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddImage((ImTextureID)(uintptr_t)m_Target.texture.id, pos, ImVec2(pos.x + w, pos.y + h), ImVec2(0, 1), ImVec2(1, 0));
        if (help) {
            ImVec2 ts = ImGui::CalcTextSize(help);
            dl->AddText(ImVec2(pos.x + 8, pos.y + h - ts.y - 6), IM_COL32(255, 255, 255, 140), help);
        }

        bool overlayBusy = overlay ? overlay(pos, (float)w, (float)h) : false;
        ImGui::SetCursorScreenPos(pos);
        ImGui::Dummy(ImVec2((float)w, (float)h));
        bool hovered = ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(pos, ImVec2(pos.x + w, pos.y + h));
        bool navBusy = DrawNavGizmo(pos, (float)w, hovered && !ImGuizmo::IsUsingAny());
        overlayBusy |= navBusy;
        for (int b = 0; b < 3; b++)
            if (hovered && ImGui::IsMouseClicked(b) && !(b == 0 && overlayBusy)) m_MouseCaptured = true;
        if (!ImGui::IsMouseDown(0) && !ImGui::IsMouseDown(1) && !ImGui::IsMouseDown(2)) m_MouseCaptured = false;
        ImVec2 mouse = ImGui::GetIO().MousePos;
        m_LocalMouse = { (mouse.x - pos.x) * m_PixelScale, (mouse.y - pos.y) * m_PixelScale };
        m_Hovered = hovered;

        // a click is a left press and release without dragging, not on a gizmo, not orbiting
        ImGuiIO& io = ImGui::GetIO();
        m_Clicked = false;
        if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !overlayBusy && !io.KeyAlt) m_PressPos = { mouse.x, mouse.y };
        if (hovered && ImGui::IsMouseReleased(ImGuiMouseButton_Left) && !overlayBusy && !io.KeyAlt && !ImGuizmo::IsUsingAny() &&
            hypotf(mouse.x - m_PressPos.x, mouse.y - m_PressPos.y) < 4.0f)
            m_Clicked = true;
        HandleInput(hovered, m_MouseCaptured);
    }

    bool Viewport::Begin3D(Color background, bool grid) {
        if (m_TexWidth <= 0) return false;
        UpdateCamera();
        BeginTextureMode(m_Target);
        ClearBackground(background);
        BeginMode3D(m_Camera);
        if (grid) DrawGrid(20, 0.25f);
        rlDrawRenderBatchActive();
        return true;
    }

    void Viewport::End3D() {
        rlDrawRenderBatchActive();
        EndMode3D();
        EndTextureMode();
    }

    bool Viewport::DrawNavGizmo(ImVec2 pos, float w, bool hovered) {
        const float R = 40.0f, ball = 10.0f;
        ImVec2 c(pos.x + w - R - 22, pos.y + R + 22);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImGuiIO& io = ImGui::GetIO();
        Vector3 fwd, right, up;
        CameraBasis(&fwd, &right, &up);
        struct Ball { int axis; bool neg; ImVec2 p; float depth; };
        Ball balls[6];
        for (int i = 0; i < 6; i++) {
            Vector3 a = { 0, 0, 0 };
            (&a.x)[i / 2] = (i % 2) ? -1.0f : 1.0f;
            balls[i] = { i / 2, (i % 2) == 1,
                         ImVec2(c.x + Vector3DotProduct(a, right) * R, c.y - Vector3DotProduct(a, up) * R),
                         Vector3DotProduct(a, fwd) };
        }
        std::sort(balls, balls + 6, [](const Ball& a, const Ball& b) { return a.depth > b.depth; });  // far first
        ImVec2 m = io.MousePos;
        bool overArea = hovered && hypotf(m.x - c.x, m.y - c.y) < R + ball + 4;
        int hot = -1;
        if (overArea && !m_NavDragging)
            for (int i = 0; i < 6; i++)
                if (hypotf(m.x - balls[i].p.x, m.y - balls[i].p.y) < ball) hot = i;
        if (overArea || m_NavDragging) dl->AddCircleFilled(c, R + ball + 4, IM_COL32(255, 255, 255, 28), 48);
        const ImU32 colors[3] = { IM_COL32(232, 72, 85, 255), IM_COL32(135, 200, 60, 255), IM_COL32(64, 132, 232, 255) };
        const char* names[3] = { "X", "Y", "Z" };
        for (int i = 0; i < 6; i++) {
            const Ball& b = balls[i];
            ImU32 col = colors[b.axis];
            if (!b.neg) {
                dl->AddLine(c, b.p, col, 2.5f);
                dl->AddCircleFilled(b.p, ball, col, 24);
                if (i == hot) dl->AddCircle(b.p, ball, IM_COL32_WHITE, 24, 2.0f);
                ImVec2 ts = ImGui::CalcTextSize(names[b.axis]);
                dl->AddText(ImVec2(b.p.x - ts.x * 0.5f, b.p.y - ts.y * 0.5f), IM_COL32(20, 20, 20, 255), names[b.axis]);
            } else {
                dl->AddCircleFilled(b.p, ball, (col & 0x00FFFFFF) | (i == hot ? 0xC0000000 : 0x50000000), 24);
                dl->AddCircle(b.p, ball, col, 24, 1.5f);
                if (i == hot) {
                    char label[3] = { '-', names[b.axis][0], 0 };
                    ImVec2 ts = ImGui::CalcTextSize(label);
                    dl->AddText(ImVec2(b.p.x - ts.x * 0.5f, b.p.y - ts.y * 0.5f), IM_COL32(20, 20, 20, 255), label);
                }
            }
        }
        if (overArea && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            if (hot >= 0) SnapView(balls[hot].axis, balls[hot].neg, true);
            else m_NavDragging = true;
        }
        if (m_NavDragging) {
            if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                if (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f) Orbit(io.MouseDelta.x, io.MouseDelta.y);
            } else {
                m_NavDragging = false;
            }
        }
        return overArea || m_NavDragging;
    }
}
