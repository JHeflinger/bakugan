#include "core/app.hpp"
#include "util/utils.hpp"
#include <imgui_internal.h>
#include <ImGuizmo.h>
#include <algorithm>
#include <raymath.h>
#include <rlgl.h>
#include <cfloat>
#include <cstring>
#include <cmath>
#include <unordered_set>

namespace Bakugan {
    static void ToFloat16(const Matrix& m, float* out) {
        float16 f = MatrixToFloatV(m);
        memcpy(out, f.v, sizeof(f.v));
    }

    static Matrix FromFloat16(const float* f) {
        return { f[0], f[4], f[8],  f[12],
                 f[1], f[5], f[9],  f[13],
                 f[2], f[6], f[10], f[14],
                 f[3], f[7], f[11], f[15] };
    }

    static Matrix BasisMatrix(Vector3 x, Vector3 y, Vector3 z, Vector3 p) {
        Matrix m = MatrixIdentity();
        m.m0 = x.x; m.m1 = x.y; m.m2 = x.z;
        m.m4 = y.x; m.m5 = y.y; m.m6 = y.z;
        m.m8 = z.x; m.m9 = z.y; m.m10 = z.z;
        m.m12 = p.x; m.m13 = p.y; m.m14 = p.z;
        return m;
    }

    static Vector3 Column(const Matrix& m, int c) {
        if (c == 0) return { m.m0, m.m1, m.m2 };
        if (c == 1) return { m.m4, m.m5, m.m6 };
        if (c == 2) return { m.m8, m.m9, m.m10 };
        return { m.m12, m.m13, m.m14 };
    }

    // ---- part meshes -----------------------------------------------------------------------------------------------

    Model& App::PartMesh(const Part& p) {
        PartModel& pm = m_PartModels[p.id];
        if (!pm.loaded || !pm.shape.SameShape(p.shape)) {
            if (pm.loaded) UnloadModel(pm.model);
            const SuperFrustum sf = p.shape;
            TriMesh t = Utils::MeshImplicit([&](Vector3 q) { return sf.LocalDistance(q); }, sf.LocalBounds(), 40);
            pm.model = Utils::TriMeshToModel(t, m_Shader);
            pm.shape = p.shape;
            pm.loaded = true;
        }
        return pm.model;
    }

    void App::PurgePartMeshes(const Design& shown) {
        std::unordered_set<int> keep;
        for (const Part& p : shown.parts) keep.insert(p.id);
        for (const Part& p : m_Design.parts) keep.insert(p.id);
        for (auto it = m_PartModels.begin(); it != m_PartModels.end();) {
            if (!keep.count(it->first)) {
                if (it->second.loaded) UnloadModel(it->second.model);
                it = m_PartModels.erase(it);
            } else {
                ++it;
            }
        }
    }

    int App::Pick(Viewport& vp, const Design& d, const std::vector<Rigid>& xf) {
        Ray ray = vp.MouseRay();
        float best = FLT_MAX;
        int hit = -1;
        for (int i = 0; i < (int)d.parts.size(); i++) {
            Model& m = PartMesh(d.parts[i]);
            Matrix t = d.parts[i].shape.Pose().Then(xf[i]).ToMatrix();
            RayCollision c = GetRayCollisionMesh(ray, m.meshes[0], t);
            if (c.hit && c.distance < best) { best = c.distance; hit = i; }
        }
        return hit;
    }

    // ---- drawing -----------------------------------------------------------------------------------------------------

    void App::DrawInput() {
        if (!m_HasInput || !m_ShowInput) return;
        m_Input.transform = m_InputTransform;
        if (m_InputAlpha >= 0.97f) {
            DrawModel(m_Input, { 0, 0, 0 }, 1.0f, Color{ 200, 200, 205, 255 });
            return;
        }
        rlDrawRenderBatchActive();
        rlDisableDepthMask();
        DrawModel(m_Input, { 0, 0, 0 }, 1.0f, ColorAlpha(Color{ 210, 210, 220, 255 }, m_InputAlpha));
        rlDrawRenderBatchActive();
        rlEnableDepthMask();
    }

    void App::DrawSuperfrustums(const Design& d, const std::vector<Rigid>& xf, float alpha, int highlight) {
        int n = (int)d.parts.size();
        if (n == 0) return;
        Vector3 eye = m_Figure.CameraPosition();
        std::vector<int> order(n);
        std::vector<float> dist(n);
        std::vector<Matrix> mats(n);
        for (int i = 0; i < n; i++) {
            order[i] = i;
            Rigid pose = d.parts[i].shape.Pose().Then(xf[i]);
            mats[i] = pose.ToMatrix();
            dist[i] = Vector3DistanceSqr(pose.position, eye);
        }
        std::sort(order.begin(), order.end(), [&](int a, int b) { return dist[a] > dist[b]; });
        bool translucent = alpha < 0.97f;
        rlDrawRenderBatchActive();
        if (translucent) rlDisableDepthMask();
        for (int i : order) {
            Model& m = PartMesh(d.parts[i]);
            m.transform = mats[i];
            Color c = d.parts[i].color;
            float a = alpha;
            if (highlight >= 0 && i != highlight && translucent) a *= 0.6f;
            if (i == highlight) { c = ColorBrightness(c, 0.25f); a = fminf(1.0f, alpha + 0.25f); }
            DrawModel(m, { 0, 0, 0 }, 1.0f, ColorAlpha(c, a));
            rlDrawRenderBatchActive();
        }
        if (translucent) rlEnableDepthMask();
        if (highlight >= 0 && highlight < n) {
            Model& m = PartMesh(d.parts[highlight]);
            m.transform = mats[highlight];
            DrawModelWires(m, { 0, 0, 0 }, 1.0f, ColorAlpha(WHITE, 0.18f));
        }
        rlDrawRenderBatchActive();
    }

    void App::DrawRestored(const Design& d, const std::vector<Rigid>& xf, float alpha) {
        rlDrawRenderBatchActive();
        if (alpha < 0.97f) rlDisableDepthMask();
        for (RestoredModel& r : m_Restored) {
            int i = d.IndexOf(r.partId);
            if (i < 0) continue;
            r.model.transform = xf[i].ToMatrix();
            Color c = d.parts[i].color;
            if (i == m_Selected) c = ColorBrightness(c, 0.25f);
            DrawModel(r.model, { 0, 0, 0 }, 1.0f, ColorAlpha(c, alpha));
        }
        rlDrawRenderBatchActive();
        if (alpha < 0.97f) rlEnableDepthMask();
    }

    void App::DrawHinges(const Design& d, const std::vector<Rigid>& xf, float t) {
        rlDrawRenderBatchActive();
        rlDisableDepthTest();
        for (int i = 0; i < (int)d.parts.size(); i++) {
            const Part& p = d.parts[i];
            if (p.hinge.parent < 0) continue;
            Vector3 pivot, axis;
            d.HingeInWorld(i, xf, &pivot, &axis);
            axis = Vector3Normalize(axis);
            const Vector3& s = p.shape.size;
            float L = Clamp(0.55f * fminf(s.x, fminf(s.y, s.z)), 0.06f, 0.3f);
            bool sel = i == m_Selected;
            if (p.hinge.locked) {
                DrawCube(pivot, L * 0.25f, L * 0.25f, L * 0.25f, sel ? WHITE : GRAY);
                continue;
            }
            Color col = sel ? WHITE : Color{ 255, 205, 60, 255 };
            float r = L * 0.06f;
            DrawCylinderEx(Vector3Subtract(pivot, Vector3Scale(axis, L * 0.5f)), Vector3Add(pivot, Vector3Scale(axis, L * 0.5f)),
                           r, r, 8, col);
            DrawSphereEx(pivot, r * 2.2f, 8, 8, col);
            // arrow: the direction this part turns when it folds closed
            Vector3 centre = xf[i].Apply(p.shape.position);
            Vector3 u = Vector3Subtract(centre, pivot);
            u = Vector3Subtract(u, Vector3Scale(axis, Vector3DotProduct(u, axis)));
            if (Vector3Length(u) < 1e-5f) u = Utils::AnyPerpendicular(axis);
            u = Vector3Normalize(u);
            float rho = L * 0.6f, sweep = 70.0f * DEG2RAD;
            Vector3 prev = Vector3Add(pivot, Vector3Scale(u, rho));
            const int N = 10;
            for (int k = 1; k <= N; k++) {
                Vector3 dir = Vector3RotateByQuaternion(u, QuaternionFromAxisAngle(axis, sweep * k / N));
                Vector3 q = Vector3Add(pivot, Vector3Scale(dir, rho));
                if (k < N) DrawCylinderEx(prev, q, r * 0.6f, r * 0.6f, 5, ColorAlpha(col, 0.85f));
                else DrawCylinderEx(prev, Vector3Add(prev, Vector3Scale(Vector3Normalize(Vector3Subtract(q, prev)), r * 6.0f)),
                                    r * 2.2f, 0.0f, 8, col);
                prev = q;
            }
        }
        rlDrawRenderBatchActive();
        rlEnableDepthTest();
        (void)t;
    }

    void App::RenderFigure() {
        if (!m_Figure.Begin3D(Color{ 38, 40, 46, 255 }, m_ShowGrid)) return;
        SetShaderValue(m_Shader, m_ViewPositionLocation, &m_Figure.GetCamera().position, SHADER_UNIFORM_VEC3);
        const Design& d = Shown();
        std::vector<Rigid> open(d.parts.size());
        if (m_ShowPiecesInFigure && !m_Restored.empty()) {
            std::vector<Rigid> o2(m_RestoredFor.parts.size());
            DrawRestored(m_RestoredFor, o2, 1.0f);
        }
        if (m_ShowParts) DrawSuperfrustums(d, open, m_PartsAlpha, m_Selected);
        DrawInput();
        if (m_Target == EditTarget::Sphere && m_Mode == Mode::Manual)
            DrawSphereWires(d.ball.center, d.ball.radius, 14, 24, ColorAlpha(SKYBLUE, 0.35f));
        if (m_ShowHinges) DrawHinges(d, open, 0.0f);
        m_Figure.End3D();
    }

    void App::RenderBall() {
        if (!m_Ball.Begin3D(Color{ 32, 36, 44, 255 }, m_ShowGrid)) return;
        SetShaderValue(m_Shader, m_ViewPositionLocation, &m_Ball.GetCamera().position, SHADER_UNIFORM_VEC3);
        const Design& d = Shown();
        float t = m_Fold;
        bool pieces = m_ShowPieces && !m_Restored.empty() && !Busy() && RestoredCurrent();
        if (pieces) DrawRestored(m_RestoredFor, m_RestoredFor.Transforms(t), 1.0f);
        else DrawSuperfrustums(d, d.Transforms(t), 0.92f, m_Selected);
        if (m_ShowSphere && !d.parts.empty()) {
            float a = 0.12f + 0.3f * t;
            DrawSphereWires(d.ball.center, d.ball.radius, 14, 24, ColorAlpha(SKYBLUE, a));
        }
        if (t > 0.98f && m_AnalysisValid) {
            rlDrawRenderBatchActive();
            float s = m_FoldModel.Spacing() * 0.35f;
            if (m_ShowShell)
                for (size_t k = 0; k < m_Visual.shell.size(); k++) {
                    uint8_t st = m_Visual.shellState[k];
                    Color c = st == 0 ? Color{ 255, 60, 60, 255 } : st == 2 ? Color{ 255, 190, 40, 255 } : Color{ 70, 220, 120, 255 };
                    if (st == 1 && pieces) continue;
                    DrawCubeV(m_Visual.shell[k], { s, s, s }, c);
                }
            if (m_ShowLost) {
                rlDisableDepthTest();
                for (const Vector3& p : m_Visual.lost) DrawCubeV(p, { s, s, s }, Color{ 255, 70, 200, 255 });
                rlDrawRenderBatchActive();
                rlEnableDepthTest();
            }
        }
        if (m_ShowHinges && m_Mode == Mode::Manual) DrawHinges(d, d.Transforms(t), t);
        m_Ball.End3D();
    }

    // ---- panels ------------------------------------------------------------------------------------------------------

    void App::DrawViewportWindow(Viewport& vp, ImVec2 pos, ImVec2 size, bool figure) {
        ImGui::SetNextWindowPos(pos);
        ImGui::SetNextWindowSize(size);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4, 4));
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                                 ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
        if (figure) ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.25f, 0.18f, 0.08f, 1));
        ImGui::Begin(vp.Title().c_str(), nullptr, flags);
        if (figure) ImGui::PopStyleColor();
        const char* help = figure
            ? (m_Mode == Mode::Manual ? "Click: select  W/E/R: move/rotate/scale  Q: part/hinge/sphere  Shift+A/D: add/duplicate  Del: delete\n"
                                        "RMB/Alt+drag: orbit  MMB/Shift+RMB: pan  Wheel: zoom  F: frame  Cmd/Ctrl+Z: undo"
                                      : "Click: select a part   RMB/Alt+drag: orbit  MMB/Shift+RMB: pan  Wheel: zoom  F: frame")
            : (m_Mode == Mode::Manual ? "Space: pop open / close   Drag the ring: turn the selected hinge   Q: sphere gizmo\n"
                                        "RMB/Alt+drag: orbit  MMB/Shift+RMB: pan  Wheel: zoom  F: frame"
                                      : "Space: pop open / close   RMB/Alt+drag: orbit  MMB/Shift+RMB: pan  Wheel: zoom");
        if (figure)
            vp.DrawPanel([this] { FigureToolbar(); }, [this](ImVec2 p, float w, float h) { return FigureOverlay(p, w, h); }, help);
        else
            vp.DrawPanel([this] { BallToolbar(); }, [this](ImVec2 p, float w, float h) { return BallOverlay(p, w, h); }, help);
        AcceptAssetDrop(vp);

        if (vp.Clicked() && !Busy()) {
            const Design& d = m_Design;
            std::vector<Rigid> xf = figure ? std::vector<Rigid>(d.parts.size()) : d.Transforms(m_Fold);
            Select(Pick(vp, d, xf));
        }
        ImGui::End();
        ImGui::PopStyleVar();
    }

    void App::FigureToolbar() {
        if (m_Mode == Mode::Manual) {
            if (ImGui::RadioButton("Move", m_GizmoOp == GizmoOp::Translate)) m_GizmoOp = GizmoOp::Translate;
            ImGui::SameLine();
            if (ImGui::RadioButton("Rotate", m_GizmoOp == GizmoOp::Rotate)) m_GizmoOp = GizmoOp::Rotate;
            ImGui::SameLine();
            if (ImGui::RadioButton("Scale", m_GizmoOp == GizmoOp::Scale)) m_GizmoOp = GizmoOp::Scale;
            ImGui::SetItemTooltip("W / E / R");
            ImGui::SameLine();
            ImGui::TextDisabled("|");
            ImGui::SameLine();
            if (ImGui::RadioButton("Part", m_Target == EditTarget::Part)) m_Target = EditTarget::Part;
            ImGui::SameLine();
            if (ImGui::RadioButton("Hinge", m_Target == EditTarget::Hinge)) m_Target = EditTarget::Hinge;
            ImGui::SameLine();
            if (ImGui::RadioButton("Sphere", m_Target == EditTarget::Sphere)) m_Target = EditTarget::Sphere;
            ImGui::SetItemTooltip("Q cycles what the gizmo edits");
            ImGui::SameLine();
            ImGui::Checkbox("Local", &m_LocalGizmo);
        }
        if (ImGui::Button("Frame (F)")) m_Figure.Frame(m_InputBounds);
        ImGui::SameLine();
        ImGui::Checkbox("Model", &m_ShowInput);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(60);
        ImGui::SliderFloat("##ia", &m_InputAlpha, 0.05f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("Model opacity");
        ImGui::SameLine();
        ImGui::Checkbox("Parts", &m_ShowParts);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(60);
        ImGui::SliderFloat("##pa", &m_PartsAlpha, 0.05f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("Superfrustum opacity");
        ImGui::SameLine();
        ImGui::Checkbox("Hinges", &m_ShowHinges);
        ImGui::SameLine();
        if (ImGui::Button("View...")) ImGui::OpenPopup("figure_view");
        if (ImGui::BeginPopup("figure_view")) {
            ImGui::Checkbox("Grid", &m_ShowGrid);
            bool ortho = m_Figure.Ortho();
            if (ImGui::Checkbox("Orthographic (5)", &ortho)) m_Figure.SetOrtho(ortho);
            ImGui::BeginDisabled(m_Restored.empty());
            ImGui::Checkbox("Show built pieces", &m_ShowPiecesInFigure);
            ImGui::EndDisabled();
            ImGui::EndPopup();
        }
    }

    void App::BallToolbar() {
        ImGui::SetNextItemWidth(160);
        if (ImGui::SliderFloat("##fold", &m_Fold, 0.0f, 1.0f, m_Fold >= 0.999f ? "closed" : m_Fold <= 0.001f ? "open" : "%.2f folded"))
            m_FoldAnim = 0;
        ImGui::SetItemTooltip("Fold fraction: every hinge at this share of its closed angle");
        ImGui::SameLine();
        if (ImGui::Button("Pop open")) { m_FoldAnim = 1; m_FoldVelocity = 0.0f; if (m_Fold < 0.01f) m_Fold = 1.0f; }
        ImGui::SetItemTooltip("Space. Release the ball: the springs snap every hinge open.");
        ImGui::SameLine();
        if (ImGui::Button("Close")) m_FoldAnim = 2;
        ImGui::SameLine();
        ImGui::Checkbox("Loop", &m_Loop);
        if (ImGui::Button("Frame (F)")) m_Ball.Frame(m_InputBounds);
        ImGui::SameLine();
        ImGui::Checkbox("Sphere", &m_ShowSphere);
        ImGui::SameLine();
        ImGui::Checkbox("Shell gaps", &m_ShowShell);
        ImGui::SetItemTooltip("When closed: red = shell not covered, amber = covered twice, green = covered");
        ImGui::SameLine();
        ImGui::Checkbox("Lost material", &m_ShowLost);
        ImGui::SetItemTooltip("When closed: material outside the sphere or colliding with another part");
        ImGui::SameLine();
        ImGui::BeginDisabled(m_Restored.empty());
        ImGui::Checkbox("Pieces", &m_ShowPieces);
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("Show the built pieces (restored detail) instead of the superfrustums");
        ImGui::SameLine();
        if (ImGui::Button("View...")) ImGui::OpenPopup("ball_view");
        if (ImGui::BeginPopup("ball_view")) {
            ImGui::Checkbox("Grid", &m_ShowGrid);
            ImGui::Checkbox("Hinges", &m_ShowHinges);
            bool ortho = m_Ball.Ortho();
            if (ImGui::Checkbox("Orthographic (5)", &ortho)) m_Ball.SetOrtho(ortho);
            ImGui::EndPopup();
        }
    }

    bool App::FigureOverlay(ImVec2 pos, float w, float h) {
        const Design& shown = Shown();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        if (m_HasInput) dl->AddText(ImVec2(pos.x + 8, pos.y + 6), IM_COL32(255, 255, 255, 140), m_InputName.c_str());
        if (m_Selected >= 0 && m_Selected < (int)shown.parts.size()) {
            Vector2 sp = GetWorldToScreenEx(shown.parts[m_Selected].shape.position, m_Figure.GetCamera(), (int)w, (int)h);
            dl->AddText(ImVec2(pos.x + sp.x + 10, pos.y + sp.y - 8), IM_COL32(255, 255, 255, 220), shown.parts[m_Selected].name.c_str());
        }
        if (m_Mode != Mode::Manual || Busy()) return false;

        float view[16], proj[16], xf[16];
        m_Figure.GizmoMatrices(view, proj);
        ImGuizmo::SetOrthographic(m_Figure.Ortho());
        ImGuizmo::SetGizmoSizeClipSpace(0.2f);
        ImGuizmo::SetDrawlist();
        ImGuizmo::SetRect(pos.x, pos.y, w, h);
        ImGuizmo::PushID(1);
        ImGuizmo::MODE mode = m_LocalGizmo ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
        bool used = false;

        if (m_Target == EditTarget::Sphere) {
            Ball& b = m_Design.ball;
            Matrix m = BasisMatrix({ b.radius, 0, 0 }, { 0, b.radius, 0 }, { 0, 0, b.radius }, b.center);
            ToFloat16(m, xf);
            ImGuizmo::OPERATION op = m_GizmoOp == GizmoOp::Scale ? ImGuizmo::SCALE_X : ImGuizmo::TRANSLATE;
            if (ImGuizmo::Manipulate(view, proj, op, ImGuizmo::LOCAL, xf)) {
                Matrix r = FromFloat16(xf);
                b.center = Column(r, 3);
                if (m_GizmoOp == GizmoOp::Scale && m_FoldModel.Valid())
                    b.radius = Clamp(Vector3Length(Column(r, 0)), m_FoldModel.RadiusMin(), m_FoldModel.RadiusMax());
            }
        } else if (m_Selected >= 0 && m_Selected < (int)m_Design.parts.size()) {
            Part& p = m_Design.parts[m_Selected];
            if (m_Target == EditTarget::Part) {
                SuperFrustum& f = p.shape;
                Vector3 ax = f.Axis(0), ay = f.Axis(1), az = f.Axis(2);
                Matrix m = BasisMatrix(Vector3Scale(ax, f.size.x), Vector3Scale(ay, f.size.y), Vector3Scale(az, f.size.z), f.position);
                ToFloat16(m, xf);
                ImGuizmo::OPERATION op = m_GizmoOp == GizmoOp::Translate ? ImGuizmo::TRANSLATE
                                       : m_GizmoOp == GizmoOp::Rotate ? ImGuizmo::ROTATE : ImGuizmo::SCALE;
                if (ImGuizmo::Manipulate(view, proj, op, m_GizmoOp == GizmoOp::Scale ? ImGuizmo::LOCAL : mode, xf)) {
                    Matrix r = FromFloat16(xf);
                    Vector3 cx = Column(r, 0), cy = Column(r, 1), cz = Column(r, 2);
                    f.size = { Vector3Length(cx), Vector3Length(cy), Vector3Length(cz) };
                    Vector3 nx = Vector3Normalize(cx), nz = Vector3Normalize(cz);
                    Vector3 ny = Vector3Normalize(Vector3CrossProduct(nz, nx));
                    nz = Vector3CrossProduct(nx, ny);
                    f.rotation = QuaternionNormalize(QuaternionFromMatrix(BasisMatrix(nx, ny, nz, { 0, 0, 0 })));
                    f.position = Column(r, 3);
                    f.Repair(0.01f);
                }
            } else if (p.hinge.parent >= 0) {
                Hinge& hg = p.hinge;
                Vector3 ax = Vector3Normalize(hg.axis);
                Vector3 ay = Utils::AnyPerpendicular(ax);
                Vector3 az = Vector3CrossProduct(ax, ay);
                Matrix m = BasisMatrix(ax, ay, az, hg.pivot);
                ToFloat16(m, xf);
                ImGuizmo::OPERATION op = m_GizmoOp == GizmoOp::Rotate ? ImGuizmo::ROTATE : ImGuizmo::TRANSLATE;
                if (ImGuizmo::Manipulate(view, proj, op, ImGuizmo::LOCAL, xf)) {
                    Matrix r = FromFloat16(xf);
                    hg.pivot = Column(r, 3);
                    if (m_GizmoOp == GizmoOp::Rotate) hg.axis = Vector3Normalize(Column(r, 0));
                }
            }
        }
        used = ImGuizmo::IsOver() || ImGuizmo::IsUsing();
        ImGuizmo::PopID();
        return used;
    }

    bool App::BallOverlay(ImVec2 pos, float w, float h) {
        if (m_Mode != Mode::Manual || Busy()) return false;
        float view[16], proj[16], xf[16];
        m_Ball.GizmoMatrices(view, proj);
        ImGuizmo::SetOrthographic(m_Ball.Ortho());
        ImGuizmo::SetGizmoSizeClipSpace(0.2f);
        ImGuizmo::SetDrawlist();
        ImGuizmo::SetRect(pos.x, pos.y, w, h);
        ImGuizmo::PushID(2);
        if (m_Target == EditTarget::Sphere) {
            Ball& b = m_Design.ball;
            Matrix m = BasisMatrix({ b.radius, 0, 0 }, { 0, b.radius, 0 }, { 0, 0, b.radius }, b.center);
            ToFloat16(m, xf);
            ImGuizmo::OPERATION op = m_GizmoOp == GizmoOp::Scale ? ImGuizmo::SCALE_X : ImGuizmo::TRANSLATE;
            if (ImGuizmo::Manipulate(view, proj, op, ImGuizmo::LOCAL, xf)) {
                Matrix r = FromFloat16(xf);
                b.center = Column(r, 3);
                if (m_GizmoOp == GizmoOp::Scale && m_FoldModel.Valid())
                    b.radius = Clamp(Vector3Length(Column(r, 0)), m_FoldModel.RadiusMin(), m_FoldModel.RadiusMax());
            }
        } else if (m_Selected >= 0 && m_Selected < (int)m_Design.parts.size() && m_Fold > 0.05f) {
            // a ring about the hinge axis: dragging it turns the hinge (scaled by the fold fraction)
            Design& d = m_Design;
            Hinge& hg = d.parts[m_Selected].hinge;
            if (hg.parent >= 0 && !hg.locked) {
                std::vector<Rigid> t = d.Transforms(m_Fold);
                Vector3 pivot, axis;
                d.HingeInWorld(m_Selected, t, &pivot, &axis);
                axis = Vector3Normalize(axis);
                Vector3 centre = t[m_Selected].Apply(d.parts[m_Selected].shape.position);
                Vector3 u = Vector3Subtract(centre, pivot);
                u = Vector3Subtract(u, Vector3Scale(axis, Vector3DotProduct(u, axis)));
                u = Vector3Length(u) > 1e-5f ? Vector3Normalize(u) : Utils::AnyPerpendicular(axis);
                Vector3 v = Vector3CrossProduct(axis, u);
                ToFloat16(BasisMatrix(axis, u, v, pivot), xf);
                if (ImGuizmo::Manipulate(view, proj, ImGuizmo::ROTATE_X, ImGuizmo::LOCAL, xf)) {
                    Vector3 nu = Vector3Normalize(Column(FromFloat16(xf), 1));
                    float delta = atan2f(Vector3DotProduct(Vector3CrossProduct(u, nu), axis), Vector3DotProduct(u, nu));
                    float maxA = fminf(hg.maxAngle, m_FoldParams.maxHingeAngle);
                    hg.angle = Clamp(hg.angle + delta / m_Fold, 0.0f, maxA);
                }
            }
        }
        bool used = ImGuizmo::IsOver() || ImGuizmo::IsUsing();
        ImGuizmo::PopID();
        return used;
    }
}
