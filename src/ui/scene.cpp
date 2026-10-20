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

    static Quaternion RotationOf(const Matrix& r) {
        Vector3 nx = Vector3Normalize(Column(r, 0)), nz = Vector3Normalize(Column(r, 2));
        Vector3 ny = Vector3Normalize(Vector3CrossProduct(nz, nx));
        nz = Vector3CrossProduct(nx, ny);
        return QuaternionNormalize(QuaternionFromMatrix(BasisMatrix(nx, ny, nz, { 0, 0, 0 })));
    }

    static Color Mix(Color a, Color b, float t) {
        return { (unsigned char)(a.r + (b.r - a.r) * t), (unsigned char)(a.g + (b.g - a.g) * t),
                 (unsigned char)(a.b + (b.b - a.b) * t), (unsigned char)(a.a + (b.a - a.a) * t) };
    }

    static const Color Danger = { 255, 50, 50, 255 };

    static float Pulse() { return 0.55f + 0.45f * sinf((float)GetTime() * 7.0f); }

    // ---- shape meshes ------------------------------------------------------------------------------------------------

    ShapeModel& App::ShapeMesh(const Shape& s) {
        ShapeModel& sm = m_ShapeModels[s.id];
        if (!sm.loaded || !sm.shape.SameShape(s.sf)) {
            if (sm.loaded) UnloadModel(sm.model);
            const SuperFrustum sf = s.sf;
            sm.mesh = Utils::MeshImplicit([&](Vector3 q) { return sf.LocalDistance(q); }, sf.LocalBounds(), 52);
            sm.model = Utils::TriMeshToModel(sm.mesh, m_Shader);
            sm.shape = s.sf;
            sm.loaded = true;
        }
        return sm;
    }

    void App::PurgeShapeMeshes() {
        std::unordered_set<int> keep;
        for (const Shape& s : m_Design.shapes) keep.insert(s.id);
        for (auto it = m_ShapeModels.begin(); it != m_ShapeModels.end();) {
            if (!keep.count(it->first)) {
                if (it->second.loaded) UnloadModel(it->second.model);
                it = m_ShapeModels.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Material wins; a cut is picked only where its ghost is in front of the material (or with no material hit).
    int App::PickShape(Viewport& vp, const std::vector<Rigid>& xf, bool cuts) {
        Ray ray = vp.MouseRay();
        float bestSolid = FLT_MAX, bestCut = FLT_MAX;
        int solid = -1, cut = -1;
        for (int i = 0; i < (int)m_Design.shapes.size(); i++) {
            const Shape& s = m_Design.shapes[i];
            bool isCut = !m_Design.AddsMaterial(i);
            if (isCut && !cuts) continue;
            int g = m_Design.GroupIndex(s.group);
            ShapeModel& sm = ShapeMesh(s);
            Matrix t = s.sf.Pose().Then(g >= 0 ? xf[g] : Rigid{}).ToMatrix();
            RayCollision c = GetRayCollisionMesh(ray, sm.model.meshes[0], t);
            if (!c.hit) continue;
            if (isCut && c.distance < bestCut) { bestCut = c.distance; cut = i; }
            if (!isCut && c.distance < bestSolid) { bestSolid = c.distance; solid = i; }
        }
        return cut >= 0 && bestCut < bestSolid ? cut : solid;
    }

    // ---- drawing -----------------------------------------------------------------------------------------------------

    // Scale, then turn about the mesh's own centre, then place it.
    Matrix App::ReferenceMatrix() const {
        float s = m_ReferenceScale;
        Vector3 c = Vector3Scale(Vector3Scale(Vector3Add(m_ReferenceBox.min, m_ReferenceBox.max), 0.5f), s);
        Matrix m = MatrixScale(s, s, s);
        m = MatrixMultiply(m, MatrixTranslate(-c.x, -c.y, -c.z));
        m = MatrixMultiply(m, MatrixRotateXYZ(Vector3Scale(m_ReferenceRotation, DEG2RAD)));
        m = MatrixMultiply(m, MatrixTranslate(c.x + m_ReferenceOffset.x, c.y + m_ReferenceOffset.y, c.z + m_ReferenceOffset.z));
        return m;
    }

    BoundingBox App::ReferenceBounds() const {
        Matrix m = ReferenceMatrix();
        BoundingBox b = m_ReferenceBox;
        Vector3 lo = { 1e30f, 1e30f, 1e30f }, hi = { -1e30f, -1e30f, -1e30f };
        for (int i = 0; i < 8; i++) {
            Vector3 c = { (i & 1) ? b.max.x : b.min.x, (i & 2) ? b.max.y : b.min.y, (i & 4) ? b.max.z : b.min.z };
            Vector3 w = Vector3Transform(c, m);
            lo = Vector3Min(lo, w);
            hi = Vector3Max(hi, w);
        }
        return { lo, hi };
    }

    // The solid modes draw first and write depth, so shapes are hidden inside the mesh and show where they poke out;
    // the ghost mode draws last, see-through and without depth.
    void App::DrawReference(bool solidPass) {
        if (!m_HasReference || !m_ShowReference) return;
        bool solid = m_ReferenceMode != ReferenceMode::Ghost;
        if (solid != solidPass) return;
        m_Reference.transform = ReferenceMatrix();
        rlDrawRenderBatchActive();
        // imported meshes often wind some (or all) faces the wrong way: draw both sides so the nearest surface wins
        rlDisableBackfaceCulling();
        if (solid) {
            DrawModel(m_Reference, { 0, 0, 0 }, 1.0f, m_ReferenceColor);
        } else {
            rlDisableDepthMask();
            DrawModel(m_Reference, { 0, 0, 0 }, 1.0f, ColorAlpha(m_ReferenceColor, m_ReferenceAlpha));
            rlDrawRenderBatchActive();
            rlEnableDepthMask();
        }
        rlDrawRenderBatchActive();
        rlEnableBackfaceCulling();
    }

    // X-ray: the material shapes again, faint and through everything, so the parts buried in the mesh stay visible.
    void App::DrawShapeGhosts() {
        if (!m_HasReference || !m_ShowReference || m_ReferenceMode != ReferenceMode::SolidXray) return;
        std::vector<int> selected = SelectedShapeIndices();
        std::unordered_set<int> sel(selected.begin(), selected.end());
        rlDrawRenderBatchActive();
        rlDisableDepthTest();
        rlDisableDepthMask();
        for (int i = 0; i < (int)m_Design.shapes.size(); i++) {
            if (!m_Design.AddsMaterial(i)) continue;
            const Shape& s = m_Design.shapes[i];
            ShapeModel& sm = ShapeMesh(s);
            sm.model.transform = s.sf.Pose().ToMatrix();
            bool isSel = sel.count(i) > 0;
            DrawModel(sm.model, { 0, 0, 0 }, 1.0f, ColorAlpha(isSel ? ColorBrightness(s.color, 0.3f) : s.color, isSel ? 0.35f : 0.18f));
            rlDrawRenderBatchActive();
        }
        rlEnableDepthMask();
        rlEnableDepthTest();
    }

    void App::DrawShapes(const std::vector<Rigid>& xf, bool output, float alpha) {
        const Design& d = m_Design;
        int n = (int)d.shapes.size();
        if (n == 0) return;
        Vector3 eye = (output ? m_OutputView : m_DesignView).CameraPosition();
        std::vector<int> selected = SelectedShapeIndices();
        std::unordered_set<int> sel(selected.begin(), selected.end());
        // shapes involved in a collision: anywhere in the motion (design view), or right now (output view)
        std::unordered_set<int> hot;
        if (m_HasSweep && m_ShowContacts) {
            if (output) {
                if (const SweepStep* st = CurrentStep())
                    for (int id : st->shapes) hot.insert(id);
            } else {
                for (const Collision& c : m_Sweep.collisions) { hot.insert(c.shapeA); hot.insert(c.shapeB); }
            }
        }
        struct Item { int i; float dist; Matrix m; bool cut; };
        std::vector<Item> items;
        for (int i = 0; i < n; i++) {
            const Shape& s = d.shapes[i];
            bool cut = !d.AddsMaterial(i);
            if (cut && (output || !m_ShowCuts)) continue;
            int g = d.GroupIndex(s.group);
            Rigid pose = s.sf.Pose().Then(g >= 0 ? xf[g] : Rigid{});
            items.push_back({ i, Vector3DistanceSqr(pose.position, eye), pose.ToMatrix(), cut });
        }
        std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.dist > b.dist; });
        float pulse = Pulse();
        // material first (translucent only when asked)
        bool translucent = alpha < 0.97f;
        rlDrawRenderBatchActive();
        if (translucent) rlDisableDepthMask();
        for (const Item& it : items) {
            if (it.cut) continue;
            const Shape& s = d.shapes[it.i];
            ShapeModel& sm = ShapeMesh(s);
            sm.model.transform = it.m;
            Color c = s.color;
            float a = alpha;
            if (sel.count(it.i)) { c = ColorBrightness(c, 0.3f); a = fminf(1.0f, alpha + 0.2f); }
            if (hot.count(s.id)) c = Mix(c, Danger, output ? 0.85f * pulse : 0.55f);
            DrawModel(sm.model, { 0, 0, 0 }, 1.0f, ColorAlpha(c, a));
            rlDrawRenderBatchActive();
        }
        if (translucent) rlEnableDepthMask();
        // cuts as red ghosts where they reach out of the material; selected ones x-rayed through it
        rlDisableDepthMask();
        for (const Item& it : items) {
            if (!it.cut) continue;
            const Shape& s = d.shapes[it.i];
            ShapeModel& sm = ShapeMesh(s);
            sm.model.transform = it.m;
            bool isSel = sel.count(it.i) > 0;
            if (isSel) rlDisableDepthTest();
            DrawModel(sm.model, { 0, 0, 0 }, 1.0f, ColorAlpha(Color{ 235, 80, 80, 255 }, isSel ? 0.3f : 0.2f));
            rlDrawRenderBatchActive();
            if (isSel) rlEnableDepthTest();
        }
        rlEnableDepthMask();
        // outlines: a single selected shape (whole groups are told apart by colour alone)
        for (const Item& it : items) {
            const Shape& s = d.shapes[it.i];
            bool isSel = sel.count(it.i) > 0;
            if (!isSel || selected.size() > 1) continue;
            ShapeModel& sm = ShapeMesh(s);
            sm.model.transform = it.m;
            Color c = ColorAlpha(it.cut ? Color{ 255, 140, 140, 255 } : WHITE, 0.14f);
            if (it.cut && isSel) rlDisableDepthTest();
            DrawModelWires(sm.model, { 0, 0, 0 }, 1.0f, c);
            rlDrawRenderBatchActive();
            if (it.cut && isSel) rlEnableDepthTest();
        }
        rlDrawRenderBatchActive();
    }

    void App::DrawHingeHardware(const std::vector<Rigid>& xf, bool output) {
        const Design& d = m_Design;
        bool built = output && m_ShowBuilt && m_HasParts && !m_PartModels.empty() && PartsCurrent();
        const SweepStep* st = output ? CurrentStep() : nullptr;
        rlDrawRenderBatchActive();
        for (int h = 0; h < (int)d.hinges.size(); h++) {
            const Hinge& hg = d.hinges[h];
            int pi = d.GroupIndex(hg.parent), ci = d.GroupIndex(hg.child);
            if (pi < 0 || ci < 0) continue;
            Vector3 pivot, axis;
            d.HingeInWorld(h, xf, &pivot, &axis);
            axis = Vector3Normalize(axis);
            bool sel = hg.id == m_Sel.hinge;
            bool hot = false;
            if (st) hot = std::find(st->hinges.begin(), st->hinges.end(), hg.id) != st->hinges.end();
            else if (!output && m_HasSweep && m_ShowContacts)
                for (const Collision& c : m_Sweep.collisions) hot |= c.hingeA == hg.id || c.hingeB == hg.id;
            // the knuckles (the built parts already have them)
            if (hg.hardware && !built) {
                Color pc = d.groups[pi].color, cc = d.groups[ci].color;
                for (const Knuckle& k : d.Knuckles(h)) {
                    Vector3 a = Vector3Add(pivot, Vector3Scale(axis, k.lo)), b = Vector3Add(pivot, Vector3Scale(axis, k.hi));
                    Color c = k.child ? cc : pc;
                    if (sel) c = ColorBrightness(c, 0.3f);
                    if (hot) c = Mix(c, Danger, 0.6f);
                    DrawCylinderEx(a, b, hg.radius, hg.radius, 20, c);
                    DrawCylinderWiresEx(a, b, hg.radius, hg.radius, 10, ColorAlpha(BLACK, 0.25f));
                }
            }
            // the axis
            float half = hg.length * 0.5f + hg.radius * 1.5f;
            Color axisCol = sel ? WHITE : Color{ 255, 205, 60, 255 };
            if (!output || sel) {
                rlDrawRenderBatchActive();
                rlDisableDepthTest();
                DrawCylinderEx(Vector3Subtract(pivot, Vector3Scale(axis, half)), Vector3Add(pivot, Vector3Scale(axis, half)),
                               fmaxf(hg.radius * 0.12f, 0.25f), fmaxf(hg.radius * 0.12f, 0.25f), 8, ColorAlpha(axisCol, sel ? 0.95f : 0.6f));
                rlDrawRenderBatchActive();
                rlEnableDepthTest();
            }
            // the range: where the child's leaf points from open (green) to closed (orange)
            if (!output && (sel || m_Sel.Empty())) {
                Vector3 leaf = d.LeafDirection(h, true);
                float rho = fmaxf(hg.radius * 2.6f, hg.leafLength + hg.radius);
                float sweep = hg.end - hg.start;
                int N = std::max(8, (int)(fabsf(sweep) * RAD2DEG / 6.0f));
                Vector3 end = Vector3Add(pivot, Vector3Scale(axis, -hg.length * 0.5f - hg.radius));
                rlDrawRenderBatchActive();
                rlDisableDepthTest();
                Vector3 prev = {};
                for (int k = 0; k <= N; k++) {
                    float a = hg.start + sweep * k / N;
                    Vector3 dir = Vector3RotateByQuaternion(leaf, QuaternionFromAxisAngle(axis, a));
                    Vector3 q = Vector3Add(end, Vector3Scale(dir, rho));
                    Color c = Mix(Color{ 90, 220, 120, 255 }, Color{ 255, 150, 40, 255 }, k / (float)N);
                    if (k > 0) DrawLine3D(prev, q, ColorAlpha(c, sel ? 1.0f : 0.5f));
                    if (k == 0 || k == N) DrawLine3D(end, q, ColorAlpha(c, sel ? 1.0f : 0.5f));
                    prev = q;
                }
                rlDrawRenderBatchActive();
                rlEnableDepthTest();
            }
        }
        rlDrawRenderBatchActive();
    }

    void App::DrawCollisions() {
        const SweepStep* st = CurrentStep();
        if (!st) return;
        float s = fmaxf(0.5f, m_PartsVoxel * 1.6f);
        Color c = ColorAlpha(Danger, 0.5f + 0.5f * Pulse());
        rlDrawRenderBatchActive();
        rlDisableDepthTest();
        for (const Vector3& p : st->points) DrawCubeV(p, { s, s, s }, c);
        rlDrawRenderBatchActive();
        rlEnableDepthTest();
    }

    // Where each collision happens, marked on the bodies as modelled.
    void App::DrawCollisionMarks() {
        if (!m_HasSweep || !m_ShowContacts) return;
        float r = fmaxf(0.8f, m_PartsVoxel * 2.5f);
        Color c = ColorAlpha(Danger, 0.55f + 0.45f * Pulse());
        rlDrawRenderBatchActive();
        rlDisableDepthTest();
        for (const Collision& col : m_Sweep.collisions) {
            DrawSphereEx(col.local, r, 8, 8, c);
            DrawSphereWires(col.local, r * 2.2f, 6, 10, ColorAlpha(Danger, 0.6f));
        }
        rlDrawRenderBatchActive();
        rlEnableDepthTest();
    }

    void App::RenderDesign() {
        if (!m_DesignView.Begin3D(Color{ 38, 40, 46, 255 }, m_ShowGrid)) return;
        SetShaderValue(m_Shader, m_ViewPositionLocation, &m_DesignView.GetCamera().position, SHADER_UNIFORM_VEC3);
        std::vector<Rigid> rest(m_Design.groups.size());
        DrawReference(true);
        DrawShapes(rest, false, m_ShapesAlpha);
        DrawShapeGhosts();
        if (m_ShowHinges) DrawHingeHardware(rest, false);
        DrawCollisionMarks();
        DrawReference(false);
        m_DesignView.End3D();
    }

    void App::RenderOutput() {
        if (!m_OutputView.Begin3D(Color{ 32, 36, 44, 255 }, m_ShowGrid)) return;
        SetShaderValue(m_Shader, m_ViewPositionLocation, &m_OutputView.GetCamera().position, SHADER_UNIFORM_VEC3);
        const Design& d = m_Design;
        std::vector<Rigid> xf = d.Transforms(m_T, m_Ease);
        const SweepStep* st = CurrentStep();
        float pulse = Pulse();
        if (m_ReferenceInOutput) DrawReference(true);
        int selBody = -1;
        if (SelectedGroup() >= 0) selBody = d.BodyOf(m_Sel.group);
        else if (SelectedShape() >= 0) selBody = d.ShapeBody(SelectedShape());
        // the built parts once they match the design; the raw shapes while an edit is being rebuilt
        bool built = m_ShowBuilt && m_HasParts && !m_PartModels.empty() && PartsCurrent();
        if (built) {
            rlDrawRenderBatchActive();
            for (PartModel& pm : m_PartModels) {
                int g = d.GroupIndex(pm.body);
                Color c = g >= 0 ? d.groups[g].color : Color{ 200, 200, 205, 255 };
                if (pm.pinHinge >= 0) {
                    int h = d.HingeIndex(pm.pinHinge);
                    g = h >= 0 ? d.GroupIndex(d.hinges[h].parent) : -1;
                    c = Color{ 205, 205, 210, 255 };
                }
                if (g < 0) continue;
                if (pm.body >= 0 && pm.body == selBody) c = ColorBrightness(c, 0.25f);
                if (st && pm.body >= 0 && std::find(st->bodies.begin(), st->bodies.end(), pm.body) != st->bodies.end())
                    c = Mix(c, Danger, 0.35f + 0.35f * pulse);
                pm.model.transform = xf[g].ToMatrix();
                DrawModel(pm.model, { 0, 0, 0 }, 1.0f, c);
            }
            rlDrawRenderBatchActive();
        } else {
            DrawShapes(xf, true, 1.0f);
        }
        if (m_ShowHinges) DrawHingeHardware(xf, true);
        DrawCollisions();
        if (m_ReferenceInOutput) DrawReference(false);
        m_OutputView.End3D();
    }

    // ---- panels ------------------------------------------------------------------------------------------------------

    void App::DrawViewportWindow(Viewport& vp, ImVec2 pos, ImVec2 size, bool design) {
        ImGui::SetNextWindowPos(pos);
        ImGui::SetNextWindowSize(size);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4, 4));
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                                 ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
        if (design) ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.25f, 0.18f, 0.08f, 1));
        ImGui::Begin(vp.Title().c_str(), nullptr, flags);
        if (design) ImGui::PopStyleColor();
        const char* help = design
            ? "Click: select (Ctrl/Shift adds)  Q: shape/feature/body  W/E/R: move/turn/scale\n"
              "X: cut  Ctrl+G: group  Shift+A/D: add/copy  Del  RMB: orbit  MMB: pan  F: frame"
            : "Space: close / open   Hinge selected: drag its ring at open or closed\n"
              "RMB/Alt+drag: orbit  MMB/Shift+RMB: pan  Wheel: zoom  F: frame";
        if (design)
            vp.DrawPanel([this] { DesignToolbar(); }, [this](ImVec2 p, float w, float h) { return DesignOverlay(p, w, h); }, help);
        else
            vp.DrawPanel([this] { OutputToolbar(); }, [this](ImVec2 p, float w, float h) { return OutputOverlay(p, w, h); }, help);

        if (vp.Clicked()) {
            ImGuiIO& io = ImGui::GetIO();
            bool additive = io.KeyCtrl || io.KeySuper || io.KeyShift;
            std::vector<Rigid> xf = design ? std::vector<Rigid>(m_Design.groups.size()) : m_Design.Transforms(m_T, m_Ease);
            int hit = PickShape(vp, xf, design && m_ShowCuts);
            if (hit < 0) {
                if (!additive) ClearSelection();
            } else {
                const Shape& s = m_Design.shapes[hit];
                PickMode mode = design ? m_PickMode : PickMode::Body;
                if (mode == PickMode::Shape) SelectShape(s.id, additive);
                else if (mode == PickMode::Feature) SelectGroup(s.group);
                else SelectGroup(m_Design.BodyOf(s.group));
            }
        }
        ImGui::End();
        ImGui::PopStyleVar();
    }

    void App::DesignToolbar() {
        if (ImGui::RadioButton("Move", m_GizmoOp == GizmoOp::Translate)) m_GizmoOp = GizmoOp::Translate;
        ImGui::SameLine();
        if (ImGui::RadioButton("Rotate", m_GizmoOp == GizmoOp::Rotate)) m_GizmoOp = GizmoOp::Rotate;
        ImGui::SameLine();
        if (ImGui::RadioButton("Scale", m_GizmoOp == GizmoOp::Scale)) m_GizmoOp = GizmoOp::Scale;
        ImGui::SetItemTooltip("W / E / R");
        ImGui::SameLine();
        ImGui::Checkbox("Local", &m_LocalGizmo);
        ImGui::SameLine();
        ImGui::Checkbox("Snap", &m_Snap);
        ImGui::SetItemTooltip("Snap moves to %.1f mm and turns to %.0f degrees (set in View...)", m_SnapMove, m_SnapAngle);
        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();
        ImGui::TextUnformatted("Pick");
        ImGui::SameLine();
        if (ImGui::RadioButton("Shape", m_PickMode == PickMode::Shape)) m_PickMode = PickMode::Shape;
        ImGui::SameLine();
        if (ImGui::RadioButton("Feature", m_PickMode == PickMode::Feature)) m_PickMode = PickMode::Feature;
        ImGui::SameLine();
        if (ImGui::RadioButton("Body", m_PickMode == PickMode::Body)) m_PickMode = PickMode::Body;
        ImGui::SetItemTooltip("Q cycles what a click selects: a shape, the group it is in, or its whole body");
        if (ImGui::Button("Frame (F)")) FrameView(m_DesignView);
        ImGui::SameLine();
        ImGui::Checkbox("Cuts", &m_ShowCuts);
        ImGui::SetItemTooltip("Show cutting shapes as red ghosts");
        ImGui::SameLine();
        ImGui::Checkbox("Hinges", &m_ShowHinges);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(60);
        ImGui::SliderFloat("##sa", &m_ShapesAlpha, 0.1f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("Shape opacity");
        ImGui::SameLine();
        if (!m_HasReference) {
            if (ImGui::Button("Mesh...")) BrowseReference();
            ImGui::SetItemTooltip("Import a mesh (.obj .gltf .glb .iqm .vox .m3d) to fit superfrustums to.\nYou can also drop one onto a view.");
        } else {
            ImGui::Checkbox("Mesh", &m_ShowReference);
            ImGui::SetItemTooltip("Show the reference mesh");
            ImGui::SameLine();
            int mode = (int)m_ReferenceMode;
            const char* modes[] = { "ghost", "solid", "x-ray" };
            ImGui::SetNextItemWidth(70);
            if (ImGui::Combo("##refmode", &mode, modes, 3)) m_ReferenceMode = (ReferenceMode)mode;
            ImGui::SetItemTooltip("ghost: see-through\nsolid: opaque, shapes show where they poke out of it\n"
                                  "x-ray: opaque, with the shapes inside it shown through");
        }
        ImGui::SameLine();
        if (ImGui::Button("View...")) ImGui::OpenPopup("design_view");
        if (ImGui::BeginPopup("design_view")) {
            ImGui::Checkbox("Grid (1 cm)", &m_ShowGrid);
            bool ortho = m_DesignView.Ortho();
            if (ImGui::Checkbox("Orthographic (5)", &ortho)) m_DesignView.SetOrtho(ortho);
            ImGui::Checkbox("Collision highlights", &m_ShowContacts);
            ImGui::SetNextItemWidth(120);
            ImGui::DragFloat("Snap move", &m_SnapMove, 0.1f, 0.1f, 50.0f, "%.1f mm");
            ImGui::SetNextItemWidth(120);
            ImGui::DragFloat("Snap angle", &m_SnapAngle, 0.5f, 1.0f, 90.0f, "%.0f deg");
            ImGui::EndPopup();
        }
    }

    void App::DrawTimeline(float width) {
        ImVec2 p = ImGui::GetCursorScreenPos();
        float h = 22.0f;
        ImGui::InvisibleButton("##timeline", ImVec2(width, h));
        bool active = ImGui::IsItemActive();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(p, ImVec2(p.x + width, p.y + h), IM_COL32(28, 30, 36, 255), 4.0f);
        // collisions along the motion
        if (m_HasSweep && m_ShowContacts && !m_Sweep.steps.empty()) {
            int n = (int)m_Sweep.steps.size();
            float cell = width / (float)n;
            for (int k = 0; k < n; k++) {
                if (m_Sweep.steps[k].depth <= 0.0f) continue;
                float x0 = p.x + k * cell, x1 = x0 + cell + 1.0f;
                dl->AddRectFilled(ImVec2(x0, p.y + 2), ImVec2(x1, p.y + h - 2), SweepCurrent() ? IM_COL32(230, 50, 50, 220) : IM_COL32(170, 90, 90, 160));
            }
        }
        // when the selected hinge moves
        int hs = SelectedHinge();
        if (hs >= 0) {
            const Hinge& hg = m_Design.hinges[hs];
            dl->AddRectFilled(ImVec2(p.x + hg.t0 * width, p.y + h - 5), ImVec2(p.x + hg.t1 * width, p.y + h - 1), IM_COL32(255, 205, 60, 230));
        }
        float x = p.x + m_T * width;
        dl->AddLine(ImVec2(x, p.y), ImVec2(x, p.y + h), IM_COL32_WHITE, 2.0f);
        dl->AddTriangleFilled(ImVec2(x - 5, p.y), ImVec2(x + 5, p.y), ImVec2(x, p.y + 6), IM_COL32_WHITE);
        dl->AddText(ImVec2(p.x + 6, p.y + 4), IM_COL32(150, 230, 160, 200), "open");
        ImVec2 ts = ImGui::CalcTextSize("closed");
        dl->AddText(ImVec2(p.x + width - ts.x - 6, p.y + 4), IM_COL32(255, 180, 90, 200), "closed");
        if (active) {
            m_T = Clamp((ImGui::GetIO().MousePos.x - p.x) / width, 0.0f, 1.0f);
            m_Anim = 0;
        }
        if (ImGui::IsItemHovered()) {
            float t = Clamp((ImGui::GetIO().MousePos.x - p.x) / width, 0.0f, 1.0f);
            int k = m_HasSweep ? m_Sweep.StepAt(t) : -1;
            if (k >= 0 && m_Sweep.steps[k].depth > 0.0f)
                ImGui::SetTooltip("%.0f%% closed: collision, %.2f mm deep", t * 100.0f, m_Sweep.steps[k].depth);
            else
                ImGui::SetTooltip("%.0f%% closed. Drag to scrub. Red = parts collide there.", t * 100.0f);
        }
    }

    void App::OutputToolbar() {
        if (ImGui::Button(m_Anim < 0 ? "Opening..." : "Open")) { m_Anim = -1; if (m_T <= 0.0f) m_T = 1.0f; }
        ImGui::SetItemTooltip("Play the opening motion (closed -> open)");
        ImGui::SameLine();
        if (ImGui::Button(m_Anim > 0 ? "Closing..." : "Close")) { m_Anim = 1; if (m_T >= 1.0f) m_T = 0.0f; }
        ImGui::SetItemTooltip("Space. Play the closing motion (open -> closed)");
        ImGui::SameLine();
        if (ImGui::Button("Stop")) m_Anim = 0;
        ImGui::SameLine();
        ImGui::Checkbox("Loop", &m_Loop);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(70);
        ImGui::SliderFloat("##dur", &m_Duration, 0.5f, 8.0f, "%.1f s");
        ImGui::SetItemTooltip("Length of the whole motion");
        ImGui::SameLine();
        ImGui::Checkbox("Ease", &m_Ease);
        ImGui::SetItemTooltip("Each hinge eases in and out of its move");
        ImGui::SameLine();
        ImGui::Checkbox("Built", &m_ShowBuilt);
        ImGui::SetItemTooltip("Show the built parts (with the hinge hardware, cuts and clearances) instead of the raw shapes");
        ImGui::SameLine();
        if (ImGui::Button("View...")) ImGui::OpenPopup("output_view");
        if (ImGui::BeginPopup("output_view")) {
            ImGui::Checkbox("Grid (1 cm)", &m_ShowGrid);
            ImGui::Checkbox("Hinge axes", &m_ShowHinges);
            ImGui::Checkbox("Collision highlights", &m_ShowContacts);
            bool ortho = m_OutputView.Ortho();
            if (ImGui::Checkbox("Orthographic (5)", &ortho)) m_OutputView.SetOrtho(ortho);
            if (ImGui::Button("Frame (F)")) FrameView(m_OutputView);
            ImGui::EndPopup();
        }
        DrawTimeline(ImGui::GetContentRegionAvail().x);
    }

    // ---- gizmos ------------------------------------------------------------------------------------------------------

    // Moves, turns or scales a group or several shapes together. ImGuizmo gives the gizmo's new matrix; the change
    // since the drag started is applied to a snapshot of the shapes, so nothing drifts.
    bool App::GroupGizmo(const float* view, const float* proj) {
        std::vector<int> idx = SelectedShapeIndices();
        if (idx.empty()) return false;
        if (!m_GroupDrag) {
            Vector3 lo = { 1e30f, 1e30f, 1e30f }, hi = { -1e30f, -1e30f, -1e30f };
            for (int s : idx) {
                BoundingBox b = m_Design.shapes[s].sf.WorldBounds();
                lo = Vector3Min(lo, b.min);
                hi = Vector3Max(hi, b.max);
            }
            ToFloat16(MatrixTranslate((lo.x + hi.x) * 0.5f, (lo.y + hi.y) * 0.5f, (lo.z + hi.z) * 0.5f), m_GroupMatrix);
        }
        float before[16];
        memcpy(before, m_GroupMatrix, sizeof(before));
        ImGuizmo::OPERATION op = m_GizmoOp == GizmoOp::Translate ? ImGuizmo::TRANSLATE : m_GizmoOp == GizmoOp::Rotate ? ImGuizmo::ROTATE
                                                                                                                      : ImGuizmo::SCALEU;
        float snap[3] = { m_SnapMove, m_SnapMove, m_SnapMove };
        if (m_GizmoOp == GizmoOp::Rotate) snap[0] = snap[1] = snap[2] = m_SnapAngle;
        if (m_GizmoOp == GizmoOp::Scale) snap[0] = snap[1] = snap[2] = 0.1f;
        ImGuizmo::Manipulate(view, proj, op, m_GizmoOp == GizmoOp::Scale ? ImGuizmo::LOCAL : ImGuizmo::WORLD, m_GroupMatrix, nullptr,
                             m_Snap ? snap : nullptr);
        if (!ImGuizmo::IsUsing()) {
            m_GroupDrag = false;
            return false;
        }
        if (!m_GroupDrag) {
            m_GroupDrag = true;
            m_GroupStart = FromFloat16(before);
            m_DragStart = m_Design;
            m_DragShapes.clear();
            m_DragHinges.clear();
            for (int s : idx) m_DragShapes.push_back(m_Design.shapes[s].id);
            // a body brings the hinge it hangs from
            int g = SelectedGroup();
            if (g >= 0 && m_Design.groups[g].parent < 0) {
                int h = m_Design.ParentHinge(m_Design.groups[g].id);
                if (h >= 0) m_DragHinges.push_back(m_Design.hinges[h].id);
            }
        }
        Matrix now = FromFloat16(m_GroupMatrix);
        Matrix total = MatrixMultiply(MatrixInvert(m_GroupStart), now);   // start -> now
        Quaternion q = RotationOf(total);
        float scale = Vector3Length(Column(total, 0));
        if (m_GizmoOp != GizmoOp::Scale) scale = 1.0f;
        for (int id : m_DragShapes) {
            int s0 = m_DragStart.ShapeIndex(id), s = m_Design.ShapeIndex(id);
            if (s0 < 0 || s < 0) continue;
            SuperFrustum f = m_DragStart.shapes[s0].sf;
            f.position = Vector3Transform(f.position, total);
            f.rotation = QuaternionNormalize(QuaternionMultiply(q, f.rotation));
            f.size = Vector3Scale(f.size, scale);
            f.dilation *= scale;
            f.Repair(0.05f);
            m_Design.shapes[s].sf = f;
        }
        for (int id : m_DragHinges) {
            int h0 = m_DragStart.HingeIndex(id), h = m_Design.HingeIndex(id);
            if (h0 < 0 || h < 0) continue;
            m_Design.hinges[h].pivot = Vector3Transform(m_DragStart.hinges[h0].pivot, total);
            m_Design.hinges[h].axis = Vector3Normalize(Vector3RotateByQuaternion(m_DragStart.hinges[h0].axis, q));
        }
        return true;
    }

    bool App::DesignOverlay(ImVec2 pos, float w, float h) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const Camera3D& cam = m_DesignView.GetCamera();
        auto label = [&](Vector3 p, const std::string& text, ImU32 col) {
            Vector2 sp = GetWorldToScreenEx(p, cam, (int)w, (int)h);
            dl->AddText(ImVec2(pos.x + sp.x + 10, pos.y + sp.y - 8), col, text.c_str());
        };
        if (m_HasReference && m_ShowReference) dl->AddText(ImVec2(pos.x + 8, pos.y + 6), IM_COL32(255, 255, 255, 120), m_ReferenceName.c_str());
        if (m_HasSweep && m_ShowContacts) {
            std::vector<Vector3> placed;
            for (const Collision& c : m_Sweep.collisions) {
                if (placed.size() >= 6) break;
                bool close = false;
                for (const Vector3& p : placed) close |= Vector3Distance(p, c.local) < 4.0f;
                if (close) continue;
                placed.push_back(c.local);
                char buf[64];
                if (c.firstT == c.lastT) snprintf(buf, sizeof(buf), "hits at %.0f%%", c.firstT * 100.0f);
                else snprintf(buf, sizeof(buf), "hits %.0f-%.0f%%", c.firstT * 100.0f, c.lastT * 100.0f);
                label(c.local, buf, IM_COL32(255, 110, 110, 230));
            }
        }
        int ss = SelectedShape(), sg = SelectedGroup(), sh = SelectedHinge();
        if (ss >= 0) label(m_Design.shapes[ss].sf.position, m_Design.shapes[ss].name, IM_COL32(255, 255, 255, 220));
        if (sg >= 0) {
            BoundingBox b = m_Design.BodyBounds(m_Design.groups[sg].id);
            std::vector<int> in = m_Design.GroupShapes(m_Design.groups[sg].id);
            if (!in.empty()) {
                Vector3 lo = { 1e30f, 1e30f, 1e30f }, hi = { -1e30f, -1e30f, -1e30f };
                for (int s : in) {
                    BoundingBox sb = m_Design.shapes[s].sf.WorldBounds();
                    lo = Vector3Min(lo, sb.min);
                    hi = Vector3Max(hi, sb.max);
                }
                b = { lo, hi };
            }
            label({ (b.min.x + b.max.x) * 0.5f, b.max.y, (b.min.z + b.max.z) * 0.5f }, m_Design.GroupPath(m_Design.groups[sg].id),
                  IM_COL32(255, 255, 255, 220));
        }
        if (sh >= 0) {
            const Hinge& hg = m_Design.hinges[sh];
            label(hg.pivot, hg.name, IM_COL32(255, 220, 120, 230));
            Vector3 leaf = m_Design.LeafDirection(sh, true), axis = Vector3Normalize(hg.axis);
            float rho = fmaxf(hg.radius * 2.6f, hg.leafLength + hg.radius);
            Vector3 end = Vector3Add(hg.pivot, Vector3Scale(axis, -hg.length * 0.5f - hg.radius));
            Vector3 o = Vector3Add(end, Vector3Scale(Vector3RotateByQuaternion(leaf, QuaternionFromAxisAngle(axis, hg.start)), rho));
            Vector3 c = Vector3Add(end, Vector3Scale(Vector3RotateByQuaternion(leaf, QuaternionFromAxisAngle(axis, hg.end)), rho));
            label(o, "open", IM_COL32(120, 230, 140, 230));
            label(c, "closed", IM_COL32(255, 170, 70, 230));
        }

        float view[16], proj[16], xf[16];
        m_DesignView.GizmoMatrices(view, proj);
        ImGuizmo::SetOrthographic(m_DesignView.Ortho());
        ImGuizmo::SetGizmoSizeClipSpace(0.2f);
        ImGuizmo::SetDrawlist();
        ImGuizmo::SetRect(pos.x, pos.y, w, h);
        ImGuizmo::PushID(1);
        ImGuizmo::MODE mode = m_LocalGizmo ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
        float snapMove[3] = { m_SnapMove, m_SnapMove, m_SnapMove };
        float snapAngle[3] = { m_SnapAngle, m_SnapAngle, m_SnapAngle };
        const float* snap = !m_Snap ? nullptr : m_GizmoOp == GizmoOp::Rotate ? snapAngle : m_GizmoOp == GizmoOp::Translate ? snapMove : nullptr;

        if (sh >= 0) {
            Hinge& hg = m_Design.hinges[sh];
            Vector3 ax = Vector3Normalize(hg.axis);
            Vector3 ay = Utils::AnyPerpendicular(ax);
            Vector3 az = Vector3CrossProduct(ax, ay);
            ToFloat16(BasisMatrix(ax, ay, az, hg.pivot), xf);
            ImGuizmo::OPERATION op = m_GizmoOp == GizmoOp::Rotate ? ImGuizmo::ROTATE : ImGuizmo::TRANSLATE;
            if (ImGuizmo::Manipulate(view, proj, op, m_GizmoOp == GizmoOp::Rotate ? ImGuizmo::LOCAL : mode, xf, nullptr, snap)) {
                Matrix r = FromFloat16(xf);
                hg.pivot = Column(r, 3);
                if (m_GizmoOp == GizmoOp::Rotate) hg.axis = Vector3Normalize(Column(r, 0));
            }
        } else if (ss >= 0) {
            SuperFrustum& f = m_Design.shapes[ss].sf;
            Vector3 ax = f.Axis(0), ay = f.Axis(1), az = f.Axis(2);
            Matrix m = BasisMatrix(Vector3Scale(ax, f.size.x), Vector3Scale(ay, f.size.y), Vector3Scale(az, f.size.z), f.position);
            ToFloat16(m, xf);
            ImGuizmo::OPERATION op = m_GizmoOp == GizmoOp::Translate ? ImGuizmo::TRANSLATE
                                   : m_GizmoOp == GizmoOp::Rotate ? ImGuizmo::ROTATE : ImGuizmo::SCALE;
            if (ImGuizmo::Manipulate(view, proj, op, m_GizmoOp == GizmoOp::Scale ? ImGuizmo::LOCAL : mode, xf, nullptr, snap)) {
                Matrix r = FromFloat16(xf);
                f.size = { Vector3Length(Column(r, 0)), Vector3Length(Column(r, 1)), Vector3Length(Column(r, 2)) };
                f.rotation = RotationOf(r);
                f.position = Column(r, 3);
                f.Repair(0.05f);
            }
        } else if (sg >= 0 || m_Sel.shapes.size() > 1) {
            GroupGizmo(view, proj);
        }
        bool used = ImGuizmo::IsOver() || ImGuizmo::IsUsing();
        ImGuizmo::PopID();
        return used;
    }

    bool App::OutputOverlay(ImVec2 pos, float w, float h) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        // a banner while parts collide right now
        const SweepStep* st = CurrentStep();
        if (st && st->depth > 0.0f) {
            char buf[96];
            snprintf(buf, sizeof(buf), "COLLISION  %.2f mm  at %.0f%% closed", st->depth, st->t * 100.0f);
            ImVec2 ts = ImGui::CalcTextSize(buf);
            ImVec2 a(pos.x + (w - ts.x) * 0.5f - 10, pos.y + 8), b(pos.x + (w + ts.x) * 0.5f + 10, pos.y + 14 + ts.y);
            dl->AddRectFilled(a, b, IM_COL32(200, 30, 30, (int)(150 + 90 * Pulse())), 4.0f);
            dl->AddText(ImVec2(a.x + 10, a.y + 3), IM_COL32_WHITE, buf);
        }
        bool building = m_BuildJob.Busy() || m_SweepJob.Busy();
        if (building || !PartsCurrent()) {
            const char* msg = m_BuildJob.Busy() ? "building parts..." : m_SweepJob.Busy() ? "checking motion..." : "waiting to rebuild...";
            dl->AddText(ImVec2(pos.x + 8, pos.y + 6), IM_COL32(255, 220, 140, 200), msg);
        }

        // a ring about the selected hinge's axis: at the open or closed end it sets that angle
        int sh = SelectedHinge();
        if (sh < 0) return false;
        bool atOpen = m_T <= 0.001f, atClosed = m_T >= 0.999f;
        Design& d = m_Design;
        std::vector<Rigid> t = d.Transforms(m_T, m_Ease);
        Vector3 pivot, axis;
        d.HingeInWorld(sh, t, &pivot, &axis);
        axis = Vector3Normalize(axis);
        Vector2 sp = GetWorldToScreenEx(pivot, m_OutputView.GetCamera(), (int)w, (int)h);
        const char* hint = atOpen ? "drag the ring: open angle" : atClosed ? "drag the ring: closed angle" : "scrub to open or closed to edit";
        dl->AddText(ImVec2(pos.x + sp.x + 12, pos.y + sp.y + 4), IM_COL32(255, 220, 120, 220), hint);
        if (!atOpen && !atClosed) return false;
        Hinge& hg = d.hinges[sh];
        int ci = d.GroupIndex(hg.child);
        Vector3 centre = pivot;
        if (ci >= 0) {
            BoundingBox b = d.BodyBounds(hg.child);
            centre = t[ci].Apply(Vector3Scale(Vector3Add(b.min, b.max), 0.5f));
        }
        Vector3 u = Vector3Subtract(centre, pivot);
        u = Vector3Subtract(u, Vector3Scale(axis, Vector3DotProduct(u, axis)));
        u = Vector3Length(u) > 1e-5f ? Vector3Normalize(u) : Utils::AnyPerpendicular(axis);
        Vector3 v = Vector3CrossProduct(axis, u);
        float view[16], proj[16], xf[16];
        m_OutputView.GizmoMatrices(view, proj);
        ImGuizmo::SetOrthographic(m_OutputView.Ortho());
        ImGuizmo::SetGizmoSizeClipSpace(0.2f);
        ImGuizmo::SetDrawlist();
        ImGuizmo::SetRect(pos.x, pos.y, w, h);
        ImGuizmo::PushID(2);
        ToFloat16(BasisMatrix(axis, u, v, pivot), xf);
        if (ImGuizmo::Manipulate(view, proj, ImGuizmo::ROTATE_X, ImGuizmo::LOCAL, xf)) {
            Vector3 nu = Vector3Normalize(Column(FromFloat16(xf), 1));
            float delta = atan2f(Vector3DotProduct(Vector3CrossProduct(u, nu), axis), Vector3DotProduct(u, nu));
            float& a = atOpen ? hg.start : hg.end;
            float other = atOpen ? hg.end : hg.start;
            a = Clamp(a + delta, -2 * PI, 2 * PI);
            a = Clamp(a, other - 2 * PI, other + 2 * PI);    // at most one full turn of travel
        }
        bool used = ImGuizmo::IsOver() || ImGuizmo::IsUsing();
        ImGuizmo::PopID();
        return used;
    }
}
