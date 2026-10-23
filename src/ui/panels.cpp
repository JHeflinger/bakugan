#include "core/app.hpp"
#include "util/utils.hpp"
#include <imgui_internal.h>
#include <algorithm>
#include <raymath.h>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <functional>
#include <unordered_set>

namespace Bakugan {
    static bool ColorEdit(const char* id, Color* c) {
        float f[3] = { c->r / 255.0f, c->g / 255.0f, c->b / 255.0f };
        if (!ImGui::ColorEdit3(id, f, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel)) return false;
        *c = { (unsigned char)(f[0] * 255), (unsigned char)(f[1] * 255), (unsigned char)(f[2] * 255), 255 };
        return true;
    }

    static void Swatch(Color c, bool hollow = false) {
        ImVec2 p = ImGui::GetCursorScreenPos();
        float s = ImGui::GetTextLineHeight();
        ImU32 col = IM_COL32(c.r, c.g, c.b, 255);
        if (hollow) ImGui::GetWindowDrawList()->AddRect(ImVec2(p.x + 1, p.y + 2), ImVec2(p.x + s - 1, p.y + s - 2), col, 3.0f, 0, 1.5f);
        else ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(p.x, p.y + 1), ImVec2(p.x + s, p.y + s - 1), col, 3.0f);
        ImGui::Dummy(ImVec2(s, s));
    }

    static void Hint(const char* text) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::TextWrapped("%s", text);
        ImGui::PopStyleColor();
    }

    static void Warn(const char* text) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.78f, 0.35f, 1.0f));
        ImGui::TextWrapped("%s", text);
        ImGui::PopStyleColor();
    }

    // A small eye drawn at the right end of the last item's row; returns true when clicked. `inherited`: shown dim
    // because an enclosing group is hidden.
    static bool EyeButton(bool visible, bool inherited) {
        float h = ImGui::GetTextLineHeight();
        float w = h * 1.5f;
        ImVec2 rowMin = ImGui::GetItemRectMin(), rowMax = ImGui::GetItemRectMax();
        ImGui::SameLine();
        ImVec2 pos(rowMax.x - w - 2.0f, rowMin.y + (rowMax.y - rowMin.y - h) * 0.5f);
        ImGui::SetCursorScreenPos(pos);
        bool clicked = ImGui::InvisibleButton("##eye", ImVec2(w, h));
        bool hovered = ImGui::IsItemHovered();
        ImGui::SetItemTooltip(visible ? "Hide (H)" : "Show");
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 c(pos.x + w * 0.5f, pos.y + h * 0.5f);
        float r = h * 0.55f;
        int alpha = hovered ? 255 : (visible && !inherited ? 200 : 110);
        ImU32 col = IM_COL32(220, 225, 235, alpha);
        if (visible) {
            dl->AddBezierQuadratic(ImVec2(c.x - r, c.y), ImVec2(c.x, c.y - r * 1.1f), ImVec2(c.x + r, c.y), col, 1.4f);
            dl->AddBezierQuadratic(ImVec2(c.x - r, c.y), ImVec2(c.x, c.y + r * 1.1f), ImVec2(c.x + r, c.y), col, 1.4f);
            dl->AddCircleFilled(c, r * 0.32f, col, 12);
        } else {
            // a closed lid with lashes
            dl->AddBezierQuadratic(ImVec2(c.x - r, c.y - r * 0.1f), ImVec2(c.x, c.y + r * 0.9f), ImVec2(c.x + r, c.y - r * 0.1f), col, 1.4f);
            for (int k = -1; k <= 1; k++) {
                float x = c.x + k * r * 0.5f, y = c.y + r * (k == 0 ? 0.4f : 0.3f);
                dl->AddLine(ImVec2(x, y), ImVec2(x + k * r * 0.2f, y + r * 0.35f), col, 1.2f);
            }
        }
        return clicked;
    }

    // True when the mouse is over the eye column of the last item's row (so a click there doesn't select the row).
    static bool OverEye() {
        float w = ImGui::GetTextLineHeight() * 1.5f + 4.0f;
        return ImGui::GetIO().MousePos.x > ImGui::GetItemRectMax().x - w;
    }

    static const ImVec4 Red = ImVec4(1.0f, 0.42f, 0.4f, 1.0f);
    static const ImVec4 Green = ImVec4(0.4f, 0.85f, 0.5f, 1.0f);
    static const ImVec4 Amber = ImVec4(1.0f, 0.78f, 0.35f, 1.0f);

    // Things the tree asks for, applied after it is drawn (the tree reads the design while it draws).
    struct TreeAction {
        enum Kind { None, MoveShape, MoveGroup, NewBodyFrom, NewBodyFromGroup, ToggleShapeCut, ToggleGroupCut, MakeBody, AddFeature,
                    Delete, Duplicate, Ungroup } kind = None;
        int id = -1;
        int target = -1;
    };

    bool App::DrawJob(const JobControl* c, bool busy, const char* id, bool stoppable) {
        if (!busy || !c) return false;
        ImGui::PushID(id);
        std::string stage = c->Stage();
        ImGui::ProgressBar(c->Progress(), ImVec2(stoppable ? ImGui::GetContentRegionAvail().x - 60 : -FLT_MIN, 0), stage.c_str());
        if (stoppable) {
            ImGui::SameLine();
            if (ImGui::Button("Stop", ImVec2(-FLT_MIN, 0))) m_ExportJob.Cancel();
        }
        ImGui::PopID();
        return true;
    }

    std::string App::ItemName(int shapeId, int hingeId) const {
        int s = m_Design.ShapeIndex(shapeId);
        if (s >= 0) return m_Design.shapes[s].name;
        int h = m_Design.HingeIndex(hingeId);
        if (h >= 0) return m_Design.hinges[h].name + " hardware";
        return "?";
    }

    void App::DrawDesignHeader() {
        if (!ImGui::CollapsingHeader("Design", ImGuiTreeNodeFlags_DefaultOpen)) return;
        std::vector<std::string> names = TemplateNames();
        float w = ImGui::GetContentRegionAvail().x;
        ImGui::SetNextItemWidth(w * 0.62f);
        if (ImGui::BeginCombo("##template", names[m_TemplateIndex].c_str())) {
            for (int i = 0; i < (int)names.size(); i++)
                if (ImGui::Selectable(names[i].c_str(), i == m_TemplateIndex)) m_TemplateIndex = i;
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        if (ImGui::Button("New from template", ImVec2(-FLT_MIN, 0))) LoadTemplate(m_TemplateIndex);
        ImGui::SetItemTooltip("Replaces the design (Cmd/Ctrl+Z brings the old one back)");
        float third = (ImGui::GetContentRegionAvail().x - 2 * ImGui::GetStyle().ItemSpacing.x) / 3.0f;
        if (ImGui::Button("Open...", ImVec2(third, 0))) BrowseDesign(false);
        ImGui::SetItemTooltip("Open a design .txt (you can also drop one onto a view)");
        ImGui::SameLine();
        if (ImGui::Button("Save", ImVec2(third, 0))) SaveDesign();
        ImGui::SetItemTooltip("Cmd/Ctrl+S  ->  %s", m_DesignPath);
        ImGui::SameLine();
        if (ImGui::Button("Save as...", ImVec2(-FLT_MIN, 0))) BrowseDesign(true);
    }

    void App::DrawEditBar() {
        ImGui::BeginDisabled(m_UndoStack.empty());
        if (ImGui::Button("Undo")) Undo();
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("Cmd/Ctrl+Z  (%d steps)", (int)m_UndoStack.size());
        ImGui::SameLine();
        ImGui::BeginDisabled(m_RedoStack.empty());
        if (ImGui::Button("Redo")) Redo();
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("Cmd/Ctrl+Shift+Z or Ctrl+Y  (%d steps)", (int)m_RedoStack.size());
        ImGui::SameLine();
        ImGui::TextDisabled("%d shapes, %d bodies, %d hinges", (int)m_Design.shapes.size(), (int)m_Design.Bodies().size(),
                            (int)m_Design.hinges.size());
    }

    void App::DrawSidebar(float width, float height) {
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(width, height));
        ImGui::Begin("Bakugan Hinges", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                                                ImGuiWindowFlags_NoSavedSettings);
        DrawDesignHeader();
        DrawEditBar();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.75f, 0.85f, 1.0f, 1.0f));
        ImGui::TextWrapped("%s", m_Status.c_str());
        ImGui::PopStyleColor();
        // the motion check, always in sight
        if (!m_HasSweep) {
            ImGui::TextDisabled("Checking the closing motion...");
        } else if (m_Sweep.collisions.empty()) {
            ImGui::TextColored(Green, "Closing motion is clear.");
        } else {
            const Collision& worst = m_Sweep.collisions[0];
            ImGui::TextColored(Red, "!! %d collision%s while closing", (int)m_Sweep.collisions.size(), m_Sweep.collisions.size() == 1 ? "" : "s");
            ImGui::SetItemTooltip("Deepest: %.2f mm at %.0f%% closed. Details in the Hinges tab.", worst.depth, worst.worstT * 100.0f);
            ImGui::SameLine();
            if (ImGui::SmallButton("Show")) { m_T = worst.worstT; m_Anim = 0; }
            ImGui::SetItemTooltip("Jump the output view to the worst moment (details in the Hinges tab)");
        }
        if (m_HasSweep && !SweepCurrent()) {
            ImGui::SameLine();
            ImGui::TextDisabled("(updating)");
        }
        if (ImGui::BeginTabBar("tabs")) {
            if (ImGui::BeginTabItem("Model")) {
                DrawModelTab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Hinges")) {
                DrawHingesTab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Print")) {
                DrawPrintTab();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::End();
    }

    // ---- model -----------------------------------------------------------------------------------------------------

    void App::DrawAddButtons() {
        int ctx = ContextGroup();
        std::string where = ctx >= 0 ? m_Design.GroupPath(ctx) : std::string("a new body");
        ImGui::SeparatorText("Add superfrustum");
        struct P { const char* label; Preset p; const char* tip; };
        const P presets[] = { { "Box", Preset::Box, "Rectangular block (Shift+A)" },
                              { "Cyl", Preset::Cylinder, "Cylinder (rounding 1)" },
                              { "Cone", Preset::Cone, "Cone (taper 0)" },
                              { "Ball", Preset::Ball, "Sphere (dilated point)" },
                              { "Wedge", Preset::Wedge, "Wedge / ridge (taper 0, flat section)" },
                              { "Tube", Preset::Tube, "Hollow tube (onion)" },
                              { "Arc", Preset::Arc, "Bent rod (bulge)" } };
        float bw = (ImGui::GetContentRegionAvail().x - 6 * ImGui::GetStyle().ItemSpacing.x) / 7.0f;
        for (int i = 0; i < 7; i++) {
            if (i) ImGui::SameLine();
            if (ImGui::Button(presets[i].label, ImVec2(bw, 0))) AddPreset(presets[i].p);
            ImGui::SetItemTooltip("%s\nGoes into %s", presets[i].tip, where.c_str());
        }
        float fifth = (ImGui::GetContentRegionAvail().x - 4 * ImGui::GetStyle().ItemSpacing.x) / 5.0f;
        if (ImGui::Button("New body", ImVec2(fifth, 0))) NewBody();
        ImGui::SetItemTooltip("An empty body: a rigid object that hinges and prints as one part");
        ImGui::SameLine();
        ImGui::BeginDisabled(m_Sel.shapes.empty() && m_Sel.group < 0);
        if (ImGui::Button("Group", ImVec2(fifth, 0))) GroupSelection();
        ImGui::SetItemTooltip("Ctrl+G. Shapes from one body become a feature of it (several superfrustums acting as one);\n"
                              "shapes from different bodies become a new body.");
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(SelectedGroup() < 0);
        if (ImGui::Button("Ungroup", ImVec2(fifth, 0))) UngroupSelection();
        ImGui::SetItemTooltip("Ctrl+Shift+G");
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(m_Sel.Empty() || SelectedHinge() >= 0);
        if (ImGui::Button("Copy", ImVec2(fifth, 0))) DuplicateSelection();
        ImGui::SetItemTooltip("Duplicate (Shift+D / Ctrl+D)");
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(m_Sel.Empty());
        if (ImGui::Button("Delete", ImVec2(-FLT_MIN, 0))) DeleteSelection();
        ImGui::SetItemTooltip("Del");
        ImGui::EndDisabled();
    }

    void App::DrawModelTab() {
        DrawAddButtons();
        DrawTree();
        if (SelectedShape() >= 0) DrawShapeInspector();
        else if (SelectedGroup() >= 0) DrawGroupInspector();
        else if (m_Sel.shapes.size() > 1) {
            ImGui::SeparatorText("Selection");
            ImGui::Text("%d shapes selected", (int)m_Sel.shapes.size());
            if (ImGui::Button("Group them (Ctrl+G)")) GroupSelection();
            ImGui::SameLine();
            if (ImGui::Button("Toggle cut (X)")) ToggleCut();
            Hint("The gizmo moves, turns and scales them together.");
        } else {
            Hint("Select a shape or a group: click it in the design view (Q picks shape / feature / body) or in the list.");
        }
        DrawReferencePanel();
    }

    void App::DrawTree() {
        Design& d = m_Design;
        TreeAction act;
        std::unordered_set<int> hot;
        if (m_HasSweep)
            for (const Collision& c : m_Sweep.collisions) { hot.insert(c.shapeA); hot.insert(c.shapeB); }
        auto dragSource = [](int kind, int id, const char* name) {
            if (ImGui::BeginDragDropSource()) {
                int payload[2] = { kind, id };
                ImGui::SetDragDropPayload("ITEM", payload, sizeof(payload));
                ImGui::Text("Move %s", name);
                ImGui::EndDragDropSource();
            }
        };
        auto dropTarget = [&](int groupId) {
            if (!ImGui::BeginDragDropTarget()) return;
            if (const ImGuiPayload* p = ImGui::AcceptDragDropPayload("ITEM")) {
                const int* v = (const int*)p->Data;
                act = { v[0] == 0 ? TreeAction::MoveShape : TreeAction::MoveGroup, v[1], groupId };
            }
            ImGui::EndDragDropTarget();
        };
        ImGui::SeparatorText("Bodies, features and shapes");
        if (ImGui::BeginChild("tree", ImVec2(0, 210), ImGuiChildFlags_Borders | ImGuiChildFlags_ResizeY)) {
            if (d.groups.empty()) ImGui::TextDisabled("Nothing yet. Add a shape above, or start\nfrom a template.");
            std::function<void(int)> shapeNode = [&](int s) {
                const Shape& sh = d.shapes[s];
                ImGui::PushID(sh.id);
                bool sel = std::find(m_Sel.shapes.begin(), m_Sel.shapes.end(), sh.id) != m_Sel.shapes.end();
                ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_SpanAvailWidth |
                                           ImGuiTreeNodeFlags_AllowOverlap;
                if (sel) flags |= ImGuiTreeNodeFlags_Selected;
                Swatch(sh.op == Op::Cut ? Color{ 235, 90, 90, 255 } : sh.color, sh.op == Op::Cut);
                ImGui::SameLine();
                bool isHot = hot.count(sh.id) > 0;
                bool hidden = ShapeHidden(s);
                if (isHot) ImGui::PushStyleColor(ImGuiCol_Text, Red);
                else if (hidden) ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                ImGui::TreeNodeEx("shape", flags, "%s%s%s", sh.name.c_str(), sh.op == Op::Cut ? "  (cut)" : "", isHot ? "  !!" : "");
                if (isHot || hidden) ImGui::PopStyleColor();
                if (isHot) ImGui::SetItemTooltip("Collides while closing (see the Hinges tab)");
                if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen() && !OverEye()) {
                    ImGuiIO& io = ImGui::GetIO();
                    SelectShape(sh.id, io.KeyCtrl || io.KeySuper || io.KeyShift);
                }
                dragSource(0, sh.id, sh.name.c_str());
                if (ImGui::BeginPopupContextItem()) {
                    if (!sel) SelectShape(sh.id, false);
                    if (ImGui::MenuItem(sh.op == Op::Cut ? "Make it add material" : "Make it cut (X)")) act = { TreeAction::ToggleShapeCut, sh.id };
                    if (ImGui::MenuItem("Move to a new body")) act = { TreeAction::NewBodyFrom, sh.id };
                    if (ImGui::MenuItem("Duplicate")) act = { TreeAction::Duplicate, sh.id };
                    if (ImGui::MenuItem("Delete")) act = { TreeAction::Delete, sh.id };
                    ImGui::EndPopup();
                }
                if (EyeButton(!m_Hidden.count(sh.id), GroupHidden(sh.group))) ToggleHidden(sh.id);
                ImGui::PopID();
            };
            std::function<void(int)> groupNode = [&](int g) {
                const Group& grp = d.groups[g];
                bool body = grp.parent < 0;
                ImGui::PushID(grp.id);
                ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth |
                                           ImGuiTreeNodeFlags_AllowOverlap;
                if (m_Sel.group == grp.id) flags |= ImGuiTreeNodeFlags_Selected;
                bool hidden = GroupHidden(grp.id);
                Swatch(grp.color, !body);
                ImGui::SameLine();
                char label[160];
                if (body) {
                    int h = d.ParentHinge(grp.id);
                    snprintf(label, sizeof(label), "%s%s", grp.name.c_str(), h >= 0 ? "   (hinged)" : "   (body)");
                } else {
                    snprintf(label, sizeof(label), "%s   [feature%s%s]", grp.name.c_str(), grp.op == Op::Cut ? ", cut" : "",
                             grp.blend > 0.0f ? ", blended" : "");
                }
                ImVec4 text = hidden ? ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled) : body ? ImVec4(1.0f, 0.92f, 0.75f, 1.0f)
                                                                                                : ImGui::GetStyleColorVec4(ImGuiCol_Text);
                ImGui::PushStyleColor(ImGuiCol_Text, text);
                bool open = ImGui::TreeNodeEx("group", flags, "%s", label);
                ImGui::PopStyleColor();
                if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen() && !OverEye()) SelectGroup(grp.id);
                dragSource(1, grp.id, grp.name.c_str());
                dropTarget(grp.id);
                if (ImGui::BeginPopupContextItem()) {
                    SelectGroup(grp.id);
                    if (ImGui::MenuItem("New feature inside")) act = { TreeAction::AddFeature, grp.id };
                    if (!body && ImGui::MenuItem(grp.op == Op::Cut ? "Make it add material" : "Make it cut")) act = { TreeAction::ToggleGroupCut, grp.id };
                    if (!body && ImGui::MenuItem("Make it a body")) act = { TreeAction::MakeBody, grp.id };
                    if (ImGui::MenuItem("Duplicate")) act = { TreeAction::Duplicate, grp.id };
                    if (ImGui::MenuItem("Ungroup")) act = { TreeAction::Ungroup, grp.id };
                    if (ImGui::MenuItem("Delete (with contents)")) act = { TreeAction::Delete, grp.id };
                    ImGui::EndPopup();
                }
                int up = grp.parent;
                if (EyeButton(!m_Hidden.count(grp.id), up >= 0 && GroupHidden(up))) ToggleHidden(grp.id);
                if (open) {
                    for (int c : d.ChildGroups(grp.id)) groupNode(c);
                    for (int s : d.DirectShapes(grp.id)) shapeNode(s);
                    ImGui::TreePop();
                }
                ImGui::PopID();
            };
            for (int g : d.Bodies()) groupNode(g);
            // dropping on the empty space below makes a new body
            ImVec2 rest = ImGui::GetContentRegionAvail();
            ImGui::InvisibleButton("##newbodydrop", ImVec2(fmaxf(rest.x, 1.0f), fmaxf(rest.y, 18.0f)));
            if (ImGui::BeginDragDropTarget()) {
                if (const ImGuiPayload* p = ImGui::AcceptDragDropPayload("ITEM")) {
                    const int* v = (const int*)p->Data;
                    act = { v[0] == 0 ? TreeAction::NewBodyFrom : TreeAction::NewBodyFromGroup, v[1] };
                }
                ImGui::EndDragDropTarget();
            }
        }
        ImGui::EndChild();
        if (!m_Hidden.empty()) {
            if (ImGui::SmallButton("Show all (Alt+H)")) m_Hidden.clear();
            ImGui::SameLine();
            ImGui::TextDisabled("hidden parts still export");
        }
        Hint("Drag items onto a group to move them there, or below the list to make a new body. The eye hides a part in the views.");

        // apply
        switch (act.kind) {
            case TreeAction::MoveShape: d.MoveShape(d.ShapeIndex(act.id), act.target); break;
            case TreeAction::MoveGroup:
                if (!d.MoveGroup(d.GroupIndex(act.id), act.target)) m_Status = "A group can't go inside itself.";
                break;
            case TreeAction::NewBodyFrom: {
                int s = d.ShapeIndex(act.id);
                if (s < 0) break;
                int b = d.AddGroup(d.shapes[s].name + " body", -1);
                d.MoveShape(s, d.groups[b].id);
                SelectGroup(d.groups[b].id);
                break;
            }
            case TreeAction::NewBodyFromGroup: d.MoveGroup(d.GroupIndex(act.id), -1); break;
            case TreeAction::ToggleShapeCut: ToggleCut(); break;
            case TreeAction::ToggleGroupCut: {
                int g = d.GroupIndex(act.id);
                if (g >= 0) d.groups[g].op = d.groups[g].op == Op::Cut ? Op::Union : Op::Cut;
                break;
            }
            case TreeAction::MakeBody: d.MoveGroup(d.GroupIndex(act.id), -1); break;
            case TreeAction::AddFeature: {
                int g = d.AddGroup("Feature", act.id);
                SelectGroup(d.groups[g].id);
                m_Status = "New feature: shapes you add now go into it.";
                break;
            }
            case TreeAction::Delete: DeleteSelection(); break;
            case TreeAction::Duplicate: DuplicateSelection(); break;
            case TreeAction::Ungroup: UngroupSelection(); break;
            default: break;
        }
    }

    void App::DrawShapeInspector() {
        int si = SelectedShape();
        Shape& s = m_Design.shapes[si];
        SuperFrustum& f = s.sf;
        ImGui::SeparatorText("Superfrustum");
        ColorEdit("##col", &s.color);
        ImGui::SameLine();
        char name[64];
        snprintf(name, sizeof(name), "%s", s.name.c_str());
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::InputText("##name", name, sizeof(name))) s.name = name;
        std::string path = m_Design.GroupPath(s.group);
        if (ImGui::BeginCombo("Group", path.c_str())) {
            for (int g = 0; g < (int)m_Design.groups.size(); g++) {
                std::string p = m_Design.GroupPath(m_Design.groups[g].id);
                ImGui::PushID(g);
                if (ImGui::Selectable(p.c_str(), m_Design.groups[g].id == s.group)) s.group = m_Design.groups[g].id;
                ImGui::PopID();
            }
            ImGui::EndCombo();
        }
        ImGui::SetItemTooltip("The body (or feature of a body) it belongs to");
        int op = (int)s.op;
        if (ImGui::RadioButton("Adds material", op == 0)) { s.op = Op::Union; int g = m_Design.GroupIndex(s.group); if (g >= 0) s.color = m_Design.groups[g].color; }
        ImGui::SameLine();
        if (ImGui::RadioButton("Cuts (X)", op == 1)) { s.op = Op::Cut; s.color = { 225, 90, 90, 255 }; }
        ImGui::SetItemTooltip("A cut removes its volume from the rest of its group (holes, hollows, slots)");

        ImGui::DragFloat3("Position", &f.position.x, 0.1f, 0, 0, "%.2f mm");
        if (m_EulerShape != s.id || !QuaternionEquals(m_EulerQuaternion, f.rotation)) {
            m_EulerDegrees = Utils::QuaternionToEulerDegrees(f.rotation);
            m_EulerShape = s.id;
            m_EulerQuaternion = f.rotation;
        }
        if (ImGui::DragFloat3("Rotation", &m_EulerDegrees.x, 0.5f, -180, 180, "%.1f deg")) {
            f.rotation = Utils::EulerDegreesToQuaternion(m_EulerDegrees);
            m_EulerQuaternion = f.rotation;
        }
        ImGui::SetItemTooltip("The superfrustum sweeps along its local Z axis (blue gizmo arrow).");
        if (ImGui::DragFloat3("Size", &f.size.x, 0.1f, 0.05f, 1000.0f, "%.2f mm")) f.Repair(0.05f);
        ImGui::SetItemTooltip("x, y: cross section   z: length along the axis");
        ImGui::SliderFloat("Rounding", &f.rounding, 0.0f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("0 = box cross section, 1 = round (cylinder / cone / capsule)");
        ImGui::SliderFloat("Taper", &f.taper, 0.0f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("1 = prism, 0 = the top shrinks to a point or ridge");
        ImGui::SliderFloat("Bulge", &f.bulge, -1.0f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("Bends the axis into an arc");
        ImGui::DragFloat("Dilation", &f.dilation, 0.05f, 0.0f, 100.0f, "%.2f mm");
        ImGui::SetItemTooltip("Grows the solid outward, rounding every edge");
        ImGui::SliderFloat("Onion", &f.onion, 0.0f, 0.95f, "%.2f");
        ImGui::SetItemTooltip("0 = solid, otherwise a hollow shell (open at both ends)");
        f.Repair(0.05f);
    }

    void App::DrawGroupInspector() {
        int gi = SelectedGroup();
        Group& g = m_Design.groups[gi];
        bool body = g.parent < 0;
        ImGui::SeparatorText(body ? "Body" : "Feature");
        Color old = g.color;
        if (ColorEdit("##gcol", &g.color)) {
            // recolour the shapes that wore the old colour
            for (int s : m_Design.GroupShapes(g.id)) {
                Color& c = m_Design.shapes[s].color;
                if (m_Design.shapes[s].op == Op::Union && c.r == old.r && c.g == old.g && c.b == old.b) c = g.color;
            }
            for (Group& sub : m_Design.groups)
                if (sub.id != g.id && m_Design.Inside(sub.id, g.id) && sub.color.r == old.r && sub.color.g == old.g && sub.color.b == old.b)
                    sub.color = g.color;
        }
        ImGui::SameLine();
        char name[64];
        snprintf(name, sizeof(name), "%s", g.name.c_str());
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::InputText("##gname", name, sizeof(name))) g.name = name;
        std::vector<int> shapes = m_Design.GroupShapes(g.id);
        if (body) {
            ImGui::TextWrapped("A body is one rigid object: it moves as a whole on its hinge and prints as one part.");
        } else {
            ImGui::TextWrapped("A feature combines its shapes into one design element, then adds it to or cuts it from %s.",
                               m_Design.GroupPath(g.parent).c_str());
            int op = (int)g.op;
            if (ImGui::RadioButton("Adds material", op == 0)) g.op = Op::Union;
            ImGui::SameLine();
            if (ImGui::RadioButton("Cuts", op == 1)) g.op = Op::Cut;
        }
        ImGui::DragFloat("Blend", &g.blend, 0.05f, 0.0f, 50.0f, g.blend > 0.0f ? "%.2f mm" : "off");
        ImGui::SetItemTooltip("Smooth union: fillets where the shapes of this group meet");
        if (!shapes.empty()) {
            BoundingBox b = m_Design.BodyBounds(m_Design.BodyOf(g.id));
            if (!body) {
                Vector3 lo = { 1e30f, 1e30f, 1e30f }, hi = { -1e30f, -1e30f, -1e30f };
                for (int s : shapes) { BoundingBox sb = m_Design.shapes[s].sf.WorldBounds(); lo = Vector3Min(lo, sb.min); hi = Vector3Max(hi, sb.max); }
                b = { lo, hi };
            }
            Vector3 sz = Vector3Subtract(b.max, b.min);
            ImGui::TextDisabled("%d shapes   %.1f x %.1f x %.1f mm", (int)shapes.size(), sz.x, sz.y, sz.z);
        }
        Hint("The gizmo moves, turns and scales the whole group.");
        if (!body) {
            if (ImGui::Button("Make it a body")) m_Design.MoveGroup(gi, -1);
            return;
        }
        // hinges of this body
        int parentHinge = m_Design.ParentHinge(g.id);
        if (parentHinge >= 0) {
            const Hinge& h = m_Design.hinges[parentHinge];
            int p = m_Design.GroupIndex(h.parent);
            ImGui::Text("Hangs from %s on", p >= 0 ? m_Design.groups[p].name.c_str() : "?");
            ImGui::SameLine();
            if (ImGui::SmallButton(h.name.c_str())) SelectHinge(h.id);
        }
        int bodyId = g.id;
        for (const Hinge& h : m_Design.hinges) {
            if (h.parent != bodyId) continue;
            int c = m_Design.GroupIndex(h.child);
            ImGui::Text("Carries %s on", c >= 0 ? m_Design.groups[c].name.c_str() : "?");
            ImGui::SameLine();
            ImGui::PushID(h.id);
            if (ImGui::SmallButton(h.name.c_str())) { SelectHinge(h.id); ImGui::PopID(); return; }
            ImGui::PopID();
        }
        if (ImGui::BeginCombo("##hingeto", "Hinge a body to this one...")) {
            for (int b : m_Design.Bodies()) {
                const Group& other = m_Design.groups[b];
                if (other.id == bodyId) continue;
                bool ok = m_Design.CanHinge(bodyId, other.id);
                ImGui::PushID(other.id);
                if (ImGui::Selectable(other.name.c_str(), false, ok ? 0 : ImGuiSelectableFlags_Disabled)) {
                    AddHingeBetween(bodyId, other.id);
                    ImGui::PopID();
                    break;
                }
                ImGui::PopID();
            }
            ImGui::EndCombo();
        }
        ImGui::SetItemTooltip("This body stays, the picked one turns on a new hinge placed where they meet");
    }

    void App::DrawReferencePanel() {
        if (m_ReferenceOpen) { ImGui::SetNextItemOpen(true); m_ReferenceOpen = false; }
        if (!ImGui::CollapsingHeader("Reference mesh")) return;
        Hint("A mesh to fit superfrustums to by hand. Shown in the design view, never exported or checked for collisions.");
        float w = ImGui::GetContentRegionAvail().x;
        if (ImGui::Button("Import mesh...", ImVec2(w * 0.4f, 0))) BrowseReference();
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-ImGui::CalcTextSize("Load").x - ImGui::GetStyle().FramePadding.x * 2 - ImGui::GetStyle().ItemSpacing.x);
        bool enter = ImGui::InputTextWithHint("##refpath", "or type a path", m_ReferencePath, sizeof(m_ReferencePath),
                                              ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::SameLine();
        if ((ImGui::Button("Load") || enter) && m_ReferencePath[0]) LoadReference(m_ReferencePath);
        if (!m_HasReference) return;
        ImGui::Checkbox("Show", &m_ShowReference);
        ImGui::SameLine();
        ColorEdit("##refcol", &m_ReferenceColor);
        ImGui::SameLine();
        ImGui::TextUnformatted(m_ReferenceName.c_str());
        int mode = (int)m_ReferenceMode;
        if (ImGui::RadioButton("Solid", mode == (int)ReferenceMode::Solid)) m_ReferenceMode = ReferenceMode::Solid;
        ImGui::SetItemTooltip("Opaque: superfrustums show only where they poke out of the mesh, so you can see how\n"
                              "well they hug it. Shapes stay clickable through it.");
        ImGui::SameLine();
        if (ImGui::RadioButton("X-ray", mode == (int)ReferenceMode::SolidXray)) m_ReferenceMode = ReferenceMode::SolidXray;
        ImGui::SetItemTooltip("Opaque, with the shapes buried inside it drawn faintly through it");
        ImGui::SameLine();
        if (ImGui::RadioButton("Ghost", mode == (int)ReferenceMode::Ghost)) m_ReferenceMode = ReferenceMode::Ghost;
        ImGui::SetItemTooltip("See-through, drawn over the shapes");
        if (m_ReferenceMode == ReferenceMode::Ghost) ImGui::SliderFloat("Mesh opacity", &m_ReferenceAlpha, 0.05f, 0.95f, "%.2f");
        ImGui::SliderFloat("Shape opacity", &m_ShapesAlpha, 0.05f, 1.0f, m_ShapesAlpha >= 0.995f ? "solid" : "%.2f");
        ImGui::SetItemTooltip("See-through superfrustums show the mesh behind them (also in the design view's toolbar)");
        ImGui::DragFloat("Scale", &m_ReferenceScale, 0.01f, 0.001f, 1000.0f, "x%.3f");
        ImGui::DragFloat3("Offset", &m_ReferenceOffset.x, 0.1f, 0, 0, "%.1f mm");
        ImGui::DragFloat3("Rotation##ref", &m_ReferenceRotation.x, 0.5f, -360.0f, 360.0f, "%.1f deg");
        ImGui::SetItemTooltip("Turns the mesh about its own centre");
        ImGui::Checkbox("Also in the output view", &m_ReferenceInOutput);
        float half = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
        if (ImGui::Button("Drop to ground", ImVec2(half, 0))) m_ReferenceOffset.y -= ReferenceBounds().min.y;
        ImGui::SetItemTooltip("Move it so its lowest point sits on the grid");
        ImGui::SameLine();
        if (ImGui::Button("Remove", ImVec2(-FLT_MIN, 0))) {
            UnloadModel(m_Reference);
            m_HasReference = false;
        }
    }

    // ---- hinges ----------------------------------------------------------------------------------------------------

    static bool BodyCombo(const char* label, const Design& d, int* id) {
        int gi = d.GroupIndex(*id);
        bool changed = false;
        if (ImGui::BeginCombo(label, gi >= 0 ? d.groups[gi].name.c_str() : "(pick a body)")) {
            for (int b : d.Bodies()) {
                ImGui::PushID(b);
                if (ImGui::Selectable(d.groups[b].name.c_str(), d.groups[b].id == *id)) { *id = d.groups[b].id; changed = true; }
                ImGui::PopID();
            }
            ImGui::EndCombo();
        }
        return changed;
    }

    void App::DrawHingesTab() {
        Design& d = m_Design;
        std::vector<int> bodies = d.Bodies();
        ImGui::SeparatorText("New hinge");
        if (bodies.size() < 2) {
            Hint("Hinges join two bodies. Make at least two in the Model tab (New body, or group shapes from different bodies).");
        } else {
            if (d.GroupIndex(m_NewHingeParent) < 0) m_NewHingeParent = d.groups[bodies[0]].id;
            if (d.GroupIndex(m_NewHingeChild) < 0 || m_NewHingeChild == m_NewHingeParent)
                for (int b : bodies)
                    if (d.groups[b].id != m_NewHingeParent && d.ParentHinge(d.groups[b].id) < 0) { m_NewHingeChild = d.groups[b].id; break; }
            ImGui::PushID("new");
            float half = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
            ImGui::SetNextItemWidth(half - ImGui::CalcTextSize("stays").x - 8);
            BodyCombo("stays", d, &m_NewHingeParent);
            ImGui::SetItemTooltip("Parent body: stays where it is");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(half - ImGui::CalcTextSize("turns").x - 8);
            BodyCombo("turns", d, &m_NewHingeChild);
            ImGui::SetItemTooltip("Child body: turns about the hinge (and carries anything hinged to it)");
            bool ok = d.CanHinge(m_NewHingeParent, m_NewHingeChild);
            ImGui::BeginDisabled(!ok);
            if (ImGui::Button("Add hinge", ImVec2(-FLT_MIN, 0))) AddHingeBetween(m_NewHingeParent, m_NewHingeChild);
            ImGui::EndDisabled();
            if (!ok) Hint("Each body turns on at most one hinge, and a body can't hang from its own child.");
            ImGui::PopID();
        }
        DrawHingeList();
        if (SelectedHinge() >= 0) DrawHingeInspector();
        DrawCollisionPanel();
    }

    void App::DrawHingeList() {
        const Design& d = m_Design;
        ImGui::SeparatorText("Hinges");
        if (d.hinges.empty()) { ImGui::TextDisabled("No hinges yet."); return; }
        std::unordered_set<int> hot;
        if (m_HasSweep)
            for (const Collision& c : m_Sweep.collisions) { hot.insert(c.hingeA); hot.insert(c.hingeB); }
        if (ImGui::BeginChild("hinges", ImVec2(0, std::min(130.0f, 26.0f + 21.0f * d.hinges.size())), ImGuiChildFlags_Borders | ImGuiChildFlags_ResizeY)) {
            for (const Hinge& h : d.hinges) {
                int p = d.GroupIndex(h.parent), c = d.GroupIndex(h.child);
                char label[200];
                snprintf(label, sizeof(label), "%s   %s -> %s   %.0f..%.0f deg", h.name.c_str(), p >= 0 ? d.groups[p].name.c_str() : "?",
                         c >= 0 ? d.groups[c].name.c_str() : "?", h.start * RAD2DEG, h.end * RAD2DEG);
                ImGui::PushID(h.id);
                bool isHot = hot.count(h.id) > 0;
                if (isHot) ImGui::PushStyleColor(ImGuiCol_Text, Red);
                if (ImGui::Selectable(label, m_Sel.hinge == h.id)) SelectHinge(h.id);
                if (isHot) ImGui::PopStyleColor();
                ImGui::PopID();
            }
        }
        ImGui::EndChild();
    }

    void App::DrawHingeInspector() {
        Design& d = m_Design;
        int hi = SelectedHinge();
        Hinge& h = d.hinges[hi];
        ImGui::PushID(h.id);
        ImGui::SeparatorText("Hinge");
        char name[64];
        snprintf(name, sizeof(name), "%s", h.name.c_str());
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::InputText("##hname", name, sizeof(name))) h.name = name;
        int parent = h.parent, child = h.child;
        float half = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
        ImGui::SetNextItemWidth(half - ImGui::CalcTextSize("stays").x - 8);
        if (BodyCombo("stays", d, &parent) && d.CanHinge(parent, h.child, hi)) h.parent = parent;
        ImGui::SameLine();
        ImGui::SetNextItemWidth(half - ImGui::CalcTextSize("turns").x - 8);
        if (BodyCombo("turns", d, &child) && d.CanHinge(h.parent, child, hi)) h.child = child;

        ImGui::DragFloat3("Pivot", &h.pivot.x, 0.1f, 0, 0, "%.2f mm");
        ImGui::SetItemTooltip("A point on the hinge axis, in the design pose (also: select the hinge and use the gizmo)");
        if (ImGui::DragFloat3("Axis", &h.axis.x, 0.01f, -1.0f, 1.0f, "%.2f")) {
            float len = Vector3Length(h.axis);
            h.axis = len > 1e-4f ? Vector3Scale(h.axis, 1.0f / len) : Vector3{ 1, 0, 0 };
        }
        ImGui::SetItemTooltip("Positive angles turn the child about +axis by the right-hand rule");
        float fifth = (ImGui::GetContentRegionAvail().x - 4 * ImGui::GetStyle().ItemSpacing.x) / 5.0f;
        if (ImGui::Button("X", ImVec2(fifth, 0))) h.axis = { 1, 0, 0 };
        ImGui::SameLine();
        if (ImGui::Button("Y", ImVec2(fifth, 0))) h.axis = { 0, 1, 0 };
        ImGui::SameLine();
        if (ImGui::Button("Z", ImVec2(fifth, 0))) h.axis = { 0, 0, 1 };
        ImGui::SameLine();
        if (ImGui::Button("Flip", ImVec2(fifth, 0))) { h.axis = Vector3Negate(h.axis); h.start = -h.start; h.end = -h.end; }
        ImGui::SetItemTooltip("Reverse the axis (the angles flip with it, so the motion stays the same)");
        ImGui::SameLine();
        if (ImGui::Button("Place", ImVec2(-FLT_MIN, 0))) PlaceHinge(hi);
        ImGui::SetItemTooltip("Put it back on the edge where the two bodies meet (resets the angles)");

        ImGui::SeparatorText("Motion");
        float start = h.start * RAD2DEG, end = h.end * RAD2DEG;
        if (ImGui::SliderFloat("Open angle", &start, -360.0f, 360.0f, "%.1f deg")) {
            h.start = Clamp(start, end - 360.0f, end + 360.0f) * DEG2RAD;
        }
        ImGui::SetItemTooltip("Starting position (fully open), relative to the design pose (0)");
        if (ImGui::SliderFloat("Closed angle", &end, -360.0f, 360.0f, "%.1f deg")) {
            h.end = Clamp(end, start - 360.0f, start + 360.0f) * DEG2RAD;
        }
        ImGui::SetItemTooltip("Ending position (fully closed). Up to a full turn from the open angle.");
        float travel = fabsf(h.end - h.start) * RAD2DEG;
        ImGui::TextDisabled("Travel %.1f deg of 360", travel);
        ImGui::SameLine();
        if (ImGui::SmallButton("Swap")) std::swap(h.start, h.end);
        ImGui::SameLine();
        if (ImGui::SmallButton("Show open")) { m_T = 0.0f; m_Anim = 0; }
        ImGui::SameLine();
        if (ImGui::SmallButton("Show closed")) { m_T = 1.0f; m_Anim = 0; }
        float t0 = h.t0 * 100.0f, t1 = h.t1 * 100.0f;
        if (ImGui::DragFloatRange2("Moves during", &t0, &t1, 0.5f, 0.0f, 100.0f, "from %.0f%%", "to %.0f%%")) {
            h.t0 = Clamp(t0 / 100.0f, 0.0f, 1.0f);
            h.t1 = Clamp(t1 / 100.0f, h.t0, 1.0f);
        }
        ImGui::SetItemTooltip("Share of the whole closing motion while this hinge turns (sequence hinges one after another).\n"
                              "Shown in yellow under the timeline.");

        ImGui::SeparatorText("Door hinge");
        ImGui::Checkbox("Build hinge hardware", &h.hardware);
        ImGui::SetItemTooltip("Knuckles alternating between the two bodies around a pin, each set on a leaf fused into its body,\n"
                              "with the print clearance cut around the other body's knuckles. Off: an axis only.");
        ImGui::BeginDisabled(!h.hardware);
        ImGui::DragFloat("Length", &h.length, 0.1f, 2.0f, 500.0f, "%.1f mm");
        ImGui::SliderInt("Knuckles", &h.knuckles, 2, 9);
        ImGui::SetItemTooltip("Alternating: the parent (stays) takes the first, third, ...");
        ImGui::DragFloat("Knuckle radius", &h.radius, 0.05f, 0.5f, 50.0f, "%.2f mm");
        ImGui::DragFloat("Pin radius", &h.pinRadius, 0.02f, 0.2f, 40.0f, "%.2f mm");
        ImGui::DragFloat("Clearance", &h.clearance, 0.005f, 0.05f, 3.0f, "%.2f mm");
        ImGui::SetItemTooltip("Gap between the moving surfaces of the two bodies. FDM: 0.3 - 0.5 mm, resin: 0.15 - 0.25 mm.");
        h.radius = fmaxf(h.radius, h.pinRadius + h.clearance + 0.4f);
        int pin = (int)h.pin;
        const char* pins[] = { "Print in place (fused pin)", "Separate pin with head (assemble)" };
        if (ImGui::Combo("Pin", &pin, pins, 2)) h.pin = (PinStyle)pin;
        ImGui::SetItemTooltip("Print in place: the pin belongs to the parent and runs through the child's knuckles, so the two\n"
                              "bodies print together. Separate: every knuckle is bored and the pin is exported as its own part.");
        ImGui::DragFloat("Leaf length", &h.leafLength, 0.1f, 0.0f, 200.0f, h.leafLength > 0.0f ? "%.1f mm" : "none");
        ImGui::SetItemTooltip("Plates that tie each set of knuckles into its body");
        ImGui::DragFloat("Leaf thickness", &h.leafThickness, 0.05f, 0.2f, 50.0f, "%.2f mm");
        if (ImGui::Checkbox("Aim leaves at their bodies", &h.autoLeaves) && !h.autoLeaves) {
            h.leafAngle[0] = d.AutoLeafAngle(hi, false);
            h.leafAngle[1] = d.AutoLeafAngle(hi, true);
        }
        if (!h.autoLeaves) {
            ImGui::SliderAngle("Parent leaf", &h.leafAngle[0], -180.0f, 180.0f);
            ImGui::SliderAngle("Child leaf", &h.leafAngle[1], -180.0f, 180.0f);
        }
        ImGui::EndDisabled();
        if (h.hardware && h.pin == PinStyle::Fused)
            Warn("Print in place: the two bodies must print together, so export with 'Combined assembly'.");
        ImGui::PopID();
    }

    void App::DrawCollisionPanel() {
        ImGui::SeparatorText("Collisions while closing");
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.4f);
        ImGui::SliderInt("Checks", &m_SweepParams.steps, 16, 240);
        ImGui::SetItemTooltip("Moments checked between open and closed");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-ImGui::CalcTextSize("Tolerance").x - 8);
        ImGui::DragFloat("Tolerance", &m_SweepParams.tolerance, 0.005f, 0.0f, 2.0f, "%.2f mm");
        ImGui::SetItemTooltip("Overlaps shallower than this count as touching");
        DrawJob(m_SweepJob.Control(), m_SweepJob.Busy(), "sweep", false);
        if (!m_SweepStats.empty()) ImGui::TextDisabled("%s", m_SweepStats.c_str());
        if (!m_HasSweep) return;
        if (m_Sweep.collisions.empty()) {
            ImGui::TextColored(Green, "Nothing collides between open and closed.");
            return;
        }
        const Design& d = m_Design;
        auto bodyName = [&](int id) { int g = d.GroupIndex(id); return g >= 0 ? d.groups[g].name : std::string("?"); };
        int shown = 0;
        for (const Collision& c : m_Sweep.collisions) {
            if (shown++ >= 40) { ImGui::TextDisabled("... and %d more", (int)m_Sweep.collisions.size() - 40); break; }
            ImGui::PushID(shown);
            char line[256];
            snprintf(line, sizeof(line), "%s / %s  x  %s / %s", bodyName(c.bodyA).c_str(), ItemName(c.shapeA, c.hingeA).c_str(),
                     bodyName(c.bodyB).c_str(), ItemName(c.shapeB, c.hingeB).c_str());
            ImGui::PushStyleColor(ImGuiCol_Text, Red);
            if (ImGui::Selectable(line)) {
                m_T = c.worstT;
                m_Anim = 0;
                m_Sel = Selection{};
                if (d.ShapeIndex(c.shapeA) >= 0) m_Sel.shapes.push_back(c.shapeA);
                if (d.ShapeIndex(c.shapeB) >= 0) m_Sel.shapes.push_back(c.shapeB);
                if (m_Sel.shapes.empty()) m_Sel.hinge = c.hingeA >= 0 ? c.hingeA : c.hingeB;
            }
            ImGui::PopStyleColor();
            ImGui::SetItemTooltip("Click: jump to the worst moment and select both");
            if (c.firstT == c.lastT)
                ImGui::TextDisabled("    at %.0f%% closed, %.2f mm deep", c.firstT * 100.0f, c.depth);
            else
                ImGui::TextDisabled("    from %.0f%% to %.0f%% closed, up to %.2f mm deep", c.firstT * 100.0f, c.lastT * 100.0f, c.depth);
            ImGui::PopID();
        }
    }

    // ---- print -----------------------------------------------------------------------------------------------------

    void App::DrawPrintTab() {
        ImGui::SeparatorText("Parts");
        ImGui::SliderInt("Preview detail", &m_PreviewResolution, 80, 400, "%d voxels");
        ImGui::SetItemTooltip("Resolution of the parts in the output view (voxels along the design's longest side)");
        DrawJob(m_BuildJob.Control(), m_BuildJob.Busy(), "build", false);
        if (m_HasParts && m_Parts) {
            if (!PartsCurrent()) ImGui::TextColored(Amber, "Out of date: rebuilding after your edit.");
            if (ImGui::BeginTable("parts", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("part");
                ImGui::TableSetupColumn("size (mm)");
                ImGui::TableSetupColumn("triangles");
                ImGui::TableHeadersRow();
                for (const BuiltPart& p : *m_Parts) {
                    Vector3 s = Vector3Subtract(p.mesh.box.max, p.mesh.box.min);
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn(); ImGui::TextUnformatted(p.name.c_str());
                    ImGui::TableNextColumn(); ImGui::Text("%.1f x %.1f x %.1f", s.x, s.y, s.z);
                    ImGui::TableNextColumn(); ImGui::Text("%zu", p.mesh.indices.size() / 3);
                }
                ImGui::EndTable();
            }
            ImGui::TextDisabled("preview voxel %.2f mm   %s", m_PartsVoxel, m_BuildStats.c_str());
        }

        ImGui::SeparatorText("Export STL");
        ExportParams& p = m_ExportParams;
        ImGui::SetNextItemWidth(-ImGui::CalcTextSize("Browse...").x - ImGui::GetStyle().FramePadding.x * 2 - ImGui::GetStyle().ItemSpacing.x);
        ImGui::InputText("##prefix", m_ExportPath, sizeof(m_ExportPath));
        ImGui::SetItemTooltip("File prefix: writes <prefix>_<part>.stl, <prefix>_assembly.stl and <prefix>_notes.txt");
        ImGui::SameLine();
        if (ImGui::Button("Browse...")) BrowseExport();
        ImGui::SliderFloat("Print voxel", &p.voxel, 0.05f, 1.0f, "%.2f mm");
        ImGui::SetItemTooltip("Mesh resolution of the exported parts. Smaller is finer and slower.");
        float minGap = 1e9f;
        for (const Hinge& h : m_Design.hinges)
            if (h.hardware) minGap = fminf(minGap, h.clearance);
        if (minGap < 1e8f && p.voxel > minGap * 0.6f) {
            char buf[128];
            snprintf(buf, sizeof(buf), "The voxel is coarse for the %.2f mm hinge clearance: try %.2f mm or less.", minGap, minGap * 0.5f);
            Warn(buf);
        }
        ImGui::SliderFloat("Simplify", &p.adaptivity, 0.0f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("OpenVDB mesh adaptivity: merges flat areas into fewer triangles (0 = full resolution)");
        int pose = (int)p.pose;
        ImGui::TextUnformatted("Pose");
        ImGui::SameLine();
        if (ImGui::RadioButton("Open", pose == (int)ExportPose::Open)) p.pose = ExportPose::Open;
        ImGui::SameLine();
        if (ImGui::RadioButton("Closed", pose == (int)ExportPose::Closed)) p.pose = ExportPose::Closed;
        ImGui::SameLine();
        if (ImGui::RadioButton("As modelled", pose == (int)ExportPose::Design)) p.pose = ExportPose::Design;
        ImGui::SetItemTooltip("The pose every part is exported in (they stay assembled relative to each other)");
        ImGui::Checkbox("One file per part", &p.separate);
        ImGui::SameLine();
        ImGui::Checkbox("Combined assembly", &p.combined);
        ImGui::SetItemTooltip("Every part in one STL, as separate shells: what print-in-place hinges need");
        ImGui::Checkbox("Keep bodies apart", &p.keepApart);
        ImGui::SetItemTooltip("Bodies that touch in the export pose are trimmed back from each other to this gap,\n"
                              "so parts printed together don't fuse (hinge knuckles already have their own clearance)");
        ImGui::SameLine();
        ImGui::BeginDisabled(!p.keepApart);
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::DragFloat("##gap", &p.gap, 0.005f, 0.05f, 2.0f, "gap %.2f mm");
        ImGui::EndDisabled();
        ImGui::Checkbox("Z up (slicers)", &p.zUp);
        ImGui::SameLine();
        ImGui::Checkbox("Drop to bed", &p.dropToBed);
        ImGui::DragFloat("Scale", &p.scale, 0.01f, 0.01f, 100.0f, "x%.2f");
        ImGui::SetItemTooltip("Design units are millimetres");
        bool fused = false;
        for (const Hinge& h : m_Design.hinges) fused |= h.hardware && h.pin == PinStyle::Fused;
        if (fused && !p.combined)
            Warn("Print-in-place hinges need their bodies printed together: tick 'Combined assembly'.");
        if (!p.separate && !p.combined) Warn("Pick at least one of the file options.");
        ImGui::BeginDisabled(m_ExportJob.Busy() || (!p.separate && !p.combined));
        if (ImGui::Button("Export STL", ImVec2(-FLT_MIN, 30))) StartExport();
        ImGui::EndDisabled();
        DrawJob(m_ExportJob.Control(), m_ExportJob.Busy(), "export", true);
        if (!m_ExportedFiles.empty()) {
            ImGui::TextColored(Green, "Wrote:");
            for (const std::string& f : m_ExportedFiles) ImGui::BulletText("%s", f.c_str());
        }
        DrawFilesPanel();
    }

    void App::DrawFilesPanel() {
        if (!ImGui::CollapsingHeader("Design file")) return;
        ImGui::InputText("Path", m_DesignPath, sizeof(m_DesignPath));
        float half = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
        if (ImGui::Button("Save design", ImVec2(half, 0))) SaveDesign();
        ImGui::SetItemTooltip("Cmd/Ctrl+S. Shapes, groups and hinges as text.");
        ImGui::SameLine();
        if (ImGui::Button("Load design", ImVec2(-FLT_MIN, 0))) LoadDesign();
    }
}
