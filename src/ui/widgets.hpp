#pragma once
// Small ImGui helpers shared by the sidebar's tabs.
#include <imgui.h>
#include <raylib.h>

namespace Bakugan::Widgets {
    inline const ImVec4 Red = ImVec4(1.0f, 0.42f, 0.4f, 1.0f);
    inline const ImVec4 Green = ImVec4(0.4f, 0.85f, 0.5f, 1.0f);
    inline const ImVec4 Amber = ImVec4(1.0f, 0.78f, 0.35f, 1.0f);

    inline void ColoredWrapped(const ImVec4& color, const char* text) {
        ImGui::PushStyleColor(ImGuiCol_Text, color);
        ImGui::TextWrapped("%s", text);
        ImGui::PopStyleColor();
    }
    // A wrapped line of help, dimmed.
    inline void Hint(const char* text) { ColoredWrapped(ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled), text); }
    // A wrapped warning, amber.
    inline void Warn(const char* text) { ColoredWrapped(Amber, text); }

    inline bool ColorEdit(const char* id, Color* c) {
        float f[3] = { c->r / 255.0f, c->g / 255.0f, c->b / 255.0f };
        if (!ImGui::ColorEdit3(id, f, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel)) return false;
        *c = { (unsigned char)(f[0] * 255), (unsigned char)(f[1] * 255), (unsigned char)(f[2] * 255), 255 };
        return true;
    }

    // Width left for one item after a button with this label on the same row.
    inline float WidthBeside(const char* buttonLabel) {
        return ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(buttonLabel).x - ImGui::GetStyle().FramePadding.x * 2.0f -
               ImGui::GetStyle().ItemSpacing.x;
    }
    // Half the row, for two buttons side by side.
    inline float HalfWidth() { return (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f; }
}
