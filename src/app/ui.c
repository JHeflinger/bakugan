#include "ui.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

Color UI_INK = { 228, 230, 236, 255 };
Color UI_MUTED = { 150, 155, 168, 255 };
Color UI_ACCENT = { 92, 150, 235, 255 };
Color UI_BG = { 30, 32, 38, 255 };
Color UI_PANEL = { 40, 43, 50, 255 };
Color UI_WARN = { 235, 170, 80, 255 };

static struct {
    Rectangle area;
    float* scroll;
    float y;                 // next row's top (content space)
    float contentH;
    Rectangle last;
    const char* tip;
    int activeSlider;        // id of the slider being dragged
    int id;
} U;

static const int kFont = 16, kRow = 24, kPad = 10;

static Rectangle row(float h) {
    Rectangle r = { U.area.x + kPad, U.area.y + U.y - *U.scroll, U.area.width - 2 * kPad, h };
    U.y += h + 4;
    U.last = r;
    U.id++;
    return r;
}

static bool visible(Rectangle r) { return r.y + r.height >= U.area.y && r.y <= U.area.y + U.area.height; }
static bool hovered(Rectangle r) { return ui_mouse_in_panel() && CheckCollisionPointRec(GetMousePosition(), r); }

bool ui_mouse_in_panel(void) { return CheckCollisionPointRec(GetMousePosition(), U.area); }

void ui_begin(Rectangle area, float* scroll) {
    U.area = area;
    U.scroll = scroll;
    U.y = kPad;
    U.id = 0;
    U.tip = NULL;
    DrawRectangleRec(area, UI_PANEL);
    if (ui_mouse_in_panel()) *scroll -= GetMouseWheelMove() * 40.0f;
    BeginScissorMode((int)area.x, (int)area.y, (int)area.width, (int)area.height);
}

void ui_end(void) {
    EndScissorMode();
    U.contentH = U.y + kPad;
    float maxScroll = U.contentH - U.area.height;
    if (maxScroll < 0) maxScroll = 0;
    if (*U.scroll > maxScroll) *U.scroll = maxScroll;
    if (*U.scroll < 0) *U.scroll = 0;
    if (maxScroll > 0) {
        float frac = U.area.height / U.contentH;
        float barH = U.area.height * frac, barY = U.area.y + (*U.scroll / U.contentH) * U.area.height;
        DrawRectangle((int)(U.area.x + U.area.width - 4), (int)barY, 3, (int)barH, (Color){ 90, 95, 110, 255 });
    }
    if (!IsMouseButtonDown(MOUSE_BUTTON_LEFT)) U.activeSlider = 0;
    if (U.tip) {
        Vector2 m = GetMousePosition();
        int w = MeasureText(U.tip, 14) + 12;
        float x = m.x + 14;
        if (x + w > GetScreenWidth()) x = GetScreenWidth() - w - 4;
        DrawRectangle((int)x, (int)m.y + 16, w, 22, (Color){ 15, 16, 20, 240 });
        DrawText(U.tip, (int)x + 6, (int)m.y + 20, 14, UI_INK);
    }
}

void ui_tooltip_last(const char* text) {
    if (text && hovered(U.last)) U.tip = text;
}

void ui_space(float h) { U.y += h; }

void ui_section(const char* title) {
    U.y += 6;
    Rectangle r = row((float)kRow);
    if (!visible(r)) return;
    DrawText(title, (int)r.x, (int)r.y + 4, kFont, UI_ACCENT);
    DrawLine((int)r.x, (int)(r.y + r.height), (int)(r.x + r.width), (int)(r.y + r.height), (Color){ 70, 75, 88, 255 });
}

bool ui_header(const char* title, bool* open) {
    U.y += 2;
    Rectangle r = row((float)kRow);
    bool hov = hovered(r);
    if (visible(r)) {
        DrawRectangleRec(r, hov ? (Color){ 58, 62, 74, 255 } : (Color){ 50, 54, 64, 255 });
        DrawText(*open ? "-" : "+", (int)r.x + 6, (int)r.y + 4, kFont, UI_MUTED);
        DrawText(title, (int)r.x + 22, (int)r.y + 4, kFont, UI_INK);
    }
    if (hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) *open = !*open;
    return *open;
}

void ui_label(const char* text, Color c) {
    Rectangle r = row(18);
    if (visible(r)) DrawText(text, (int)r.x, (int)r.y + 2, 14, c);
}

void ui_text_wrapped(const char* text, Color c) {
    // words onto lines that fit
    const float width = U.area.width - 2 * kPad;
    char line[512];
    int n = 0;
    const char* s = text;
    while (*s) {
        const char* w = s;
        while (*w && *w != ' ' && *w != '\n') w++;
        int wl = (int)(w - s);
        char trial[512];
        snprintf(trial, sizeof(trial), "%.*s%s%.*s", n, line, n ? " " : "", wl, s);
        if (n && MeasureText(trial, 14) > width) {
            line[n] = 0;
            ui_label(line, c);
            n = 0;
            snprintf(line, sizeof(line), "%.*s", wl, s);
            n = wl;
        } else {
            snprintf(line, sizeof(line), "%s", trial);
            n = (int)strlen(line);
        }
        s = w;
        if (*s == '\n') {
            line[n] = 0;
            ui_label(line, c);
            n = 0;
        }
        if (*s) s++;
    }
    if (n) {
        line[n] = 0;
        ui_label(line, c);
    }
}

bool ui_button(const char* text, bool enabled) {
    Rectangle r = row((float)kRow + 4);
    bool hov = enabled && hovered(r);
    if (visible(r)) {
        DrawRectangleRec(r, !enabled ? (Color){ 52, 55, 62, 255 } : hov ? (Color){ 82, 120, 190, 255 } : (Color){ 64, 96, 156, 255 });
        int tw = MeasureText(text, kFont);
        DrawText(text, (int)(r.x + (r.width - tw) / 2), (int)r.y + 6, kFont, enabled ? UI_INK : UI_MUTED);
    }
    return hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
}

bool ui_button_row(const char** texts, int n, bool* enabled, int* clicked) {
    Rectangle r = row((float)kRow + 4);
    *clicked = -1;
    float w = (r.width - (n - 1) * 6) / n;
    for (int k = 0; k < n; k++) {
        Rectangle b = { r.x + k * (w + 6), r.y, w, r.height };
        bool en = enabled ? enabled[k] : true;
        bool hov = en && hovered(b);
        if (visible(b)) {
            DrawRectangleRec(b, !en ? (Color){ 52, 55, 62, 255 } : hov ? (Color){ 82, 120, 190, 255 } : (Color){ 64, 96, 156, 255 });
            int tw = MeasureText(texts[k], 14);
            DrawText(texts[k], (int)(b.x + (b.width - tw) / 2), (int)b.y + 7, 14, en ? UI_INK : UI_MUTED);
        }
        if (hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) *clicked = k;
    }
    return *clicked >= 0;
}

bool ui_checkbox(const char* text, bool* value) {
    Rectangle r = row(20);
    Rectangle box = { r.x, r.y + 2, 16, 16 };
    bool hov = hovered(r);
    if (visible(r)) {
        DrawRectangleLinesEx(box, 1, hov ? UI_ACCENT : UI_MUTED);
        if (*value) DrawRectangle((int)box.x + 4, (int)box.y + 4, 8, 8, UI_ACCENT);
        DrawText(text, (int)r.x + 24, (int)r.y + 3, 14, UI_INK);
    }
    if (hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        *value = !*value;
        return true;
    }
    return false;
}

// a slider; wide ranges move along a cubic so small values get room
static bool slider(const char* text, float* value, float lo, float hi, const char* shown, bool cubic) {
    Rectangle r = row(34);
    int id = U.id;
    Rectangle track = { r.x, r.y + 20, r.width, 10 };
    bool hov = hovered(r);
    float t = hi > lo ? (*value - lo) / (hi - lo) : 0;
    if (t < 0) t = 0;
    if (t > 1) t = 1;
    if (cubic) t = cbrtf(t);
    if (hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) U.activeSlider = id;
    bool changed = false;
    if (U.activeSlider == id && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        float nt = (GetMousePosition().x - track.x) / track.width;
        if (nt < 0) nt = 0;
        if (nt > 1) nt = 1;
        float v = lo + (hi - lo) * (cubic ? nt * nt * nt : nt);
        if (v != *value) {
            *value = v;
            changed = true;
        }
        t = nt;
    }
    if (visible(r)) {
        DrawText(text, (int)r.x, (int)r.y + 2, 14, UI_INK);
        int vw = MeasureText(shown, 14);
        DrawText(shown, (int)(r.x + r.width - vw), (int)r.y + 2, 14, UI_MUTED);
        DrawRectangleRec(track, (Color){ 58, 62, 74, 255 });
        DrawRectangle((int)track.x, (int)track.y, (int)(track.width * t), (int)track.height, U.activeSlider == id || hov ? UI_ACCENT : (Color){ 72, 110, 170, 255 });
    }
    return changed;
}

bool ui_slider_float(const char* text, float* value, float lo, float hi, const char* fmt) {
    char shown[64];
    snprintf(shown, sizeof(shown), fmt ? fmt : "%.3g", *value);
    bool cubic = lo >= 0 && hi > 50 * (lo > 0 ? lo : 1) && hi > 20;
    return slider(text, value, lo, hi, shown, cubic);
}

bool ui_slider_int(const char* text, int* value, int lo, int hi) {
    char shown[64];
    snprintf(shown, sizeof(shown), "%d", *value);
    float v = (float)*value;
    bool cubic = lo >= 0 && hi > 1000;
    bool changed = slider(text, &v, (float)lo, (float)hi, shown, cubic);
    if (changed) *value = (int)lrintf(v);
    return changed;
}

Rectangle ui_block(float h) { return row(h); }
