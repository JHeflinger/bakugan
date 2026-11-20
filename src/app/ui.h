#ifndef FB_UI_H
#define FB_UI_H
// A small immediate mode UI on raylib: a panel of rows (labels, buttons, checkboxes, sliders), scrolled.
#include <raylib.h>
#include <stdbool.h>

void ui_begin(Rectangle area, float* scroll);
void ui_end(void);
void ui_section(const char* title);
bool ui_header(const char* title, bool* open);      // a collapsible section; returns *open
void ui_label(const char* text, Color c);
void ui_text_wrapped(const char* text, Color c);
bool ui_button(const char* text, bool enabled);
bool ui_button_row(const char** texts, int n, bool* enabled, int* clicked);
bool ui_checkbox(const char* text, bool* value);
bool ui_slider_float(const char* text, float* value, float lo, float hi, const char* fmt);
bool ui_slider_int(const char* text, int* value, int lo, int hi);
void ui_space(float h);
void ui_tooltip_last(const char* text);
// a custom block: returns its rectangle (h tall) for drawing into
Rectangle ui_block(float h);
bool ui_mouse_in_panel(void);
extern Color UI_INK, UI_MUTED, UI_ACCENT, UI_BG, UI_PANEL, UI_WARN;
#endif
