#ifndef FB_GUI_H
#define FB_GUI_H
// The window: a 3D view of the design (any pose between closed and open) and a panel to load, optimize, post
// process and export. mesh: a .obj or .fold to open with (or NULL).
int gui_run(const char* mesh);
#endif
