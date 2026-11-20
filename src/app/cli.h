#ifndef FB_CLI_H
#define FB_CLI_H
// The command line. Returns the exit code, or -1 when the GUI should open (no command, or "gui"); then *guiMesh is
// the mesh to open with (or NULL).
int cli_main(int argc, char** argv, const char** guiMesh);
#endif
