// foldball: turns a mesh into a hinged shape that folds into a ball.
//   foldball               the GUI
//   foldball help          the command line (headless optimization, post processing, export)
#include "app/cli.h"
#include "app/gui.h"
#include "core/util.h"

int main(int argc, char** argv) {
    const char* mesh = NULL;
    int code = cli_main(argc, argv, &mesh);
    if (code >= 0) return code;
    return gui_run(mesh);
}
