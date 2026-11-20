#include "core/design.h"
#include "core/util.h"
#include "opt/optimizer.h"
#include "post/preview.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void runner(void* a) { void** v = a; optimize_run(v[0], v[1], v[2]); }
int main(int argc, char** argv) {
    fb_threads_init(0);
    TriMesh m; char err[256]; Params P; params_default(&P);
    for (int k = 3; k < argc; k++) if (!params_set(&P, argv[k], err, sizeof err)) { printf("%s\n", err); return 1; }
    mesh_load_obj(argv[1], &m, err, sizeof err); mesh_weld(&m, 1e-6f); mesh_normalize(&m, 2.0f);
    Design D; design_init(&D, &m, P.sizeMm, P.knuckleRadiusMm);
    P.steps = atoi(argv[2]);
    OptShared* sh = opt_shared_create();
    void* args[3] = { &P, &D, sh };
    FbThread* t = fb_thread_start(runner, args);
    double t0 = fb_now();
    while (1) { fb_sleep_ms(2000); fb_mutex_lock(sh->lock); bool fin = sh->finished; fb_mutex_unlock(sh->lock); char* s = opt_status_line(sh); printf("%s\n", s); free(s); if (fin) break; }
    fb_thread_join(t);
    printf("done in %.1f s; %d bodies %d hinges\n", fb_now() - t0, D.nbodies, D.nhinges);
    for (int h = 0; h < D.nhinges; h++) printf("  hinge %d: %d->%d angle %.1f deg order %d\n", D.hinges[h].id, D.hinges[h].parent, D.hinges[h].child, D.hinges[h].angle * 180 / FB_PI, D.hinges[h].order);
    design_save(&D, "/tmp/opt.fold", err, sizeof err);
    DesignMeshes dm = design_meshes(&D, 80); FbImage im = preview_sheet(&D, &dm, 220); png_write("/tmp/opt.png", &im);
    return 0;
}
