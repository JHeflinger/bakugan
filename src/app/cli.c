#include "cli.h"
#include "core/config.h"
#include "core/util.h"
#include "opt/optimizer.h"
#include "post/post.h"
#include "post/preview.h"
#include "suite.h"
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#endif

static void usage(void) {
    printf("foldball: turn a mesh into a hinged shape that folds into a ball.\n\n"
           "  foldball                              open the GUI\n"
           "  foldball gui [mesh.obj]               open the GUI (with a mesh loaded)\n"
           "  foldball optimize <mesh.obj> [options]  optimize, headless\n"
           "  foldball optimize --start <design.fold> [mesh.obj] [options]\n"
           "                                        carry on from a saved design (a mesh given replaces its target)\n"
           "  foldball post <design.fold> [-o out.fold] [options]   hinges, carving, ledges, latch\n"
           "  foldball export <design.fold> [-o dir] [options]      STL parts\n"
           "  foldball preview <design.fold> [-o out.png]           a picture of it, open to closed\n"
           "  foldball config [-o settings.ini]     write every setting with its default (and what it does)\n"
           "  foldball suite <folder> [-c settings.ini] [options]\n"
           "                                        every .obj in the folder optimized and recorded, into one video (suite.mp4,\n"
           "                                        needs ffmpeg) with summary.csv and each result, in <out>/suite_<n>\n\n"
           "Options:\n"
           "  -c, --config <file.ini>   settings (see foldball config)\n"
           "  -s, --set key=value       one setting, e.g. -s optimizer.steps=4000 (repeatable)\n"
           "  -o, --out <path>          output folder (optimize, export) or file\n"
           "  --steps <n>               the same as -s optimizer.steps=<n>\n"
           "  --post                    optimize: post process the result and export STL parts too\n"
           "  -q, --quiet               no progress lines\n\n"
           "While optimizing, type (then Enter):  p  pause / resume   s  status   w  write the best design so far now\n"
           "                                       q  stop (keeping the best so far)   h  this help\n"
           "It also keeps <run>/status.txt and <run>/best.fold (and best.png) up to date, so another shell can look in.\n"
           "Ctrl-C stops the same as q; a second Ctrl-C quits at once.\n");
}

typedef struct {
    const char* cmd;
    const char* mesh;
    const char* design;
    const char* out;
    bool post, quiet;
    Params P;
} Args;

static bool parse(int argc, char** argv, Args* a, char* err, int errSize) {
    memset(a, 0, sizeof(*a));
    params_default(&a->P);
    a->cmd = argc > 1 ? argv[1] : NULL;
    // the config first, so -s overrides it whatever the order
    for (int k = 2; k < argc; k++)
        if ((!strcmp(argv[k], "-c") || !strcmp(argv[k], "--config")) && k + 1 < argc) {
            if (!params_load(&a->P, argv[k + 1], err, errSize)) return false;
            k++;
        }
    for (int k = 2; k < argc; k++) {
        const char* s = argv[k];
        if (!strcmp(s, "-c") || !strcmp(s, "--config")) { k++; continue; }
        if ((!strcmp(s, "-s") || !strcmp(s, "--set")) && k + 1 < argc) {
            if (!params_set(&a->P, argv[++k], err, errSize)) return false;
        } else if ((!strcmp(s, "-o") || !strcmp(s, "--out")) && k + 1 < argc) {
            a->out = argv[++k];
        } else if (!strcmp(s, "--start") && k + 1 < argc) {
            a->design = argv[++k];
        } else if (!strcmp(s, "--steps") && k + 1 < argc) {
            char buf[64];
            snprintf(buf, sizeof(buf), "optimizer.steps=%s", argv[++k]);
            if (!params_set(&a->P, buf, err, errSize)) return false;
        } else if (!strcmp(s, "--post")) {
            a->post = true;
        } else if (!strcmp(s, "-q") || !strcmp(s, "--quiet")) {
            a->quiet = true;
        } else if (s[0] == '-') {
            snprintf(err, (size_t)errSize, "unknown option %s (foldball help)", s);
            return false;
        } else {
            const char* dot = strrchr(s, '.');
            if (dot && (!strcmp(dot, ".fold") || !strcmp(dot, ".FOLD"))) a->design = s;
            else a->mesh = s;
        }
    }
    return true;
}

// ---- loading what to start from

static bool load_mesh_normalized(const char* path, TriMesh* m, char* err, int errSize) {
    if (!mesh_load_obj(path, m, err, errSize)) return false;
    Box3 b = mesh_bounds(m);
    V3 e = v3sub(b.max, b.min);
    mesh_weld(m, 1e-6f * fb_max(e.x, fb_max(e.y, e.z)));
    mesh_normalize(m, 2.0f);
    if (m->nt == 0) { snprintf(err, (size_t)errSize, "%s has no triangles", path); return false; }
    return true;
}

static bool start_design(const Args* a, Design* d, char* err, int errSize) {
    if (a->design) {
        if (!design_load(d, a->design, err, errSize)) return false;
        if (a->mesh) {
            TriMesh m;
            if (!load_mesh_normalized(a->mesh, &m, err, errSize)) return false;
            mesh_free(&d->target);
            mesh_free(&d->target0);
            d->target = mesh_copy(&m);
            d->target0 = m;
        }
        d->mmPerUnit = a->P.sizeMm / 2.0f;
        return true;
    }
    if (!a->mesh) { snprintf(err, (size_t)errSize, "no mesh given (foldball optimize <mesh.obj>)"); return false; }
    TriMesh m;
    if (!load_mesh_normalized(a->mesh, &m, err, errSize)) return false;
    design_init(d, &m, a->P.sizeMm, a->P.knuckleRadiusMm);
    mesh_free(&m);
    return true;
}

// ---- optimize

static OptShared* gShared;
static SuiteShared* gSuite;
static volatile int gInterrupts;
static void on_sigint(int s) {
    (void)s;
    if (++gInterrupts >= 2) _Exit(130);
    if (gShared) gShared->stopRequested = 1;
    if (gSuite) gSuite->stopRequested = 1;
}
#ifndef _WIN32
static volatile int gWriteNow;
static void on_sigusr1(int s) { (void)s; gWriteNow = 1; }
#endif

typedef struct {
    const Params* P;
    Design* d;
    OptShared* sh;
} RunArgs;
static void run_thread(void* v) {
    RunArgs* r = v;
    optimize_run(r->P, r->d, r->sh);
}

typedef struct {
    OptShared* sh;
    volatile int done;
    volatile int writeNow, statusNow;
} Keys;
static void keys_thread(void* v) {
    Keys* k = v;
    char line[256];
    while (!k->done && fgets(line, sizeof(line), stdin)) {
        char c = line[0];
        if (c == 'p' || c == 'P') {
            k->sh->pauseRequested = !k->sh->pauseRequested;
            printf(k->sh->pauseRequested ? "Pausing (p again to go on).\n" : "Going on.\n");
        } else if (c == 'q' || c == 'Q') {
            k->sh->stopRequested = 1;
            k->sh->pauseRequested = 0;
            printf("Stopping: the best design so far will be kept.\n");
        } else if (c == 'w' || c == 'W') {
            k->writeNow = 1;
        } else if (c == 's' || c == 'S') {
            k->statusNow = 1;
        } else if (c == 'h' || c == 'H' || c == '?') {
            printf("p pause/resume, s status, w write the best so far, q stop\n");
        }
        fflush(stdout);
    }
}

static char* make_run_dir(const char* base, const char* stem) {
    fb_make_dir(base);
    for (int k = 1; k < 100000; k++) {
        char* p = fb_format("%s/%s_%d", base, stem, k);
        if (!fb_file_exists(p)) {
            fb_make_dir(p);
            return p;
        }
        free(p);
    }
    return fb_format("%s/%s", base, stem);
}

static void write_snapshot(OptShared* sh, const char* dir, bool picture) {
    Design* copy = NULL;
    fb_mutex_lock(sh->lock);
    if (sh->best) copy = design_clone(sh->best);
    fb_mutex_unlock(sh->lock);
    if (!copy) return;
    char err[256];
    char* path = fb_format("%s/best.fold", dir);
    design_save(copy, path, err, sizeof(err));
    free(path);
    if (picture) {
        DesignMeshes m = design_meshes(copy, 64);
        FbImage im = preview_sheet(copy, &m, 200);
        char* png = fb_format("%s/best.png", dir);
        png_write(png, &im);
        free(png);
        image_free(&im);
        design_meshes_free(&m);
    }
    design_free(copy);
    free(copy);
}

static void write_status(OptShared* sh, const char* dir) {
    char* line = opt_status_line(sh);
    char* path = fb_format("%s/status.txt", dir);
    fb_mutex_lock(sh->lock);
    char* text = fb_format("%s\n\nlast message: %s\n\nThe best design so far: best.fold (picture: best.png). In the running shell: p pause/resume, "
                           "s status, w write now, q stop.\n",
                           line, sh->message[0] ? sh->message : "(none)");
    fb_mutex_unlock(sh->lock);
    fb_write_file(path, text, strlen(text));
    free(text);
    free(path);
    free(line);
}

static int cmd_optimize(Args* a) {
    char err[512];
    Design* d = FB_NEW(Design, 1);
    if (!start_design(a, d, err, sizeof(err))) { fprintf(stderr, "foldball: %s\n", err); return 1; }
    char* stem = fb_stem(a->mesh ? a->mesh : a->design);
    char* dir = make_run_dir(a->out ? a->out : a->P.outDir, stem);
    char* cfg = fb_format("%s/settings.ini", dir);
    params_save(&a->P, cfg);
    free(cfg);
    if (!a->quiet)
        printf("foldball: %s, %d steps, %d bodies to start; results in %s\nType p (pause/resume), s (status), w (write now), q (stop), then Enter.\n",
               a->mesh ? a->mesh : a->design, a->P.steps, d->nbodies, dir);
    OptShared* sh = opt_shared_create();
    gShared = sh;
    signal(SIGINT, on_sigint);
#ifndef _WIN32
    signal(SIGUSR1, on_sigusr1);
#endif
    RunArgs ra = { &a->P, d, sh };
    FbThread* worker = fb_thread_start(run_thread, &ra);
    Keys keys = { sh, 0, 0, 0 };
    FbThread* keyThread = fb_thread_start(keys_thread, &keys);
    double lastStatus = fb_now(), lastSnap = fb_now();
    for (;;) {
        fb_sleep_ms(100);
        fb_mutex_lock(sh->lock);
        bool fin = sh->finished;
        fb_mutex_unlock(sh->lock);
        double now = fb_now();
        bool statusDue = a->P.statusSeconds > 0 && now - lastStatus >= a->P.statusSeconds;
        if (statusDue || keys.statusNow || fin) {
            write_status(sh, dir);
            if (!a->quiet || keys.statusNow) {
                char* line = opt_status_line(sh);
                printf("%s\n", line);
                fflush(stdout);
                free(line);
            }
            keys.statusNow = 0;
            lastStatus = now;
        }
#ifndef _WIN32
        if (gWriteNow) { keys.writeNow = 1; gWriteNow = 0; }
#endif
        bool snapDue = a->P.snapshotSeconds > 0 && now - lastSnap >= a->P.snapshotSeconds;
        if (snapDue || keys.writeNow) {
            write_snapshot(sh, dir, true);
            if (keys.writeNow) printf("Wrote %s/best.fold and best.png.\n", dir);
            keys.writeNow = 0;
            lastSnap = now;
        }
        if (fin) break;
    }
    fb_thread_join(worker);
    keys.done = 1;
    (void)keyThread;                 // (left blocked on stdin: it ends with the process)
    gShared = NULL;

    // the results
    char* path = fb_format("%s/%s.fold", dir, stem);
    if (!design_save(d, path, err, sizeof(err))) fprintf(stderr, "foldball: %s\n", err);
    char* hist = fb_format("%s/history.csv", dir);
    FILE* hf = fopen(hist, "w");
    if (hf) {
        fprintf(hf, "step,loss\n");
        fb_mutex_lock(sh->lock);
        for (int k = 0; k < sh->nhist; k++) fprintf(hf, "%d,%.6f\n", k + 1, sh->history[k]);
        fb_mutex_unlock(sh->lock);
        fclose(hf);
    }
    free(hist);
    DesignMeshes m = design_meshes(d, 96);
    FbImage im = preview_sheet(d, &m, 260);
    char* png = fb_format("%s/%s.png", dir, stem);
    png_write(png, &im);
    image_free(&im);
    design_meshes_free(&m);
    char* status = opt_status_line(sh);
    char* report = fb_format("%s\n\n%s\n", status, sh->message);
    if (a->post) {
        PostReport pr;
        post_process(d, &a->P, &pr);
        char* built = fb_format("%s/%s_built.fold", dir, stem);
        design_save(d, built, err, sizeof(err));
        char msg[512];
        export_stl(d, &a->P, dir, stem, msg, sizeof(msg));
        char* r2 = fb_format("%s\n%s\n%s\n", report, pr.text, msg);
        free(report);
        report = r2;
        if (!a->quiet) printf("%s\n%s\n", pr.text, msg);
        DesignMeshes mb = design_meshes(d, 128);
        FbImage ib = preview_sheet(d, &mb, 260);
        char* pngb = fb_format("%s/%s_built.png", dir, stem);
        png_write(pngb, &ib);
        free(pngb);
        image_free(&ib);
        design_meshes_free(&mb);
        free(built);
    }
    char* rp = fb_format("%s/report.txt", dir);
    fb_write_file(rp, report, strlen(report));
    if (!a->quiet) printf("Done: %s (and %s.png, report.txt, history.csv) in %s\n", fb_basename(path), stem, dir);
    free(rp);
    free(report);
    free(status);
    free(png);
    free(path);
    free(stem);
    free(dir);
    opt_shared_free(sh);
    design_free(d);
    free(d);
    return 0;
}

// ---- the test suite

typedef struct {
    SuiteJob job;
    SuiteShared* sh;
} SuiteArgs;
static void suite_thread(void* v) {
    SuiteArgs* s = v;
    suite_run(&s->job, s->sh);
}

typedef struct {
    SuiteShared* sh;
    volatile int done, statusNow;
} SuiteKeys;
static void suite_keys_thread(void* v) {
    SuiteKeys* k = v;
    char line[256];
    while (!k->done && fgets(line, sizeof(line), stdin)) {
        char c = line[0];
        if (c == 'p' || c == 'P') {
            k->sh->pauseRequested = !k->sh->pauseRequested;
            printf(k->sh->pauseRequested ? "Pausing (p again to go on).\n" : "Going on.\n");
        } else if (c == 'n' || c == 'N') {
            k->sh->skipRequested = 1;
            printf("Ending this mesh's run (its best so far is kept), on to the next.\n");
        } else if (c == 'q' || c == 'Q') {
            k->sh->stopRequested = 1;
            k->sh->pauseRequested = 0;
            printf("Stopping: the meshes run so far still go into the video.\n");
        } else if (c == 's' || c == 'S') {
            k->statusNow = 1;
        } else if (c == 'h' || c == 'H' || c == '?') {
            printf("p pause/resume, n next mesh, s status, q stop (and make the video of what's done)\n");
        }
        fflush(stdout);
    }
}

static int cmd_suite(Args* a) {
    if (!a->mesh) { fprintf(stderr, "foldball: suite needs a folder of .obj files (foldball suite <folder>)\n"); return 2; }
    int n = suite_count(a->mesh);
    if (n < 0) { fprintf(stderr, "foldball: can't read the folder %s\n", a->mesh); return 1; }
    if (n == 0) { fprintf(stderr, "foldball: there are no .obj files in %s\n", a->mesh); return 1; }
    SuiteArgs sa;
    memset(&sa, 0, sizeof(sa));
    sa.job.P = a->P;
    snprintf(sa.job.folder, sizeof(sa.job.folder), "%s", a->mesh);
    snprintf(sa.job.outBase, sizeof(sa.job.outBase), "%s", a->out ? a->out : a->P.outDir);
    sa.sh = suite_shared_create();
    gSuite = sa.sh;
    signal(SIGINT, on_sigint);
    if (!a->quiet)
        printf("foldball suite: %d meshes in %s, %d steps each, a %d px square each at %d fps.\n"
               "Type p (pause/resume), n (next mesh), s (status), q (stop and make the video so far), then Enter.\n",
               n, a->mesh, a->P.steps, a->P.suiteTile, a->P.suiteFps);
    FbThread* t = fb_thread_start(suite_thread, &sa);
    SuiteKeys keys = { sa.sh, 0, 0 };
    fb_thread_start(suite_keys_thread, &keys);
    double last = fb_now();
    size_t logShown = 0;
    for (;;) {
        fb_sleep_ms(200);
        fb_mutex_lock(sa.sh->lock);
        bool fin = sa.sh->finished;
        size_t len = strlen(sa.sh->log);
        if (len > logShown) {
            printf("%s", sa.sh->log + logShown);
            logShown = len;
        }
        fb_mutex_unlock(sa.sh->lock);
        if (fin) break;
        if ((!a->quiet && a->P.statusSeconds > 0 && fb_now() - last >= fmax(a->P.statusSeconds, 5.0)) || keys.statusNow) {
            char* line = suite_status_line(sa.sh);
            printf("%s\n", line);
            free(line);
            last = fb_now();
            keys.statusNow = 0;
        }
        fflush(stdout);
    }
    fb_thread_join(t);
    keys.done = 1;
    gSuite = NULL;
    bool ok = sa.sh->video[0] != 0;
    suite_shared_free(sa.sh);
    return ok ? 0 : 1;
}

static int cmd_post(Args* a) {
    char err[512];
    if (!a->design) { fprintf(stderr, "foldball: post needs a design (.fold)\n"); return 1; }
    Design d = { 0 };
    if (!design_load(&d, a->design, err, sizeof(err))) { fprintf(stderr, "foldball: %s\n", err); return 1; }
    PostReport pr;
    post_process(&d, &a->P, &pr);
    char* stem = fb_stem(a->design);
    char* out = a->out ? fb_strdup(a->out) : fb_format("%s_built.fold", stem);
    if (!design_save(&d, out, err, sizeof(err))) { fprintf(stderr, "foldball: %s\n", err); return 1; }
    printf("%s\nWritten: %s\n", pr.text, out);
    free(out);
    free(stem);
    design_free(&d);
    return 0;
}

static int cmd_export(Args* a) {
    char err[512];
    if (!a->design) { fprintf(stderr, "foldball: export needs a design (.fold)\n"); return 1; }
    Design d = { 0 };
    if (!design_load(&d, a->design, err, sizeof(err))) { fprintf(stderr, "foldball: %s\n", err); return 1; }
    char* stem = fb_stem(a->design);
    char msg[512];
    int n = export_stl(&d, &a->P, a->out ? a->out : ".", stem, msg, sizeof(msg));
    printf("%s\n", msg);
    free(stem);
    design_free(&d);
    return n > 0 ? 0 : 1;
}

static int cmd_preview(Args* a) {
    char err[512];
    if (!a->design) { fprintf(stderr, "foldball: preview needs a design (.fold)\n"); return 1; }
    Design d = { 0 };
    if (!design_load(&d, a->design, err, sizeof(err))) { fprintf(stderr, "foldball: %s\n", err); return 1; }
    DesignMeshes m = design_meshes(&d, 128);
    FbImage im = preview_sheet(&d, &m, 300);
    char* stem = fb_stem(a->design);
    char* out = a->out ? fb_strdup(a->out) : fb_format("%s.png", stem);
    bool ok = png_write(out, &im);
    printf(ok ? "Written: %s\n" : "Couldn't write %s\n", out);
    free(out);
    free(stem);
    image_free(&im);
    design_meshes_free(&m);
    design_free(&d);
    return ok ? 0 : 1;
}

int cli_main(int argc, char** argv, const char** guiMesh) {
    *guiMesh = NULL;
    if (argc < 2) return -1;
    const char* cmd = argv[1];
    if (!strcmp(cmd, "help") || !strcmp(cmd, "-h") || !strcmp(cmd, "--help")) {
        usage();
        return 0;
    }
    Args a;
    char err[512];
    if (!strcmp(cmd, "gui")) {
        *guiMesh = argc > 2 ? argv[2] : NULL;
        return -1;
    }
    if (!parse(argc, argv, &a, err, sizeof(err))) {
        fprintf(stderr, "foldball: %s\n", err);
        return 2;
    }
    fb_threads_init(a.P.threads);
    if (!strcmp(cmd, "optimize")) return cmd_optimize(&a);
    if (!strcmp(cmd, "post")) return cmd_post(&a);
    if (!strcmp(cmd, "export")) return cmd_export(&a);
    if (!strcmp(cmd, "preview")) return cmd_preview(&a);
    if (!strcmp(cmd, "suite")) return cmd_suite(&a);
    if (!strcmp(cmd, "config")) {
        const char* out = a.out ? a.out : "foldball.ini";
        if (!params_save(&a.P, out)) { fprintf(stderr, "foldball: can't write %s\n", out); return 1; }
        printf("Written: %s\n", out);
        return 0;
    }
    // a bare mesh: the GUI with it
    const char* dot = strrchr(cmd, '.');
    if (dot && !strcmp(dot, ".obj")) {
        *guiMesh = cmd;
        return -1;
    }
    fprintf(stderr, "foldball: unknown command %s\n\n", cmd);
    usage();
    return 2;
}
