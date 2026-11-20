#include "suite.h"
#include "post/post.h"
#include "post/preview.h"
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

SuiteShared* suite_shared_create(void) {
    SuiteShared* s = FB_NEW(SuiteShared, 1);
    s->lock = fb_mutex_create();
    return s;
}

void suite_shared_free(SuiteShared* s) {
    if (!s) return;
    free(s->tile);
    fb_mutex_destroy(s->lock);
    free(s);
}

int suite_count(const char* folder) {
    char** names;
    int n = fb_list_dir(folder, ".obj", &names);
    for (int k = 0; k < n; k++) free(names[k]);
    free(names);
    return n;
}

static void say(SuiteShared* sh, const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fb_mutex_lock(sh->lock);
    size_t n = strlen(sh->log);
    snprintf(sh->log + n, sizeof(sh->log) - n, "%s\n", buf);
    fb_mutex_unlock(sh->lock);
}

static void set_stage(SuiteShared* sh, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fb_mutex_lock(sh->lock);
    vsnprintf(sh->stage, sizeof(sh->stage), fmt, ap);
    fb_mutex_unlock(sh->lock);
    va_end(ap);
}

// ---- recording one mesh's run

typedef struct {
    SuiteShared* sh;
    const Params* P;
    FILE* frames;                    // tile x tile RGB frames, one after another
    int count;                       // frames written
    const char* name;
    int index, total;
    FbImage im;                      // the tile being drawn
    DesignMeshes meshes;             // the last design's (reused while it doesn't change)
    unsigned long long meshedFor;
    bool hasMeshes;
} Recorder;

// what decides a design's shape (its meshes are the same while this is)
static unsigned long long shape_signature(const Design* d) {
    unsigned long long h = 1469598103934665603ull;
    #define MIX(ptr, bytes) do { const unsigned char* p_ = (const unsigned char*)(ptr); \
            for (size_t i_ = 0; i_ < (size_t)(bytes); i_++) { h ^= p_[i_]; h *= 1099511628211ull; } } while (0)
    MIX(&d->nplanes, sizeof(int));
    for (int k = 0; k < d->nplanes; k++) { MIX(&d->planes[k].n, sizeof(V3)); MIX(&d->planes[k].d, sizeof(float)); MIX(&d->planes[k].k, sizeof(float)); }
    MIX(&d->nbodies, sizeof(int));
    for (int b = 0; b < d->nbodies; b++) {
        MIX(&d->bodies[b].ncells, sizeof(int));
        for (int c = 0; c < d->bodies[b].ncells; c++) {
            const Cell* cell = &d->bodies[b].cells[c];
            MIX(&cell->n, sizeof(int));
            MIX(cell->plane, sizeof(short) * (size_t)cell->n);
            MIX(cell->sign, (size_t)cell->n);
        }
        MIX(&d->bodies[b].detail.v, sizeof(void*));
        MIX(&d->bodies[b].carve.v, sizeof(void*));
    }
    MIX(&d->nhollows, sizeof(int));
    for (int k = 0; k < d->nhollows; k++) { MIX(&d->hollows[k].a, sizeof(V3)); MIX(&d->hollows[k].b, sizeof(V3)); MIX(&d->hollows[k].r, sizeof(float)); }
    MIX(&d->nextras, sizeof(int));
    MIX(&d->gap, sizeof(float));
    #undef MIX
    return h;
}

static int text_scale(int tile) { return tile >= 480 ? 3 : (tile >= 240 ? 2 : 1); }

static void draw_frame(Recorder* r, const Design* d, float s, float yaw, const char* line1, const char* line2) {
    const int T = r->P->suiteTile;
    unsigned long long sig = shape_signature(d);
    if (!r->hasMeshes || sig != r->meshedFor) {
        if (r->hasMeshes) design_meshes_free(&r->meshes);
        int cells = fb_mini(fb_maxi(T / 5, 32), 96);
        for (int b = 0; b < d->nbodies; b++)
            if (d->bodies[b].detail.v) cells = fb_maxi(cells, fb_mini(T / 2, 192));
        r->meshes = design_meshes(d, cells);
        r->meshedFor = sig;
        r->hasMeshes = true;
    }
    preview_tile(&r->im, 0, 0, T, d, &r->meshes, s, yaw, r->P->suiteGhost);
    const int sc = text_scale(T);
    // (each line as big as fits)
    #define FIT(str) fb_maxi(1, fb_mini(sc, (T - 10) / (6 * fb_maxi(1, (int)strlen(str)))))
    const int s1 = fb_mini(line1 ? FIT(line1) : sc, line2 ? FIT(line2) : sc), s2 = s1;      // (both lines alike)
    image_text(&r->im, 5 * sc, 4 * sc, FIT(r->name), r->name, 235, 238, 245);
    if (line1) image_text(&r->im, 5 * sc, T - 9 * s2 - 9 * s1 - 2 * sc, s1, line1, 190, 200, 215);
    if (line2) image_text(&r->im, 5 * sc, T - 9 * s2 - 2 * sc, s2, line2, 150, 160, 180);
    #undef FIT
    // a progress bar along the top
    char counter[32];
    snprintf(counter, sizeof(counter), "%d/%d", r->index + 1, r->total);
    image_text(&r->im, T - (int)strlen(counter) * 6 * sc - 5 * sc, 4 * sc, sc, counter, 150, 160, 180);
    fwrite(r->im.rgb, 1, (size_t)T * T * 3, r->frames);
    r->count++;
    // for the GUI
    fb_mutex_lock(r->sh->lock);
    if (r->sh->tileSize != T) {
        free(r->sh->tile);
        r->sh->tile = FB_NEW(unsigned char, (size_t)T * T * 3);
        r->sh->tileSize = T;
    }
    memcpy(r->sh->tile, r->im.rgb, (size_t)T * T * 3);
    r->sh->tileVersion++;
    fb_mutex_unlock(r->sh->lock);
}

static void record_cb(void* ctx, const Design* best, const Score* s, int step, int steps) {
    Recorder* r = ctx;
    char l1[128], l2[128];
    snprintf(l1, sizeof(l1), "step %d/%d loss %.3f", step, steps, s->total);
    snprintf(l2, sizeof(l2), "cover %.0f%%, %d parts", 100.0f * s->openCover, best->nbodies);
    float yaw = 0.5f * (float)step / fb_maxi(steps, 1);       // (a slow turn, so it reads as 3D)
    draw_frame(r, best, 1.0f, yaw, l1, l2);
}

// ---- running one mesh

typedef struct {
    const Params* P;
    Design* d;
    OptShared* os;
} Worker;
static void worker_thread(void* v) {
    Worker* w = v;
    optimize_run(w->P, w->d, w->os);
}

static bool load_mesh(const char* path, TriMesh* m, char* err, int errSize) {
    if (!mesh_load_obj(path, m, err, errSize)) return false;
    Box3 b = mesh_bounds(m);
    V3 e = v3sub(b.max, b.min);
    mesh_weld(m, 1e-6f * fb_max(e.x, fb_max(e.y, e.z)));
    mesh_normalize(m, 2.0f);
    if (m->nt == 0) { snprintf(err, (size_t)errSize, "%s has no triangles", path); return false; }
    return true;
}

// ---- the video

static bool encode(const SuiteJob* job, SuiteShared* sh, char** stems, char** framePaths, const int* counts, int n, const char* outDir,
                   char* note, int noteSize) {
    const int T = job->P.suiteTile, fps = job->P.suiteFps;
    const int cols = (int)ceil(sqrt((double)n)), rows = (n + cols - 1) / cols;
    const int W = cols * T, H = rows * T;
    int total = 0;
    for (int k = 0; k < n; k++) total = fb_maxi(total, counts[k]);
    if (total == 0) { snprintf(note, (size_t)noteSize, "no frames were recorded"); return false; }
    const int hold = 2 * fps;
    char* video = fb_format("%s/suite.mp4", outDir);
    char* cmd = fb_format("ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 -s %dx%d -r %d -i - -c:v libx264 -preset medium -crf 20 "
                          "-pix_fmt yuv420p -movflags +faststart \"%s\"", W, H, fps, video);
    FILE* ff = popen(cmd, "w");
    free(cmd);
    FILE** in = FB_NEW(FILE*, n);
    for (int k = 0; k < n; k++) in[k] = counts[k] > 0 ? fopen(framePaths[k], "rb") : NULL;
    unsigned char* frame = FB_NEW(unsigned char, (size_t)W * H * 3);
    unsigned char* tile = FB_NEW(unsigned char, (size_t)T * T * 3);
    bool ok = ff != NULL;
    for (int f = 0; f < total + hold && ok; f++) {
        if (f % 10 == 0) set_stage(sh, "Encoding the video: frame %d of %d", f + 1, total + hold);
        memset(frame, 18, (size_t)W * H * 3);
        for (int k = 0; k < n; k++) {
            if (!in[k]) continue;
            int which = fb_mini(f, counts[k] - 1);
            fseek(in[k], (long)((size_t)which * T * T * 3), SEEK_SET);
            if (fread(tile, 1, (size_t)T * T * 3, in[k]) != (size_t)T * T * 3) continue;
            int cx = (k % cols) * T, cy = (k / cols) * T;
            for (int y = 0; y < T; y++) memcpy(frame + ((size_t)(cy + y) * W + cx) * 3, tile + (size_t)y * T * 3, (size_t)T * 3);
        }
        ok = fwrite(frame, 1, (size_t)W * H * 3, ff) == (size_t)W * H * 3;
        // the last frame as a picture too
        if (f == total - 1) {
            FbImage im = { W, H, frame };
            char* png = fb_format("%s/suite.png", outDir);
            png_write(png, &im);
            free(png);
        }
    }
    if (ff) ok = pclose(ff) == 0 && ok;
    for (int k = 0; k < n; k++)
        if (in[k]) fclose(in[k]);
    free(in);
    free(frame);
    free(tile);
    (void)stems;
    if (ok) {
        snprintf(note, (size_t)noteSize, "%s (%dx%d, %d frames at %d fps)", video, W, H, total + hold, fps);
        fb_mutex_lock(sh->lock);
        snprintf(sh->video, sizeof(sh->video), "%s", video);
        fb_mutex_unlock(sh->lock);
    } else {
        snprintf(note, (size_t)noteSize, "the video couldn't be made: is ffmpeg on the PATH? (Arch: pacman -S ffmpeg). The last frame is suite.png, "
                                         "every mesh's frames are kept as <stem>/frames.rgb (%dx%d RGB, raw)", T, T);
    }
    free(video);
    return ok;
}

// ---- the suite

static char* new_dir(const char* base) {
    fb_make_dir(base);
    for (int k = 1; k < 100000; k++) {
        char* p = fb_format("%s/suite_%d", base, k);
        if (!fb_file_exists(p)) {
            fb_make_dir(p);
            return p;
        }
        free(p);
    }
    return fb_format("%s/suite", base);
}

bool suite_run(const SuiteJob* job, SuiteShared* sh) {
    const Params* P = &job->P;
    char** names = NULL;
    int n = fb_list_dir(job->folder, ".obj", &names);
    fb_mutex_lock(sh->lock);
    sh->running = true;
    sh->finished = false;
    sh->started = fb_now();
    sh->count = n > 0 ? n : 0;
    sh->log[0] = 0;
    sh->video[0] = 0;
    fb_mutex_unlock(sh->lock);
    if (n <= 0) {
        say(sh, n < 0 ? "Can't read the folder %s." : "There are no .obj files in %s.", job->folder);
        fb_mutex_lock(sh->lock);
        sh->running = false;
        sh->finished = true;
        snprintf(sh->stage, sizeof(sh->stage), "Nothing to do");
        fb_mutex_unlock(sh->lock);
        free(names);
        return false;
    }
    char* outDir = new_dir(job->outBase[0] ? job->outBase : P->outDir);
    fb_mutex_lock(sh->lock);
    snprintf(sh->outDir, sizeof(sh->outDir), "%s", outDir);
    fb_mutex_unlock(sh->lock);
    char* ini = fb_format("%s/settings.ini", outDir);
    params_save(P, ini);
    free(ini);
    char* csvPath = fb_format("%s/summary.csv", outDir);
    FILE* csv = fopen(csvPath, "w");
    if (csv) fprintf(csv, "mesh,steps,seconds,loss,open,cover,extra,collision,parts,hinges,hollows,surface_carved,status\n");
    char** stems = FB_NEW(char*, n);
    char** framePaths = FB_NEW(char*, n);
    int* counts = FB_NEW(int, n);
    int done = 0;
    for (int i = 0; i < n && !sh->stopRequested; i++) {
        stems[i] = fb_stem(names[i]);
        char* meshPath = fb_format("%s/%s", job->folder, names[i]);
        char* dir = fb_format("%s/%s", outDir, stems[i]);
        fb_make_dir(dir);
        framePaths[i] = fb_format("%s/frames.rgb", dir);
        fb_mutex_lock(sh->lock);
        sh->index = i;
        snprintf(sh->current, sizeof(sh->current), "%s", names[i]);
        fb_mutex_unlock(sh->lock);
        set_stage(sh, "Optimizing %s", names[i]);
        sh->skipRequested = 0;
        char err[512];
        TriMesh m;
        if (!load_mesh(meshPath, &m, err, sizeof(err))) {
            say(sh, "%s: skipped (%s)", names[i], err);
            if (csv) fprintf(csv, "%s,0,0,,,,,,,,,,\"%s\"\n", names[i], err);
            free(meshPath);
            free(dir);
            continue;
        }
        const double t0 = fb_now();
        Design* d = FB_NEW(Design, 1);
        design_init(d, &m, P->sizeMm, P->knuckleRadiusMm);
        mesh_free(&m);
        Recorder rec;
        memset(&rec, 0, sizeof(rec));
        rec.sh = sh;
        rec.P = P;
        rec.frames = fopen(framePaths[i], "wb");
        rec.name = stems[i];
        rec.index = i;
        rec.total = n;
        rec.im = image_create(P->suiteTile, P->suiteTile, 30);
        OptShared* os = opt_shared_create();
        if (rec.frames) {
            os->recordFrames = P->suiteFrames;
            os->record = record_cb;
            os->recordCtx = &rec;
        }
        fb_mutex_lock(sh->lock);
        sh->opt = os;
        fb_mutex_unlock(sh->lock);
        Worker w = { P, d, os };
        FbThread* t = fb_thread_start(worker_thread, &w);
        for (;;) {
            fb_sleep_ms(50);
            os->pauseRequested = sh->pauseRequested;
            if (sh->stopRequested || sh->skipRequested) {
                os->stopRequested = 1;
                os->pauseRequested = 0;
            }
            fb_mutex_lock(os->lock);
            bool fin = os->finished;
            fb_mutex_unlock(os->lock);
            if (fin) break;
        }
        fb_thread_join(t);
        const bool cut = sh->stopRequested || sh->skipRequested;
        Score sc;
        fb_mutex_lock(os->lock);
        sc = os->bestScore;
        int steps = os->step;
        fb_mutex_unlock(os->lock);
        // the result, folding shut and open again
        PostReport pr;
        bool built = false;
        if (P->suitePost && !sh->stopRequested) {
            set_stage(sh, "Post processing %s", names[i]);
            Design* b = design_clone(d);
            post_process(b, P, &pr);
            char* bp = fb_format("%s/%s_built.fold", dir, stems[i]);
            design_save(b, bp, err, sizeof(err));
            free(bp);
            design_free(d);
            free(d);
            d = b;
            built = true;
        }
        if (rec.frames && P->suiteFoldFrames > 0 && !sh->stopRequested) {
            set_stage(sh, "Recording %s folding", names[i]);
            const int F = P->suiteFoldFrames;
            char l1[128], l2[128];
            snprintf(l1, sizeof(l1), "result%s  loss %.3f", built ? " (built)" : "", sc.total);
            for (int f = 0; f < F && !sh->stopRequested; f++) {
                float x = (float)f / fb_maxi(F - 1, 1);
                float s = 0.5f + 0.5f * cosf(2.0f * FB_PI * x);       // open, shut, open
                snprintf(l2, sizeof(l2), "%s", s < 0.02f ? "closed" : (x < 0.5f ? "folding" : "opening"));
                draw_frame(&rec, d, s, 0.5f + 0.5f * x, l1, l2);
            }
        }
        if (rec.frames) fclose(rec.frames);
        counts[i] = rec.count;
        if (rec.hasMeshes) design_meshes_free(&rec.meshes);
        image_free(&rec.im);
        // the results
        char* fp = fb_format("%s/%s.fold", dir, stems[i]);
        design_save(d, fp, err, sizeof(err));
        free(fp);
        DesignMeshes dm = design_meshes(d, 96);
        FbImage sheet = preview_sheet(d, &dm, 240);
        char* png = fb_format("%s/%s.png", dir, stems[i]);
        png_write(png, &sheet);
        free(png);
        image_free(&sheet);
        design_meshes_free(&dm);
        const double secs = fb_now() - t0;
        const char* status = sh->stopRequested ? "stopped" : (cut ? "skipped (best so far kept)" : "done");
        say(sh, "%s: %s in %.0f s, loss %.4f (open %.3f, %.0f%% covered), %d parts, %d hinges, %d hollows", names[i], status, secs, sc.total, sc.open,
            100.0f * sc.openCover, d->nbodies, d->nhinges, d->nhollows);
        if (built) say(sh, "  %s", pr.text);
        if (csv) {
            fprintf(csv, "%s,%d,%.1f,%.6f,%.6f,%.4f,%.4f,%.6f,%d,%d,%d,%.4f,%s\n", names[i], steps, secs, sc.total, sc.open, sc.openCover, sc.openExtra,
                    sc.collision, d->nbodies, d->nhinges, d->nhollows, sc.surfaceCarved, status);
            fflush(csv);
        }
        fb_mutex_lock(sh->lock);
        sh->opt = NULL;
        fb_mutex_unlock(sh->lock);
        opt_shared_free(os);
        design_free(d);
        free(d);
        free(meshPath);
        free(dir);
        done = i + 1;
    }
    if (csv) fclose(csv);
    // the video, of every mesh run (even when stopped)
    char note[1400];
    bool video = false;
    if (done > 0) {
        set_stage(sh, "Encoding the video");
        video = encode(job, sh, stems, framePaths, counts, done, outDir, note, sizeof(note));
        if (video)
            for (int k = 0; k < done; k++) remove(framePaths[k]);
        say(sh, "Video: %s", note);
    }
    say(sh, "Results in %s (summary.csv; per mesh <stem>.fold and <stem>.png).", outDir);
    fb_mutex_lock(sh->lock);
    sh->running = false;
    sh->finished = true;
    snprintf(sh->stage, sizeof(sh->stage), "%s", sh->stopRequested ? "Stopped" : "Done");
    fb_mutex_unlock(sh->lock);
    for (int k = 0; k < done; k++) {
        free(stems[k]);
        free(framePaths[k]);
    }
    for (int k = done; k < n; k++)
        if (stems[k]) { free(stems[k]); free(framePaths[k]); }
    for (int k = 0; k < n; k++) free(names[k]);
    free(names);
    free(stems);
    free(framePaths);
    free(counts);
    free(csvPath);
    free(outDir);
    return video;
}

char* suite_status_line(SuiteShared* sh) {
    fb_mutex_lock(sh->lock);
    OptShared* os = sh->opt;
    char* inner = NULL;
    fb_mutex_unlock(sh->lock);
    // (the run's own line: its lock, not the suite's; the run can't go away while the suite thread waits on it... but
    // it can between meshes, so only while it's still the current one)
    fb_mutex_lock(sh->lock);
    if (os && sh->opt == os) inner = opt_status_line(os);
    char* line = fb_format("[suite %s] mesh %d of %d: %s, %.0f s in%s%s", sh->stage, sh->index + 1, sh->count, sh->current, fb_now() - sh->started,
                           inner ? " | " : "", inner ? inner : "");
    fb_mutex_unlock(sh->lock);
    free(inner);
    return line;
}
