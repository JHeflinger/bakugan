#include "gui.h"
#include "ui.h"
#include "core/config.h"
#include "core/util.h"
#include "opt/optimizer.h"
#include "post/mesher.h"
#include "post/post.h"
#include "post/preview.h"
#include "suite.h"
#include <raylib.h>
#include <raymath.h>
#include <rlgl.h>
#include <stdlib.h>
#include <string.h>
#include <nfd.h>

// ---- file dialogs (nfd): a path chosen into out, or false (cancelled, or the dialog failed: said in the status)

static void dialog_error(void);
static bool pick_open(const char* filters, char* out, size_t size) {
    nfdchar_t* path = NULL;
    nfdresult_t r = NFD_OpenDialog(filters, NULL, &path);
    if (r == NFD_ERROR) dialog_error();
    if (r != NFD_OKAY || !path) return false;
    snprintf(out, size, "%s", path);
    free(path);
    return true;
}
// (nfd's save dialog takes no file name: the extension is added when it was left off)
static bool pick_save(const char* filters, const char* ext, char* out, size_t size) {
    nfdchar_t* path = NULL;
    nfdresult_t r = NFD_SaveDialog(filters, NULL, &path);
    if (r == NFD_ERROR) dialog_error();
    if (r != NFD_OKAY || !path) return false;
    size_t n = strlen(path), e = strlen(ext);
    snprintf(out, size, "%s%s", path, n > e && !strcmp(path + n - e, ext) ? "" : ext);
    free(path);
    return true;
}
static bool pick_folder(char* out, size_t size) {
    nfdchar_t* path = NULL;
    nfdresult_t r = NFD_PickFolder(NULL, &path);
    if (r == NFD_ERROR) dialog_error();
    if (r != NFD_OKAY || !path) return false;
    snprintf(out, size, "%s", path);
    free(path);
    return true;
}

// ---- shading

static const char* kVS =
    "#version 330\n"
    "in vec3 vertexPosition; in vec3 vertexNormal;\n"
    "uniform mat4 mvp; uniform mat4 matModel; uniform mat4 matNormal;\n"
    "out vec3 fragNormal; out vec3 fragPos;\n"
    "void main() { fragPos = vec3(matModel * vec4(vertexPosition, 1.0)); fragNormal = normalize(vec3(matNormal * vec4(vertexNormal, 0.0)));\n"
    "  gl_Position = mvp * vec4(vertexPosition, 1.0); }\n";
static const char* kFS =
    "#version 330\n"
    "in vec3 fragNormal; in vec3 fragPos; uniform vec4 colDiffuse; uniform vec3 viewPos; out vec4 finalColor;\n"
    "void main() { vec3 n = normalize(fragNormal); vec3 v = normalize(viewPos - fragPos); if (dot(n, v) < 0.0) n = -n;\n"
    "  vec3 l = normalize(vec3(-0.4, 0.8, 0.45)); float d = max(dot(n, l), 0.0); float f = max(dot(n, v), 0.0);\n"
    "  float s = pow(max(dot(n, normalize(l + v)), 0.0), 40.0);\n"
    "  finalColor = vec4(colDiffuse.rgb * (0.22 + 0.55 * d + 0.25 * f) + 0.18 * s, colDiffuse.a); }\n";

// ---- the app's state

typedef struct {
    Params P;
    Design* design;                  // what is shown (and what Start starts from)
    bool haveDesign;
    char meshPath[1024], designPath[1024];
    // the optimization
    OptShared* sh;
    FbThread* worker;
    Design* working;                 // the worker's own copy
    bool running;
    int seenVersion;
    // post processing in the background
    FbThread* postThread;
    volatile int postDone;
    bool posting;
    Design* postDesign;
    PostReport postReport;
    // meshing the bodies in the background
    FbThread* mesher;
    volatile int meshDone;
    bool meshing, meshWanted;
    Design* meshDesign;
    DesignMeshes meshes;             // the mesher's result, waiting for upload
    Mesh* gpu;                       // uploaded body meshes
    int ngpu;
    Mesh targetMesh;
    bool haveTarget;
    int targetVersion;
    // the view
    Camera3D cam;
    float yaw, pitch, dist;
    V3 focus;
    float pose;                      // 0 closed .. 1 open
    bool animate, showTarget, showHinges, showSphere;
    float animT;
    float scroll;
    bool open[12];
    // the test suite
    SuiteShared* suite;
    FbThread* suiteThread;
    bool suiteRunning;
    char suiteFolder[1024], suiteIni[1024];
    int suiteCount;
    bool suiteUseIni;
    Texture2D suiteTex;
    int suiteTexVersion;
    Shader shader;
    int locView;
    RenderTexture2D rt;              // the 3D view, drawn at its own size
    Texture2D pics[2];
    int picVersion;
    char status[2048];
} App;

static App A;

static void dialog_error(void) {
    const char* e = NFD_GetError();
    snprintf(A.status, sizeof(A.status), "The file dialog failed: %s", e ? e : "(no reason given)");
}

// ---- meshes

static Mesh to_gpu(const TriMesh* m) {
    Mesh g = { 0 };
    if (m->nt == 0) return g;
    // smooth normals on the indexed mesh, then flattened (raylib's indices are 16 bit)
    V3* n = FB_NEW(V3, m->nv);
    for (int t = 0; t < m->nt; t++) {
        V3 a = m->pos[m->idx[3 * t]], b = m->pos[m->idx[3 * t + 1]], c = m->pos[m->idx[3 * t + 2]];
        V3 fn = v3cross(v3sub(b, a), v3sub(c, a));
        for (int k = 0; k < 3; k++) n[m->idx[3 * t + k]] = v3add(n[m->idx[3 * t + k]], fn);
    }
    g.vertexCount = 3 * m->nt;
    g.triangleCount = m->nt;
    g.vertices = MemAlloc(sizeof(float) * 3 * g.vertexCount);
    g.normals = MemAlloc(sizeof(float) * 3 * g.vertexCount);
    for (int t = 0; t < m->nt; t++)
        for (int k = 0; k < 3; k++) {
            int i = m->idx[3 * t + k], o = 3 * (3 * t + k);
            V3 p = m->pos[i], nn = v3norm(n[i]);
            g.vertices[o] = p.x; g.vertices[o + 1] = p.y; g.vertices[o + 2] = p.z;
            g.normals[o] = nn.x; g.normals[o + 1] = nn.y; g.normals[o + 2] = nn.z;
        }
    free(n);
    UploadMesh(&g, false);
    return g;
}

static void mesher_thread(void* v) {
    (void)v;
    // (finer where features were restored: as fine as they were)
    int cells = 160;
    for (int b = 0; b < A.meshDesign->nbodies; b++)
        if (A.meshDesign->bodies[b].detail.v) cells = fb_maxi(cells, A.P.featureResolution);
    A.meshes = design_meshes(A.meshDesign, cells);
    A.meshDone = 1;
}

static void request_meshes(const Design* d) {
    if (A.meshing) {
        A.meshWanted = true;         // (again when this one is done)
        return;
    }
    if (A.meshDesign) {
        design_free(A.meshDesign);
        free(A.meshDesign);
    }
    A.meshDesign = design_clone(d);
    A.meshDone = 0;
    A.meshing = true;
    A.meshWanted = false;
    A.mesher = fb_thread_start(mesher_thread, NULL);
}

static void poll_meshes(void) {
    if (!A.meshing || !A.meshDone) return;
    fb_thread_join(A.mesher);
    A.meshing = false;
    for (int k = 0; k < A.ngpu; k++)
        if (A.gpu[k].vertexCount) UnloadMesh(A.gpu[k]);
    free(A.gpu);
    A.ngpu = A.meshes.n;
    A.gpu = FB_NEW(Mesh, A.ngpu > 0 ? A.ngpu : 1);
    for (int k = 0; k < A.ngpu; k++) A.gpu[k] = to_gpu(&A.meshes.bodies[k]);
    design_meshes_free(&A.meshes);
    if (A.meshWanted) request_meshes(A.design);
}

static void set_target_mesh(const TriMesh* t) {
    if (A.haveTarget) UnloadMesh(A.targetMesh);
    A.targetMesh = to_gpu(t);
    A.haveTarget = true;
}

static void show_design(Design* d) {
    if (A.design && A.design != d) {
        design_free(A.design);
        free(A.design);
    }
    A.design = d;
    A.haveDesign = true;
    set_target_mesh(&d->target);
    request_meshes(d);
}

// ---- files

static void load_mesh(const char* path) {
    TriMesh m;
    char err[512];
    if (!mesh_load_obj(path, &m, err, sizeof(err))) { snprintf(A.status, sizeof(A.status), "%s", err); return; }
    Box3 b = mesh_bounds(&m);
    V3 e = v3sub(b.max, b.min);
    mesh_weld(&m, 1e-6f * fb_max(e.x, fb_max(e.y, e.z)));
    mesh_normalize(&m, 2.0f);
    Design* d = FB_NEW(Design, 1);
    design_init(d, &m, A.P.sizeMm, A.P.knuckleRadiusMm);
    mesh_free(&m);
    show_design(d);
    snprintf(A.meshPath, sizeof(A.meshPath), "%s", path);
    snprintf(A.status, sizeof(A.status), "Loaded %s: %d triangles. Press Start to optimize.", fb_basename(path), d->target.nt);
}

static void load_design(const char* path) {
    Design* d = FB_NEW(Design, 1);
    char err[512];
    if (!design_load(d, path, err, sizeof(err))) {
        free(d);
        snprintf(A.status, sizeof(A.status), "%s", err);
        return;
    }
    show_design(d);
    snprintf(A.designPath, sizeof(A.designPath), "%s", path);
    snprintf(A.status, sizeof(A.status), "Loaded %s: %d bodies, %d hinges%s. Start carries the optimization on from it.", fb_basename(path), d->nbodies,
             d->nhinges, d->built ? " (post processed)" : "");
}

// ---- running

typedef struct { Params P; Design* d; OptShared* sh; } Job;
static Job gJob;
static void opt_thread(void* v) {
    Job* j = v;
    optimize_run(&j->P, j->d, j->sh);
}

static void start(void) {
    if (!A.haveDesign || A.running || A.suiteRunning) return;
    if (A.sh) opt_shared_free(A.sh);
    A.sh = opt_shared_create();
    A.working = design_clone(A.design);
    A.seenVersion = -1;
    A.picVersion = -1;
    gJob.P = A.P;
    gJob.d = A.working;
    gJob.sh = A.sh;
    fb_threads_init(A.P.threads);
    A.worker = fb_thread_start(opt_thread, &gJob);
    A.running = true;
    snprintf(A.status, sizeof(A.status), "Optimizing...");
}

static void poll_run(void) {
    if (!A.sh) return;
    fb_mutex_lock(A.sh->lock);
    bool fin = A.sh->finished;
    Design* best = NULL;
    if (A.sh->best && A.sh->version != A.seenVersion) {
        best = design_clone(A.sh->best);
        A.seenVersion = A.sh->version;
    }
    fb_mutex_unlock(A.sh->lock);
    if (best) {
        // (the target mesh only changes with the semantic deformation)
        bool targetChanged = !A.design || best->target.nv != A.design->target.nv ||
                             (best->target.nv && memcmp(best->target.pos, A.design->target.pos, sizeof(V3) * (size_t)best->target.nv) != 0);
        if (A.design) {
            design_free(A.design);
            free(A.design);
        }
        A.design = best;
        if (targetChanged) set_target_mesh(&best->target);
        request_meshes(best);
    }
    if (A.running && fin) {
        fb_thread_join(A.worker);
        A.running = false;
        if (A.design) {
            design_free(A.design);
            free(A.design);
        }
        A.design = A.working;
        A.working = NULL;
        request_meshes(A.design);
        char* line = opt_status_line(A.sh);
        snprintf(A.status, sizeof(A.status), "%s", line);
        free(line);
    }
}

static void post_thread(void* v) {
    (void)v;
    post_process(A.postDesign, &A.P, &A.postReport);
    A.postDone = 1;
}

// ---- the 3D view

static Matrix rigid_matrix(Rigid r) {
    Matrix m = QuaternionToMatrix((Quaternion){ r.q.x, r.q.y, r.q.z, r.q.w });
    m.m12 = r.t.x;
    m.m13 = r.t.y;
    m.m14 = r.t.z;
    return m;
}

static void draw_view(Rectangle vp) {
    // the camera: orbit with the left button, pan with the right (or middle), zoom with the wheel
    Vector2 m = GetMousePosition();
    bool inside = CheckCollisionPointRec(m, vp);
    if (inside) {
        Vector2 dm = GetMouseDelta();
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            A.yaw -= dm.x * 0.008f;
            A.pitch = fb_clamp(A.pitch + dm.y * 0.008f, -1.5f, 1.5f);
        }
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT) || IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
            V3 fwd = v3(cosf(A.pitch) * sinf(A.yaw), sinf(A.pitch), cosf(A.pitch) * cosf(A.yaw));
            V3 right = v3norm(v3cross(v3(0, 1, 0), fwd)), up = v3cross(fwd, right);
            A.focus = v3add(A.focus, v3add(v3scale(right, dm.x * 0.002f * A.dist), v3scale(up, dm.y * 0.002f * A.dist)));
        }
        A.dist = fb_clamp(A.dist * powf(0.9f, GetMouseWheelMove()), 0.3f, 30.0f);
    }
    V3 eye = v3add(A.focus, v3scale(v3(cosf(A.pitch) * sinf(A.yaw), sinf(A.pitch), cosf(A.pitch) * cosf(A.yaw)), A.dist));
    A.cam.position = (Vector3){ eye.x, eye.y, eye.z };
    A.cam.target = (Vector3){ A.focus.x, A.focus.y, A.focus.z };
    A.cam.up = (Vector3){ 0, 1, 0 };
    A.cam.fovy = 40;
    A.cam.projection = CAMERA_PERSPECTIVE;
    float viewPos[3] = { eye.x, eye.y, eye.z };
    SetShaderValue(A.shader, A.locView, viewPos, SHADER_UNIFORM_VEC3);

    if (!A.rt.id || A.rt.texture.width != (int)vp.width || A.rt.texture.height != (int)vp.height) {
        if (A.rt.id) UnloadRenderTexture(A.rt);
        A.rt = LoadRenderTexture((int)vp.width, (int)vp.height);
    }
    BeginTextureMode(A.rt);
    ClearBackground((Color){ 22, 24, 29, 255 });
    BeginMode3D(A.cam);
    DrawGrid(20, 0.25f);
    if (A.haveDesign && A.design) {
        Design* d = A.design;
        Rigid xf[FB_MAX_BODIES];
        design_transforms(d, A.pose, xf);
        Material mat = LoadMaterialDefault();
        mat.shader = A.shader;
        for (int b = 0; b < A.ngpu && b < d->nbodies; b++) {
            if (!A.gpu[b].vertexCount) continue;
            V3 c = body_color(b);
            mat.maps[MATERIAL_MAP_DIFFUSE].color = (Color){ (unsigned char)(255 * c.x), (unsigned char)(255 * c.y), (unsigned char)(255 * c.z), 255 };
            DrawMesh(A.gpu[b], mat, rigid_matrix(xf[b]));
        }
        if (A.showTarget && A.haveTarget) {
            // the target, where the open pose is meant to match it (a ghost to compare with)
            rlEnableWireMode();
            mat.maps[MATERIAL_MAP_DIFFUSE].color = (Color){ 200, 200, 210, 60 };
            const float is = 1.0f / fb_max(d->scale, 1e-4f);
            DrawMesh(A.targetMesh, mat, MatrixMultiply(MatrixTranslate(-d->offset.x, -d->offset.y, -d->offset.z), MatrixScale(is, is, is)));
            rlDisableWireMode();
        }
        if (A.showHinges)
            for (int h = 0; h < d->nhinges; h++) {
                const Hinge* hg = &d->hinges[h];
                int p = design_body(d, hg->parent);
                if (p < 0) continue;
                V3 a = rapply(xf[p], v3mad(hg->pivot, hg->axis, -0.15f)), b = rapply(xf[p], v3mad(hg->pivot, hg->axis, 0.15f));
                DrawCylinderEx((Vector3){ a.x, a.y, a.z }, (Vector3){ b.x, b.y, b.z }, 0.012f, 0.012f, 8, (Color){ 250, 220, 90, 255 });
            }
        if (A.showSphere && A.pose < 0.05f)
            DrawSphereWires((Vector3){ 0, 0, 0 }, d->R * 1.002f, 12, 18, (Color){ 120, 200, 255, 90 });
    }
    EndMode3D();
    EndTextureMode();
    DrawTextureRec(A.rt.texture, (Rectangle){ 0, 0, (float)A.rt.texture.width, -(float)A.rt.texture.height }, (Vector2){ vp.x, vp.y }, WHITE);
    if (A.showTarget && A.haveDesign) DrawText("target (wireframe), over the open pose", (int)vp.x + 10, (int)vp.y + 10, 14, UI_MUTED);
    DrawText("left drag: orbit   right drag: pan   wheel: zoom", (int)vp.x + 10, (int)(vp.y + vp.height - 22), 14, UI_MUTED);
}

// the latest pictures: design red, target green (both: white)
static void update_pictures(void) {
    if (!A.sh) return;
    fb_mutex_lock(A.sh->lock);
    if (A.sh->picVersion == A.picVersion || !A.sh->pics) {
        fb_mutex_unlock(A.sh->lock);
        return;
    }
    A.picVersion = A.sh->picVersion;
    // every view, in a grid of up to 4 a row
    const int nv = A.sh->nv, res = A.sh->res, px = res * res, cols = nv < 4 ? nv : 4, rows = (nv + cols - 1) / cols;
    for (int pose = 0; pose < 1; pose++) {
        Image im = GenImageColor(cols * res, rows * res, (Color){ 30, 32, 38, 255 });
        Color* c = im.data;
        for (int v = 0; v < nv; v++)
            for (int k = 0; k < px; k++) {
                float D = A.sh->pics[((size_t)v) * px + k], T = A.sh->pics[((size_t)nv + v) * px + k];
                int x = k % res, y = k / res;
                c[(y + (v / cols) * res) * cols * res + (v % cols) * res + x] = (Color){ (unsigned char)(255 * D), (unsigned char)(255 * T), (unsigned char)(255 * fb_min(D, T)), 255 };
            }
        if (A.pics[pose].id) UnloadTexture(A.pics[pose]);
        A.pics[pose] = LoadTextureFromImage(im);
        UnloadImage(im);
    }
    fb_mutex_unlock(A.sh->lock);
}

// ---- the panel

static void params_panel(void) {
    int n;
    const ParamInfo* t = params_table(&n);
    const char* sections[] = { "model", "optimizer", "loss", "constraints", "grammar", "semantic", "post", "export", "suite" };
    for (int s = 0; s < 9; s++) {
        char title[64];
        snprintf(title, sizeof(title), "Settings: %s", sections[s]);
        if (!ui_header(title, &A.open[s])) continue;
        for (int k = 0; k < n; k++) {
            if (strcmp(t[k].section, sections[s]) != 0) continue;
            char* base = (char*)&A.P + t[k].offset;
            switch (t[k].type) {
            case PT_FLOAT: ui_slider_float(t[k].key, (float*)base, t[k].lo, t[k].hi, NULL); break;
            case PT_INT: ui_slider_int(t[k].key, (int*)base, (int)t[k].lo, (int)t[k].hi); break;
            case PT_BOOL: ui_checkbox(t[k].key, (bool*)base); break;
            default: continue;
            }
            ui_tooltip_last(t[k].help);
        }
    }
}

// ---- the test suite

static SuiteJob gSuiteJob;
static void suite_thread_fn(void* v) {
    (void)v;
    suite_run(&gSuiteJob, A.suite);
}

static void suite_start(void) {
    if (A.suiteRunning || A.running || !A.suiteFolder[0]) return;
    memset(&gSuiteJob, 0, sizeof(gSuiteJob));
    gSuiteJob.P = A.P;
    if (A.suiteUseIni) {
        char err[512];
        params_default(&gSuiteJob.P);
        if (!A.suiteIni[0] || !params_load(&gSuiteJob.P, A.suiteIni, err, sizeof(err))) {
            snprintf(A.status, sizeof(A.status), "The suite's settings: %s", A.suiteIni[0] ? err : "choose an .ini file first");
            return;
        }
    }
    snprintf(gSuiteJob.folder, sizeof(gSuiteJob.folder), "%s", A.suiteFolder);
    snprintf(gSuiteJob.outBase, sizeof(gSuiteJob.outBase), "%s", gSuiteJob.P.outDir);
    if (A.suite) suite_shared_free(A.suite);
    A.suite = suite_shared_create();
    A.suiteTexVersion = -1;
    fb_threads_init(gSuiteJob.P.threads);
    A.suiteThread = fb_thread_start(suite_thread_fn, NULL);
    A.suiteRunning = true;
    snprintf(A.status, sizeof(A.status), "Running the test suite on %d meshes in %s...", A.suiteCount, A.suiteFolder);
}

static void suite_poll(void) {
    if (!A.suite) return;
    fb_mutex_lock(A.suite->lock);
    bool fin = A.suite->finished;
    if (A.suite->tile && A.suite->tileVersion != A.suiteTexVersion) {
        A.suiteTexVersion = A.suite->tileVersion;
        Image im = { A.suite->tile, A.suite->tileSize, A.suite->tileSize, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8 };
        if (A.suiteTex.id && (A.suiteTex.width != A.suite->tileSize)) {
            UnloadTexture(A.suiteTex);
            A.suiteTex.id = 0;
        }
        if (!A.suiteTex.id) A.suiteTex = LoadTextureFromImage(im);
        else UpdateTexture(A.suiteTex, A.suite->tile);
    }
    // the current mesh's best design, in the view
    static OptShared* seenOpt;
    static int seenVersion = -1;
    Design* best = NULL;
    OptShared* os = A.suite->opt;
    if (os) {
        if (os != seenOpt) { seenOpt = os; seenVersion = -1; }
        fb_mutex_lock(os->lock);
        if (os->best && os->version != seenVersion) {
            best = design_clone(os->best);
            seenVersion = os->version;
        }
        fb_mutex_unlock(os->lock);
    }
    fb_mutex_unlock(A.suite->lock);
    if (best) {
        bool targetChanged = !A.design || best->target.nv != A.design->target.nv ||
                             (best->target.nv && memcmp(best->target.pos, A.design->target.pos, sizeof(V3) * (size_t)best->target.nv) != 0);
        if (A.design) {
            design_free(A.design);
            free(A.design);
        }
        A.design = best;
        A.haveDesign = true;
        if (targetChanged) set_target_mesh(&best->target);
        request_meshes(best);
    }
    if (A.suiteRunning && fin) {
        fb_thread_join(A.suiteThread);
        A.suiteRunning = false;
        fb_mutex_lock(A.suite->lock);
        if (A.suite->video[0]) snprintf(A.status, sizeof(A.status), "The test suite is done: %s", A.suite->video);
        else snprintf(A.status, sizeof(A.status), "The test suite ended without a video (see its log).");
        fb_mutex_unlock(A.suite->lock);
    }
}

static void suite_panel(Rectangle area) {
    ui_section("Test suite");
    {
        char label[1200];
        snprintf(label, sizeof(label), "%s", A.suiteFolder[0] ? fb_basename(A.suiteFolder) : "(no folder)");
        const char* labels[] = { "Folder of .obj files...", "Settings .ini..." };
        bool en[] = { !A.suiteRunning, !A.suiteRunning };
        int c;
        if (ui_button_row(labels, 2, en, &c)) {
            char path[1024];
            if (c == 0 && pick_folder(path, sizeof(path))) {
                snprintf(A.suiteFolder, sizeof(A.suiteFolder), "%s", path);
                A.suiteCount = suite_count(path);
            } else if (c == 1 && pick_open("ini", path, sizeof(path))) {
                snprintf(A.suiteIni, sizeof(A.suiteIni), "%s", path);
                A.suiteUseIni = true;
            }
        }
        ui_tooltip_last("Every .obj in the folder is optimized in turn with the same settings, recorded, and put into one video");
        if (A.suiteFolder[0]) {
            snprintf(label, sizeof(label), "%s: %d mesh%s", A.suiteFolder, A.suiteCount, A.suiteCount == 1 ? "" : "es");
            ui_text_wrapped(label, A.suiteCount > 0 ? UI_MUTED : UI_WARN);
        }
        bool useIni = A.suiteUseIni;
        if (ui_checkbox(A.suiteIni[0] ? "use the .ini's settings (else the current ones)" : "use an .ini's settings (choose one)", &useIni) && !A.suiteRunning)
            A.suiteUseIni = useIni && A.suiteIni[0];
        if (A.suiteUseIni) ui_text_wrapped(A.suiteIni, UI_MUTED);
    }
    {
        const char* labels[] = { "Run suite", A.suite && A.suite->pauseRequested ? "Resume" : "Pause", "Next mesh", "Stop" };
        bool en[] = { !A.suiteRunning && !A.running && A.suiteCount > 0, A.suiteRunning, A.suiteRunning, A.suiteRunning };
        int c;
        if (ui_button_row(labels, 4, en, &c)) {
            if (c == 0) suite_start();
            else if (c == 1) A.suite->pauseRequested = !A.suite->pauseRequested;
            else if (c == 2) A.suite->skipRequested = 1;
            else {
                A.suite->stopRequested = 1;
                A.suite->pauseRequested = 0;
            }
        }
        ui_tooltip_last("Next mesh ends this mesh's run (its best so far is kept); Stop ends the suite, and the meshes done still go into the video");
    }
    if (A.suite) {
        char* line = suite_status_line(A.suite);
        ui_text_wrapped(line, UI_MUTED);
        free(line);
        if (A.suiteTex.id) {
            float w = fminf(area.width - 20, 300);
            Rectangle r = ui_block(w);
            DrawTexturePro(A.suiteTex, (Rectangle){ 0, 0, (float)A.suiteTex.width, (float)A.suiteTex.height }, (Rectangle){ r.x, r.y, w, w }, (Vector2){ 0, 0 }, 0, WHITE);
        }
        fb_mutex_lock(A.suite->lock);
        if (A.suite->log[0]) ui_text_wrapped(A.suite->log, UI_INK);
        fb_mutex_unlock(A.suite->lock);
    }
}

static void panel(Rectangle area) {
    ui_begin(area, &A.scroll);
    ui_section("Files");
    char picked[1024];
    {
        const char* labels[] = { "Load mesh", "Load design", "Save design" };
        bool en[] = { !A.running, !A.running, A.haveDesign };
        int c;
        if (ui_button_row(labels, 3, en, &c)) {
            if (c == 0) {
                if (pick_open("obj", picked, sizeof(picked))) load_mesh(picked);
            } else if (c == 1) {
                if (pick_open("fold", picked, sizeof(picked))) load_design(picked);
            } else if (c == 2 && A.design) {
                char err[512];
                if (pick_save("fold", ".fold", picked, sizeof(picked))) {
                    bool ok = design_save(A.design, picked, err, sizeof(err));
                    snprintf(A.status, sizeof(A.status), ok ? "Saved %s" : "%s", ok ? picked : err);
                }
            }
        }
    }
    {
        const char* labels[] = { "Load settings", "Save settings" };
        int c;
        if (ui_button_row(labels, 2, NULL, &c)) {
            char err[512];
            if (c == 0) {
                if (pick_open("ini", picked, sizeof(picked))) {
                    Params q = A.P;
                    if (params_load(&q, picked, err, sizeof(err))) {
                        A.P = q;
                        snprintf(A.status, sizeof(A.status), "Loaded settings from %s", picked);
                    } else snprintf(A.status, sizeof(A.status), "%s", err);
                }
            } else if (pick_save("ini", ".ini", picked, sizeof(picked))) {
                snprintf(A.status, sizeof(A.status), params_save(&A.P, picked) ? "Saved settings to %s" : "Couldn't write %s", picked);
            }
        }
    }

    ui_section("Optimize");
    {
        bool paused = A.sh && A.sh->pauseRequested;
        const char* labels[] = { "Start", paused ? "Resume" : "Pause", "Stop" };
        bool en[] = { A.haveDesign && !A.running && !A.posting && !A.suiteRunning, A.running, A.running };
        int c;
        if (ui_button_row(labels, 3, en, &c)) {
            if (c == 0) start();
            else if (c == 1) A.sh->pauseRequested = !A.sh->pauseRequested;
            else if (c == 2) {
                A.sh->stopRequested = 1;
                A.sh->pauseRequested = 0;
            }
        }
        ui_tooltip_last("Start optimizes from the design shown (a loaded mesh, a loaded design, or the last result)");
    }
    if (A.sh) {
        char* line = opt_status_line(A.sh);
        ui_text_wrapped(line, UI_MUTED);
        free(line);
        fb_mutex_lock(A.sh->lock);
        const Score s = A.sh->bestScore;
        char buf[256];
        snprintf(buf, sizeof(buf), "open %.3f  cover %.0f%%  extra %.0f%%  collision %.4f (x%.1f)  hinges %.4f", s.open, 100.0f * s.openCover, 100.0f * s.openExtra, s.collision,
                 s.collisionScale, s.hinge);
        ui_label(buf, UI_INK);
        snprintf(buf, sizeof(buf), "holding %.3f  apart %.4f (%d)  surface carved %.1f%%  %.0f ms/eval", s.dependency, s.fragments, s.fragmentParts, 100.0f * s.surfaceCarved, s.ms);
        ui_label(buf, UI_INK);
        if (A.sh->message[0]) ui_text_wrapped(A.sh->message, UI_MUTED);
        // the loss over the steps
        Rectangle plot = ui_block(90);
        DrawRectangleRec(plot, (Color){ 30, 32, 38, 255 });
        if (A.sh->nhist > 1) {
            float lo = FLT_MAX, hi = -FLT_MAX;
            for (int k = 0; k < A.sh->nhist; k++) {
                lo = fb_min(lo, A.sh->history[k]);
                hi = fb_max(hi, A.sh->history[k]);
            }
            if (hi <= lo) hi = lo + 1e-6f;
            int step = A.sh->nhist / (int)plot.width + 1;
            Vector2 prev = { 0, 0 };
            for (int k = 0; k < A.sh->nhist; k += step) {
                Vector2 p = { plot.x + plot.width * k / (float)(A.sh->nhist - 1), plot.y + plot.height - 4 - (plot.height - 8) * (A.sh->history[k] - lo) / (hi - lo) };
                if (k) DrawLineV(prev, p, UI_ACCENT);
                prev = p;
            }
            snprintf(buf, sizeof(buf), "%.3f", hi);
            DrawText(buf, (int)plot.x + 4, (int)plot.y + 2, 12, UI_MUTED);
            snprintf(buf, sizeof(buf), "%.3f", lo);
            DrawText(buf, (int)plot.x + 4, (int)(plot.y + plot.height - 14), 12, UI_MUTED);
        }
        fb_mutex_unlock(A.sh->lock);
        // the pictures: open against the mesh (closed is the sphere by construction)
        for (int pose = 0; pose < 1; pose++) {
            if (!A.pics[pose].id) continue;
            ui_label("open vs target (red design, green target, white both)", UI_MUTED);
            float w = area.width - 20, hh = w * A.pics[pose].height / A.pics[pose].width;
            Rectangle r = ui_block(hh);
            DrawTexturePro(A.pics[pose], (Rectangle){ 0, 0, (float)A.pics[pose].width, (float)A.pics[pose].height }, r, (Vector2){ 0, 0 }, 0, WHITE);
        }
    }

    ui_section("View");
    ui_slider_float("pose (0 closed, 1 open)", &A.pose, 0, 1, "%.2f");
    ui_checkbox("animate opening and closing", &A.animate);
    ui_checkbox("show the target", &A.showTarget);
    ui_checkbox("show hinges", &A.showHinges);
    ui_checkbox("show the sphere (closed)", &A.showSphere);

    ui_section("Finish");
    {
        const char* labels[] = { A.posting ? "Post processing..." : "Post process", "Export STL" };
        bool en[] = { A.haveDesign && !A.running && !A.posting && !A.suiteRunning, A.haveDesign && !A.running && !A.posting && !A.suiteRunning };
        int c;
        if (ui_button_row(labels, 2, en, &c)) {
            if (c == 0) {
                A.postDesign = design_clone(A.design);
                A.postDone = 0;
                A.posting = true;
                fb_threads_init(A.P.threads);
                A.postThread = fb_thread_start(post_thread, NULL);
                snprintf(A.status, sizeof(A.status), "Post processing: hinges, carving, ledges, latch...");
            } else {
                char dir[1024];
                if (pick_folder(dir, sizeof(dir))) {
                    char msg[512];
                    char* stem = A.meshPath[0] ? fb_stem(A.meshPath) : fb_strdup("foldball");
                    export_stl(A.design, &A.P, dir, stem, msg, sizeof(msg));
                    snprintf(A.status, sizeof(A.status), "%s", msg);
                    free(stem);
                }
            }
        }
        ui_tooltip_last("Post processing builds the hinges, carves leftover collisions, adds ledges and the latch");
    }
    suite_panel(area);
    params_panel();
    ui_section("Status");
    ui_text_wrapped(A.status, UI_INK);
    ui_end();
}

int gui_run(const char* mesh) {
    memset(&A, 0, sizeof(A));
    params_default(&A.P);
    fb_threads_init(A.P.threads);
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(1500, 900, "foldball");
    SetTargetFPS(60);
    A.shader = LoadShaderFromMemory(kVS, kFS);
    A.locView = GetShaderLocation(A.shader, "viewPos");
    A.yaw = 0.6f;
    A.pitch = 0.35f;
    A.dist = 4.5f;
    A.pose = 1.0f;
    A.showHinges = true;
    A.showSphere = true;
    A.showTarget = true;
    A.open[1] = true;
    snprintf(A.status, sizeof(A.status), "Load a mesh (.obj) or a design (.fold) to start.");
    if (mesh) {
        const char* dot = strrchr(mesh, '.');
        if (dot && !strcmp(dot, ".fold")) load_design(mesh);
        else load_mesh(mesh);
    }
    // (a test hook: FOLDBALL_SCREENSHOT=<file> takes a picture after FOLDBALL_FRAMES frames and quits; FOLDBALL_START
    // starts optimizing at once)
    const char* shot = getenv("FOLDBALL_SCREENSHOT");
    int frames = getenv("FOLDBALL_FRAMES") ? atoi(getenv("FOLDBALL_FRAMES")) : 120, frame = 0;
    if (getenv("FOLDBALL_START")) start();
    if (getenv("FOLDBALL_POSE")) A.pose = (float)atof(getenv("FOLDBALL_POSE"));
    if (getenv("FOLDBALL_SUITE")) {          // (another hook: run the test suite on this folder at once)
        snprintf(A.suiteFolder, sizeof(A.suiteFolder), "%s", getenv("FOLDBALL_SUITE"));
        A.suiteCount = suite_count(A.suiteFolder);
        suite_start();
    }
    while (!WindowShouldClose()) {
        poll_meshes();
        poll_run();
        suite_poll();
        update_pictures();
        if (A.posting && A.postDone) {
            fb_thread_join(A.postThread);
            A.posting = false;
            show_design(A.postDesign);
            A.postDesign = NULL;
            snprintf(A.status, sizeof(A.status), "%s", A.postReport.text);
        }
        if (A.animate) {
            A.animT += GetFrameTime() * 0.25f;
            float x = fmodf(A.animT, 2.0f);
            A.pose = x < 1.0f ? x : 2.0f - x;
        }
        const int W = GetScreenWidth(), H = GetScreenHeight(), panelW = 440;
        BeginDrawing();
        ClearBackground(UI_BG);
        draw_view((Rectangle){ 0, 0, (float)(W - panelW), (float)H });
        panel((Rectangle){ (float)(W - panelW), 0, (float)panelW, (float)H });
        EndDrawing();
        if (shot && ++frame == frames) {
            TakeScreenshot(shot);
            break;
        }
    }
    if (A.running) {
        A.sh->stopRequested = 1;
        A.sh->pauseRequested = 0;
        fb_thread_join(A.worker);
    }
    if (A.suiteRunning) {
        A.suite->stopRequested = 1;
        A.suite->pauseRequested = 0;
        fb_thread_join(A.suiteThread);
    }
    if (A.meshing) fb_thread_join(A.mesher);
    if (A.posting) fb_thread_join(A.postThread);
    CloseWindow();
    return 0;
}
