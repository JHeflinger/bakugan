#include "eval.h"
#include "core/util.h"
#include <stdlib.h>
#include <string.h>

#define MAX_VIEWS 32
#define MAX_POSES 1100

// A slope: d loss / d (body's field) at a point, in design units. The point is x (design space, the pose's), or
// (src >= 0) p in body src's frame carried along with it (a collision sample). depth: the weight is d loss / d
// depth along dir, to be turned into d loss / d field by -1 / (grad f . dir).
typedef struct {
    int pose, body, src;
    V3 x, p;
    float w;
    V3 dir;
    int depth;
} Slope;
typedef FB_ARRAY(Slope) SlopeList;

typedef struct {
    float s;                 // opening progress (0 closed, 1 open)
    int hinge;               // the hinge (index) turning, -2 open, -1 closed
    float weight;
} Pose;

struct Evaluator {
    Params P;
    int nv, res;
    V3 dirs[MAX_VIEWS];
    View open[MAX_VIEWS];        // in the mesh's space
    float *openT, *openZ;
    Grid target;                 // the target's distance grid (drawn the way the design is)
    float *picD, *picT;
};

// ---- the lattice

typedef struct { const Design* d; Lattice* L; } LatCtx;
static void lattice_task(void* vc, int k) {
    LatCtx* c = vc;
    const Design* d = c->d;
    const int M = c->L->M;
    for (int j = 0; j < M; j++)
        for (int i = 0; i < M; i++) {
            V3 p = lattice_point(c->L, i, j, k);
            int lab = -1;
            if (v3len(p) < d->R) {
                bool hollow = false;
                for (int h = 0; h < d->nhollows && !hollow; h++) hollow = hollow_field(&d->hollows[h], p) < 0.0f;
                if (!hollow)
                    for (int b = 0; b < d->nbodies && lab < 0; b++)
                        if (body_cell(d, b, p, NULL) <= 0.0f) lab = b;
            }
            c->L->label[(k * M + j) * M + i] = lab;
        }
}

void lattice_build(const Design* d, int M, Lattice* L) {
    L->M = M;
    L->cell = 2.0f * d->R / M;
    L->origin = v3(-d->R + 0.5f * L->cell + 0.137f * L->cell, -d->R + 0.5f * L->cell + 0.071f * L->cell, -d->R + 0.5f * L->cell + 0.193f * L->cell);
    L->label = FB_NEW(int, (size_t)M * M * M);
    LatCtx c = { d, L };
    fb_parallel_for(M, lattice_task, &c);
}

void lattice_free(Lattice* L) {
    free(L->label);
    L->label = NULL;
}

// ---- drawing (a pose of the design, or the target's grid), soft silhouettes and depth

typedef struct {
    const Design* d;
    const Grid* grid;                // the target instead of the design
    const Rigid* xf;
    const Box3* box;
    View w;                          // (in design space for the design)
    float k;
    float* D;
    int* bodyD;
    V3* xD;
    float* Z;
    int* bodyZ;
    V3* xZ;
} RenderJob;
typedef struct { RenderJob* jobs; int res; } RenderCtx;

static inline float field_of(const RenderJob* J, int b, V3 q) { return J->grid ? grid_sample(J->grid, q) : body_field(J->d, b, q, NULL); }

static void render_row(void* vc, int task) {
    RenderCtx* c = vc;
    RenderJob* J = &c->jobs[task / c->res];
    const int j = task % c->res;
    const View* w = &J->w;
    const int nb = J->grid ? 1 : J->d->nbodies;
    const float k = J->k, far = 4.0f * w->half + 4.0f;
    int act[FB_MAX_BODIES];
    float tn[FB_MAX_BODIES], tf[FB_MAX_BODIES];
    V3 ol[FB_MAX_BODIES], dl[FB_MAX_BODIES];
    for (int i = 0; i < w->res; i++) {
        V3 pix = view_pixel(w, (float)i, (float)j);
        V3 o = v3mad(pix, w->dir, -far);
        int na = 0;
        float t0 = FLT_MAX, t1 = -FLT_MAX;
        for (int b = 0; b < nb; b++) {
            if (!box_valid(J->box[b])) continue;
            Rigid T = J->xf[b];
            V3 lo = rinv_apply(T, o), ld = rinv_rot(T, w->dir);
            float a0 = 0.0f, a1 = 2.0f * far + 10.0f;
            if (!box_ray(box_grow(J->box[b], 4.0f * k), lo, ld, &a0, &a1)) continue;
            act[na] = b;
            tn[na] = a0;
            tf[na] = a1;
            ol[na] = lo;
            dl[na] = ld;
            na++;
            t0 = fb_min(t0, a0);
            t1 = fb_max(t1, a1);
        }
        int id = j * w->res + i;
        J->D[id] = 0.0f;
        J->bodyD[id] = -1;
        J->xD[id] = pix;
        J->Z[id] = FLT_MAX;
        J->bodyZ[id] = -1;
        J->xZ[id] = pix;
        if (na == 0) continue;
        #define UNION_AT(tt, outv, outb) do { \
                float best_ = FLT_MAX; int bb_ = -1; \
                for (int q_ = 0; q_ < na; q_++) { \
                    if ((tt) < tn[q_] || (tt) > tf[q_]) continue; \
                    float v_ = field_of(J, act[q_], v3mad(ol[q_], dl[q_], (tt))); \
                    if (v_ < best_) { best_ = v_; bb_ = act[q_]; } \
                } \
                (outv) = best_; (outb) = bb_; } while (0)
        float t = t0, dmin = FLT_MAX, tmin = t0, lastOut = -1.0f, firstIn = -1.0f;
        int bmin = -1, bin = -1;
        for (int it = 0; it < 220 && t <= t1; it++) {
            float v;
            int bv;
            UNION_AT(t, v, bv);
            if (bv < 0) {
                float next = FLT_MAX;
                for (int q = 0; q < na; q++)
                    if (tn[q] > t) next = fb_min(next, tn[q]);
                if (next == FLT_MAX) break;
                t = next;
                continue;
            }
            if (v < dmin) { dmin = v; tmin = t; bmin = bv; }
            if (v < 0.0f) {
                if (firstIn < 0.0f) { firstIn = t; bin = bv; }
                if (v < -4.0f * k) break;
            } else if (firstIn < 0.0f) lastOut = t;
            t += v > k ? 0.95f * v : 0.5f * k;
        }
        if (bmin < 0) continue;
        if (fabsf(dmin) < 4.0f * k) {
            float a = tmin - 0.5f * k, bb = tmin + 0.5f * k;
            for (int it = 0; it < 6; it++) {
                float c1 = bb - 0.618034f * (bb - a), c2 = a + 0.618034f * (bb - a), f1, f2;
                int b1, b2;
                UNION_AT(c1, f1, b1);
                UNION_AT(c2, f2, b2);
                if (b1 >= 0 && f1 < dmin) { dmin = f1; tmin = c1; bmin = b1; }
                if (b2 >= 0 && f2 < dmin) { dmin = f2; tmin = c2; bmin = b2; }
                if (f1 < f2) bb = c2; else a = c1;
            }
        }
        J->D[id] = 1.0f / (1.0f + expf(fb_clamp(dmin / k, -30.0f, 30.0f)));
        J->bodyD[id] = bmin;
        J->xD[id] = v3mad(o, w->dir, tmin);
        if (firstIn >= 0.0f) {
            float lo = lastOut >= 0.0f ? lastOut : firstIn - 0.5f * k, hi = firstIn;
            for (int it = 0; it < 12; it++) {
                float mid = 0.5f * (lo + hi), v;
                int bv;
                UNION_AT(mid, v, bv);
                if (bv >= 0 && v < 0.0f) { hi = mid; bin = bv; } else lo = mid;
            }
            V3 xh = v3mad(o, w->dir, hi);
            J->Z[id] = v3dot(v3sub(xh, w->center), w->dir);
            J->bodyZ[id] = bin;
            J->xZ[id] = xh;
        }
        #undef UNION_AT
    }
}

static void render(RenderJob* jobs, int n, int res) {
    RenderCtx rc = { jobs, res };
    fb_parallel_for(n * res, render_row, &rc);
}

// ---- setting up

void eval_set_target(Evaluator* e, const TriMesh* target) {
    Box3 b = mesh_bounds(target);
    V3 c = v3scale(v3add(b.min, b.max), 0.5f), ext = v3sub(b.max, b.min);
    float longest = fb_max(ext.x, fb_max(ext.y, ext.z));
    float half = 0.6f * longest + 0.05f;
    for (int v = 0; v < e->nv; v++) e->open[v] = view_make(e->dirs[v], c, half, e->res);
    // the target's distance grid, around it (the mesh is centred at the origin by loading; a deformed one may move)
    grid_free(&e->target);
    float B = fb_max(fb_max(fabsf(b.min.x), fabsf(b.max.x)), fb_max(fb_max(fabsf(b.min.y), fabsf(b.max.y)), fb_max(fabsf(b.min.z), fabsf(b.max.z))));
    e->target = grid_create(80, B + 0.08f * longest + 1e-3f);
    grid_from_mesh(&e->target, target);
    const int nv = e->nv, res = e->res, px = res * res;
    RenderJob* jobs = FB_NEW(RenderJob, nv);
    int* bd = FB_NEW(int, (size_t)nv * px);
    int* bz = FB_NEW(int, (size_t)nv * px);
    V3* xd = FB_NEW(V3, (size_t)nv * px);
    V3* xz = FB_NEW(V3, (size_t)nv * px);
    Rigid xf[1] = { rident() };
    Box3 box[1] = { { v3(-e->target.B, -e->target.B, -e->target.B), v3(e->target.B, e->target.B, e->target.B) } };
    for (int v = 0; v < nv; v++) {
        RenderJob* J = &jobs[v];
        memset(J, 0, sizeof(*J));
        J->grid = &e->target;
        J->xf = xf;
        J->box = box;
        J->w = e->open[v];
        J->k = e->P.softness * 2.0f * J->w.half / res * 0.5f;
        J->D = e->openT + (size_t)v * px;
        J->Z = e->openZ + (size_t)v * px;
        J->bodyD = bd + (size_t)v * px;
        J->bodyZ = bz + (size_t)v * px;
        J->xD = xd + (size_t)v * px;
        J->xZ = xz + (size_t)v * px;
    }
    render(jobs, nv, res);
    for (size_t k = 0; k < (size_t)nv * px; k++)
        if (e->openZ[k] == FLT_MAX) e->openZ[k] = 0.0f;
    free(jobs);
    free(bd);
    free(bz);
    free(xd);
    free(xz);
}

Evaluator* eval_create(const Params* p, const Design* d) {
    Evaluator* e = FB_NEW(Evaluator, 1);
    e->P = *p;
    e->nv = fb_mini(fb_maxi(p->views, 1), MAX_VIEWS);
    e->res = p->viewPixels;
    view_directions(e->nv, e->dirs);
    const size_t px = (size_t)e->nv * e->res * e->res;
    e->openT = FB_NEW(float, px);
    e->openZ = FB_NEW(float, px);
    e->picD = FB_NEW(float, px);
    e->picT = FB_NEW(float, px);
    eval_set_target(e, &d->target);
    return e;
}

void eval_free(Evaluator* e) {
    if (!e) return;
    free(e->openT);
    free(e->openZ);
    free(e->picD);
    free(e->picT);
    grid_free(&e->target);
    free(e);
}

int eval_views(const Evaluator* e) { return e->nv; }
int eval_resolution(const Evaluator* e) { return e->res; }
void eval_picture(const Evaluator* e, int view, float* design, float* target) {
    const size_t px = (size_t)e->res * e->res;
    memcpy(design, e->picD + view * px, sizeof(float) * px);
    memcpy(target, e->picT + view * px, sizeof(float) * px);
}

Grad* grad_create(const Design* d) {
    (void)d;
    return FB_NEW(Grad, 1);
}
void grad_free(Grad* g) { free(g); }
void grad_clear(Grad* g, const Design* d) {
    (void)d;
    memset(g, 0, sizeof(*g));
}

// ---- collisions

typedef struct {
    int pose, a, b;
    float weight;
} PairJob;

typedef struct {
    const Design* d;
    Rigid* const* xf;
    const Box3* box;
    V3* const* samples;
    const int* nsamples;
    PairJob* jobs;
    SlopeList* out;
    double* cost;
    float sampleVolume, scale, depthScale;
} CollCtx;

static void collide_one(CollCtx* c, int job, int a, int b) {
    const PairJob* J = &c->jobs[job];
    const Rigid Ta = c->xf[J->pose][a], Tb = c->xf[J->pose][b];
    const Box3 bb = c->box[b];
    for (int s = 0; s < c->nsamples[a]; s++) {
        V3 p = c->samples[a][s];
        V3 x = rapply(Ta, p);
        V3 q = rinv_apply(Tb, x);
        if (q.x < bb.min.x || q.y < bb.min.y || q.z < bb.min.z || q.x > bb.max.x || q.y > bb.max.y || q.z > bb.max.z) continue;
        float f = body_field(c->d, b, q, NULL);
        if (f >= 0.0f) continue;
        float depth = -f;
        float cost = c->sampleVolume * (depth + depth * depth / c->depthScale);
        float slope = c->sampleVolume * (1.0f + 2.0f * depth / c->depthScale);
        c->cost[job] += J->weight * cost;
        Slope sl = { J->pose, b, a, x, p, -J->weight * slope * c->scale, v3(0, 0, 0), 0 };
        fb_push(&c->out[job], sl);
    }
}

static void collide_task(void* vc, int job) {
    CollCtx* c = vc;
    collide_one(c, job, c->jobs[job].a, c->jobs[job].b);
    collide_one(c, job, c->jobs[job].b, c->jobs[job].a);
}

// ---- slopes into the gradient

// a parameter the transforms depend on: a hinge's phi or angle, or a plane's (a hinge sits on it)
typedef struct { int kind, index, comp; } TParam;

typedef struct {
    const Design* d;
    const Pose* poses;
    Rigid* const* xf;
    Rigid*** pert;                   // [pose][param * 2 + side] -> transforms (pose 1, closed, has none)
    unsigned long long** moved;      // [pose][param]: which bodies the parameter moves
    const TParam* tp;
    int ntp;
    const Slope* slopes;
    int nslopes, chunks;
    Grad** part;
    float scale;                     // the placement's scale
} SlopeCtx;

static const float kEps = 1e-3f;

static void hollow_param_grad(const Hollow* h, V3 q, float w, float* g) {
    Hollow t = *h;
    float* v[7] = { &t.a.x, &t.a.y, &t.a.z, &t.b.x, &t.b.y, &t.b.z, &t.r };
    for (int c = 0; c < 7; c++) {
        float keep = *v[c];
        *v[c] = keep + kEps;
        float fp = hollow_field(&t, q);
        *v[c] = keep - kEps;
        float fm = hollow_field(&t, q);
        *v[c] = keep;
        // the body's field there is -hollow
        g[c] += w * -(fp - fm) / (2.0f * kEps);
    }
}

static V3 hollow_grad(const Hollow* h, V3 q) {
    const float e = 1e-4f;
    return v3((hollow_field(h, v3(q.x + e, q.y, q.z)) - hollow_field(h, v3(q.x - e, q.y, q.z))) / (2 * e),
              (hollow_field(h, v3(q.x, q.y + e, q.z)) - hollow_field(h, v3(q.x, q.y - e, q.z))) / (2 * e),
              (hollow_field(h, v3(q.x, q.y, q.z + e)) - hollow_field(h, v3(q.x, q.y, q.z - e))) / (2 * e));
}

static void slope_task(void* vc, int chunk) {
    SlopeCtx* c = vc;
    const Design* d = c->d;
    Grad* g = c->part[chunk];
    const int from = (int)((long long)c->nslopes * chunk / c->chunks), to = (int)((long long)c->nslopes * (chunk + 1) / c->chunks);
    for (int k = from; k < to; k++) {
        const Slope* s = &c->slopes[k];
        const Rigid* T = c->xf[s->pose];
        V3 x = s->src >= 0 ? rapply(T[s->src], s->p) : s->x;
        V3 q = rinv_apply(T[s->body], x);
        Active act;
        float f = body_field(d, s->body, q, &act);
        V3 gf;
        if (act.kind == 0) gf = v3norm(q);
        else if (act.kind == 1) gf = v3scale(d->planes[act.index].n, (float)act.sign);
        else if (act.kind == 2) gf = v3scale(hollow_grad(&d->hollows[act.index], q), -1.0f);
        else gf = body_gradient(d, s->body, q);
        float w = s->w;
        if (s->depth) {
            float gd = v3dot(gf, rinv_rot(T[s->body], s->dir));
            if (gd > -0.2f) gd = -0.2f;
            w = s->w * (-1.0f / gd);
        }
        if (act.kind == 1) {
            float sg = (float)act.sign;
            g->plane[act.index][0] += w * sg * q.x;
            g->plane[act.index][1] += w * sg * q.y;
            g->plane[act.index][2] += w * sg * q.z;
            g->plane[act.index][3] -= w * sg;
        } else if (act.kind == 2) {
            hollow_param_grad(&d->hollows[act.index], q, w, g->hollow[act.index]);
        }
        // the placement (the open pose's pictures): world field = scale * f((x - offset) / scale)
        if (s->pose == 0 && s->src < 0) {
            V3 gy = rrot(T[s->body], gf);
            float ww = w / c->scale;
            g->place[0] += ww * (f - v3dot(gy, x));
            g->place[1] -= ww * gy.x;
            g->place[2] -= ww * gy.y;
            g->place[3] -= ww * gy.z;
        }
        // the transforms
        if (!c->pert[s->pose]) continue;
        for (int t = 0; t < c->ntp; t++) {
            unsigned long long mv = c->moved[s->pose][t];
            bool mb = (mv >> s->body) & 1ull, ms = s->src >= 0 ? (mv >> s->src) & 1ull : false;
            if (!mb && !ms) continue;
            const Rigid* Tp = c->pert[s->pose][2 * t];
            const Rigid* Tm = c->pert[s->pose][2 * t + 1];
            V3 xp = s->src >= 0 ? rapply(Tp[s->src], s->p) : s->x;
            V3 xm = s->src >= 0 ? rapply(Tm[s->src], s->p) : s->x;
            V3 qp = rinv_apply(Tp[s->body], xp), qm = rinv_apply(Tm[s->body], xm);
            float dv = w * v3dot(gf, v3sub(qp, qm)) / (2.0f * kEps);
            const TParam* P = &c->tp[t];
            if (P->kind == 0) g->hinge[P->index][P->comp] += dv;
            else g->plane[P->index][P->comp] += dv;
        }
    }
}

// ---- the whole loss

static float ramp(float c) { return fb_clamp((c - 0.5f) / 0.4f, 0.0f, 1.0f); }
static float ramp_slope(float c) { return c > 0.5f && c < 0.9f ? 2.5f : 0.0f; }
static float smooth_abs(float x) { return sqrtf(x * x + 1e-4f) - 0.01f; }
static float smooth_sign(float x) { return x / sqrtf(x * x + 1e-4f); }

// the transforms with one parameter nudged
static void transforms_nudged(const Design* d, const TParam* tp, float delta, float s, Rigid* out) {
    Plane planes[FB_MAX_PLANES];
    Hinge hinges[FB_MAX_BODIES];
    memcpy(planes, d->planes, sizeof(Plane) * (size_t)d->nplanes);
    memcpy(hinges, d->hinges, sizeof(Hinge) * (size_t)d->nhinges);
    if (tp->kind == 0) {
        if (tp->comp == 0) hinges[tp->index].phi += delta;
        else hinges[tp->index].angle += delta;
    } else {
        if (tp->comp < 3) v3set(&planes[tp->index].n, tp->comp, v3get(planes[tp->index].n, tp->comp) + delta);
        else planes[tp->index].d += delta;
    }
    design_transforms_h(d, planes, hinges, s, out);
}

Score eval_run(Evaluator* e, const Design* d, Grad* grad, float collisionScale, float phase) {
    const double began = fb_now();
    const Params* P = &e->P;
    Score sc;
    memset(&sc, 0, sizeof(sc));
    sc.collisionScale = collisionScale;
    const int NB = d->nbodies;
    const float R = d->R;

    // 1. the lattice: whose each point is, the bodies' boxes, collision samples, volume
    Lattice L;
    lattice_build(d, fb_maxi(P->samples, 8), &L);
    const int M = L.M;
    const float cell = L.cell;
    Box3 box[FB_MAX_BODIES];
    for (int b = 0; b < NB; b++) box[b] = box_empty();
    V3* samples[FB_MAX_BODIES] = { 0 };
    int nsamples[FB_MAX_BODIES] = { 0 }, cap[FB_MAX_BODIES] = { 0 };
    int total = 0;
    for (int k = 0; k < M; k++)
        for (int j = 0; j < M; j++)
            for (int i = 0; i < M; i++) {
                int b = L.label[(k * M + j) * M + i];
                if (b < 0) continue;
                V3 p = lattice_point(&L, i, j, k);
                box[b] = box_add(box[b], p);
                total++;
                if (nsamples[b] >= cap[b]) {
                    cap[b] = cap[b] ? cap[b] * 2 : 256;
                    samples[b] = fb_realloc(samples[b], sizeof(V3) * (size_t)cap[b]);
                }
                samples[b][nsamples[b]++] = p;
            }
    for (int b = 0; b < NB; b++)
        if (box_valid(box[b])) {
            box[b] = box_grow(box[b], cell);
            // (never past the sphere)
            box[b].min = v3max(box[b].min, v3(-R, -R, -R));
            box[b].max = v3min(box[b].max, v3(R, R, R));
        }
    const double volume = total * (double)cell * cell * cell;
    sc.volume = (float)volume;

    // 2. the poses: open, closed, and checks through each hinge's turn
    Pose poses[MAX_POSES];
    int np = 0;
    poses[np++] = (Pose){ 1.0f, -2, 1.0f };
    poses[np++] = (Pose){ 0.0f, -1, 1.0f };
    int order[FB_MAX_BODIES];
    design_order(d, order);
    const int m = d->nhinges, K = fb_maxi(1, P->checksPerHinge);
    for (int slot = 0; slot < m && P->collisionWeight > 0.0f; slot++)
        for (int j = 0; j < K && np < MAX_POSES; j++) {
            float frac = (j + fmodf(phase * 0.618034f * (j + 1) + 0.5f * phase, 1.0f)) / K;
            poses[np++] = (Pose){ (slot + frac) / m, order[slot], 1.0f / K };
        }
    Rigid** xf = FB_NEW(Rigid*, np);
    for (int q = 0; q < np; q++) {
        xf[q] = FB_NEW(Rigid, FB_MAX_BODIES);
        design_transforms(d, poses[q].s, xf[q]);
    }

    SlopeList slopes = { 0 };
    // 3. the open pose drawn, in the mesh's space through the placement: done in design space (the views brought
    // into it: the same pictures)
    const int nv = e->nv, res = e->res, px = res * res;
    const float s = fb_max(d->scale, 1e-4f);
    RenderJob* jobs = FB_NEW(RenderJob, nv);
    float* bufD = FB_NEW(float, (size_t)nv * px);
    float* bufZ = FB_NEW(float, (size_t)nv * px);
    int* bufBD = FB_NEW(int, (size_t)nv * px);
    int* bufBZ = FB_NEW(int, (size_t)nv * px);
    V3* bufXD = FB_NEW(V3, (size_t)nv * px);
    V3* bufXZ = FB_NEW(V3, (size_t)nv * px);
    for (int v = 0; v < nv; v++) {
        RenderJob* J = &jobs[v];
        memset(J, 0, sizeof(*J));
        size_t off = (size_t)v * px;
        const View* ww = &e->open[v];
        J->d = d;
        J->xf = xf[0];
        J->box = box;
        J->w = *ww;
        J->w.center = v3scale(v3sub(ww->center, d->offset), 1.0f / s);
        J->w.half = ww->half / s;
        J->k = P->softness * 2.0f * ww->half / res * 0.5f / s;
        J->D = bufD + off;
        J->Z = bufZ + off;
        J->bodyD = bufBD + off;
        J->bodyZ = bufBZ + off;
        J->xD = bufXD + off;
        J->xZ = bufXZ + off;
    }
    render(jobs, nv, res);
    {
        const float depthScale = 0.04f * 2.0f * e->open[0].half;     // (world units)
        const float poseW = 1.0f / nv;
        double lossSum = 0, coverSum = 0, extraSum = 0;
        // the depth rasters' look (fit_look 2): coverage times nearness, nearness 1 at the front of the frame to 0.25 at its
        // back (so a covered pixel never looks like nothing), along the view in the mesh's units
        const float zHalf = e->open[0].half, zA = 0.75f / (2.0f * zHalf);
        #define NEAR(z) (1.0f - zA * fb_clamp((z) + zHalf, 0.0f, 2.0f * zHalf))
        for (int v = 0; v < nv; v++) {
            const RenderJob* J = &jobs[v];
            const float* T = e->openT + (size_t)v * px;
            const float* TZ = e->openZ + (size_t)v * px;
            memcpy(e->picD + (size_t)v * px, J->D, sizeof(float) * px);
            memcpy(e->picT + (size_t)v * px, T, sizeof(float) * px);
            if (P->fitLook == 2) {
                // design: where its ray enters the open pose (or, missing it, comes nearest); target: its front surface
                double sumT = 1e-6, sumAbs = 0, hit = 0, extra = 0;
                for (int q = 0; q < px; q++) {
                    const float D = J->D[q];
                    float zd = J->Z[q] < FLT_MAX ? s * J->Z[q] : s * v3dot(v3sub(J->xD[q], J->w.center), J->w.dir);
                    float vD = D * NEAR(zd), vT = T[q] * NEAR(TZ[q]);
                    e->picD[(size_t)v * px + q] = vD;
                    e->picT[(size_t)v * px + q] = vT;
                    sumT += vT;
                    sumAbs += smooth_abs(vD - vT);
                    hit += fb_min(T[q], D);
                    extra += fb_max(0.0f, D - T[q]);
                }
                lossSum += sumAbs / sumT;
                double sumTT = 1e-6;
                for (int q = 0; q < px; q++) sumTT += T[q];
                coverSum += hit / sumTT;
                extraSum += extra / sumTT;
                if (!grad) continue;
                for (int q = 0; q < px; q++) {
                    const float D = J->D[q];
                    const bool entered = J->Z[q] < FLT_MAX && J->bodyZ[q] >= 0;
                    float zd = entered ? s * J->Z[q] : s * v3dot(v3sub(J->xD[q], J->w.center), J->w.dir);
                    float vD = D * NEAR(zd), vT = T[q] * NEAR(TZ[q]);
                    float dLdv = smooth_sign(vD - vT) / (float)sumT;
                    // through the coverage (the edge)
                    if (J->bodyD[q] >= 0 && D * (1.0f - D) > 1e-4f) {
                        Slope sl = { 0, J->bodyD[q], -1, J->xD[q], v3(0, 0, 0), poseW * dLdv * NEAR(zd) * (-D * (1.0f - D) / J->k), v3(0, 0, 0), 0 };
                        fb_push(&slopes, sl);
                    }
                    // through the depth (where the ray enters: d near / d world depth, times the scale: per design unit)
                    const float zc = zd + zHalf;
                    if (entered && zc > 0.0f && zc < 2.0f * zHalf) {
                        Slope sl = { 0, J->bodyZ[q], -1, J->xZ[q], v3(0, 0, 0), poseW * dLdv * D * -zA * s, J->w.dir, 1 };
                        fb_push(&slopes, sl);
                    }
                }
                continue;
            }
            double sumT = 1e-6, sumAbs = 0, hit = 0, extra = 0, zsum = 0;
            for (int q = 0; q < px; q++) {
                sumT += T[q];
                sumAbs += smooth_abs(T[q] - J->D[q]);
                hit += fb_min(T[q], J->D[q]);
                extra += fb_max(0.0f, J->D[q] - T[q]);
                if (P->fitLook == 1 && J->D[q] > 0.5f && T[q] > 0.5f && J->Z[q] < FLT_MAX) {
                    float ez = s * J->Z[q] - TZ[q], s2 = depthScale * depthScale;
                    zsum += ramp(J->D[q]) * ramp(T[q]) * ez * ez / (ez * ez + s2);
                }
            }
            lossSum += sumAbs / sumT + P->depthWeight * zsum / sumT;
            coverSum += hit / sumT;
            extraSum += extra / sumT;
            if (!grad) continue;
            for (int q = 0; q < px; q++) {
                const float D = J->D[q];
                const float dDdf = -D * (1.0f - D) / J->k;
                if (J->bodyD[q] >= 0 && D * (1.0f - D) > 1e-4f) {
                    float dLdD = smooth_sign(D - T[q]) / (float)sumT;
                    Slope sl = { 0, J->bodyD[q], -1, J->xD[q], v3(0, 0, 0), poseW * dLdD * dDdf, v3(0, 0, 0), 0 };
                    fb_push(&slopes, sl);
                }
                if (P->fitLook == 1 && D > 0.5f && T[q] > 0.5f && J->Z[q] < FLT_MAX && J->bodyZ[q] >= 0 && P->depthWeight > 0.0f) {
                    float ez = s * J->Z[q] - TZ[q], s2 = depthScale * depthScale, den = ez * ez + s2;
                    float mD = ramp(D), mT = ramp(T[q]), rho = ez * ez / den;
                    // d loss / d world depth; the slope wants d loss / d design depth (world = scale * design)
                    float dz = P->depthWeight * mD * mT * 2.0f * ez * s2 / (den * den) / (float)sumT;
                    Slope sl = { 0, J->bodyZ[q], -1, J->xZ[q], v3(0, 0, 0), poseW * dz * s, J->w.dir, 1 };
                    fb_push(&slopes, sl);
                    float dm = P->depthWeight * rho * mT * ramp_slope(D) / (float)sumT;
                    if (dm != 0.0f && J->bodyD[q] >= 0) {
                        Slope sm = { 0, J->bodyD[q], -1, J->xD[q], v3(0, 0, 0), poseW * dm * dDdf, v3(0, 0, 0), 0 };
                        fb_push(&slopes, sm);
                    }
                }
            }
        }
        #undef NEAR
        sc.open = (float)(lossSum / nv);
        sc.openCover = (float)(coverSum / nv);
        sc.openExtra = (float)(extraSum / nv);
    }

    // 4. collisions: open, and through each hinge's turn (closed, the bodies partition the sphere)
    {
        FB_ARRAY(PairJob) pj = { 0 };
        for (int q = 0; q < np; q++) {
            if (poses[q].hinge == -1) continue;
            Box3 wb[FB_MAX_BODIES];
            for (int b = 0; b < NB; b++) wb[b] = box_transform(box[b], xf[q][b]);
            for (int a = 0; a < NB; a++)
                for (int b = a + 1; b < NB; b++) {
                    if (!box_valid(wb[a]) || !box_valid(wb[b]) || !box_overlap(wb[a], wb[b])) continue;
                    if (poses[q].hinge >= 0 && design_below(d, poses[q].hinge, a) == design_below(d, poses[q].hinge, b)) continue;
                    PairJob J = { q, a, b, poses[q].weight };
                    fb_push(&pj, J);
                }
        }
        const float norm = (float)fb_max(volume, 1e-6) * 0.05f * R;
        CollCtx cc = { d, xf, box, samples, nsamples, pj.data, FB_NEW(SlopeList, pj.count + 1), FB_NEW(double, pj.count + 1),
                       cell * cell * cell, P->collisionWeight * collisionScale / norm, 0.1f * R };
        fb_parallel_for(pj.count, collide_task, &cc);
        double openCost = 0, motionCost = 0;
        for (int k = 0; k < pj.count; k++) {
            if (poses[pj.data[k].pose].hinge == -2) openCost += cc.cost[k];
            else {
                motionCost += cc.cost[k];
                sc.hingeCollision[poses[pj.data[k].pose].hinge] += (float)(cc.cost[k] / norm);
            }
            if (grad)
                for (int q = 0; q < cc.out[k].count; q++) fb_push(&slopes, cc.out[k].data[q]);
            fb_free_array(&cc.out[k]);
        }
        free(cc.out);
        free(cc.cost);
        fb_free_array(&pj);
        sc.collisionOpen = (float)(openCost / norm);
        sc.collisionMotion = (float)(motionCost / norm);
        sc.collision = sc.collisionOpen + sc.collisionMotion;
    }

    // 5. hinges: both bodies right at the barrel (the hinge sits on its seam; where along it, both must be there)
    if (P->hingeWeight > 0.0f) {
        const float Lh = P->hingeLength * R, slack = 0.25f * d->inset;
        const float sj[5] = { -0.5f, -0.25f, 0.0f, 0.25f, 0.5f };
        for (int hi = 0; hi < d->nhinges; hi++) {
            const Hinge* hg = &d->hinges[hi];
            int bs[2] = { design_body(d, hg->parent), design_body(d, hg->child) };
            // the barrel's cost for a hinge frame (pivot, axis)
            #define BARREL_COST(pv, ax, out) do { \
                    float c_ = 0; \
                    for (int k_ = 0; k_ < 5; k_++) { \
                        V3 pt_ = v3mad((pv), (ax), sj[k_] * Lh); \
                        for (int s_ = 0; s_ < 2; s_++) \
                            if (bs[s_] >= 0) c_ += fb_max(0.0f, body_field(d, bs[s_], pt_, NULL) - slack); \
                    } \
                    (out) = c_ / (5.0f * R); } while (0)
            float cost;
            BARREL_COST(hg->pivot, hg->axis, cost);
            sc.hinge += cost;
            if (!grad || cost <= 0.0f) continue;
            // the barrel moves with phi and the plane
            const float w = P->hingeWeight;
            Plane planes[FB_MAX_PLANES];
            memcpy(planes, d->planes, sizeof(Plane) * (size_t)d->nplanes);
            for (int comp = 0; comp < 5; comp++) {
                float cp, cm;
                for (int side = 0; side < 2; side++) {
                    float delta = side == 0 ? kEps : -kEps;
                    Hinge h2 = *hg;
                    Plane* pl = &planes[hg->plane];
                    Plane keep = *pl;
                    if (comp == 0) h2.phi += delta;
                    else if (comp < 4) v3set(&pl->n, comp - 1, v3get(pl->n, comp - 1) + delta);
                    else pl->d += delta;
                    V3 pv, ax;
                    hinge_frame(d, planes, &h2, &pv, &ax);
                    *pl = keep;
                    BARREL_COST(pv, ax, *(side == 0 ? &cp : &cm));
                }
                float gcomp = w * (cp - cm) / (2.0f * kEps);
                if (comp == 0) grad->hinge[hi][0] += gcomp;
                else grad->plane[hg->plane][comp - 1] += gcomp;
            }
            // and the bodies' fields there (closed: the design itself)
            for (int k = 0; k < 5; k++) {
                V3 pt = v3mad(hg->pivot, hg->axis, sj[k] * Lh);
                for (int s2 = 0; s2 < 2; s2++)
                    if (bs[s2] >= 0 && body_field(d, bs[s2], pt, NULL) > slack) {
                        Slope sl = { 1, bs[s2], -1, pt, v3(0, 0, 0), w / (5.0f * R), v3(0, 0, 0), 0 };
                        fb_push(&slopes, sl);
                    }
            }
            #undef BARREL_COST
        }
    }

    // 6. holding: each hinge but the first to open next (closed) to a body that opens before it
    if (P->dependencyWeight > 0.0f && d->nhinges > 1) {
        const float slack = 1.5f * cell;
        for (int slot = 1; slot < m; slot++) {
            const int hi = order[slot];
            const int C = design_body(d, d->hinges[hi].child);
            if (C < 0 || nsamples[C] == 0) continue;
            float bestGap = FLT_MAX;
            int bestH = -1, bestS = -1;
            for (int H = 0; H < NB; H++) {
                if (H == C || design_below(d, hi, H) || nsamples[H] == 0) continue;
                bool frees = false;
                for (int s2 = 0; s2 < slot && !frees; s2++) frees = design_below(d, order[s2], H) != design_below(d, order[s2], C);
                if (!frees) continue;
                for (int q = 0; q < nsamples[C]; q += 2) {
                    // (near the surface: where a ledge can go)
                    if (v3len(samples[C][q]) < 0.75f * R) continue;
                    float f = body_field(d, H, samples[C][q], NULL);
                    if (f < bestGap) { bestGap = f; bestH = H; bestS = q; }
                }
            }
            if (bestH < 0) {
                sc.dependency += 1.0f;
                continue;
            }
            if (bestGap <= slack) continue;
            sc.dependency += fb_min((bestGap - slack) / R, 1.0f);
            if (grad) {
                Slope sl = { 1, bestH, -1, samples[C][bestS], v3(0, 0, 0), P->dependencyWeight / R, v3(0, 0, 0), 0 };
                fb_push(&slopes, sl);
            }
        }
    }

    // 7. one piece: per body, what lies apart from its largest part (6-connected lattice points)
    if (P->onePieceWeight > 0.0f) {
        const size_t N = (size_t)M * M * M;
        int* comp = FB_NEW(int, N);
        int* stack = FB_NEW(int, N);
        for (size_t k = 0; k < N; k++) comp[k] = -1;
        int ncomp = 0;
        FB_ARRAY(int) size = { 0 };
        FB_ARRAY(int) owner = { 0 };
        for (size_t s0 = 0; s0 < N; s0++) {
            if (comp[s0] >= 0 || L.label[s0] < 0) continue;
            int b = L.label[s0], cnt = 0, top = 0;
            stack[top++] = (int)s0;
            comp[s0] = ncomp;
            while (top) {
                int c = stack[--top];
                cnt++;
                int i = c % M, j = (c / M) % M, k = c / (M * M);
                int nb[6] = { i > 0 ? c - 1 : -1, i < M - 1 ? c + 1 : -1, j > 0 ? c - M : -1, j < M - 1 ? c + M : -1, k > 0 ? c - M * M : -1,
                              k < M - 1 ? c + M * M : -1 };
                for (int q = 0; q < 6; q++)
                    if (nb[q] >= 0 && comp[nb[q]] < 0 && L.label[nb[q]] == b) {
                        comp[nb[q]] = ncomp;
                        stack[top++] = nb[q];
                    }
            }
            fb_push(&size, cnt);
            fb_push(&owner, b);
            ncomp++;
        }
        int largest[FB_MAX_BODIES];
        for (int b = 0; b < NB; b++) largest[b] = -1;
        for (int c = 0; c < ncomp; c++)
            if (largest[owner.data[c]] < 0 || size.data[c] > size.data[largest[owner.data[c]]]) largest[owner.data[c]] = c;
        double apart = 0;
        for (int c = 0; c < ncomp; c++)
            if (c != largest[owner.data[c]]) {
                apart += size.data[c];
                sc.fragmentParts++;
            }
        sc.fragments = (float)(apart / fb_maxi(total, 1));
        sc.onePiece = sc.fragments;
        if (grad && apart > 0) {
            const float w = -P->onePieceWeight / (float)fb_maxi(total, 1) / cell;
            for (size_t s0 = 0; s0 < N; s0++) {
                int c = comp[s0];
                if (c < 0 || c == largest[owner.data[c]]) continue;
                int i = (int)(s0 % M), j = (int)((s0 / M) % M), k = (int)(s0 / ((size_t)M * M));
                bool edge = i == 0 || j == 0 || k == 0 || i == M - 1 || j == M - 1 || k == M - 1 || comp[s0 - 1] != c || comp[s0 + 1] != c ||
                            comp[s0 - M] != c || comp[s0 + M] != c || comp[s0 - (size_t)M * M] != c || comp[s0 + (size_t)M * M] != c;
                if (!edge) continue;
                Slope sl = { 1, owner.data[c], -1, lattice_point(&L, i, j, k), v3(0, 0, 0), w, v3(0, 0, 0), 0 };
                fb_push(&slopes, sl);
            }
        }
        fb_free_array(&size);
        fb_free_array(&owner);
        free(comp);
        free(stack);
    }

    // 8. the surface: hollows carving into the sphere's surface cost (it should stay a sphere, to roll)
    if (P->surfaceWeight > 0.0f && d->nhollows > 0) {
        const int NS = 1500;
        const float scale = 0.02f * R, under = 0.997f * R;
        double sum = 0;
        int in = 0;
        for (int i = 0; i < NS; i++) {
            float y = 1.0f - (i + 0.5f) / NS * 2.0f, rr = sqrtf(fb_max(0.0f, 1.0f - y * y)), phi = i * 2.39996323f;
            V3 p = v3scale(v3(rr * cosf(phi), y, rr * sinf(phi)), under);
            int which = -1;
            float best = 0.0f;
            for (int h = 0; h < d->nhollows; h++) {
                float f = hollow_field(&d->hollows[h], p);
                if (f < best) { best = f; which = h; }
            }
            if (which < 0) continue;
            in++;
            float dd = -best;
            sum += dd / (dd + scale);
            if (grad) {
                // d cost / d hollow field = -scale / (dd + scale)^2 (the hollow's field is -dd there)
                float w = -P->surfaceWeight * scale / ((dd + scale) * (dd + scale)) / NS;
                // (hollow_param_grad takes the body field's slope: the body's field is minus the hollow's)
                hollow_param_grad(&d->hollows[which], p, -w, grad->hollow[which]);
            }
        }
        sc.surface = (float)(sum / NS);
        sc.surfaceCarved = (float)in / NS;
    }

    sc.total = sc.open + P->collisionWeight * collisionScale * sc.collision + P->hingeWeight * sc.hinge + P->dependencyWeight * sc.dependency +
               P->onePieceWeight * sc.onePiece + P->surfaceWeight * sc.surface;
    sc.valid = true;

    // 9. the slopes into the gradient
    if (grad) {
        // the parameters the transforms depend on: each hinge's phi and angle, each plane a hinge sits on
        TParam* tp = FB_NEW(TParam, 2 * FB_MAX_BODIES + 4 * FB_MAX_PLANES);
        int ntp = 0;
        bool planeUsed[FB_MAX_PLANES] = { 0 };
        for (int h = 0; h < d->nhinges; h++) {
            tp[ntp++] = (TParam){ 0, h, 0 };
            tp[ntp++] = (TParam){ 0, h, 1 };
            planeUsed[d->hinges[h].plane] = true;
        }
        for (int p = 0; p < d->nplanes; p++)
            if (planeUsed[p])
                for (int c = 0; c < 4; c++) tp[ntp++] = (TParam){ 1, p, c };
        bool need[MAX_POSES] = { 0 };
        for (int k = 0; k < slopes.count; k++) need[slopes.data[k].pose] = true;
        Rigid*** pert = FB_NEW(Rigid**, np);
        unsigned long long** moved = FB_NEW(unsigned long long*, np);
        for (int q = 0; q < np; q++) {
            if (!need[q] || q == 1 || ntp == 0) continue;
            pert[q] = FB_NEW(Rigid*, 2 * ntp);
            moved[q] = FB_NEW(unsigned long long, ntp);
            for (int t = 0; t < ntp; t++) {
                for (int side = 0; side < 2; side++) {
                    pert[q][2 * t + side] = FB_NEW(Rigid, FB_MAX_BODIES);
                    transforms_nudged(d, &tp[t], side == 0 ? kEps : -kEps, poses[q].s, pert[q][2 * t + side]);
                }
                unsigned long long mv = 0;
                for (int b = 0; b < NB; b++)
                    if (memcmp(&pert[q][2 * t][b], &xf[q][b], sizeof(Rigid)) != 0 || memcmp(&pert[q][2 * t + 1][b], &xf[q][b], sizeof(Rigid)) != 0)
                        mv |= 1ull << b;
                moved[q][t] = mv;
            }
        }
        const int chunks = fb_threads_count() * 2;
        Grad** part = FB_NEW(Grad*, chunks);
        for (int c = 0; c < chunks; c++) part[c] = FB_NEW(Grad, 1);
        SlopeCtx sctx = { d, poses, xf, pert, moved, tp, ntp, slopes.data, slopes.count, chunks, part, s };
        fb_parallel_for(chunks, slope_task, &sctx);
        for (int c = 0; c < chunks; c++) {
            const Grad* pg = part[c];
            for (int pl = 0; pl < d->nplanes; pl++)
                for (int q = 0; q < 4; q++) grad->plane[pl][q] += pg->plane[pl][q];
            for (int hi = 0; hi < d->nhinges; hi++)
                for (int q = 0; q < 2; q++) grad->hinge[hi][q] += pg->hinge[hi][q];
            for (int ho = 0; ho < d->nhollows; ho++)
                for (int q = 0; q < 7; q++) grad->hollow[ho][q] += pg->hollow[ho][q];
            for (int q = 0; q < 4; q++) grad->place[q] += pg->place[q];
            free(part[c]);
        }
        free(part);
        for (int q = 0; q < np; q++) {
            if (!pert[q]) continue;
            for (int k = 0; k < 2 * ntp; k++) free(pert[q][k]);
            free(pert[q]);
            free(moved[q]);
        }
        free(pert);
        free(moved);
        free(tp);
    }

    fb_free_array(&slopes);
    for (int q = 0; q < np; q++) free(xf[q]);
    free(xf);
    for (int b = 0; b < NB; b++) free(samples[b]);
    free(jobs);
    free(bufD);
    free(bufZ);
    free(bufBD);
    free(bufBZ);
    free(bufXD);
    free(bufXZ);
    lattice_free(&L);
    sc.ms = (fb_now() - began) * 1e3;
    return sc;
}
