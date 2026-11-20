#include "post.h"
#include "core/util.h"
#include "opt/eval.h"
#include "features.h"
#include <stdlib.h>
#include <string.h>

// ---- helpers

typedef FB_ARRAY(V3) Pts;

// the lattice post processing works on (and carves on): over the sphere, carveResolution a side
static Grid g_lat;
static void lattice_set(const Design* d, const Params* p) {
    g_lat.n = fb_mini(fb_maxi(p->carveResolution, 24), 256);
    g_lat.B = 1.05f * d->R;
    g_lat.h = 2.0f * g_lat.B / (g_lat.n - 1);
    g_lat.v = NULL;
}

// lattice nodes inside body b (its whole field), every `stride`-th along each axis
typedef struct { const Design* d; int b, stride, nk; Pts* slices; } NodesCtx;
static void nodes_task(void* vc, int s) {
    NodesCtx* c = vc;
    const Grid* g = &g_lat;
    const int k = s * c->stride;
    for (int j = 0; j < g->n; j += c->stride)
        for (int i = 0; i < g->n; i += c->stride) {
            V3 q = grid_node(g, i, j, k);
            if (body_field(c->d, c->b, q, NULL) < 0.0f) fb_push(&c->slices[s], q);
        }
}
static Pts body_nodes(const Design* d, int b, int stride) {
    const int nk = (g_lat.n + stride - 1) / stride;
    NodesCtx c = { d, b, stride, nk, FB_NEW(Pts, nk) };
    fb_parallel_for(nk, nodes_task, &c);
    Pts p = { 0 };
    for (int s = 0; s < nk; s++) {
        for (int q = 0; q < c.slices[s].count; q++) fb_push(&p, c.slices[s].data[q]);
        fb_free_array(&c.slices[s]);
    }
    free(c.slices);
    return p;
}

// which lattice nodes are inside body b
typedef struct { const Design* d; int b; unsigned char* in; } MaskCtx;
static void mask_task(void* vc, int k) {
    MaskCtx* c = vc;
    const int n = g_lat.n;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) c->in[((size_t)k * n + j) * n + i] = body_field(c->d, c->b, grid_node(&g_lat, i, j, k), NULL) < 0.0f;
}

static float total_volume(const Design* d) {
    const Grid* g = &g_lat;
    int in = 0;
    for (int k = 0; k < g->n; k++)
        for (int j = 0; j < g->n; j++)
            for (int i = 0; i < g->n; i++) {
                V3 q = grid_node(g, i, j, k);
                for (int b = 0; b < d->nbodies; b++)
                    if (body_field(d, b, q, NULL) < 0.0f) { in++; break; }
            }
    return in * g->h * g->h * g->h;
}

// A rotation taking z to dir.
static Quat quat_z_to(V3 dir) {
    V3 z = v3(0, 0, 1), a = v3norm(dir);
    float c = v3dot(z, a);
    if (c > 0.99999f) return qident();
    if (c < -0.99999f) return qaxis(v3(1, 0, 0), FB_PI);
    return qaxis(v3cross(z, a), acosf(fb_clamp(c, -1, 1)));
}
// A rotation taking x, y, z to the given (orthonormal, right handed) axes.
static Quat quat_frame(V3 ex, V3 ey, V3 ez) {
    float m00 = ex.x, m01 = ey.x, m02 = ez.x, m10 = ex.y, m11 = ey.y, m12 = ez.y, m20 = ex.z, m21 = ey.z, m22 = ez.z;
    float tr = m00 + m11 + m22;
    Quat q;
    if (tr > 0) {
        float s = sqrtf(tr + 1.0f) * 2;
        q.w = 0.25f * s; q.x = (m21 - m12) / s; q.y = (m02 - m20) / s; q.z = (m10 - m01) / s;
    } else if (m00 > m11 && m00 > m22) {
        float s = sqrtf(1.0f + m00 - m11 - m22) * 2;
        q.w = (m21 - m12) / s; q.x = 0.25f * s; q.y = (m01 + m10) / s; q.z = (m02 + m20) / s;
    } else if (m11 > m22) {
        float s = sqrtf(1.0f + m11 - m00 - m22) * 2;
        q.w = (m02 - m20) / s; q.x = (m01 + m10) / s; q.y = 0.25f * s; q.z = (m12 + m21) / s;
    } else {
        float s = sqrtf(1.0f + m22 - m00 - m11) * 2;
        q.w = (m10 - m01) / s; q.x = (m02 + m20) / s; q.y = (m12 + m21) / s; q.z = 0.25f * s;
    }
    return qnorm(q);
}

static int add_extra(Design* d, int bodyId, int shape, int op, Rigid pose, V3 size, float round, int tag) {
    if (d->nextras >= FB_MAX_EXTRAS) return -1;
    Extra* e = &d->extras[d->nextras];
    e->body = bodyId;
    e->shape = shape;
    e->op = op;
    e->pose = pose;
    e->size = size;
    e->round = round;
    e->tag = tag;
    return d->nextras++;
}

// a box given in the world (closed pose), as an extra of a body in its own frame
static int add_world_box(Design* d, int b, const Rigid* xf, V3 centre, V3 ex, V3 ey, V3 ez, V3 half, int op, float round, int tag) {
    Rigid w = { quat_frame(ex, ey, ez), centre };
    Rigid local = rthen(w, rinverse(xf[b]));
    return add_extra(d, d->bodies[b].id, EX_BOX, op, local, half, round, tag);
}

// how much of body a's samples (given in its frame) lie inside body b, with the transforms given
typedef struct { const Design* d; const Pts* sa; int a, b; const Rigid* xf; int chunks; int* out; } CountCtx;
static void count_task(void* vc, int chunk) {
    CountCtx* c = vc;
    const int from = (int)((long long)c->sa->count * chunk / c->chunks), to = (int)((long long)c->sa->count * (chunk + 1) / c->chunks);
    int n = 0;
    for (int k = from; k < to; k++) {
        V3 x = rapply(c->xf[c->a], c->sa->data[k]);
        if (body_field(c->d, c->b, rinv_apply(c->xf[c->b], x), NULL) < 0.0f) n++;
    }
    c->out[chunk] = n;
}
static int count_inside(const Design* d, const Pts* sa, int a, int b, const Rigid* xf) {
    int out[64] = { 0 };
    const int chunks = sa->count < 2048 ? 1 : 64;
    CountCtx c = { d, sa, a, b, xf, chunks, out };
    fb_parallel_for(chunks, count_task, &c);
    int n = 0;
    for (int k = 0; k < chunks; k++) n += out[k];
    return n;
}

// the gradient of a body's cells alone (not the sphere): across its seams, near the surface
static V3 cell_gradient(const Design* d, int b, V3 q) {
    const float e = 1e-3f * d->R;
    return v3((body_cell(d, b, v3(q.x + e, q.y, q.z), NULL) - body_cell(d, b, v3(q.x - e, q.y, q.z), NULL)) / (2 * e),
              (body_cell(d, b, v3(q.x, q.y + e, q.z), NULL) - body_cell(d, b, v3(q.x, q.y - e, q.z), NULL)) / (2 * e),
              (body_cell(d, b, v3(q.x, q.y, q.z + e), NULL) - body_cell(d, b, v3(q.x, q.y, q.z - e), NULL)) / (2 * e));
}

// ---- overlap along the motion

float post_overlap(const Design* d, int checks) {
    if (d->nbodies < 2) return 0;
    if (g_lat.n == 0) {
        g_lat.n = 72;
        g_lat.B = 1.05f * d->R;
        g_lat.h = 2.0f * g_lat.B / (g_lat.n - 1);
    }
    const float h = g_lat.h, cellVol = 8.0f * h * h * h;
    Pts* s = FB_NEW(Pts, d->nbodies);
    for (int b = 0; b < d->nbodies; b++) s[b] = body_nodes(d, b, 2);
    float total = 0;
    for (int b = 0; b < d->nbodies; b++) total += s[b].count * cellVol;
    int order[FB_MAX_BODIES];
    design_order(d, order);
    Rigid xf[FB_MAX_BODIES];
    float worst = 0;
    const int m = d->nhinges;
    for (int q = -1; q < m * checks; q++) {
        float sv = q < 0 ? 0.0f : ((q / checks) + (q % checks + 0.5f) / checks) / m;
        int moving = q < 0 ? -1 : order[q / checks];
        design_transforms(d, sv, xf);
        int over = 0;
        for (int a = 0; a < d->nbodies; a++)
            for (int b = 0; b < d->nbodies; b++) {
                if (a == b) continue;
                if (moving >= 0 && design_below(d, moving, a) == design_below(d, moving, b)) continue;
                over += count_inside(d, &s[a], a, b, xf);
            }
        worst = fb_max(worst, 0.5f * over * cellVol / fb_max(total, 1e-9f));
    }
    for (int b = 0; b < d->nbodies; b++) fb_free_array(&s[b]);
    free(s);
    return worst;
}

// ---- 1. fragments

static void carve_ensure(Design* d, int b) {
    Grid* cg = &d->bodies[b].carve;
    if (cg->v && cg->n == g_lat.n && fabsf(cg->B - g_lat.B) < 1e-6f) return;
    Grid ng = grid_create(g_lat.n, g_lat.B);
    const size_t N = (size_t)ng.n * ng.n * ng.n;
    for (size_t k = 0; k < N; k++) {
        V3 q = grid_node(&ng, (int)(k % ng.n), (int)((k / ng.n) % ng.n), (int)(k / ((size_t)ng.n * ng.n)));
        ng.v[k] = cg->v ? grid_sample(cg, q) : 4.0f * d->R;
    }
    grid_free(cg);
    *cg = ng;
}

static int remove_fragments(Design* d) {
    const Grid* g = &g_lat;
    const int n = g->n;
    const size_t N = (size_t)n * n * n;
    int removed = 0;
    int* comp = FB_NEW(int, N);
    int* stack = FB_NEW(int, N);
    unsigned char* in = FB_NEW(unsigned char, N);
    for (int b = 0; b < d->nbodies; b++) {
        MaskCtx mc = { d, b, in };
        fb_parallel_for(n, mask_task, &mc);
        for (size_t k = 0; k < N; k++) comp[k] = -1;
        int nc = 0, best = -1, bestSize = 0;
        // (parts of a few nodes are the lattice missing a thin edge of the body, not pieces apart from it: left be)
        FB_ARRAY(int) sizes = { 0 };
        for (size_t s0 = 0; s0 < N; s0++) {
            if (!in[s0] || comp[s0] >= 0) continue;
            int top = 0, size = 0;
            stack[top++] = (int)s0;
            comp[s0] = nc;
            while (top) {
                int c = stack[--top];
                size++;
                int i = c % n, j = (c / n) % n, k = c / (n * n);
                int nb[6] = { i > 0 ? c - 1 : -1, i < n - 1 ? c + 1 : -1, j > 0 ? c - n : -1, j < n - 1 ? c + n : -1, k > 0 ? c - n * n : -1,
                              k < n - 1 ? c + n * n : -1 };
                for (int q = 0; q < 6; q++)
                    if (nb[q] >= 0 && in[nb[q]] && comp[nb[q]] < 0) {
                        comp[nb[q]] = nc;
                        stack[top++] = nb[q];
                    }
            }
            if (size > bestSize) { bestSize = size; best = nc; }
            fb_push(&sizes, size);
            nc++;
        }
        bool real = false;
        for (int c = 0; c < nc; c++) real |= c != best && sizes.data[c] >= 6;
        if (!real) { fb_free_array(&sizes); continue; }
        // everything but the largest part carved away (through the body's carve grid, on the same lattice)
        carve_ensure(d, b);
        float* cv = d->bodies[b].carve.v;
        bool* gone = FB_NEW(bool, nc);
        for (size_t k = 0; k < N; k++)
            if (comp[k] >= 0 && comp[k] != best && sizes.data[comp[k]] >= 6) {
                cv[k] = -g->h;
                gone[comp[k]] = true;
            }
        // (and the empty nodes around them, so the smooth carve takes them whole)
        for (size_t k = 0; k < N; k++) {
            if (comp[k] >= 0) continue;
            int i = (int)(k % n), j = (int)((k / n) % n), kk = (int)(k / ((size_t)n * n));
            bool near = false;
            for (int dz = -1; dz <= 1 && !near; dz++)
                for (int dy = -1; dy <= 1 && !near; dy++)
                    for (int dx = -1; dx <= 1 && !near; dx++) {
                        int a2 = i + dx, b2 = j + dy, c2 = kk + dz;
                        if (a2 < 0 || b2 < 0 || c2 < 0 || a2 >= n || b2 >= n || c2 >= n) continue;
                        int q = comp[((size_t)c2 * n + b2) * n + a2];
                        near = q >= 0 && q != best && sizes.data[q] >= 6;
                    }
            if (near) cv[k] = -g->h;
        }
        for (int c = 0; c < nc; c++) removed += gone[c];
        fb_free_array(&sizes);
        if (getenv("FOLDBALL_DEBUG")) {
            int* sz = FB_NEW(int, nc);
            for (size_t k = 0; k < N; k++) if (comp[k] >= 0) sz[comp[k]]++;
            fprintf(stderr, "body %d: %d parts:", b, nc);
            for (int c = 0; c < nc; c++) fprintf(stderr, " %d", sz[c]);
            fprintf(stderr, "\n");
            free(sz);
        }
        free(gone);
    }
    free(comp);
    free(stack);
    free(in);
    return removed;
}

// ---- 2. hinges

static void build_hinges(Design* d, const Params* p, PostReport* rep) {
    const float u = 1.0f / fb_max(d->mmPerUnit, 1e-6f);
    const float rk = p->knuckleRadiusMm * u, rp = fb_min(p->pinRadiusMm * u, 0.6f * rk), c = p->clearanceMm * u;
    const float L = fb_max(p->hingeLength * d->R, 6.0f * c + 0.01f);
    for (int h = 0; h < d->nhinges; h++) {
        const Hinge* hg = &d->hinges[h];
        V3 a = v3norm(hg->axis);
        Quat q = quat_z_to(a);
        const float seg = L / 3.0f, hl = 0.5f * seg - 0.5f * c;
        for (int k = 0; k < 3; k++) {
            Rigid pose = { q, v3mad(hg->pivot, a, (k - 1) * seg) };
            bool parentOwns = k != 1;
            // the knuckle, on its body
            add_extra(d, parentOwns ? hg->parent : hg->child, EX_CYLINDER, 1, pose, v3(rk, 0, hl), 0, 0);
            // room for it in the other body
            add_extra(d, parentOwns ? hg->child : hg->parent, EX_CYLINDER, -1, pose, v3(rk + c, 0, hl + c), 0, 0);
        }
        Rigid mid = { q, hg->pivot };
        // the pin fused to the parent's knuckles, through a bore in the child's
        add_extra(d, hg->parent, EX_CYLINDER, 2, mid, v3(rp, 0, 0.5f * L - 0.5f * c), 0, 0);
        add_extra(d, hg->child, EX_CYLINDER, -1, mid, v3(rp + c, 0, 0.5f * L + c), 0, 0);
        rep->hingesBuilt++;
    }
}

// ---- 3. carving

typedef struct { float s; int victim, other; } Hit;
typedef struct { const Design* d; int v; const Hit* hits; int nhits; float clearance; float* cv; Rigid* xf; Box3* box; } CarveCtx;
typedef FB_ARRAY(Hit) HitList;
typedef struct { const Design* d; const Pts* s; const int* vol; const float* sv; const int* moving; HitList* out; } HitCtx;
static void hit_task(void* vc, int qi) {
    HitCtx* c = vc;
    const Design* d = c->d;
    const float sv = c->sv[qi];
    const int moving = c->moving[qi];
    Rigid xf[FB_MAX_BODIES];
    design_transforms(d, sv, xf);
    const Pts* s = c->s;
    for (int a = 0; a < d->nbodies; a++)
        for (int b = a + 1; b < d->nbodies; b++) {
            if (moving >= 0 && design_below(d, moving, a) == design_below(d, moving, b)) continue;
            // (a sample stride keeps this quick; the carving itself is on every node)
            int over = 0;
            for (int k = 0; k < s[a].count && !over; k += 3)
                if (body_field(d, b, rinv_apply(xf[b], rapply(xf[a], s[a].data[k])), NULL) < 0.0f) over = 1;
            for (int k = 0; k < s[b].count && !over; k += 3)
                if (body_field(d, a, rinv_apply(xf[a], rapply(xf[b], s[b].data[k])), NULL) < 0.0f) over = 1;
            if (!over) continue;
            // the smaller of the two gives way
            Hit hit = { sv, c->vol[a] <= c->vol[b] ? a : b, c->vol[a] <= c->vol[b] ? b : a };
            fb_push(&c->out[qi], hit);
        }
}

static void carve_task(void* vc, int k) {
    CarveCtx* c = vc;
    const Grid* g = &g_lat;
    const int n = g->n;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            V3 q = grid_node(g, i, j, k);
            if (body_field(c->d, c->v, q, NULL) > 3.0f * g->h) continue;
            size_t id = ((size_t)k * n + j) * n + i;
            for (int hI = 0; hI < c->nhits; hI++) {
                if (c->hits[hI].victim != c->v) continue;
                const Box3 bb = c->box[hI];
                if (q.x < bb.min.x || q.y < bb.min.y || q.z < bb.min.z || q.x > bb.max.x || q.y > bb.max.y || q.z > bb.max.z) continue;
                const Rigid* xf = c->xf + (size_t)hI * FB_MAX_BODIES;
                int o = c->hits[hI].other;
                float f = body_field(c->d, o, rinv_apply(xf[o], rapply(xf[c->v], q)), NULL) - c->clearance;
                if (f < c->cv[id]) c->cv[id] = f;
            }
        }
}

static void carve(Design* d, const Params* p, PostReport* rep) {
    const Grid* g = &g_lat;
    const int n = g->n;
    const float clearance = p->clearanceMm / fb_max(d->mmPerUnit, 1e-6f);
    const int NB = d->nbodies, checks = fb_maxi(2, p->carveChecks);
    Pts* s = FB_NEW(Pts, NB);
    int vol[FB_MAX_BODIES];
    Box3 nodeBox[FB_MAX_BODIES];
    for (int b = 0; b < NB; b++) {
        s[b] = body_nodes(d, b, 1);
        vol[b] = s[b].count;
        nodeBox[b] = box_empty();
        for (int k = 0; k < s[b].count; k++) nodeBox[b] = box_add(nodeBox[b], s[b].data[k]);
        nodeBox[b] = box_grow(nodeBox[b], g->h);
    }
    int order[FB_MAX_BODIES];
    design_order(d, order);
    const int m = d->nhinges;
    // the colliding (pose, victim, other)
    FB_ARRAY(Hit) hits = { 0 };
    // the poses: closed, and along each hinge's turn finely enough that the swept volume is smooth (a step of the
    // turn moves the material a hinge's length away by under a lattice cell), carve_checks at least
    FB_ARRAY(float) svs = { 0 };
    FB_ARRAY(int) movs = { 0 };
    fb_push(&svs, 0.0f);
    fb_push(&movs, -1);
    for (int slot = 0; slot < m; slot++) {
        const float turn = fabsf(d->hinges[order[slot]].angle);
        const int K = fb_mini(fb_maxi(checks, (int)ceilf(turn * 0.6f * d->R / (0.7f * g->h))), 400);
        for (int j = 0; j < K; j++) {
            fb_push(&svs, (slot + (j + 0.5f) / K) / m);
            fb_push(&movs, order[slot]);
        }
    }
    const int nposes = svs.count;
    HitCtx hc = { d, s, vol, svs.data, movs.data, FB_NEW(HitList, nposes) };
    fb_parallel_for(nposes, hit_task, &hc);
    for (int q = 0; q < nposes; q++) {
        for (int k = 0; k < hc.out[q].count; k++) fb_push(&hits, hc.out[q].data[k]);
        fb_free_array(&hc.out[q]);
    }
    free(hc.out);
    fb_free_array(&svs);
    fb_free_array(&movs);
    // each victim: the others' bodies over its poses, grown by the clearance, as a grid in its frame
    for (int v = 0; v < NB; v++) {
        bool any = false;
        for (int k = 0; k < hits.count; k++) any |= hits.data[k].victim == v;
        if (!any) continue;
        carve_ensure(d, v);
        float* cv = d->bodies[v].carve.v;
        // (only where the victim is, or nearly)
        CarveCtx cc = { d, v, hits.data, hits.count, clearance, cv, FB_NEW(Rigid, (size_t)hits.count * FB_MAX_BODIES), FB_NEW(Box3, hits.count + 1) };
        for (int hI = 0; hI < hits.count; hI++) {
            if (hits.data[hI].victim != v) continue;
            Rigid* xf = cc.xf + (size_t)hI * FB_MAX_BODIES;
            design_transforms(d, hits.data[hI].s, xf);
            // the other body's box in the victim's frame, grown by the clearance and the carve's reach
            const int o = hits.data[hI].other;
            cc.box[hI] = box_grow(box_transform(nodeBox[o], rthen(xf[o], rinverse(xf[v]))), clearance + 4.0f * g->h);
        }
        fb_parallel_for(n, carve_task, &cc);
        free(cc.xf);
        free(cc.box);
        rep->carvedPairs++;
    }
    fb_free_array(&hits);
    for (int b = 0; b < NB; b++) fb_free_array(&s[b]);
    free(s);
}

// ---- 4. ledges

// the body (index) other than C that holds hinge h (index) shut: C, opened a little, runs into it, while it hasn't
// moved yet; -1 if none
static int holder_of(const Design* d, int h, int C, const Pts* sc, const int* order, int slot) {
    Rigid closed[FB_MAX_BODIES], ajar[FB_MAX_BODIES];
    design_transforms(d, 0.0f, closed);
    Hinge hs[FB_MAX_BODIES];
    memcpy(hs, d->hinges, sizeof(Hinge) * (size_t)d->nhinges);
    // (every hinge shut but h, a few degrees open: fully "open" with those angles)
    for (int k = 0; k < d->nhinges; k++) hs[k].angle = 0.0f;
    hs[h].angle = (d->hinges[h].angle >= 0 ? 1.0f : -1.0f) * 0.06f;
    design_transforms_h(d, d->planes, hs, 1.0f, ajar);
    int best = -1, bestCount = 0;
    for (int H = 0; H < d->nbodies; H++) {
        if (H == C || design_below(d, h, H)) continue;
        bool frees = false;
        for (int s2 = 0; s2 < slot && !frees; s2++) frees = design_below(d, order[s2], H) != design_below(d, order[s2], C);
        if (!frees) continue;
        // more of C in H ajar than closed: it pushes into H as it starts to open
        int a = count_inside(d, sc, C, H, ajar), b = count_inside(d, sc, C, H, closed);
        if (a - b > bestCount) { bestCount = a - b; best = H; }
    }
    return bestCount >= 2 ? best : -1;
}

static int freeing_hinge(const Design* d, int H, int C, const int* order, int slot) {
    for (int s2 = 0; s2 < slot; s2++)
        if (design_below(d, order[s2], H) != design_below(d, order[s2], C)) return d->hinges[order[s2]].id;
    return -1;
}

static void add_ledges(Design* d, const Params* p, PostReport* rep) {
    const float u = 1.0f / fb_max(d->mmPerUnit, 1e-6f);
    const float L = p->ledgeMm * u, c = p->clearanceMm * u, r = d->R;
    const V3 centre = v3(0, 0, 0);
    int order[FB_MAX_BODIES];
    design_order(d, order);
    Rigid closed[FB_MAX_BODIES];
    design_transforms(d, 0.0f, closed);
    for (int slot = 1; slot < d->nhinges; slot++) {
        const int h = order[slot];
        const int C = design_body(d, d->hinges[h].child);
        if (C < 0) continue;
        Pts sc = body_nodes(d, C, 2);
        int H = holder_of(d, h, C, &sc, order, slot);
        if (H >= 0) {
            d->hinges[h].holder = freeing_hinge(d, H, C, order, slot);
            rep->alreadyHeld++;
            fb_free_array(&sc);
            continue;
        }
        // candidates: C's points near the closed sphere's surface next to a body that could hold it
        bool placed = false;
        for (int H2 = 0; H2 < d->nbodies && !placed; H2++) {
            if (H2 == C || design_below(d, h, H2)) continue;
            bool frees = false;
            for (int s2 = 0; s2 < slot && !frees; s2++) frees = design_below(d, order[s2], H2) != design_below(d, order[s2], C);
            if (!frees) continue;
            int tries = 0;
            for (int k = 0; k < sc.count && tries < 24 && !placed; k += 1 + sc.count / 400) {
                V3 x = rapply(closed[C], sc.data[k]);
                V3 rel = v3sub(x, centre);
                if (v3len(rel) < 0.8f * r) continue;
                V3 qH = rinv_apply(closed[H2], x);
                float fH = body_field(d, H2, qH, NULL);
                if (fH > 1.5f * L || fH < 0) continue;
                tries++;
                V3 nr = v3norm(rel);
                V3 xs = v3add(centre, v3scale(nr, fb_min(v3len(rel), 0.98f * r)));
                // across the seam, from H toward C
                V3 gH = rrot(closed[H2], cell_gradient(d, H2, rinv_apply(closed[H2], xs)));
                V3 t2 = v3sub(gH, v3scale(nr, v3dot(gH, nr)));
                if (v3len(t2) < 1e-4f) continue;
                t2 = v3norm(t2);
                V3 t1 = v3cross(nr, t2);
                const float T = 0.5f * L;
                V3 ledgeC = v3sub(xs, v3scale(nr, 0.5f * T + 0.3f * c));
                int e1 = add_world_box(d, H2, closed, ledgeC, t1, t2, nr, v3(L, L, 0.5f * T), 1, 0, 1);
                int e2 = add_world_box(d, C, closed, ledgeC, t1, t2, nr, v3(L + c, L + c, 0.5f * T + c), -1, 0, 1);
                if (e1 < 0 || e2 < 0) break;
                // it holds C (C ajar runs into it) and lets H go (H's own turn doesn't run into C)
                Pts scNow = body_nodes(d, C, 2);
                bool holds = holder_of(d, h, C, &scNow, order, slot) == H2;
                fb_free_array(&scNow);
                bool freed = true;
                if (holds) {
                    int g = design_hinge(d, freeing_hinge(d, H2, C, order, slot));
                    Pts sh = body_nodes(d, H2, 2);
                    Rigid xf[FB_MAX_BODIES];
                    int gslot = 0;
                    for (int s2 = 0; s2 < d->nhinges; s2++)
                        if (order[s2] == g) gslot = s2;
                    for (int q = 1; q <= 4 && freed; q++) {
                        design_transforms(d, (gslot + q / 5.0f) / d->nhinges, xf);
                        freed = count_inside(d, &sh, H2, C, xf) <= 2;
                    }
                    fb_free_array(&sh);
                }
                if (holds && freed) {
                    d->hinges[h].holder = freeing_hinge(d, H2, C, order, slot);
                    placed = true;
                } else {
                    // (not this one: the two extras go, the last two added)
                    d->nextras -= 2;
                }
            }
        }
        if (placed) rep->ledges++;
        else rep->ledgesFailed++;
        fb_free_array(&sc);
    }
}

// ---- 5. the latch

static bool add_latch(Design* d, const Params* p) {
    if (d->nhinges == 0) return false;
    int order[FB_MAX_BODIES];
    design_order(d, order);
    const Hinge* h0 = &d->hinges[order[0]];
    const int P = design_body(d, h0->parent), C = design_body(d, h0->child);
    if (P < 0 || C < 0) return false;
    const float u = 1.0f / fb_max(d->mmPerUnit, 1e-6f), Lm = p->ledgeMm * u, c = p->clearanceMm * u, r = d->R;
    Rigid closed[FB_MAX_BODIES];
    design_transforms(d, 0.0f, closed);
    // the axis in the closed pose (it's the parent's: the hinge turns the child about it)
    V3 axP = rapply(closed[P], h0->pivot), axD = rrot(closed[P], v3norm(h0->axis));
    Pts sc = body_nodes(d, C, 1);
    float bestDist = -1;
    V3 best = v3(0, 0, 0);
    for (int k = 0; k < sc.count; k++) {
        V3 x = rapply(closed[C], sc.data[k]);
        if (v3dist(x, v3(0, 0, 0)) < 0.85f * r) continue;
        float fP = body_field(d, P, rinv_apply(closed[P], x), NULL);
        if (fP > 1.5f * Lm || fP < 0) continue;
        V3 rel = v3sub(x, axP);
        float dist = v3len(v3sub(rel, v3scale(axD, v3dot(rel, axD))));
        if (dist > bestDist) { bestDist = dist; best = x; }
    }
    if (getenv("FOLDBALL_DEBUG")) {
        int a = 0, b = 0;
        for (int k = 0; k < sc.count; k++) {
            if (v3len(sc.data[k]) < 0.85f * r) continue;
            a++;
            float fP = body_field(d, P, sc.data[k], NULL);
            if (fP <= 1.5f * Lm && fP >= 0) b++;
        }
        fprintf(stderr, "latch: %d points of C, %d near the surface, %d of them next to P (Lm %.3f)\n", sc.count, a, b, Lm);
    }
    fb_free_array(&sc);
    if (bestDist < 0) return false;
    V3 nr = v3norm(v3sub(best, v3(0, 0, 0)));
    V3 xs = v3add(v3(0, 0, 0), v3scale(nr, 0.98f * r));
    // across the seam, from C toward P
    V3 gC = rrot(closed[C], cell_gradient(d, C, rinv_apply(closed[C], xs)));
    V3 t2 = v3sub(gC, v3scale(nr, v3dot(gC, nr)));
    if (v3len(t2) < 1e-4f) return false;
    t2 = v3norm(t2);
    V3 t1 = v3cross(nr, t2);
    // sizes: tab (in C, its top flush with the sphere), lip (P's skin over the hook), hook, room to press
    const float wt = 2.0f * Lm, lt = 3.0f * Lm, tt = 0.6f * Lm, lip = 0.6f * Lm, hk = 0.7f * Lm, gap = 0.9f * Lm, slit = 1.5f * c;
    // under the tab and beside it: C cut free on three sides (attached at its far end)
    V3 under = v3sub(v3sub(xs, v3scale(t2, 0.45f * lt)), v3scale(nr, tt + 0.5f * gap));
    add_world_box(d, C, closed, under, t1, t2, nr, v3(0.5f * wt + slit, 0.45f * lt, 0.5f * gap), -1, 0, 2);
    for (int sgn = -1; sgn <= 1; sgn += 2) {
        V3 side = v3add(v3sub(v3sub(xs, v3scale(t2, 0.45f * lt)), v3scale(nr, 0.5f * (tt + gap))), v3scale(t1, sgn * (0.5f * wt + 0.5f * slit)));
        add_world_box(d, C, closed, side, t1, t2, nr, v3(0.5f * slit, 0.45f * lt, 0.5f * (tt + gap) + 0.02f), -1, 0, 2);
    }
    // the hook: from the tab's free end, under P's lip
    V3 hookC = v3sub(v3add(xs, v3scale(t2, 0.5f * hk)), v3scale(nr, lip + c + 0.5f * hk));
    add_world_box(d, C, closed, hookC, t1, t2, nr, v3(0.4f * wt, 0.5f * hk + 0.5f * c, 0.5f * hk), 1, 0, 2);
    // its pocket in P (with room to move in when pressed)
    V3 pocket = v3sub(v3add(xs, v3scale(t2, 0.5f * hk)), v3scale(nr, lip + 0.5f * (hk + gap) + c));
    add_world_box(d, P, closed, pocket, t1, t2, nr, v3(0.4f * wt + c, 0.5f * hk + 2.0f * c, 0.5f * (hk + gap) + c), -1, 0, 2);
    return true;
}

bool post_process(Design* d, const Params* p, PostReport* rep) {
    memset(rep, 0, sizeof(*rep));
    lattice_set(d, p);
    rep->overlapBefore = post_overlap(d, 6);
    if (p->postFragments) rep->fragmentsRemoved += remove_fragments(d);
    // every cut opened by the clearance, exactly (half on each side): the bodies print apart, with clean seams
    d->gap = 0.5f * p->clearanceMm / fb_max(d->mmPerUnit, 1e-6f);
    // the input's features, back on the parts (before the hinges, carving and the rest, which then work around them)
    if (p->postFeatures) rep->featureParts = restore_features(d, p, rep->featureText, sizeof(rep->featureText));
    if (p->postHinges) build_hinges(d, p, rep);
    if (p->postCarve) carve(d, p, rep);
    // (measured before the ledges and the latch, which are meant to be in the way until they're let go)
    rep->overlapAfter = post_overlap(d, 6);
    if (p->postLedges) add_ledges(d, p, rep);
    if (p->postLatch) rep->latched = add_latch(d, p);
    if (p->postFragments) rep->fragmentsRemoved += remove_fragments(d);
    d->built = true;
    float vol = total_volume(d);
    snprintf(rep->text, sizeof(rep->text),
             "Post processing: %d fragment%s removed; %d hinge%s built; carved %d bod%s where they still collided; %d ledge%s added "
             "(%d hinge%s already held, %d without a place for one); %s. Worst overlap along the motion %.2f%% -> %.2f%% of the volume after carving "
             "(%.1f mm^3 total).",
             rep->fragmentsRemoved, rep->fragmentsRemoved == 1 ? "" : "s", rep->hingesBuilt, rep->hingesBuilt == 1 ? "" : "s", rep->carvedPairs,
             rep->carvedPairs == 1 ? "y" : "ies", rep->ledges, rep->ledges == 1 ? "" : "s", rep->alreadyHeld, rep->alreadyHeld == 1 ? " was" : "s were",
             rep->ledgesFailed, rep->latched ? "a latch on the root hinge" : "no latch (no place on the root hinge's seam)",
             100.0f * rep->overlapBefore, 100.0f * rep->overlapAfter, vol * d->mmPerUnit * d->mmPerUnit * d->mmPerUnit);
    if (p->postFeatures) {
        size_t n = strlen(rep->text);
        snprintf(rep->text + n, sizeof(rep->text) - n, " Features: %s.", rep->featureText);
    }
    return true;
}
