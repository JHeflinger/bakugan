#include "design.h"
#include "util.h"
#include <stdlib.h>
#include <string.h>

void design_init(Design* d, const TriMesh* mesh, float sphereMm, float insetMm) {
    memset(d, 0, sizeof(*d));
    d->nextId = 1;
    d->R = 1.0f;
    d->mmPerUnit = fb_max(sphereMm, 1e-3f) / 2.0f;
    d->inset = fb_clamp(insetMm / d->mmPerUnit, 0.005f, 0.3f);
    d->target0 = mesh_copy(mesh);
    d->target = mesh_copy(mesh);
    Body* b = &d->bodies[d->nbodies++];
    b->id = design_new_id(d);
    b->ncells = 1;               // one cell with no half spaces: the whole sphere
    d->root = b->id;
    // the open pose placed in the mesh's space: the sphere's volume the mesh's, at its middle
    double vol = fabs(mesh_volume(mesh));
    Box3 bb = mesh_bounds(mesh);
    V3 e = v3sub(bb.max, bb.min);
    if (vol < 1e-9) vol = 0.1 * e.x * e.y * e.z + 1e-6;
    d->scale = (float)cbrt(vol / (4.0 / 3.0 * FB_PI));
    d->offset = v3scale(v3add(bb.min, bb.max), 0.5f);
}

void design_free(Design* d) {
    for (int b = 0; b < d->nbodies; b++) {
        grid_free(&d->bodies[b].carve);
        boxgrid_free(&d->bodies[b].detail);
    }
    mesh_free(&d->target);
    mesh_free(&d->target0);
    memset(d, 0, sizeof(*d));
}

void design_copy(Design* dst, const Design* src) {
    if (dst == src) return;
    if (dst->target.pos || dst->nbodies) design_free(dst);
    memcpy(dst, src, sizeof(Design));
    for (int b = 0; b < src->nbodies; b++)
        if (src->bodies[b].carve.v) dst->bodies[b].carve = grid_copy(&src->bodies[b].carve);
    for (int b = 0; b < src->nbodies; b++)
        if (src->bodies[b].detail.v) dst->bodies[b].detail = boxgrid_copy(&src->bodies[b].detail);
    dst->target = mesh_copy(&src->target);
    dst->target0 = mesh_copy(&src->target0);
}

Design* design_clone(const Design* src) {
    Design* d = FB_NEW(Design, 1);
    design_copy(d, src);
    return d;
}

int design_body(const Design* d, int id) {
    for (int b = 0; b < d->nbodies; b++)
        if (d->bodies[b].id == id) return b;
    return -1;
}

int design_hinge(const Design* d, int id) {
    for (int h = 0; h < d->nhinges; h++)
        if (d->hinges[h].id == id) return h;
    return -1;
}

int design_parent_hinge(const Design* d, int bodyId) {
    for (int h = 0; h < d->nhinges; h++)
        if (d->hinges[h].child == bodyId) return h;
    return -1;
}

int design_new_id(Design* d) { return d->nextId++; }

// ---- fields

float hollow_field(const Hollow* h, V3 q) {
    V3 ab = v3sub(h->b, h->a);
    float l2 = v3dot(ab, ab);
    float t = l2 > 1e-12f ? fb_clamp(v3dot(v3sub(q, h->a), ab) / l2, 0.0f, 1.0f) : 0.0f;
    return v3dist(q, v3mad(h->a, ab, t)) - h->r;
}

float body_cell(const Design* d, int b, V3 q, Active* act) {
    const Body* body = &d->bodies[b];
    float best = FLT_MAX;
    int bp = -1, bs = 0;
    for (int c = 0; c < body->ncells; c++) {
        const Cell* cell = &body->cells[c];
        float v = -FLT_MAX;
        int vp = -1, vs = 0;
        for (int k = 0; k < cell->n; k++) {
            const Plane* p = &d->planes[cell->plane[k]];
            float s = cell->sign[k] * (v3dot(p->n, q) - p->d);
            if (s > v) { v = s; vp = cell->plane[k]; vs = cell->sign[k]; }
        }
        if (v < best) { best = v; bp = vp; bs = vs; }
    }
    if (act) { act->index = bp; act->sign = bs; }
    return best;
}

static float smax(float a, float b, float k) {
    if (k <= 0) return fb_max(a, b);
    float h = fb_clamp(0.5f - 0.5f * (b - a) / k, 0.0f, 1.0f);
    return b + (a - b) * h + k * h * (1.0f - h);
}

float extra_field(const Extra* e, V3 q) {
    V3 p = rinv_apply(e->pose, q);
    if (e->shape == EX_BOX) {
        V3 s = v3sub(e->size, v3(e->round, e->round, e->round));
        V3 a = v3(fabsf(p.x) - s.x, fabsf(p.y) - s.y, fabsf(p.z) - s.z);
        V3 m = v3max(a, v3(0, 0, 0));
        return v3len(m) + fb_min(fb_max(a.x, fb_max(a.y, a.z)), 0.0f) - e->round;
    }
    float r = sqrtf(p.x * p.x + p.y * p.y) - e->size.x, z = fabsf(p.z) - e->size.z;
    float ox = fb_max(r, 0.0f), oz = fb_max(z, 0.0f);
    return sqrtf(ox * ox + oz * oz) + fb_min(fb_max(r, z), 0.0f);
}

float body_field(const Design* d, int b, V3 q, Active* act) {
    const Body* body = &d->bodies[b];
    // the sphere, its cell, the hollows (which cut every body)
    float f = v3len(q) - d->R;
    int kind = 0, index = -1, sign = 0;
    Active ca;
    float c = body_cell(d, b, q, &ca) + d->gap;
    if (c > f) { f = c; kind = 1; index = ca.index; sign = ca.sign; }
    for (int k = 0; k < d->nhollows; k++) {
        float v = -hollow_field(&d->hollows[k], q);
        if (v > f) { f = v; kind = 2; index = k; }
    }
    // the input's features restored (post: features.c): an offset, never past the sphere
    if (body->detail.v) {
        float dv = boxgrid_sample(&body->detail, q);
        if (dv != 0.0f) {
            f = fb_max(f + dv, v3len(q) - d->R);
            kind = 3;
        }
    }
    if (d->nextras) {
        for (int k = 0; k < d->nextras; k++) {
            const Extra* e = &d->extras[k];
            if (e->body != body->id || e->op != 1) continue;
            float v = fb_max(extra_field(e, q), v3len(q) - d->R);     // (never past the sphere: it must roll)
            if (v < f) { f = v; kind = 3; }
        }
        for (int k = 0; k < d->nextras; k++) {
            const Extra* e = &d->extras[k];
            if (e->body != body->id || e->op > 0) continue;
            float v = -extra_field(e, q);
            if (v > f) { f = v; kind = 3; }
        }
    }
    if (body->carve.v) {
        float v = -grid_sample(&body->carve, q);
        float g = smax(f, v, 0.5f * body->carve.h);
        if (g > f + 1e-7f) kind = 3;
        f = g;
    }
    // (added after the cuts and the carving: a hinge pin through the room cleared for the other body's knuckle)
    for (int k = 0; k < d->nextras; k++) {
        const Extra* e = &d->extras[k];
        if (e->body != body->id || e->op != 2) continue;
        float v = fb_max(extra_field(e, q), v3len(q) - d->R);
        if (v < f) { f = v; kind = 3; }
    }
    if (act) {
        act->kind = kind;
        act->index = index;
        act->sign = sign;
    }
    return f;
}

V3 body_gradient(const Design* d, int b, V3 q) {
    const float e = 1e-3f * d->R;
    V3 g = v3(body_field(d, b, v3(q.x + e, q.y, q.z), NULL) - body_field(d, b, v3(q.x - e, q.y, q.z), NULL),
              body_field(d, b, v3(q.x, q.y + e, q.z), NULL) - body_field(d, b, v3(q.x, q.y - e, q.z), NULL),
              body_field(d, b, v3(q.x, q.y, q.z + e), NULL) - body_field(d, b, v3(q.x, q.y, q.z - e), NULL));
    return v3scale(g, 0.5f / e);
}

float design_field_open(const Design* d, V3 q) {
    Rigid xf[FB_MAX_BODIES];
    design_transforms(d, 1.0f, xf);
    float f = FLT_MAX;
    for (int b = 0; b < d->nbodies; b++) f = fb_min(f, body_field(d, b, rinv_apply(xf[b], q), NULL));
    return f;
}

// ---- hinges and motion

void hinge_frame(const Design* d, const Plane* planes, const Hinge* h, V3* pivot, V3* axis) {
    const Plane* p = &planes[h->plane];
    V3 n = v3norm(p->n);
    V3 e1 = v3sub(h->ref, v3scale(n, v3dot(n, h->ref)));
    e1 = v3len(e1) > 1e-6f ? v3norm(e1) : v3perp(n);
    V3 e2 = v3cross(n, e1);
    float ri = d->R - d->inset;
    float rho = sqrtf(fb_max(ri * ri - p->d * p->d, 1e-6f));
    V3 c = v3scale(n, p->d);
    float cs = cosf(h->phi), sn = sinf(h->phi);
    *pivot = v3add(c, v3scale(v3add(v3scale(e1, cs), v3scale(e2, sn)), rho));
    *axis = v3add(v3scale(e1, -sn), v3scale(e2, cs));
}

void design_hinge_frames(Design* d) {
    for (int h = 0; h < d->nhinges; h++) hinge_frame(d, d->planes, &d->hinges[h], &d->hinges[h].pivot, &d->hinges[h].axis);
}

float hinge_angle_at(const Design* d, const Hinge* h, float s) {
    const int m = d->nhinges > 0 ? d->nhinges : 1;
    return h->angle * fb_clamp(s * m - h->order, 0.0f, 1.0f);
}

void design_transforms(const Design* d, float s, Rigid* out) { design_transforms_h(d, d->planes, d->hinges, s, out); }

void design_transforms_h(const Design* d, const Plane* planes, const Hinge* hinges, float s, Rigid* out) {
    bool done[FB_MAX_BODIES] = { 0 };
    for (int b = 0; b < d->nbodies; b++) out[b] = rident();
    int root = design_body(d, d->root);
    if (root < 0) root = 0;
    done[root] = true;
    for (int pass = 0; pass < d->nbodies; pass++) {
        bool changed = false;
        for (int h = 0; h < d->nhinges; h++) {
            const Hinge* hg = &hinges[h];
            int p = design_body(d, hg->parent), c = design_body(d, hg->child);
            if (p < 0 || c < 0 || !done[p] || done[c]) continue;
            V3 pivot, axis;
            hinge_frame(d, planes, hg, &pivot, &axis);
            out[c] = rthen(rabout(pivot, axis, hinge_angle_at(d, hg, s)), out[p]);
            done[c] = true;
            changed = true;
        }
        if (!changed) break;
    }
}

static const Design* gOrderDesign;
static int cmp_order(const void* a, const void* b) {
    const Hinge* x = &gOrderDesign->hinges[*(const int*)a];
    const Hinge* y = &gOrderDesign->hinges[*(const int*)b];
    if (x->order != y->order) return x->order < y->order ? -1 : 1;
    return x->id < y->id ? -1 : (x->id > y->id);
}

void design_order(const Design* d, int* order) {
    for (int h = 0; h < d->nhinges; h++) order[h] = h;
    gOrderDesign = d;
    qsort(order, (size_t)d->nhinges, sizeof(int), cmp_order);
}

void design_normalize_orders(Design* d) {
    int order[FB_MAX_BODIES];
    design_order(d, order);
    for (int k = 0; k < d->nhinges; k++) d->hinges[order[k]].order = k;
}

bool design_below(const Design* d, int hingeIndex, int bodyIndex) {
    int id = d->bodies[bodyIndex].id;
    const int top = d->hinges[hingeIndex].child;
    for (int guard = 0; guard <= d->nbodies; guard++) {
        if (id == top) return true;
        int h = design_parent_hinge(d, id);
        if (h < 0) return false;
        id = d->hinges[h].parent;
    }
    return false;
}

void design_tidy(Design* d) {
    int used[FB_MAX_PLANES] = { 0 };
    for (int b = 0; b < d->nbodies; b++) {
        Body* body = &d->bodies[b];
        for (int c = 0; c < body->ncells; c++)
            for (int k = 0; k < body->cells[c].n; k++) used[body->cells[c].plane[k]] = 1;
    }
    for (int h = 0; h < d->nhinges; h++) used[d->hinges[h].plane] = 1;
    int remap[FB_MAX_PLANES], n = 0;
    for (int p = 0; p < d->nplanes; p++) {
        remap[p] = used[p] ? n : -1;
        if (used[p]) d->planes[n++] = d->planes[p];
    }
    d->nplanes = n;
    for (int b = 0; b < d->nbodies; b++) {
        Body* body = &d->bodies[b];
        for (int c = 0; c < body->ncells; c++)
            for (int k = 0; k < body->cells[c].n; k++) body->cells[c].plane[k] = (short)remap[body->cells[c].plane[k]];
    }
    for (int h = 0; h < d->nhinges; h++) d->hinges[h].plane = remap[d->hinges[h].plane];
    design_normalize_orders(d);
    design_hinge_frames(d);
}

// ---- files: a text header, then any carving grids as raw little endian floats

static void write_mesh(FILE* f, const char* name, const TriMesh* m) {
    fprintf(f, "%s %d %d\n", name, m->nv, m->nt);
    for (int i = 0; i < m->nv; i++) fprintf(f, "%.6g %.6g %.6g\n", m->pos[i].x, m->pos[i].y, m->pos[i].z);
    for (int t = 0; t < m->nt; t++) fprintf(f, "%d %d %d\n", m->idx[3 * t], m->idx[3 * t + 1], m->idx[3 * t + 2]);
}

bool design_save(const Design* d, const char* path, char* err, int errSize) {
    FILE* f = fopen(path, "wb");
    if (!f) { snprintf(err, (size_t)errSize, "can't write %s", path); return false; }
    fprintf(f, "FOLDBALL 2\n");
    fprintf(f, "sphere %.9g inset %.9g scale %.9g offset %.9g %.9g %.9g\n", d->R, d->inset, d->scale, d->offset.x, d->offset.y, d->offset.z);
    fprintf(f, "gap %.9g\n", d->gap);
    fprintf(f, "mm %.9g nextid %d built %d\n", d->mmPerUnit, d->nextId, d->built ? 1 : 0);
    fprintf(f, "planes %d\n", d->nplanes);
    for (int p = 0; p < d->nplanes; p++)
        fprintf(f, "%d %.9g %.9g %.9g %.9g\n", d->planes[p].id, d->planes[p].n.x, d->planes[p].n.y, d->planes[p].n.z, d->planes[p].d);
    fprintf(f, "bodies %d %d\n", d->nbodies, d->root);
    for (int b = 0; b < d->nbodies; b++) {
        const Body* body = &d->bodies[b];
        fprintf(f, "body %d %d %d %d %.9g\n", body->id, body->ncells, body->carve.v ? body->carve.n : 0, 0, body->carve.v ? body->carve.B : 0.0f);
        for (int c = 0; c < body->ncells; c++) {
            fprintf(f, "%d", body->cells[c].n);
            for (int k = 0; k < body->cells[c].n; k++) fprintf(f, " %d %d", body->cells[c].plane[k], body->cells[c].sign[k]);
            fprintf(f, "\n");
        }
    }
    fprintf(f, "hinges %d\n", d->nhinges);
    for (int h = 0; h < d->nhinges; h++) {
        const Hinge* g = &d->hinges[h];
        fprintf(f, "%d %d %d %d %.9g %.9g %.9g %.9g %.9g %d %d\n", g->id, g->parent, g->child, g->plane, g->ref.x, g->ref.y, g->ref.z, g->phi,
                g->angle, g->order, g->holder);
    }
    fprintf(f, "hollows %d\n", d->nhollows);
    for (int k = 0; k < d->nhollows; k++) {
        const Hollow* o = &d->hollows[k];
        fprintf(f, "%d %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n", o->id, o->a.x, o->a.y, o->a.z, o->b.x, o->b.y, o->b.z, o->r);
    }
    fprintf(f, "extras %d\n", d->nextras);
    for (int k = 0; k < d->nextras; k++) {
        const Extra* e = &d->extras[k];
        fprintf(f, "%d %d %d %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %d\n", e->body, e->shape, e->op, e->pose.q.x, e->pose.q.y,
                e->pose.q.z, e->pose.q.w, e->pose.t.x, e->pose.t.y, e->pose.t.z, e->size.x, e->size.y, e->size.z, e->round, e->tag);
    }
    int ndetails = 0;
    for (int b = 0; b < d->nbodies; b++) ndetails += d->bodies[b].detail.v != NULL;
    fprintf(f, "details %d\n", ndetails);
    for (int b = 0; b < d->nbodies; b++) {
        const BoxGrid* g = &d->bodies[b].detail;
        if (g->v) fprintf(f, "%d %d %d %d %.9g %.9g %.9g %.9g\n", b, g->nx, g->ny, g->nz, g->o.x, g->o.y, g->o.z, g->h);
    }
    write_mesh(f, "target", &d->target);
    write_mesh(f, "input", &d->target0);
    fprintf(f, "data\n");
    bool ok = true;
    for (int b = 0; b < d->nbodies && ok; b++)
        if (d->bodies[b].carve.v) {
            size_t N = (size_t)d->bodies[b].carve.n * d->bodies[b].carve.n * d->bodies[b].carve.n;
            ok = fwrite(d->bodies[b].carve.v, sizeof(float), N, f) == N;
        }
    for (int b = 0; b < d->nbodies && ok; b++)
        if (d->bodies[b].detail.v) {
            size_t N = (size_t)d->bodies[b].detail.nx * d->bodies[b].detail.ny * d->bodies[b].detail.nz;
            ok = fwrite(d->bodies[b].detail.v, sizeof(float), N, f) == N;
        }
    ok = fclose(f) == 0 && ok;
    if (!ok) snprintf(err, (size_t)errSize, "writing %s failed", path);
    return ok;
}

static bool read_mesh(FILE* f, const char* name, TriMesh* m) {
    char word[64];
    if (fscanf(f, "%63s %d %d", word, &m->nv, &m->nt) != 3 || strcmp(word, name) != 0 || m->nv < 0 || m->nt < 0) return false;
    m->pos = FB_NEW(V3, m->nv ? m->nv : 1);
    m->idx = FB_NEW(int, m->nt ? 3 * m->nt : 1);
    for (int i = 0; i < m->nv; i++)
        if (fscanf(f, "%f %f %f", &m->pos[i].x, &m->pos[i].y, &m->pos[i].z) != 3) return false;
    for (int t = 0; t < 3 * m->nt; t++)
        if (fscanf(f, "%d", &m->idx[t]) != 1 || m->idx[t] < 0 || m->idx[t] >= m->nv) return false;
    return true;
}

bool design_load(Design* d, const char* path, char* err, int errSize) {
    FILE* f = fopen(path, "rb");
    if (!f) { snprintf(err, (size_t)errSize, "can't read %s", path); return false; }
    Design* n = FB_NEW(Design, 1);
    char word[64];
    int version = 0, built = 0;
    int carveN[FB_MAX_BODIES] = { 0 };
    float carveB[FB_MAX_BODIES] = { 0 };
    bool ok = fscanf(f, "%63s %d", word, &version) == 2 && strcmp(word, "FOLDBALL") == 0;
    if (ok && version != 2) {
        fclose(f);
        free(n);
        snprintf(err, (size_t)errSize, "%s is from an older foldball (the grid design): it can't be carried on", path);
        return false;
    }
    ok = ok && fscanf(f, " sphere %f inset %f scale %f offset %f %f %f", &n->R, &n->inset, &n->scale, &n->offset.x, &n->offset.y, &n->offset.z) == 6;
    if (ok && fscanf(f, " gap %f", &n->gap) != 1) n->gap = 0.0f;
    ok = ok && fscanf(f, " mm %f nextid %d built %d", &n->mmPerUnit, &n->nextId, &built) == 3;
    ok = ok && fscanf(f, " planes %d", &n->nplanes) == 1 && n->nplanes >= 0 && n->nplanes <= FB_MAX_PLANES;
    for (int p = 0; ok && p < n->nplanes; p++) {
        Plane* pl = &n->planes[p];
        ok = fscanf(f, "%d %f %f %f %f", &pl->id, &pl->n.x, &pl->n.y, &pl->n.z, &pl->d) == 5;
    }
    ok = ok && fscanf(f, " bodies %d %d", &n->nbodies, &n->root) == 2 && n->nbodies >= 1 && n->nbodies <= FB_MAX_BODIES;
    for (int b = 0; ok && b < n->nbodies; b++) {
        Body* body = &n->bodies[b];
        int unused;
        ok = fscanf(f, " body %d %d %d %d %f", &body->id, &body->ncells, &carveN[b], &unused, &carveB[b]) == 5 && body->ncells >= 0 &&
             body->ncells <= FB_MAX_CELLS && carveN[b] >= 0 && carveN[b] <= 512;
        for (int c = 0; ok && c < body->ncells; c++) {
            Cell* cell = &body->cells[c];
            ok = fscanf(f, "%d", &cell->n) == 1 && cell->n >= 0 && cell->n <= FB_MAX_HALF;
            for (int k = 0; ok && k < cell->n; k++) {
                int p, s;
                ok = fscanf(f, "%d %d", &p, &s) == 2 && p >= 0 && p < n->nplanes && (s == 1 || s == -1);
                cell->plane[k] = (short)p;
                cell->sign[k] = (signed char)s;
            }
        }
    }
    ok = ok && fscanf(f, " hinges %d", &n->nhinges) == 1 && n->nhinges >= 0 && n->nhinges < FB_MAX_BODIES;
    for (int h = 0; ok && h < n->nhinges; h++) {
        Hinge* g = &n->hinges[h];
        ok = fscanf(f, "%d %d %d %d %f %f %f %f %f %d %d", &g->id, &g->parent, &g->child, &g->plane, &g->ref.x, &g->ref.y, &g->ref.z, &g->phi,
                    &g->angle, &g->order, &g->holder) == 11 && g->plane >= 0 && g->plane < n->nplanes;
    }
    ok = ok && fscanf(f, " hollows %d", &n->nhollows) == 1 && n->nhollows >= 0 && n->nhollows <= FB_MAX_HOLLOWS;
    for (int k = 0; ok && k < n->nhollows; k++) {
        Hollow* o = &n->hollows[k];
        ok = fscanf(f, "%d %f %f %f %f %f %f %f", &o->id, &o->a.x, &o->a.y, &o->a.z, &o->b.x, &o->b.y, &o->b.z, &o->r) == 8;
    }
    ok = ok && fscanf(f, " extras %d", &n->nextras) == 1 && n->nextras >= 0 && n->nextras <= FB_MAX_EXTRAS;
    for (int k = 0; ok && k < n->nextras; k++) {
        Extra* e = &n->extras[k];
        ok = fscanf(f, "%d %d %d %f %f %f %f %f %f %f %f %f %f %f %d", &e->body, &e->shape, &e->op, &e->pose.q.x, &e->pose.q.y, &e->pose.q.z,
                    &e->pose.q.w, &e->pose.t.x, &e->pose.t.y, &e->pose.t.z, &e->size.x, &e->size.y, &e->size.z, &e->round, &e->tag) == 15;
    }
    // (feature restoration's grids: files from before it have none)
    int ndetails = 0;
    if (ok && fscanf(f, " details %d", &ndetails) == 1) {
        ok = ndetails >= 0 && ndetails <= n->nbodies;
        for (int k = 0; ok && k < ndetails; k++) {
            int b;
            BoxGrid g;
            ok = fscanf(f, "%d %d %d %d %f %f %f %f", &b, &g.nx, &g.ny, &g.nz, &g.o.x, &g.o.y, &g.o.z, &g.h) == 8 && b >= 0 && b < n->nbodies &&
                 g.nx >= 2 && g.ny >= 2 && g.nz >= 2 && g.nx <= 2048 && g.ny <= 2048 && g.nz <= 2048 && g.h > 0;
            if (ok) {
                g.v = FB_NEW(float, (size_t)g.nx * g.ny * g.nz);
                boxgrid_free(&n->bodies[b].detail);
                n->bodies[b].detail = g;
            }
        }
    }
    ok = ok && read_mesh(f, "target", &n->target) && read_mesh(f, "input", &n->target0);
    ok = ok && fscanf(f, " %63s", word) == 1 && strcmp(word, "data") == 0 && fgetc(f) == '\n';
    for (int b = 0; ok && b < n->nbodies; b++)
        if (carveN[b] > 0) {
            n->bodies[b].carve = grid_create(carveN[b], carveB[b]);
            size_t N = (size_t)carveN[b] * carveN[b] * carveN[b];
            ok = fread(n->bodies[b].carve.v, sizeof(float), N, f) == N;
        }
    for (int b = 0; ok && b < n->nbodies; b++)
        if (n->bodies[b].detail.v) {
            size_t N = (size_t)n->bodies[b].detail.nx * n->bodies[b].detail.ny * n->bodies[b].detail.nz;
            ok = fread(n->bodies[b].detail.v, sizeof(float), N, f) == N;
        }
    fclose(f);
    n->built = built != 0;
    if (!ok) {
        snprintf(err, (size_t)errSize, "%s is not a foldball design (or it is cut short)", path);
        design_free(n);
        free(n);
        return false;
    }
    design_hinge_frames(n);
    if (d->target.pos || d->nbodies) design_free(d);
    *d = *n;
    free(n);
    return true;
}
