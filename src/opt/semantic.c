#include "semantic.h"
#include "core/util.h"
#include <stdlib.h>
#include <string.h>

// ---- 3x3 matrices (row major)

typedef struct { float m[9]; } M3;
static M3 m3zero(void) { M3 r; memset(&r, 0, sizeof(r)); return r; }
static M3 m3ident(void) { M3 r = m3zero(); r.m[0] = r.m[4] = r.m[8] = 1; return r; }
static V3 m3mul_v(const M3* a, V3 v) {
    return v3(a->m[0] * v.x + a->m[1] * v.y + a->m[2] * v.z, a->m[3] * v.x + a->m[4] * v.y + a->m[5] * v.z, a->m[6] * v.x + a->m[7] * v.y + a->m[8] * v.z);
}
static float m3det(const M3* a) {
    const float* m = a->m;
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}
// the rotation nearest a (its orthogonal polar factor, Higham's iteration), a proper rotation (det +1)
static M3 m3rotation(M3 a) {
    float scale = 0;
    for (int k = 0; k < 9; k++) scale = fb_max(scale, fabsf(a.m[k]));
    if (scale < 1e-12f) return m3ident();
    for (int k = 0; k < 9; k++) a.m[k] /= scale;
    if (m3det(&a) < 0) {        // reflect it so the iteration lands on a rotation
        for (int k = 6; k < 9; k++) a.m[k] = -a.m[k];
    }
    for (int it = 0; it < 20; it++) {
        float det = m3det(&a);
        if (fabsf(det) < 1e-9f) {
            for (int k = 0; k < 9; k += 4) a.m[k] += 1e-3f;
            det = m3det(&a);
        }
        // inverse transpose
        const float* m = a.m;
        M3 it_;
        it_.m[0] = (m[4] * m[8] - m[5] * m[7]) / det;
        it_.m[1] = -(m[3] * m[8] - m[5] * m[6]) / det;
        it_.m[2] = (m[3] * m[7] - m[4] * m[6]) / det;
        it_.m[3] = -(m[1] * m[8] - m[2] * m[7]) / det;
        it_.m[4] = (m[0] * m[8] - m[2] * m[6]) / det;
        it_.m[5] = -(m[0] * m[7] - m[1] * m[6]) / det;
        it_.m[6] = (m[1] * m[5] - m[2] * m[4]) / det;
        it_.m[7] = -(m[0] * m[5] - m[2] * m[3]) / det;
        it_.m[8] = (m[0] * m[4] - m[1] * m[3]) / det;
        float change = 0;
        for (int k = 0; k < 9; k++) {
            float nv = 0.5f * (a.m[k] + it_.m[k]);
            change = fb_max(change, fabsf(nv - a.m[k]));
            a.m[k] = nv;
        }
        if (change < 1e-6f) break;
    }
    if (m3det(&a) < 0) return m3ident();
    return a;
}

// ---- the cage: the input simplified, with every input vertex's cluster

typedef struct {
    int nv;
    V3* rest;            // cage vertices as loaded
    int *nbStart, *nb;   // neighbours (CSR)
    int* clusterOf;      // per input vertex: its cage vertex
    int* attach;         // per input vertex: 3 cage vertices carrying it
    float* attachW;
} Cage;

typedef struct { long long key; int id; } Key;
static int cmp_key(const void* a, const void* b) {
    long long x = ((const Key*)a)->key, y = ((const Key*)b)->key;
    return x < y ? -1 : (x > y);
}

static Cage cage_make(const TriMesh* m, int target) {
    Cage c;
    memset(&c, 0, sizeof(c));
    Box3 b = mesh_bounds(m);
    double area = 0;
    for (int t = 0; t < m->nt; t++) {
        V3 p = m->pos[m->idx[3 * t]], q = m->pos[m->idx[3 * t + 1]], r = m->pos[m->idx[3 * t + 2]];
        area += 0.5 * v3len(v3cross(v3sub(q, p), v3sub(r, p)));
    }
    float cell = (float)sqrt(area / fb_maxi(target, 16)) * 0.9f;
    Key* keys = FB_NEW(Key, m->nv);
    c.clusterOf = FB_NEW(int, m->nv);
    for (int tries = 0; tries < 10; tries++) {
        for (int i = 0; i < m->nv; i++) {
            V3 q = v3sub(m->pos[i], b.min);
            long long x = (long long)floorf(q.x / cell), y = (long long)floorf(q.y / cell), z = (long long)floorf(q.z / cell);
            keys[i].key = (x * 2097152LL + y) * 2097152LL + z;
            keys[i].id = i;
        }
        qsort(keys, (size_t)m->nv, sizeof(Key), cmp_key);
        int n = 0;
        for (int s = 0; s < m->nv;) {
            int e = s;
            while (e < m->nv && keys[e].key == keys[s].key) c.clusterOf[keys[e++].id] = n;
            n++;
            s = e;
        }
        c.nv = n;
        if (n <= target * 1.3f || tries == 9) break;
        cell *= 1.2f;
    }
    free(keys);
    c.rest = FB_NEW(V3, c.nv);
    int* cnt = FB_NEW(int, c.nv);
    for (int i = 0; i < m->nv; i++) {
        c.rest[c.clusterOf[i]] = v3add(c.rest[c.clusterOf[i]], m->pos[i]);
        cnt[c.clusterOf[i]]++;
    }
    for (int k = 0; k < c.nv; k++) c.rest[k] = v3scale(c.rest[k], 1.0f / fb_maxi(cnt[k], 1));
    free(cnt);
    // edges from the triangles, between different clusters
    FB_ARRAY(long long) edges = { 0 };
    for (int t = 0; t < m->nt; t++)
        for (int e = 0; e < 3; e++) {
            int a = c.clusterOf[m->idx[3 * t + e]], bb = c.clusterOf[m->idx[3 * t + (e + 1) % 3]];
            if (a == bb) continue;
            fb_push(&edges, (long long)a * c.nv + bb);
            fb_push(&edges, (long long)bb * c.nv + a);
        }
    c.nbStart = FB_NEW(int, c.nv + 1);
    // (sort the pairs: counting sort by first vertex, duplicates removed)
    int* deg = FB_NEW(int, c.nv);
    for (int k = 0; k < edges.count; k++) deg[edges.data[k] / c.nv]++;
    for (int k = 0; k < c.nv; k++) c.nbStart[k + 1] = c.nbStart[k] + deg[k];
    int* fill = FB_NEW(int, c.nv);
    int* tmp = FB_NEW(int, edges.count > 0 ? edges.count : 1);
    for (int k = 0; k < edges.count; k++) {
        int a = (int)(edges.data[k] / c.nv), bb = (int)(edges.data[k] % c.nv);
        tmp[c.nbStart[a] + fill[a]++] = bb;
    }
    // remove duplicates per vertex
    int* start2 = FB_NEW(int, c.nv + 1);
    int w = 0;
    for (int a = 0; a < c.nv; a++) {
        start2[a] = w;
        for (int k = c.nbStart[a]; k < c.nbStart[a + 1]; k++) {
            bool dup = false;
            for (int q = start2[a]; q < w && !dup; q++) dup = tmp[q] == tmp[k];
            if (!dup) tmp[w++] = tmp[k];
        }
    }
    start2[c.nv] = w;
    free(c.nbStart);
    c.nbStart = start2;
    c.nb = tmp;
    free(deg);
    free(fill);
    fb_free_array(&edges);
    // every input vertex carried by its own cluster's vertex and the two nearest neighbours of it
    c.attach = FB_NEW(int, 3 * m->nv);
    c.attachW = FB_NEW(float, 3 * m->nv);
    for (int i = 0; i < m->nv; i++) {
        int own = c.clusterOf[i];
        int best[3] = { own, own, own };
        float bd[3] = { v3dist(m->pos[i], c.rest[own]), FLT_MAX, FLT_MAX };
        for (int k = c.nbStart[own]; k < c.nbStart[own + 1]; k++) {
            int q = c.nb[k];
            float dq = v3dist(m->pos[i], c.rest[q]);
            if (dq < bd[1]) { bd[2] = bd[1]; best[2] = best[1]; bd[1] = dq; best[1] = q; }
            else if (dq < bd[2]) { bd[2] = dq; best[2] = q; }
        }
        float ws = 0;
        for (int k = 0; k < 3; k++) {
            float wk = bd[k] < FLT_MAX ? 1.0f / (bd[k] + 1e-4f) : 0.0f;
            c.attach[3 * i + k] = best[k];
            c.attachW[3 * i + k] = wk;
            ws += wk;
        }
        for (int k = 0; k < 3; k++) c.attachW[3 * i + k] /= ws;
    }
    return c;
}

static void cage_free(Cage* c) {
    free(c->rest);
    free(c->nbStart);
    free(c->nb);
    free(c->clusterOf);
    free(c->attach);
    free(c->attachW);
}

// ---- ARAP with soft positional constraints (uniform weights: the cage's triangles are too uneven for cotangents)

static void arap_rotations(const Cage* c, const V3* x, M3* R) {
    for (int i = 0; i < c->nv; i++) {
        M3 S = m3zero();     // sum of e' e^T
        for (int k = c->nbStart[i]; k < c->nbStart[i + 1]; k++) {
            int j = c->nb[k];
            V3 e = v3sub(c->rest[i], c->rest[j]), f = v3sub(x[i], x[j]);
            float ev[3] = { e.x, e.y, e.z }, fv[3] = { f.x, f.y, f.z };
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++) S.m[3 * a + b] += fv[a] * ev[b];
        }
        R[i] = m3rotation(S);
    }
}

// (L + lambda I) x = rhs by conjugate gradients, from x as it is
static void arap_solve(const Cage* c, const float* lambda, const V3* rhs, V3* x) {
    const int n = c->nv;
    V3 *r = FB_NEW(V3, n), *pp = FB_NEW(V3, n), *Ap = FB_NEW(V3, n);
    #define APPLY(in, out) for (int i = 0; i < n; i++) { \
            V3 s = v3scale(in[i], (float)(c->nbStart[i + 1] - c->nbStart[i]) + lambda[i]); \
            for (int k = c->nbStart[i]; k < c->nbStart[i + 1]; k++) s = v3sub(s, in[c->nb[k]]); \
            out[i] = s; }
    APPLY(x, Ap);
    double rr[3] = { 0, 0, 0 };
    for (int i = 0; i < n; i++) {
        r[i] = v3sub(rhs[i], Ap[i]);
        pp[i] = r[i];
        rr[0] += r[i].x * r[i].x; rr[1] += r[i].y * r[i].y; rr[2] += r[i].z * r[i].z;
    }
    for (int it = 0; it < 200; it++) {
        APPLY(pp, Ap);
        double pAp[3] = { 0, 0, 0 };
        for (int i = 0; i < n; i++) { pAp[0] += pp[i].x * Ap[i].x; pAp[1] += pp[i].y * Ap[i].y; pAp[2] += pp[i].z * Ap[i].z; }
        float al[3];
        for (int a = 0; a < 3; a++) al[a] = pAp[a] > 1e-30 ? (float)(rr[a] / pAp[a]) : 0.0f;
        double nr[3] = { 0, 0, 0 };
        for (int i = 0; i < n; i++) {
            x[i] = v3add(x[i], v3(al[0] * pp[i].x, al[1] * pp[i].y, al[2] * pp[i].z));
            r[i] = v3sub(r[i], v3(al[0] * Ap[i].x, al[1] * Ap[i].y, al[2] * Ap[i].z));
            nr[0] += r[i].x * r[i].x; nr[1] += r[i].y * r[i].y; nr[2] += r[i].z * r[i].z;
        }
        if (nr[0] + nr[1] + nr[2] < 1e-14) break;
        float be[3];
        for (int a = 0; a < 3; a++) be[a] = rr[a] > 1e-30 ? (float)(nr[a] / rr[a]) : 0.0f;
        for (int i = 0; i < n; i++) pp[i] = v3add(r[i], v3(be[0] * pp[i].x, be[1] * pp[i].y, be[2] * pp[i].z));
        memcpy(rr, nr, sizeof(rr));
    }
    #undef APPLY
    free(r);
    free(pp);
    free(Ap);
}

static void arap(const Cage* c, const float* lambda, const V3* goal, V3* x, M3* R, int iterations) {
    V3* rhs = FB_NEW(V3, c->nv);
    for (int it = 0; it < iterations; it++) {
        arap_rotations(c, x, R);
        for (int i = 0; i < c->nv; i++) {
            V3 s = v3scale(goal[i], lambda[i]);
            for (int k = c->nbStart[i]; k < c->nbStart[i + 1]; k++) {
                int j = c->nb[k];
                V3 e = v3sub(c->rest[i], c->rest[j]);
                s = v3add(s, v3scale(v3add(m3mul_v(&R[i], e), m3mul_v(&R[j], e)), 0.5f));
            }
            rhs[i] = s;
        }
        arap_solve(c, lambda, rhs, x);
    }
    arap_rotations(c, x, R);
    free(rhs);
}

// ---- the checks

void semantic_measure(const TriMesh* rest, const TriMesh* def, float* strain, float* volumeChange, float* overlap) {
    // stretch: per edge, the change of length relative to it (rotations change no length)
    double s = 0;
    int ne = 0;
    for (int t = 0; t < rest->nt; t++)
        for (int e = 0; e < 3; e++) {
            int a = rest->idx[3 * t + e], b = rest->idx[3 * t + (e + 1) % 3];
            float l0 = v3dist(rest->pos[a], rest->pos[b]), l1 = v3dist(def->pos[a], def->pos[b]);
            if (l0 < 1e-7f) continue;
            s += fabsf(l1 - l0) / l0;
            ne++;
        }
    *strain = ne ? (float)(s / ne) : 0.0f;
    double v0 = fabs(mesh_volume(rest)), v1 = fabs(mesh_volume(def));
    *volumeChange = v0 > 1e-12 ? (float)fabs(v1 - v0) / (float)v0 : 0.0f;
    // running through itself: where it overlaps itself, parity counts the overlap as outside
    Box3 b = mesh_bounds(def);
    V3 ext = v3sub(b.max, b.min);
    float half = 0.55f * fb_max(ext.x, fb_max(ext.y, ext.z)) + 1e-3f;
    TriMesh moved = mesh_copy(def);
    V3 c = v3scale(v3add(b.min, b.max), 0.5f);
    for (int i = 0; i < moved.nv; i++) moved.pos[i] = v3scale(v3sub(moved.pos[i], c), 1.0f / half);
    Grid g = grid_create(48, 1.05f);
    grid_from_mesh(&g, &moved);
    int in = 0;
    for (int k = 0; k < 48 * 48 * 48; k++) in += g.v[k] < 0;
    double vox = in * (double)g.h * g.h * g.h * half * half * half;
    *overlap = v1 > 1e-12 ? (float)fabs(vox - v1) / (float)v1 : 0.0f;
    grid_free(&g);
    mesh_free(&moved);
}

// ---- trying it

// the open pose's field in the mesh's space (through the placement)
static float world_field(const Design* d, V3 x) {
    const float s = fb_max(d->scale, 1e-4f);
    return s * design_field_open(d, v3scale(v3sub(x, d->offset), 1.0f / s));
}

bool semantic_try(Evaluator* ev, Design* d, const Params* p, float collisionScale, char* msg, int msgSize) {
    const double t0 = fb_now();
    const TriMesh* input = &d->target0;
    Cage c = cage_make(input, p->arapVertices);
    const int n = c.nv;
    // where the cage is now: its clusters' average in the current target
    V3* x = FB_NEW(V3, n);
    int* cnt = FB_NEW(int, n);
    for (int i = 0; i < d->target.nv; i++) {
        x[c.clusterOf[i]] = v3add(x[c.clusterOf[i]], d->target.pos[i]);
        cnt[c.clusterOf[i]]++;
    }
    for (int k = 0; k < n; k++) x[k] = v3scale(x[k], 1.0f / fb_maxi(cnt[k], 1));
    free(cnt);
    M3* R = FB_NEW(M3, n);
    V3* goal = FB_NEW(V3, n);
    float* lambda = FB_NEW(float, n);
    const float h = 0.02f;                 // (the mesh is normalized: its longest side 2)
    for (int round = 0; round < p->semanticIterations; round++) {
        // each cage vertex: a step toward the design's surface (the union of the bodies, open), where they differ
        for (int i = 0; i < n; i++) {
            float f = world_field(d, x[i]);
            if (fabsf(f) < 0.5f * h) {
                goal[i] = x[i];
                lambda[i] = 0.5f;            // (where they agree: held where it is)
                continue;
            }
            const float e = 0.5f * h;
            V3 gr = v3((world_field(d, v3(x[i].x + e, x[i].y, x[i].z)) - world_field(d, v3(x[i].x - e, x[i].y, x[i].z))) / (2 * e),
                       (world_field(d, v3(x[i].x, x[i].y + e, x[i].z)) - world_field(d, v3(x[i].x, x[i].y - e, x[i].z))) / (2 * e),
                       (world_field(d, v3(x[i].x, x[i].y, x[i].z + e)) - world_field(d, v3(x[i].x, x[i].y, x[i].z - e))) / (2 * e));
            float gl = v3dot(gr, gr);
            V3 onSurface = gl > 1e-8f ? v3sub(x[i], v3scale(gr, f / gl)) : x[i];
            goal[i] = v3lerp(x[i], onSurface, p->semanticStep);
            lambda[i] = 0.25f;
        }
        arap(&c, lambda, goal, x, R, 4);
    }
    // the full mesh, carried by its cage: v' = sum w (x_k + R_k (v - rest_k))
    TriMesh cand = mesh_copy(input);
    for (int i = 0; i < input->nv; i++) {
        V3 s = v3(0, 0, 0);
        for (int k = 0; k < 3; k++) {
            int a = c.attach[3 * i + k];
            float w = c.attachW[3 * i + k];
            s = v3add(s, v3scale(v3add(x[a], m3mul_v(&R[a], v3sub(input->pos[i], c.rest[a]))), w));
        }
        cand.pos[i] = s;
    }
    free(x);
    free(R);
    free(goal);
    free(lambda);
    cage_free(&c);

    float strain, volume, overlap;
    semantic_measure(input, &cand, &strain, &volume, &overlap);
    bool same = strain <= p->maxStrain && volume <= p->maxVolumeChange && overlap <= 0.05f + p->maxVolumeChange;
    float before = eval_run(ev, d, NULL, collisionScale, 0.5f).total;
    float after = before;
    bool kept = false;
    if (same) {
        eval_set_target(ev, &cand);
        after = eval_run(ev, d, NULL, collisionScale, 0.5f).total;
        kept = after < before;
        if (kept) {
            mesh_free(&d->target);
            d->target = cand;
        } else {
            eval_set_target(ev, &d->target);
        }
    }
    if (!kept) mesh_free(&cand);
    snprintf(msg, (size_t)msgSize, "Semantic deformation %s: loss %.4f -> %.4f, stretch %.3f (max %.3f), volume %+.1f%%, self-overlap %.1f%% (%.0f ms)",
             kept ? "kept" : (same ? "not kept (no better)" : "not kept (not the same shape)"), before, after, strain, p->maxStrain,
             100.0f * volume, 100.0f * overlap, (fb_now() - t0) * 1e3);
    return kept;
}
