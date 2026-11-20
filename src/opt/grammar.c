#include "grammar.h"
#include "eval.h"
#include "core/util.h"
#include <stdlib.h>
#include <string.h>

const char* rule_name(int rule) {
    static const char* names[] = { "split", "merge", "join", "carve", "uncarve" };
    return rule >= 0 && rule < RULE_COUNT ? names[rule] : "?";
}

bool rule_simplifies(int rule) { return rule == RULE_MERGE || rule == RULE_JOIN || rule == RULE_UNCARVE; }

int rule_draw(const Params* p, Rng* rng) {
    float w[RULE_COUNT] = { p->wSplit, p->wMerge, p->wJoin, p->wCarve, p->wUncarve }, sum = 0;
    for (int k = 0; k < RULE_COUNT; k++) sum += fb_max(0.0f, w[k]);
    if (sum <= 0) return -1;
    float x = rng_uniform(rng) * sum;
    for (int k = 0; k < RULE_COUNT; k++) {
        x -= fb_max(0.0f, w[k]);
        if (x <= 0) return k;
    }
    return RULE_COUNT - 1;
}

static int cmp_f(const void* a, const void* b) {
    float x = *(const float*)a, y = *(const float*)b;
    return x < y ? -1 : (x > y);
}

typedef FB_ARRAY(V3) Points;

// ---- split

typedef struct {
    int body;
    V3 n, ref;
    float d, phi, cost;
} SplitPlan;

static void split_apply(Design* d, const SplitPlan* sp) {
    Plane pl;
    memset(&pl, 0, sizeof(pl));
    pl.id = design_new_id(d);
    pl.n = sp->n;
    pl.d = sp->d;
    const int pi = d->nplanes;
    d->planes[d->nplanes++] = pl;
    Body* body = &d->bodies[sp->body];
    Body* nb = &d->bodies[d->nbodies];
    memset(nb, 0, sizeof(*nb));
    nb->id = design_new_id(d);
    nb->ncells = body->ncells;
    for (int c = 0; c < body->ncells; c++) {
        nb->cells[c] = body->cells[c];
        nb->cells[c].plane[nb->cells[c].n] = (short)pi;
        nb->cells[c].sign[nb->cells[c].n++] = -1;
        body->cells[c].plane[body->cells[c].n] = (short)pi;
        body->cells[c].sign[body->cells[c].n++] = 1;
    }
    d->nbodies++;
    const int oldId = body->id, newId = nb->id;
    // the hinges the body had go with the half their pivot is in
    bool newIsParentOfOld = false;
    for (int h = 0; h < d->nhinges; h++) {
        Hinge* hg = &d->hinges[h];
        if (v3dot(sp->n, hg->pivot) <= sp->d) continue;
        if (hg->parent == oldId) hg->parent = newId;
        else if (hg->child == oldId) {
            hg->child = newId;
            newIsParentOfOld = true;
        }
    }
    Hinge* nh = &d->hinges[d->nhinges];
    memset(nh, 0, sizeof(*nh));
    nh->id = design_new_id(d);
    nh->parent = newIsParentOfOld ? newId : oldId;
    nh->child = newIsParentOfOld ? oldId : newId;
    nh->plane = pi;
    nh->ref = sp->ref;
    nh->phi = sp->phi;
    nh->angle = 0.0f;                    // it does nothing until Adam turns it
    nh->order = d->nhinges;              // it opens last
    nh->holder = -1;
    d->nhinges++;
    design_normalize_orders(d);
    design_hinge_frames(d);
}

static bool rule_split(Design* d, const Params* p, Rng* rng, struct Evaluator* ev, float collisionScale) {
    if (d->nbodies >= fb_mini(p->maxBodies, FB_MAX_BODIES) || d->nplanes >= FB_MAX_PLANES - 1) return false;
    Lattice L;
    lattice_build(d, fb_maxi(p->samples, 16), &L);
    const int M = L.M;
    const size_t N = (size_t)M * M * M;
    int count[FB_MAX_BODIES] = { 0 }, total = 0;
    for (size_t k = 0; k < N; k++)
        if (L.label[k] >= 0) { count[L.label[k]]++; total++; }
    SplitPlan plans[32];
    int nplans = 0;
    const int bodiesToTry = ev ? 3 : 1;
    const float Lh = p->hingeLength * d->R, slack = 0.25f * d->inset;
    for (int bt = 0; bt < bodiesToTry; bt++) {
        int b = -1;
        if (total > 0) {
            int x = rng_int(rng, total);
            for (int k = 0; k < d->nbodies; k++) {
                x -= count[k];
                if (x < 0) { b = k; break; }
            }
        }
        if (b < 0 || count[b] < 30) continue;
        bool room = d->bodies[b].ncells <= FB_MAX_CELLS;
        for (int c = 0; c < d->bodies[b].ncells; c++) room &= d->bodies[b].cells[c].n < FB_MAX_HALF;
        if (!room) continue;
        Points pts = { 0 };
        int* nodes = FB_NEW(int, count[b] + 1);
        int nn = 0;
        for (size_t k = 0; k < N; k++)
            if (L.label[k] == b) {
                nodes[nn++] = (int)k;
                fb_push(&pts, lattice_point(&L, (int)(k % M), (int)((k / M) % M), (int)(k / ((size_t)M * M))));
            }
        int* stack = FB_NEW(int, nn + 1);
        int* mark = FB_NEW(int, N);
        int stamp = 0;
        float* along = FB_NEW(float, pts.count);
        // directions: the axes, the body's principal axes, random ones
        V3 dirs[20];
        int ndirs = 0;
        dirs[ndirs++] = v3(1, 0, 0);
        dirs[ndirs++] = v3(0, 1, 0);
        dirs[ndirs++] = v3(0, 0, 1);
        {
            V3 mean = v3(0, 0, 0);
            for (int k = 0; k < pts.count; k++) mean = v3add(mean, pts.data[k]);
            mean = v3scale(mean, 1.0f / fb_maxi(pts.count, 1));
            float C[9] = { 0 };
            for (int k = 0; k < pts.count; k++) {
                V3 q = v3sub(pts.data[k], mean);
                float v[3] = { q.x, q.y, q.z };
                for (int a = 0; a < 3; a++)
                    for (int c = 0; c < 3; c++) C[3 * a + c] += v[a] * v[c];
            }
            for (int ax = 0; ax < 3; ax++) {
                V3 x = rng_dir(rng);
                for (int it = 0; it < 30; it++) {
                    V3 y = v3(C[0] * x.x + C[1] * x.y + C[2] * x.z, C[3] * x.x + C[4] * x.y + C[5] * x.z, C[6] * x.x + C[7] * x.y + C[8] * x.z);
                    for (int q = 0; q < ax; q++) y = v3mad(y, dirs[3 + q], -v3dot(y, dirs[3 + q]));
                    x = v3norm(y);
                }
                dirs[ndirs++] = x;
            }
        }
        while (ndirs < 12) dirs[ndirs++] = rng_dir(rng);
        const float shares[7] = { 0.15f, 0.25f, 0.35f, 0.5f, 0.65f, 0.75f, 0.85f };
        for (int di = 0; di < ndirs; di++) {
            V3 nrm = dirs[di];
            for (int k = 0; k < pts.count; k++) along[k] = v3dot(nrm, pts.data[k]);
            qsort(along, (size_t)pts.count, sizeof(float), cmp_f);
            for (int sh = 0; sh < 7; sh++) {
                const float dd = along[(int)(shares[sh] * (pts.count - 1))];
                if (fabsf(dd) > 0.9f * (d->R - d->inset)) continue;
                // each side one piece
                bool ok = true;
                for (int side = 0; side < 2 && ok; side++) {
                    int pieces = 0;
                    stamp++;
                    for (int q = 0; q < nn && ok; q++) {
                        int s0 = nodes[q];
                        V3 x0 = lattice_point(&L, s0 % M, (s0 / M) % M, s0 / (M * M));
                        if ((v3dot(nrm, x0) > dd) != (side == 1) || mark[s0] == stamp) continue;
                        int top = 0, size = 0;
                        stack[top++] = s0;
                        mark[s0] = stamp;
                        while (top) {
                            int c = stack[--top];
                            size++;
                            int ci = c % M, cj = (c / M) % M, ck = c / (M * M);
                            int nb[6] = { ci > 0 ? c - 1 : -1, ci < M - 1 ? c + 1 : -1, cj > 0 ? c - M : -1, cj < M - 1 ? c + M : -1,
                                          ck > 0 ? c - M * M : -1, ck < M - 1 ? c + M * M : -1 };
                            for (int e = 0; e < 6; e++) {
                                int q2 = nb[e];
                                if (q2 < 0 || L.label[q2] != b || mark[q2] == stamp) continue;
                                V3 x2 = lattice_point(&L, q2 % M, (q2 / M) % M, q2 / (M * M));
                                if ((v3dot(nrm, x2) > dd) != (side == 1)) continue;
                                mark[q2] = stamp;
                                stack[top++] = q2;
                            }
                        }
                        if (size > 2) pieces++;
                        if (pieces > 1) ok = false;
                    }
                    if (pieces == 0) ok = false;
                }
                if (!ok) continue;
                // the hinge: along the seam, where the body is right at its barrel
                SplitPlan best = { b, nrm, v3perp(nrm), dd, 0, FLT_MAX };
                Plane tmp[FB_MAX_PLANES];
                memcpy(tmp, d->planes, sizeof(Plane) * (size_t)d->nplanes);
                tmp[d->nplanes].n = nrm;
                tmp[d->nplanes].d = dd;
                Hinge hh;
                memset(&hh, 0, sizeof(hh));
                hh.plane = d->nplanes;
                hh.ref = best.ref;
                for (int a = 0; a < 36; a++) {
                    hh.phi = 2.0f * FB_PI * a / 36.0f;
                    V3 pv, ax;
                    hinge_frame(d, tmp, &hh, &pv, &ax);
                    float cost = 0;
                    for (int j = -2; j <= 2; j++) cost += fb_max(0.0f, body_field(d, b, v3mad(pv, ax, 0.25f * j * Lh), NULL) - slack);
                    cost += 1e-4f * rng_uniform(rng);
                    if (cost < best.cost) { best.cost = cost; best.phi = hh.phi; }
                }
                if (nplans < 32) plans[nplans++] = best;
                else {
                    int worst = 0;
                    for (int k = 1; k < nplans; k++)
                        if (plans[k].cost > plans[worst].cost) worst = k;
                    if (best.cost < plans[worst].cost) plans[worst] = best;
                }
            }
        }
        free(along);
        free(nodes);
        free(stack);
        free(mark);
        fb_free_array(&pts);
    }
    lattice_free(&L);
    if (nplans == 0) return false;
    for (int a = 0; a < nplans; a++)
        for (int c = a + 1; c < nplans; c++)
            if (plans[c].cost < plans[a].cost) { SplitPlan t = plans[a]; plans[a] = plans[c]; plans[c] = t; }
    // which, and how far it opens: among plans whose hinge fits, the one (and the angle) the loss likes best. (Split
    // at angle 0 changes nothing, and a hinge's pull there is often nil: a piece has to swing well out before it
    // starts to cover anything, so the angles are searched outright.)
    int pick = 0;
    float pickAngle = 0.0f;
    if (ev) {
        const float fits = plans[0].cost + 0.02f * d->R;
        float bestLoss = eval_run(ev, d, NULL, collisionScale, 0.5f).total;
        for (int k = 0; k < nplans && k < 8 && plans[k].cost <= fits; k++) {
            Design* t = design_clone(d);
            split_apply(t, &plans[k]);
            const int hi = design_hinge(t, t->hinges[t->nhinges - 1].id);
            for (int a = -5; a <= 6; a++) {
                if (a == 0) continue;
                t->hinges[hi].angle = a * FB_PI / 6.0f;
                float l = eval_run(ev, t, NULL, collisionScale, 0.5f).total;
                if (l < bestLoss) { bestLoss = l; pick = k; pickAngle = t->hinges[hi].angle; }
            }
            design_free(t);
            free(t);
        }
    }
    split_apply(d, &plans[pick]);
    d->hinges[d->nhinges - 1].angle = pickAngle;
    return true;
}

// ---- merge and join

static void simplify_cells(Body* body) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (int a = 0; a < body->ncells && !changed; a++)
            for (int b = a + 1; b < body->ncells && !changed; b++) {
                Cell *A = &body->cells[a], *B = &body->cells[b];
                if (A->n != B->n) continue;
                int diff = -1, ndiff = 0;
                for (int k = 0; k < A->n && ndiff <= 1; k++) {
                    int found = -1;
                    for (int q = 0; q < B->n; q++)
                        if (B->plane[q] == A->plane[k]) found = q;
                    if (found < 0) { ndiff = 2; break; }
                    if (B->sign[found] != A->sign[k]) { diff = k; ndiff++; }
                }
                if (ndiff != 1) continue;
                A->plane[diff] = A->plane[A->n - 1];
                A->sign[diff] = A->sign[A->n - 1];
                A->n--;
                body->cells[b] = body->cells[--body->ncells];
                changed = true;
            }
    }
}

static void remove_hinge(Design* d, int h) {
    for (int k = h; k + 1 < d->nhinges; k++) d->hinges[k] = d->hinges[k + 1];
    d->nhinges--;
}

// body C (index) becomes part of body P (index): its cells, its hinges
static bool absorb(Design* d, int P, int C) {
    if (d->bodies[P].ncells + d->bodies[C].ncells > FB_MAX_CELLS) return false;
    Body* parent = &d->bodies[P];
    const Body* child = &d->bodies[C];
    for (int c = 0; c < child->ncells; c++) parent->cells[parent->ncells++] = child->cells[c];
    simplify_cells(parent);
    const int pid = parent->id, cid = child->id;
    for (int k = 0; k < d->nhinges; k++)
        if (d->hinges[k].parent == cid) d->hinges[k].parent = pid;
    for (int e = 0; e < d->nextras; e++)
        if (d->extras[e].body == cid) d->extras[e].body = pid;
    grid_free(&d->bodies[C].carve);
    boxgrid_free(&d->bodies[C].detail);
    d->bodies[C] = d->bodies[--d->nbodies];
    return true;
}

static bool rule_merge(Design* d, const Params* p, Rng* rng) {
    int cand[FB_MAX_BODIES], nc = 0;
    const float lim = p->mergeAngleDeg * FB_PI / 180.0f;
    for (int h = 0; h < d->nhinges; h++)
        if (fabsf(d->hinges[h].angle) < lim) cand[nc++] = h;
    if (nc == 0) return false;
    for (int a = 0; a < nc; a++)
        for (int b = a + 1; b < nc; b++)
            if (fabsf(d->hinges[cand[b]].angle) < fabsf(d->hinges[cand[a]].angle)) { int t = cand[a]; cand[a] = cand[b]; cand[b] = t; }
    int pick = 0;
    while (pick + 1 < nc && rng_uniform(rng) < 0.5f) pick++;
    const int h = cand[pick];
    const int P = design_body(d, d->hinges[h].parent), C = design_body(d, d->hinges[h].child);
    if (P < 0 || C < 0) return false;
    if (d->bodies[P].ncells + d->bodies[C].ncells > FB_MAX_CELLS) return false;
    remove_hinge(d, h);
    if (!absorb(d, P, C)) return false;
    design_tidy(d);
    return true;
}

static bool rule_join(Design* d, const Params* p, Rng* rng) {
    if (d->nbodies < 3) return false;
    Lattice L;
    lattice_build(d, fb_maxi(p->samples, 16), &L);
    const int M = L.M;
    const size_t N = (size_t)M * M * M;
    int count[FB_MAX_BODIES] = { 0 }, total = 0;
    for (size_t k = 0; k < N; k++)
        if (L.label[k] >= 0) { count[L.label[k]]++; total++; }
    int cand[FB_MAX_BODIES], nc = 0;
    for (int b = 0; b < d->nbodies; b++) {
        if (d->bodies[b].id == d->root || design_parent_hinge(d, d->bodies[b].id) < 0) continue;
        if (total > 0 && (float)count[b] / total > p->joinShare) continue;
        cand[nc++] = b;
    }
    if (nc == 0) { lattice_free(&L); return false; }
    for (int a = 0; a < nc; a++)
        for (int b = a + 1; b < nc; b++)
            if (count[cand[b]] < count[cand[a]]) { int t = cand[a]; cand[a] = cand[b]; cand[b] = t; }
    int pick = 0;
    while (pick + 1 < nc && rng_uniform(rng) < 0.5f) pick++;
    const int X = cand[pick], xid = d->bodies[X].id;
    const int upIndex = design_parent_hinge(d, xid);
    const Hinge up = d->hinges[upIndex];
    const Rigid Rup = rabout(up.pivot, up.axis, up.angle);
    // each hinge that hung from X: its angle chosen to take its pieces (opened) as near as it can to where X's hinge
    // and its own took them (its axis stays on its seam)
    for (int h = 0; h < d->nhinges; h++) {
        Hinge* j = &d->hinges[h];
        if (j->parent != xid) continue;
        Points pts = { 0 };
        for (int b = 0; b < d->nbodies; b++) {
            if (!design_below(d, h, b)) continue;
            for (size_t k = 0; k < N; k += 7)
                if (L.label[k] == b) fb_push(&pts, lattice_point(&L, (int)(k % M), (int)((k / M) % M), (int)(k / ((size_t)M * M))));
        }
        const Rigid Rj = rabout(j->pivot, j->axis, j->angle);
        float best = j->angle, bestErr = FLT_MAX;
        for (int a = -36; a <= 36; a++) {
            float ang = j->angle + up.angle * 0.0f + a * FB_PI / 36.0f;
            Rigid r = rabout(j->pivot, j->axis, ang);
            double e = 0;
            for (int k = 0; k < pts.count; k++) {
                V3 want = rapply(Rup, rapply(Rj, pts.data[k]));
                V3 dlt = v3sub(rapply(r, pts.data[k]), want);
                e += v3dot(dlt, dlt);
            }
            if (e < bestErr) { bestErr = (float)e; best = ang; }
        }
        fb_free_array(&pts);
        while (best > FB_PI) best -= 2 * FB_PI;
        while (best < -FB_PI) best += 2 * FB_PI;
        j->angle = best;
        j->order = fb_mini(j->order, up.order);
    }
    lattice_free(&L);
    const int P = design_body(d, up.parent);
    remove_hinge(d, upIndex);
    if (P < 0 || !absorb(d, P, design_body(d, xid))) return false;
    design_tidy(d);
    return true;
}

// ---- carve and uncarve

static bool rule_carve(Design* d, const Params* p, Rng* rng) {
    if (d->nhollows >= fb_mini(p->maxHollows, FB_MAX_HOLLOWS) || d->nbodies == 0) return false;
    // a point in a body near its surface (closed: the design), toward the outside of the sphere a little less often
    Lattice L;
    lattice_build(d, fb_maxi(p->samples, 16), &L);
    const int M = L.M;
    const size_t N = (size_t)M * M * M;
    V3 at = v3(0, 0, 0);
    bool found = false;
    for (int tries = 0; tries < 400 && !found; tries++) {
        size_t k = (size_t)rng_int(rng, (int)N);
        if (L.label[k] < 0) continue;
        at = lattice_point(&L, (int)(k % M), (int)((k / M) % M), (int)(k / ((size_t)M * M)));
        found = v3len(at) < 0.85f * d->R || rng_uniform(rng) < 0.3f;
    }
    lattice_free(&L);
    if (!found) return false;
    Hollow* h = &d->hollows[d->nhollows];
    memset(h, 0, sizeof(*h));
    h->id = design_new_id(d);
    V3 dir = rng_dir(rng);
    h->a = v3mad(at, dir, -0.08f * d->R);
    h->b = v3mad(at, dir, 0.08f * d->R);
    h->r = -0.01f * d->R;            // it takes nothing yet: Adam grows it where it helps
    d->nhollows++;
    return true;
}

static bool rule_uncarve(Design* d, const Params* p, Rng* rng) {
    (void)p;
    if (d->nhollows == 0) return false;
    // the thinnest likeliest
    int pick = 0;
    for (int k = 1; k < d->nhollows; k++)
        if (d->hollows[k].r < d->hollows[pick].r) pick = k;
    if (rng_uniform(rng) < 0.3f) pick = rng_int(rng, d->nhollows);
    d->hollows[pick] = d->hollows[--d->nhollows];
    return true;
}

bool rule_apply(int rule, Design* d, const Params* p, Rng* rng, struct Evaluator* ev, float collisionScale) {
    switch (rule) {
    case RULE_SPLIT: return rule_split(d, p, rng, ev, collisionScale);
    case RULE_MERGE: return rule_merge(d, p, rng);
    case RULE_JOIN: return rule_join(d, p, rng);
    case RULE_CARVE: return rule_carve(d, p, rng);
    case RULE_UNCARVE: return rule_uncarve(d, p, rng);
    }
    return false;
}
