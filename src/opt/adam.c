#include "adam.h"
#include "core/fmath.h"
#include <math.h>

#define B1 0.9f
#define B2 0.99f

static inline float adam_one(float g, float* m, float* v, int t) {
    *m = B1 * *m + (1.0f - B1) * g;
    *v = B2 * *v + (1.0f - B2) * g * g;
    float mh = *m / (1.0f - powf(B1, (float)t)), vh = *v / (1.0f - powf(B2, (float)t));
    return mh / (sqrtf(vh) + 1e-12f + 1e-3f * sqrtf(vh));
}

void adam_step(Design* d, const Grad* g, const Params* p, float lrScale) {
    for (int k = 0; k < d->nplanes; k++) {
        Plane* pl = &d->planes[k];
        pl->t++;
        for (int c = 0; c < 3; c++) v3set(&pl->n, c, v3get(pl->n, c) - p->lrPlane * lrScale * adam_one(g->plane[k][c], &pl->m[c], &pl->v[c], pl->t));
        pl->n = v3norm(pl->n);
        pl->d -= p->lrPlane * lrScale * adam_one(g->plane[k][3], &pl->m[3], &pl->v[3], pl->t);
        pl->d = fb_clamp(pl->d, -0.95f * d->R, 0.95f * d->R);
    }
    for (int k = 0; k < d->nhinges; k++) {
        Hinge* h = &d->hinges[k];
        h->t++;
        // a new hinge warms up over its first steps (its first gradients are the noisiest), then keeps its pace
        // longer than the rest (it has had less time)
        const float warm = fb_min(1.0f, h->t / 8.0f) * fb_max(lrScale, fb_clamp(1.0f - h->t / 300.0f, 0.0f, 1.0f));
        h->phi -= p->lrPhi * warm * adam_one(g->hinge[k][0], &h->m[0], &h->v[0], h->t);
        h->angle -= p->lrAngle * warm * adam_one(g->hinge[k][1], &h->m[1], &h->v[1], h->t);
        h->angle = fb_clamp(h->angle, -FB_PI, FB_PI);
    }
    for (int k = 0; k < d->nhollows; k++) {
        Hollow* o = &d->hollows[k];
        o->t++;
        float* v[7] = { &o->a.x, &o->a.y, &o->a.z, &o->b.x, &o->b.y, &o->b.z, &o->r };
        for (int c = 0; c < 7; c++) *v[c] -= p->lrHollow * lrScale * fb_min(1.0f, o->t / 5.0f) * adam_one(g->hollow[k][c], &o->m[c], &o->v[c], o->t);
        o->r = fb_clamp(o->r, 0.0f, 0.6f * d->R);
    }
    d->pt++;
    // the scale in proportion (a step is a share of it), the offset in the mesh's units
    float ds = p->lrPlace * lrScale * adam_one(g->place[0] * d->scale, &d->pm[0], &d->pv[0], d->pt);
    d->scale *= expf(-ds);
    for (int c = 0; c < 3; c++) v3set(&d->offset, c, v3get(d->offset, c) - p->lrPlace * lrScale * adam_one(g->place[1 + c], &d->pm[1 + c], &d->pv[1 + c], d->pt));
    design_hinge_frames(d);
}
