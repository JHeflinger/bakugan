#include "optimizer.h"
#include "adam.h"
#include "semantic.h"
#include <stdlib.h>
#include <string.h>

OptShared* opt_shared_create(void) {
    OptShared* s = FB_NEW(OptShared, 1);
    s->lock = fb_mutex_create();
    return s;
}

void opt_shared_free(OptShared* s) {
    if (!s) return;
    if (s->best) {
        design_free(s->best);
        free(s->best);
    }
    free(s->history);
    free(s->pics);
    fb_mutex_destroy(s->lock);
    free(s);
}

static float phase_of(int k) { return fmodf(0.5f + 0.6180339887f * (float)k, 1.0f); }

typedef struct {
    const Params* p;
    OptShared* sh;
    Evaluator* ev;
    Grad* grad;
    int step, budget;
    Design* best;
    Score bestScore;
    double lastPics;
} Loop;

static float ramp_at(const Loop* L, int step) {
    float r = fb_max(1.0f, L->p->collisionRamp);
    return powf(r, fb_clamp((float)step / fb_maxi(1, L->budget), 0.0f, 1.0f));
}

// a score as it would be at the end of the ramp: what designs from different times are compared by
static float judged(const Loop* L, const Score* s) {
    return s->total + (fb_max(1.0f, L->p->collisionRamp) - s->collisionScale) * L->p->collisionWeight * s->collision;
}

static void set_stage(OptShared* sh, const char* stage) {
    fb_mutex_lock(sh->lock);
    snprintf(sh->stage, sizeof(sh->stage), "%s", stage);
    fb_mutex_unlock(sh->lock);
}

static void publish(Loop* L, const Design* d, const Score* s) {
    OptShared* sh = L->sh;
    bool better = s->valid && (!L->bestScore.valid || judged(L, s) < judged(L, &L->bestScore));
    if (better) {
        design_copy(L->best, d);
        L->bestScore = *s;
    }
    double now = fb_now();
    bool pics = now - L->lastPics > 0.25;
    fb_mutex_lock(sh->lock);
    sh->step = L->step;
    sh->score = *s;
    sh->bestScore = L->bestScore;
    if (sh->nhist >= sh->caphist) {
        sh->caphist = sh->caphist ? sh->caphist * 2 : 1024;
        sh->history = fb_realloc(sh->history, sizeof(float) * (size_t)sh->caphist);
    }
    sh->history[sh->nhist++] = s->total;
    if (better) {
        if (!sh->best) sh->best = FB_NEW(Design, 1);
        design_copy(sh->best, L->best);
        sh->version++;
    }
    if (pics) {
        const int nv = eval_views(L->ev), res = eval_resolution(L->ev), px = res * res;
        if (!sh->pics || sh->nv != nv || sh->res != res) {
            free(sh->pics);
            sh->pics = FB_NEW(float, (size_t)2 * nv * px);
            sh->nv = nv;
            sh->res = res;
        }
        for (int v = 0; v < nv; v++) eval_picture(L->ev, v, sh->pics + (size_t)v * px, sh->pics + ((size_t)nv + v) * px);
        sh->picVersion++;
    }
    fb_mutex_unlock(sh->lock);
    if (pics) L->lastPics = now;
    // recording: the best so far, at evenly spaced steps
    if (sh->record && sh->recordFrames > 0 && L->bestScore.valid)
        while (sh->recorded < sh->recordFrames &&
               (long long)L->step * fb_maxi(1, sh->recordFrames - 1) >= (long long)sh->recorded * fb_maxi(1, L->budget)) {
            sh->record(sh->recordCtx, L->best, &L->bestScore, L->step, L->budget);
            sh->recorded++;
        }
}

static void wait_while_paused(Loop* L) {
    OptShared* sh = L->sh;
    if (!sh->pauseRequested) return;
    double t0 = fb_now();
    fb_mutex_lock(sh->lock);
    sh->paused = true;
    sh->pausedAt = t0;
    fb_mutex_unlock(sh->lock);
    while (sh->pauseRequested && !sh->stopRequested) fb_sleep_ms(50);
    fb_mutex_lock(sh->lock);
    sh->paused = false;
    sh->pausedFor += fb_now() - t0;
    fb_mutex_unlock(sh->lock);
}

// one Adam step; returns the score of the design before it
static Score step_once(Loop* L, Design* d, int k) {
    grad_clear(L->grad, d);
    Score s = eval_run(L->ev, d, L->grad, ramp_at(L, L->step), phase_of(k));
    // learning rates: a cosine down to a tenth over the run
    float x = fb_clamp((float)L->step / fb_maxi(1, L->budget), 0.0f, 1.0f);
    adam_step(d, L->grad, L->p, 0.1f + 0.9f * 0.5f * (1.0f + cosf(FB_PI * x)));
    return s;
}

// the opening order re-sorted, least colliding hinge first, kept when it does better
static void reorder(Loop* L, Design* d, const Score* s) {
    if (d->nhinges < 2) return;
    int idx[FB_MAX_BODIES];
    for (int h = 0; h < d->nhinges; h++) idx[h] = h;
    for (int a = 0; a < d->nhinges; a++)
        for (int b = a + 1; b < d->nhinges; b++)
            if (s->hingeCollision[idx[b]] < s->hingeCollision[idx[a]] ||
                (s->hingeCollision[idx[b]] == s->hingeCollision[idx[a]] && d->hinges[idx[b]].order < d->hinges[idx[a]].order)) {
                int t = idx[a];
                idx[a] = idx[b];
                idx[b] = t;
            }
    bool same = true;
    for (int k = 0; k < d->nhinges; k++) same &= d->hinges[idx[k]].order == k;
    if (same) return;
    Design* other = design_clone(d);
    for (int k = 0; k < d->nhinges; k++) other->hinges[idx[k]].order = k;
    float scale = ramp_at(L, L->step);
    Score a = eval_run(L->ev, d, NULL, scale, 0.5f), b = eval_run(L->ev, other, NULL, scale, 0.5f);
    if (b.total < a.total - 1e-6f) {
        for (int k = 0; k < d->nhinges; k++) d->hinges[k].order = other->hinges[k].order;
        fb_mutex_lock(L->sh->lock);
        L->sh->reorders++;
        fb_mutex_unlock(L->sh->lock);
    }
    design_free(other);
    free(other);
}

void optimize_run(const Params* p, Design* d, OptShared* sh) {
    Loop L;
    memset(&L, 0, sizeof(L));
    L.p = p;
    L.sh = sh;
    L.budget = p->steps;
    L.lastPics = -1;
    fb_mutex_lock(sh->lock);
    sh->running = true;
    sh->finished = false;
    sh->steps = p->steps;
    sh->started = fb_now();
    fb_mutex_unlock(sh->lock);
    set_stage(sh, "Drawing the targets");
    L.ev = eval_create(p, d);
    L.grad = grad_create(d);
    L.best = FB_NEW(Design, 1);
    Rng rng;
    rng_seed(&rng, p->seed);
    set_stage(sh, "Optimizing");
    Score last = { 0 };
    int sinceSemantic = 0;
    while (L.step < L.budget && !sh->stopRequested) {
        wait_while_paused(&L);
        if (sh->stopRequested) break;
        Design* before = design_clone(d);
        Score s = step_once(&L, d, L.step);
        L.step++;
        publish(&L, before, &s);
        design_free(before);
        free(before);
        last = s;
        sinceSemantic++;

        // the semantic deformation of the target, now and then
        if (p->semantic && sinceSemantic >= p->semanticEvery) {
            sinceSemantic = 0;
            set_stage(sh, "Semantic deformation");
            char msg[512];
            bool kept = semantic_try(L.ev, d, p, ramp_at(&L, L.step), msg, sizeof(msg));
            fb_mutex_lock(sh->lock);
            sh->semanticTried++;
            if (kept) sh->semanticKept++;
            snprintf(sh->message, sizeof(sh->message), "%s", msg);
            fb_mutex_unlock(sh->lock);
            if (kept) L.bestScore.valid = false;    // (the target changed: earlier scores aren't comparable)
            set_stage(sh, "Optimizing");
        }
        if (p->reorderEvery > 0 && L.step % p->reorderEvery == 0) reorder(&L, d, &last);

        // a rewrite round
        if (p->rewritesPerRound <= 0 || L.step % fb_maxi(1, p->stepsPerRound) != 0 || L.step + p->lookahead > L.budget) continue;
        set_stage(sh, "Rewriting");
        Design* tries[33];
        int rules[33], nt = 0;
        tries[nt] = design_clone(d);
        rules[nt++] = -1;
        for (int k = 0; k < p->rewritesPerRound && nt < 33; k++) {
            int rule = rule_draw(p, &rng);
            if (rule < 0) break;
            Design* t = design_clone(d);
            fb_mutex_lock(sh->lock);
            sh->tried[rule]++;
            fb_mutex_unlock(sh->lock);
            if (rule_apply(rule, t, p, &rng, L.ev, ramp_at(&L, L.step))) {
                tries[nt] = t;
                rules[nt++] = rule;
            } else {
                design_free(t);
                free(t);
            }
        }
        float scores[33];
        for (int t = 0; t < nt; t++) {
            Score s2 = { 0 };
            for (int k = 0; k < p->lookahead && L.step < L.budget && !sh->stopRequested; k++) {
                wait_while_paused(&L);
                Design* b2 = design_clone(tries[t]);
                s2 = step_once(&L, tries[t], k);        // (every try looks in the same places)
                L.step++;
                publish(&L, b2, &s2);
                design_free(b2);
                free(b2);
            }
            if (!s2.valid) s2 = eval_run(L.ev, tries[t], NULL, ramp_at(&L, L.step), 0.5f);
            // a rewrite is kept over leaving the design alone when it does no worse than that by the margin (a split
            // changes nothing when it's made: what it adds is room, and merge and join take away what doesn't pay)
            float w = s2.total;
            if (rules[t] >= 0) w /= 1.0f + p->rewriteMargin;
            scores[t] = w;
        }
        int win = 0;
        for (int t = 1; t < nt; t++)
            if (scores[t] < scores[win]) win = t;
        if (getenv("FOLDBALL_DEBUG")) {
            fprintf(stderr, "round at %d:", L.step);
            for (int t = 0; t < nt; t++) fprintf(stderr, " %s %.4f%s", rules[t] < 0 ? "same" : rule_name(rules[t]), scores[t], t == win ? "*" : "");
            fprintf(stderr, "\n");
        }
        design_copy(d, tries[win]);
        for (int t = 0; t < nt; t++) {
            design_free(tries[t]);
            free(tries[t]);
        }
        fb_mutex_lock(sh->lock);
        sh->rounds++;
        if (win > 0) {
            sh->accepted++;
            sh->kept[rules[win]]++;
        }
        fb_mutex_unlock(sh->lock);
        set_stage(sh, "Optimizing");
    }
    // the result: the best design seen (scored at the end of the ramp)
    set_stage(sh, "Finishing");
    Score fin = eval_run(L.ev, d, NULL, fb_max(1.0f, p->collisionRamp), 0.5f);
    publish(&L, d, &fin);
    if (L.bestScore.valid) design_copy(d, L.best);
    Score bestFinal = eval_run(L.ev, d, NULL, fb_max(1.0f, p->collisionRamp), 0.5f);
    fb_mutex_lock(sh->lock);
    sh->bestScore = bestFinal;
    if (!sh->best) sh->best = FB_NEW(Design, 1);
    design_copy(sh->best, d);
    sh->version++;
    sh->running = false;
    sh->finished = true;
    sh->doneAt = fb_now();
    snprintf(sh->stage, sizeof(sh->stage), "%s", sh->stopRequested ? "Stopped" : "Done");
    fb_mutex_unlock(sh->lock);
    design_free(L.best);
    free(L.best);
    grad_free(L.grad);
    eval_free(L.ev);
}

char* opt_status_line(OptShared* sh) {
    fb_mutex_lock(sh->lock);
    double el = (sh->finished ? sh->doneAt : fb_now()) - sh->started - sh->pausedFor - (sh->paused ? fb_now() - sh->pausedAt : 0.0);
    const Score* b = &sh->bestScore;
    char* s = fb_format("[%s%s] step %d/%d (%.0f%%), %.1f s, %.0f ms/step | loss %.4f (best %.4f: open %.3f, cover %.0f%%, collision %.4f, hinges %.4f, surface carved %.1f%%) | "
                        "%d bodies, %d hollows | rewrites kept %d of %d rounds (split %d, merge %d, join %d, carve %d, uncarve %d) | semantic %d of %d",
                        sh->stage, sh->paused ? ", PAUSED" : "", sh->step, sh->steps, sh->steps ? 100.0 * sh->step / sh->steps : 0.0, el,
                        sh->step ? el * 1e3 / sh->step : 0.0, sh->score.total, b->total, b->open, 100.0 * b->openCover, b->collision, b->hinge,
                        100.0 * b->surfaceCarved, sh->best ? sh->best->nbodies : 1, sh->best ? sh->best->nhollows : 0, sh->accepted, sh->rounds, sh->kept[0],
                        sh->kept[1], sh->kept[2], sh->kept[3], sh->kept[4], sh->semanticKept,
                        sh->semanticTried);
    fb_mutex_unlock(sh->lock);
    return s;
}
