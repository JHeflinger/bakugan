#ifndef FB_OPTIMIZER_H
#define FB_OPTIMIZER_H
#include "eval.h"
#include "grammar.h"
#include "core/util.h"

// What a running optimization shares with whoever watches it (the GUI, the CLI). Fields below `lock` are only read
// or written with it held; the requests are plain flags the loop checks between steps.
typedef struct {
    volatile int pauseRequested, stopRequested;
    FbMutex* lock;
    bool running, paused, finished;
    int step, steps;
    int rounds, accepted, kept[RULE_COUNT], tried[RULE_COUNT];
    int reorders, semanticTried, semanticKept;
    Score score;                     // the latest step's
    Score bestScore;
    Design* best;                    // the best design so far (a copy; version bumps when it changes)
    int version;
    float* history;                  // total loss per step
    int nhist, caphist;
    char stage[128];
    char message[2048];
    int nv, res;
    float* pics;                     // [design / target 2][view][res*res] (the open pose), the latest
    int picVersion;
    double started, pausedFor, pausedAt, doneAt;
    // recording (set before the run; the test suite's): `record` is called from the optimizing thread with the best
    // design so far at recordFrames evenly spaced steps (the first at step 1), without the lock held
    int recordFrames;
    void (*record)(void* ctx, const Design* best, const Score* score, int step, int steps);
    void* recordCtx;
    int recorded;
} OptShared;

OptShared* opt_shared_create(void);
void opt_shared_free(OptShared* s);

// Runs the optimization on d (in this thread) until it is done or stopped; d is left as the best design found.
void optimize_run(const Params* p, Design* d, OptShared* sh);

// one line of where it stands (for the CLI and the status file); returns a malloc'd string
char* opt_status_line(OptShared* sh);
#endif
