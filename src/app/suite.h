#ifndef FB_SUITE_H
#define FB_SUITE_H
// The test suite: every .obj in a folder optimized in turn with one set of settings (an .ini, or the GUI's current
// ones), each run recorded (the best design so far, at evenly spaced steps, then the result folding shut and open
// again), and all of them put side by side into one video (ffmpeg, which must be on the PATH), with per mesh results
// (<stem>/<stem>.fold, <stem>.png) and summary.csv. The GUI and `foldball suite` both run it.
#include "core/config.h"
#include "core/util.h"
#include "opt/optimizer.h"

typedef struct {
    volatile int stopRequested, skipRequested, pauseRequested;
    FbMutex* lock;                   // the fields below
    bool running, finished;
    int index, count;                // the mesh being run (0 based), of how many
    char current[256];               // its name
    char stage[256];
    char log[16384];                 // a line per mesh done, and how it ended
    char outDir[1024], video[1024];
    OptShared* opt;                  // the current mesh's run (only read with the lock held)
    unsigned char* tile;             // the latest frame of the current mesh (tileSize^2 RGB)
    int tileSize, tileVersion;
    double started;
} SuiteShared;

typedef struct {
    Params P;                        // the settings for every run (and [suite]'s for the video)
    char folder[1024];               // the .obj files
    char outBase[1024];              // a new folder suite_<n> is made in it
} SuiteJob;

SuiteShared* suite_shared_create(void);
void suite_shared_free(SuiteShared* s);
// Runs the whole suite in this thread. False (and the log says why) if it couldn't start.
bool suite_run(const SuiteJob* job, SuiteShared* sh);
// Where it stands, one line (malloc'd).
char* suite_status_line(SuiteShared* sh);
// The .obj files in a folder (how many; -1: it can't be read).
int suite_count(const char* folder);
#endif
