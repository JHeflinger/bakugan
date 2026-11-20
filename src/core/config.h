#ifndef FB_CONFIG_H
#define FB_CONFIG_H
#include <stdbool.h>
#include <stddef.h>

typedef struct {
    // [model]
    float sizeMm;            // the sphere's diameter, mm
    int maxBodies, maxHollows;
    int samples;             // lattice points across the sphere for collisions, holding and one piece
    // [optimizer]
    int steps;
    unsigned seed;
    int threads;
    float lrPlane, lrPhi, lrAngle, lrHollow, lrPlace;
    int stepsPerRound, rewritesPerRound, lookahead;
    float rewriteMargin;
    // [loss]
    int views, viewPixels;
    int fitLook;             // 0 silhouettes, 1 silhouettes + depth where both cover, 2 depth rasters
    float depthWeight;
    float softness;          // soft silhouette edge, pixels
    float surfaceWeight;     // hollows carving into the sphere's surface
    // [constraints]
    float collisionWeight, collisionRamp;
    int checksPerHinge;
    float hingeWeight, hingeLength;
    float dependencyWeight;
    float onePieceWeight;
    int reorderEvery;
    // [grammar]
    float wSplit, wMerge, wJoin, wCarve, wUncarve;
    float mergeAngleDeg, joinShare;
    // [semantic]
    bool semantic;
    int semanticEvery, semanticIterations, arapVertices;
    float semanticStep, maxStrain, maxVolumeChange;
    // [post]
    bool postLedges, postCarve, postLatch, postHinges, postFragments;
    bool postFeatures;
    int featureResolution;
    float featureReachMm, featureFlatten, featureSink, featureSinkMm;
    float clearanceMm, knuckleRadiusMm, pinRadiusMm, ledgeMm;
    int carveChecks, carveResolution;
    // [export]
    int stlResolution;
    int stlPose;             // 0 open, 1 closed
    // [output]
    char outDir[512];
    float statusSeconds, snapshotSeconds;
    // [suite]
    int suiteFrames, suiteFoldFrames, suiteTile, suiteFps;
    bool suitePost, suiteGhost;
} Params;

void params_default(Params* p);
bool params_load(Params* p, const char* path, char* err, int errSize);
bool params_save(const Params* p, const char* path);
bool params_set(Params* p, const char* assignment, char* err, int errSize);

typedef enum { PT_INT, PT_FLOAT, PT_BOOL, PT_STRING, PT_UINT } ParamType;
typedef struct {
    const char* section;
    const char* key;
    ParamType type;
    size_t offset;
    float lo, hi;
    const char* help;
} ParamInfo;
const ParamInfo* params_table(int* count);
#endif
