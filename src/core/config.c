#include "config.h"
#include "util.h"
#include <ctype.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void params_default(Params* p) {
    memset(p, 0, sizeof(*p));
    p->sizeMm = 60.0f;
    p->maxBodies = 20;
    p->maxHollows = 12;
    p->samples = 30;
    p->steps = 3000;
    p->seed = 1;
    p->lrPlane = 0.01f;
    p->lrBend = 0.01f;
    p->maxBend = 0.0f;
    p->lrPhi = 0.03f;
    p->lrAngle = 0.05f;
    p->lrHollow = 0.01f;
    p->lrPlace = 0.01f;
    p->stepsPerRound = 30;
    p->rewritesPerRound = 4;
    p->lookahead = 20;
    p->rewriteMargin = 0.01f;
    p->views = 8;
    p->viewPixels = 72;
    p->fitLook = 1;
    p->depthWeight = 0.5f;
    p->softness = 0.75f;
    p->surfaceWeight = 1.0f;
    p->collisionWeight = 1.0f;
    p->collisionRamp = 10.0f;
    p->checksPerHinge = 4;
    p->hingeWeight = 1.0f;
    p->hingeLength = 0.3f;
    p->dependencyWeight = 0.1f;
    p->onePieceWeight = 5.0f;
    p->reorderEvery = 60;
    p->wSplit = 1.0f;
    p->wMerge = 0.4f;
    p->wJoin = 0.4f;
    p->wCarve = 0.4f;
    p->wUncarve = 0.2f;
    p->mergeAngleDeg = 4.0f;
    p->joinShare = 0.02f;
    p->semantic = false;
    p->semanticEvery = 150;
    p->semanticIterations = 6;
    p->arapVertices = 1500;
    p->semanticStep = 0.5f;
    p->maxStrain = 0.12f;
    p->maxVolumeChange = 0.08f;
    p->postLedges = true;
    p->postCarve = true;
    p->postLatch = true;
    p->postHinges = true;
    p->postFragments = true;
    p->postFeatures = false;
    p->featureResolution = 192;
    p->featureReachMm = 3.0f;
    p->featureFlatten = 0.75f;
    p->featureSink = 0.5f;
    p->featureSinkMm = 1.5f;
    p->clearanceMm = 0.4f;
    p->knuckleRadiusMm = 2.0f;
    p->pinRadiusMm = 0.9f;
    p->ledgeMm = 2.0f;
    p->carveChecks = 24;
    p->carveResolution = 96;
    p->stlResolution = 220;
    p->stlPose = 0;
    snprintf(p->outDir, sizeof(p->outDir), "foldball_out");
    p->statusSeconds = 2.0f;
    p->suiteFrames = 120;
    p->suiteFoldFrames = 72;
    p->suiteTile = 360;
    p->suiteFps = 30;
    p->suitePost = false;
    p->suiteGhost = true;
    p->snapshotSeconds = 30.0f;
}

#define F(sec, key, type, field, lo, hi, help) { sec, key, type, offsetof(Params, field), lo, hi, help }
static const ParamInfo kTable[] = {
    F("model", "size_mm", PT_FLOAT, sizeMm, 5, 10000, "The sphere's diameter, mm (the hinges' and clearances' mm are measured against it)."),
    F("model", "max_bodies", PT_INT, maxBodies, 1, 64, "No more bodies than this."),
    F("model", "max_hollows", PT_INT, maxHollows, 0, 64, "No more hollows (capsules carved from the sphere) than this."),
    F("model", "max_bend", PT_FLOAT, maxBend, 0, 4, "How far cuts may bend: 0 flat cuts only; otherwise each cut may curve into a piece of a sphere as tight as 1/max_bend of the ball's radius (1: as round as the ball itself), and how much each one bends is optimized (every cut starts flat)."),
    F("model", "samples", PT_INT, samples, 12, 80, "Lattice points across the sphere for collisions, holding and one piece."),
    F("optimizer", "steps", PT_INT, steps, 0, 10000000, "Adam steps (rewrite lookaheads count too)."),
    F("optimizer", "seed", PT_UINT, seed, 0, 4e9f, "Random seed."),
    F("optimizer", "threads", PT_INT, threads, 0, 1024, "Worker threads (0: every core)."),
    F("optimizer", "lr_bend", PT_FLOAT, lrBend, 0, 1, "Cut bend step (curvature, per sphere radius) when model.max_bend allows bent cuts."),
    F("optimizer", "lr_plane", PT_FLOAT, lrPlane, 0, 1, "Cut plane step (radians / sphere radii)."),
    F("optimizer", "lr_phi", PT_FLOAT, lrPhi, 0, 1, "Hinge position along its seam, step (radians)."),
    F("optimizer", "lr_angle", PT_FLOAT, lrAngle, 0, 1, "Hinge opening angle step (radians)."),
    F("optimizer", "lr_hollow", PT_FLOAT, lrHollow, 0, 1, "Hollow step (sphere radii)."),
    F("optimizer", "lr_place", PT_FLOAT, lrPlace, 0, 1, "Where the open pose sits against the mesh (its scale and offset), step."),
    F("optimizer", "steps_per_round", PT_INT, stepsPerRound, 1, 100000, "Adam steps between rewrite rounds."),
    F("optimizer", "rewrites_per_round", PT_INT, rewritesPerRound, 0, 32, "Rewrites tried each round (0: no grammar)."),
    F("optimizer", "lookahead", PT_INT, lookahead, 0, 1000, "Adam steps each try gets before they are compared."),
    F("optimizer", "rewrite_margin", PT_FLOAT, rewriteMargin, 0, 1, "A rewrite is kept when it is no worse than leaving the design alone by more than this share."),
    F("loss", "views", PT_INT, views, 1, 32, "Views of the open pose."),
    F("loss", "view_pixels", PT_INT, viewPixels, 16, 512, "Pixels across each view."),
    F("loss", "fit_look", PT_INT, fitLook, 0, 2, "How each view of the open pose is compared with the mesh's: 0 silhouettes (outlines only); 1 silhouettes, plus how far apart the front surfaces are where both cover a pixel (depth_weight); 2 depth rasters: each view as a picture of nearness (nearer brighter, nothing black), compared pixel by pixel, so outline and depth are one comparison."),
    F("loss", "depth_weight", PT_FLOAT, depthWeight, 0, 10, "fit_look 1: the depth comparison's weight on top of the silhouettes (3D shape, not just outline)."),
    F("loss", "softness", PT_FLOAT, softness, 0.1f, 5, "Soft edge of the silhouettes, pixels."),
    F("loss", "surface_weight", PT_FLOAT, surfaceWeight, 0, 1000, "Hollows carving into the sphere's surface (it should stay a sphere to roll)."),
    F("constraints", "collision_weight", PT_FLOAT, collisionWeight, 0, 1000, "Bodies overlapping, open and along the motion."),
    F("constraints", "collision_ramp", PT_FLOAT, collisionRamp, 1, 1000, "The collision weight grows to this many times itself over the run."),
    F("constraints", "checks_per_hinge", PT_INT, checksPerHinge, 1, 64, "Poses checked in each hinge's turn (a fixed number per turn: a faster turn doesn't escape them)."),
    F("constraints", "hinge_weight", PT_FLOAT, hingeWeight, 0, 1000, "A hinge's barrel must have both of its bodies right at it (where along its seam both are)."),
    F("constraints", "hinge_length", PT_FLOAT, hingeLength, 0.05f, 2, "Hinge barrel length, as a share of the sphere's radius."),
    F("constraints", "dependency_weight", PT_FLOAT, dependencyWeight, 0, 1000, "Every hinge but the root next to a piece opening before it (so a ledge can hold it)."),
    F("constraints", "one_piece_weight", PT_FLOAT, onePieceWeight, 0, 1000, "Material of a body apart from the rest of it."),
    F("constraints", "reorder_every", PT_INT, reorderEvery, 0, 100000, "Re-sort the opening order (least colliding first) this often, if it helps."),
    F("grammar", "split", PT_FLOAT, wSplit, 0, 100, "Split: a body cut in two by a plane, a hinge on the seam at angle 0."),
    F("grammar", "merge", PT_FLOAT, wMerge, 0, 100, "Merge: a hinge that hardly turns is removed, its two bodies made one."),
    F("grammar", "join", PT_FLOAT, wJoin, 0, 100, "Join: a negligible body becomes part of its parent; what hung from it is re-hung there, turned as nearly as before."),
    F("grammar", "carve", PT_FLOAT, wCarve, 0, 100, "Carve: a hollow starts (at nothing) where the open pose shows material the mesh doesn't have."),
    F("grammar", "uncarve", PT_FLOAT, wUncarve, 0, 100, "Uncarve: a hollow is removed."),
    F("grammar", "merge_angle_deg", PT_FLOAT, mergeAngleDeg, 0, 180, "A hinge turning less than this may be merged."),
    F("grammar", "join_share", PT_FLOAT, joinShare, 0, 1, "A body with less than this share of the volume may be joined into its parent."),
    F("semantic", "enabled", PT_BOOL, semantic, 0, 1, "Every few rounds, deform the input pose (ARAP) toward the design, kept when it helps and stays the same shape."),
    F("semantic", "every", PT_INT, semanticEvery, 1, 1000000, "Steps between semantic deformations."),
    F("semantic", "iterations", PT_INT, semanticIterations, 1, 100, "ARAP rounds per deformation."),
    F("semantic", "arap_vertices", PT_INT, arapVertices, 100, 20000, "The input is simplified to about this many vertices for ARAP."),
    F("semantic", "step", PT_FLOAT, semanticStep, 0.01f, 1, "How far toward the design each round pulls the input."),
    F("semantic", "max_strain", PT_FLOAT, maxStrain, 0, 1, "Most local stretch (not explained by rotation) a deformation may have."),
    F("semantic", "max_volume_change", PT_FLOAT, maxVolumeChange, 0, 1, "Most volume change a deformation may have."),
    F("post", "ledges", PT_BOOL, postLedges, 0, 1, "Add ledges so every hinge is held by one that opens before it."),
    F("post", "carve", PT_BOOL, postCarve, 0, 1, "Carve leftover collisions away (smoothly)."),
    F("post", "latch", PT_BOOL, postLatch, 0, 1, "A press latch on the root hinge, flush with the sphere."),
    F("post", "hinges", PT_BOOL, postHinges, 0, 1, "Build the hinges (knuckles and a pin)."),
    F("post", "fragments", PT_BOOL, postFragments, 0, 1, "Remove any part of a body apart from the rest of it."),
    F("post", "features", PT_BOOL, postFeatures, 0, 1, "Restore the input's features onto the parts it opens into (where they don't collide; see the feature_ settings)."),
    F("post", "feature_resolution", PT_INT, featureResolution, 32, 1024, "Feature restoration: grid points across the sphere (finer features, more memory and time)."),
    F("post", "feature_reach_mm", PT_FLOAT, featureReachMm, 0.1f, 50, "Feature restoration: how far (mm) the restored surface may lie from the part's, in or out; farther apart, the part is left as it is."),
    F("post", "feature_flatten", PT_FLOAT, featureFlatten, 0, 1, "Feature restoration: where a feature would stick out into another part (or past the sphere), how much it is flattened (0: full height, 1: flat)."),
    F("post", "feature_sink", PT_FLOAT, featureSink, 0, 1, "Feature restoration: what still sticks out after flattening: 0 squared off flush, 1 sunk whole into a recess (so the sphere keeps the feature's shape as a texture)."),
    F("post", "feature_sink_mm", PT_FLOAT, featureSinkMm, 0, 20, "Feature restoration: how far around a sunk feature the recess reaches, mm (and the flattened zone around a feature that doesn't fit)."),
    F("post", "clearance_mm", PT_FLOAT, clearanceMm, 0, 5, "Gap between parts that move against each other, mm."),
    F("post", "knuckle_radius_mm", PT_FLOAT, knuckleRadiusMm, 0.3f, 20, "Hinge knuckle radius, mm: also how deep under the surface the hinges sit."),
    F("post", "pin_radius_mm", PT_FLOAT, pinRadiusMm, 0.2f, 10, "Hinge pin radius, mm."),
    F("post", "ledge_mm", PT_FLOAT, ledgeMm, 0.3f, 20, "Ledge size, mm."),
    F("post", "carve_checks", PT_INT, carveChecks, 2, 256, "Poses per hinge turn the carving looks at."),
    F("post", "carve_resolution", PT_INT, carveResolution, 32, 256, "Grid points across the sphere for the carving."),
    F("export", "stl_resolution", PT_INT, stlResolution, 32, 1024, "Voxels across the sphere when meshing for STL."),
    F("export", "stl_pose", PT_INT, stlPose, 0, 1, "0: the open pose, 1: closed."),
    F("output", "dir", PT_STRING, outDir, 0, 0, "Where runs write their results."),
    F("output", "status_seconds", PT_FLOAT, statusSeconds, 0, 3600, "Status file and line this often."),
    F("output", "snapshot_seconds", PT_FLOAT, snapshotSeconds, 0, 86400, "Best design so far saved this often (0: never)."),
    F("suite", "frames", PT_INT, suiteFrames, 2, 100000, "Test suite: video frames of each mesh's optimization (the best design so far, at evenly spaced steps)."),
    F("suite", "fold_frames", PT_INT, suiteFoldFrames, 0, 100000, "Test suite: then frames of the result folding shut and opening again (0: none)."),
    F("suite", "tile", PT_INT, suiteTile, 64, 2048, "Test suite: pixels across each mesh's square in the video."),
    F("suite", "fps", PT_INT, suiteFps, 1, 120, "Test suite: the video's frames per second."),
    F("suite", "post", PT_BOOL, suitePost, 0, 1, "Test suite: post process each result (as [post] says) before it folds in the video, and save it built."),
    F("suite", "ghost", PT_BOOL, suiteGhost, 0, 1, "Test suite: show the target mesh faintly over the design in the video."),
};
#undef F

const ParamInfo* params_table(int* count) {
    *count = (int)(sizeof(kTable) / sizeof(kTable[0]));
    return kTable;
}

static char* trim(char* s) {
    while (*s && isspace((unsigned char)*s)) s++;
    char* e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) *--e = 0;
    return s;
}

static bool set_one(Params* p, const char* section, const char* key, const char* value, char* err, int errSize) {
    int n;
    const ParamInfo* t = params_table(&n);
    for (int k = 0; k < n; k++) {
        if (strcmp(t[k].key, key) != 0 || (section && strcmp(t[k].section, section) != 0)) continue;
        char* base = (char*)p + t[k].offset;
        char* end;
        switch (t[k].type) {
        case PT_INT: {
            long v = strtol(value, &end, 10);
            if (*trim(end) || v < t[k].lo || v > t[k].hi) goto bad;
            *(int*)base = (int)v;
            return true;
        }
        case PT_UINT: {
            unsigned long v = strtoul(value, &end, 10);
            if (*trim(end)) goto bad;
            *(unsigned*)base = (unsigned)v;
            return true;
        }
        case PT_FLOAT: {
            float v = strtof(value, &end);
            if (*trim(end) || v < t[k].lo || v > t[k].hi) goto bad;
            *(float*)base = v;
            return true;
        }
        case PT_BOOL:
            if (!strcmp(value, "true") || !strcmp(value, "1") || !strcmp(value, "yes") || !strcmp(value, "on")) *(bool*)base = true;
            else if (!strcmp(value, "false") || !strcmp(value, "0") || !strcmp(value, "no") || !strcmp(value, "off")) *(bool*)base = false;
            else goto bad;
            return true;
        case PT_STRING:
            snprintf(base, 512, "%s", value);
            return true;
        }
    bad:
        if (t[k].type == PT_STRING || t[k].type == PT_BOOL || t[k].type == PT_UINT)
            snprintf(err, (size_t)errSize, "%s.%s: \"%s\" isn't a valid value", t[k].section, key, value);
        else
            snprintf(err, (size_t)errSize, "%s.%s: \"%s\" isn't a number in %g..%g", t[k].section, key, value, t[k].lo, t[k].hi);
        return false;
    }
    snprintf(err, (size_t)errSize, "no setting %s%s%s", section ? section : "", section ? "." : "", key);
    return false;
}

bool params_set(Params* p, const char* assignment, char* err, int errSize) {
    char* s = fb_strdup(assignment);
    char* eq = strchr(s, '=');
    if (!eq) {
        snprintf(err, (size_t)errSize, "\"%s\": expected key=value", assignment);
        free(s);
        return false;
    }
    *eq = 0;
    char* key = trim(s);
    char* value = trim(eq + 1);
    char* dot = strchr(key, '.');
    bool ok;
    if (dot) {
        *dot = 0;
        ok = set_one(p, key, dot + 1, value, err, errSize);
    } else {
        ok = set_one(p, NULL, key, value, err, errSize);
    }
    free(s);
    return ok;
}

bool params_load(Params* p, const char* path, char* err, int errSize) {
    char* text = fb_read_file(path, NULL);
    if (!text) { snprintf(err, (size_t)errSize, "can't read %s", path); return false; }
    char section[64] = "";
    int lineNo = 0;
    bool ok = true;
    for (char* line = strtok(text, "\n"); line && ok; line = strtok(NULL, "\n")) {
        lineNo++;
        char* l = trim(line);
        if (!*l || *l == '#' || *l == ';') continue;
        if (*l == '[') {
            char* e = strchr(l, ']');
            if (!e) { snprintf(err, (size_t)errSize, "%s:%d: no ] closes the section", path, lineNo); ok = false; break; }
            *e = 0;
            snprintf(section, sizeof(section), "%s", trim(l + 1));
            continue;
        }
        char* eq = strchr(l, '=');
        if (!eq) { snprintf(err, (size_t)errSize, "%s:%d: expected key = value", path, lineNo); ok = false; break; }
        *eq = 0;
        char msg[256];
        if (!set_one(p, section[0] ? section : NULL, trim(l), trim(eq + 1), msg, sizeof(msg))) {
            snprintf(err, (size_t)errSize, "%s:%d: %s", path, lineNo, msg);
            ok = false;
        }
    }
    free(text);
    return ok;
}

bool params_save(const Params* p, const char* path) {
    FILE* f = fopen(path, "w");
    if (!f) return false;
    fprintf(f, "# foldball settings. Lines starting with # are comments; a missing line keeps the default.\n");
    int n;
    const ParamInfo* t = params_table(&n);
    const char* section = "";
    for (int k = 0; k < n; k++) {
        if (strcmp(section, t[k].section) != 0) {
            section = t[k].section;
            fprintf(f, "\n[%s]\n", section);
        }
        const char* base = (const char*)p + t[k].offset;
        fprintf(f, "# %s\n", t[k].help);
        switch (t[k].type) {
        case PT_INT: fprintf(f, "%s = %d\n", t[k].key, *(const int*)base); break;
        case PT_UINT: fprintf(f, "%s = %u\n", t[k].key, *(const unsigned*)base); break;
        case PT_FLOAT: fprintf(f, "%s = %g\n", t[k].key, *(const float*)base); break;
        case PT_BOOL: fprintf(f, "%s = %s\n", t[k].key, *(const bool*)base ? "true" : "false"); break;
        case PT_STRING: fprintf(f, "%s = %s\n", t[k].key, base); break;
        }
    }
    return fclose(f) == 0;
}
