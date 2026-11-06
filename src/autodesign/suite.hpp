#pragma once
#include "autodesign/autodesign.hpp"
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace Bakugan {
    // Test suite: runs the generator on every .obj in a folder and renders one square video of all of them side by
    // side, each in its own labelled box: the best design over the whole search, then the result closing and opening.
    struct SuiteParams {
        AutoParams generate;         // the generator settings (fromCurrent is ignored: every mesh starts from scratch)
        std::string folder;          // where the .obj files are; results go to <folder>/autodesign_suite/
        float meshSize = 80.0f;      // mm: each mesh is centred and scaled so its longest side is this (0: as is)
        int videoSize = 1080;        // pixels, square
        int fps = 30;
        float searchSeconds = 8.0f;  // video time for the search (one frame per evenly spaced snapshot)
        float motionSeconds = 2.0f;  // closing, and again opening
        float holdSeconds = 0.6f;    // pauses on the result, closed, and open again
    };

    // Shared with the UI while a suite runs.
    struct SuiteLive {
        std::mutex mutex;
        int index = 0;               // mesh being searched (0 based)
        int count = 0;
        std::string name;
        std::string phase;           // "Searching", "Rendering", ...
        float progress = 0.0f;       // of the phase
        std::shared_ptr<AutoLive> search;    // the search running now
    };

    struct SuiteResult {
        bool ok = false;
        bool cancelled = false;
        std::string message;
        std::string video;           // the mp4
        std::string folder;          // where the video and the designs went
    };

    std::vector<std::string> SuiteMeshes(const std::string& folder);     // the .obj files in a folder, sorted
    bool LoadObj(const std::string& path, TriMesh* out, std::string* error);
    bool FfmpegAvailable();
    SuiteResult RunSuite(const SuiteParams& p, JobControl& c, std::shared_ptr<SuiteLive> live);
}
