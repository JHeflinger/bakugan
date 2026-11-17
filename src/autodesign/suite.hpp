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
        AutoParams generate;         // the generator settings (every mesh starts from a sphere)
        std::string folder;          // where the .obj files are
        std::string outDir;          // where the results go (empty: <folder>/autodesign_suite/)
        bool video = true;           // render the video (needs ffmpeg); off: only the designs, pictures and summary are written
        bool preview = true;         // a still picture per mesh: the result open, closing and closed (<name>.png)
        int previewSize = 360;       // pixels across each pose of that picture
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
    // Centred, its longest side `size` mm (0: as it is); fills in the mesh's box.
    void NormalizeMesh(TriMesh& m, float size);
    // A png from 8 bit pixels (3: RGB, 4: RGBA), rows from the top. No window or GL context needed.
    bool WritePng(const std::string& path, const std::vector<unsigned char>& pixels, int width, int height, int channels,
                  std::string* error);
    // One still picture of a design: open, a third and two thirds of the way closed, and closed, from one fixed camera,
    // over the mesh it was fitted to (already placed where it was fitted: the caller applies AutoScore::target).
    bool WritePreviewPng(const Design& design, const TriMesh& placedMesh, int cellSize, const std::string& path, std::string* error);
    bool LoadObj(const std::string& path, TriMesh* out, std::string* error);
    bool FfmpegAvailable();
    SuiteResult RunSuite(const SuiteParams& p, JobControl& c, std::shared_ptr<SuiteLive> live);
}
