#pragma once
// The command line's settings: every parameter the Generate tab's sliders set, plus the test suite's, the input mesh and
// where the results go, in one INI style file (see foldaball.example.ini, which `foldaball config` writes).
//
//   [optimizer]
//   steps = 3000          # comments start with # (or ;)
//   seed  = 7
//
// A key can also be given whole, anywhere (`optimizer.steps = 3000`), and on the command line (`-s optimizer.steps=3000`).
// Several files may be loaded; later ones (and then the command line) win. Anything not set keeps the editor's default.
#include "autodesign/autodesign.hpp"
#include "autodesign/suite.hpp"
#include "data/build.hpp"
#include <functional>
#include <string>
#include <vector>

namespace Bakugan::Cli {
    struct InputParams {
        std::string mesh;                // the .obj to fit (optimize, score)
        float meshSize = 0.0f;           // mm: centre it and scale its longest side to this (0: as the file has it)
        std::string startDesign;         // optimize: a saved design (.txt) to carry the search on from (empty: the sphere)
    };

    struct OutputParams {
        std::string dir;                 // where results go (empty: ./foldaball_out, or <folder>/autodesign_suite for a suite)
        std::string name;                // file name stem (empty: the mesh's)
        bool preview = true;             // <name>.png: the result open, closing and closed
        int previewSize = 360;           // pixels across each pose in it
        bool viewsImage = true;          // <name>.views.png: the silhouettes scored last, against the mesh's
        bool history = true;             // <name>.history.csv: the loss over the run
        int checkpointSeconds = 60;      // <name>.partial.txt: the best design so far, this often (0: never)
        bool stl = false;                // also export the result as STL parts (needs OpenVDB, as the editor's export does)
        ExportParams stlParams;          // ... how (pose, voxel size, ...)
    };

    struct RuntimeParams {
        int threads = 0;                 // worker threads (0: every core)
        int progressSeconds = 10;        // a progress line this often (0: none)
        bool gpu = false;                // draw the views and measure the gradient on the GPU (an OpenGL context of its own)
    };

    // One setting: where it lives in the file and how to read, write and describe it.
    struct Option {
        std::string section, name;
        std::string help;                // for the commented dump
        std::string range;               // e.g. "1..64" or "fused | separate"
        std::function<bool(const std::string&, std::string*)> set;     // false and a message when the value is no good
        std::function<std::string()> get;                               // the current value, as it would be written
        std::string Key() const { return section + "." + name; }
    };

    // All the settings. Not copyable: the options point into it.
    class Config {
    public:
        Config();
        Config(const Config&) = delete;
        Config& operator=(const Config&) = delete;

        InputParams input;
        AutoParams generate;             // the generator (and the suite's: every mesh of a suite uses these)
        SuiteParams suite;               // the suite's own (its .generate is filled from `generate` when it runs)
        OutputParams output;
        RuntimeParams runtime;

        bool LoadFile(const std::string& path, std::string* error);
        // key: "section.name" (case, '-' and '_' don't matter)
        bool Set(const std::string& key, const std::string& value, std::string* error);
        // Settings that can't work together, found before hours of search go into them.
        bool Validate(std::string* error) const;
        // The whole file: every setting with its current value, a comment and its range.
        std::string Dump() const;
        const std::vector<Option>& Options() const { return m_Options; }

    private:
        std::vector<Option> m_Options;
        const Option* Find(const std::string& key) const;
        std::string Suggest(const std::string& key) const;
        void Build();
    };
}
