# foldaball on the command line

The generator and the test suite without the editor: no window, no GL context, no GPU. Everything the Generate tab's
sliders set is a line in a settings file.

```
foldaball optimize bunny.obj -c settings.ini -o results/bunny     # one mesh (the Generate button)
foldaball suite meshes/ -c settings.ini -o results/suite           # every .obj in a folder
foldaball score results/bunny/bunny.txt bunny.obj                  # score a saved design, no search
foldaball config > settings.ini                                    # every setting, with its value and a comment
```

## Building

The CLI is two new files (`cli/main_cli.cpp`, `cli/config.cpp`) plus everything the editor's generator already uses.
It needs **none** of `ui/`, `core/`, ImGui, ImGuizmo, rlImGui or NFD. Add a target next to the editor's; adjust the
target names to however your build finds raylib, TBB and OpenVDB:

```cmake
file(GLOB_RECURSE FOLDABALL_ENGINE src/autodesign/*.cpp src/data/*.cpp src/util/*.cpp)
add_executable(foldaball-cli src/cli/main_cli.cpp src/cli/config.cpp ${FOLDABALL_ENGINE})
target_include_directories(foldaball-cli PRIVATE src src/vendor/onnxruntime)
target_compile_features(foldaball-cli PRIVATE cxx_std_17)
target_link_libraries(foldaball-cli PRIVATE raylib TBB::tbb OpenVDB::openvdb Threads::Threads ${CMAKE_DL_LIBS})
```

raylib is linked because `util/utils.cpp` uses its model helpers, but nothing here opens a window. If linking raylib on the
server asks for X11/GL development packages, those are only needed to link, not to run.

## Settings

An INI file; `#` or `;` start a comment. Every key is also a command line option (`-s section.key=value`), and the command
line beats the files, which beat the editor's defaults. Several `-c` files may be given; later ones win.

```ini
[optimizer]
steps = 6000
seed = 7

[goal]
views = 8
dependencies = true
```

`foldaball config` prints **every** setting with its current value, a comment and its allowed range
(`foldaball.example.ini` is that output; `server.example.ini` is a short realistic one). `foldaball config -c mine.ini` shows
what a file ends up meaning. Mistakes are named by file and line, with a suggestion:

```
settings: mine.ini:4: unknown setting 'goal.viewz' (did you mean 'goal.views'?)
settings: goal.views: must be between 1 and 64, got 0
```

| section | what | the editor's |
|---|---|---|
| `[input]` | `mesh`, `mesh_size` (centre and scale to this many mm; 0 = as the file is) | Model tab |
| `[goal]` | views, collisions, hinges, dependencies, surface, mesh move/scale | Goal |
| `[start]` | `sphere_diameter` | Start from |
| `[semantic]` | DINOv3 loss, silhouette weight | Semantic loss |
| `[optimizer]` | steps, learning rate, rounds, lookahead, limits, seed | Optimizer |
| `[grammar]` | one weight per rewrite rule | Grammar |
| `[finishing]` | hinges, carving, latch | Finishing |
| `[suite]` | folder, mesh size, video | the suite's panel |
| `[output]` | where results go, which files | new |
| `[runtime]` | threads, progress interval | new |

Relative paths are relative to where you run foldaball. The suite uses the same `[goal]`...`[finishing]` settings as
`optimize`, as the editor's suite uses the Generate tab's.

## Results

`optimize` writes to `-o DIR` (default `./foldaball_out`), each named after the mesh (`<name>`):

| file | what |
|---|---|
| `<name>.txt` | the design: open it in the editor |
| `<name>.png` | the result open, a third and two thirds closed, and closed |
| `<name>.report.txt` | the message and score in words |
| `<name>.result.json` | the same, for scripts |
| `<name>.history.csv` | loss at every step |
| `<name>.views.png` | the silhouettes scored last, against the placed mesh's |
| `<name>.config.ini` | the settings that were used: running with this file repeats the run |
| `<name>.partial.txt` | the best so far, saved every `checkpoint_seconds`; removed when the run finishes |
| `<name>_<pose>_<part>.stl` | with `output.stl = true` |

`suite` writes to `-o DIR` (default `<folder>/autodesign_suite`): per mesh `<name>.txt`, `.png`, `.report.txt`, plus
`summary.csv` (one row per mesh, written as each finishes), `suite.config.ini` and, with `suite.video = true`, `suite.mp4`.
**ffmpeg is only needed for the video**: `--no-video` (or `suite.video = false`) skips it, and so the check for it.

## Running on a server

- **Stopping.** Ctrl+C or SIGTERM (a scheduler's time limit) stops the search at its next step and writes everything for the
  best design found so far (exit status 130). A second signal quits at once. `checkpoint_seconds` bounds what a hard kill loses.
- **Progress** goes to stderr every `progress_seconds`; results and the final summary to stdout. `-q` silences progress.
- **Exit status**: 0 done, 1 failed, 2 bad usage or settings, 130 stopped early.
- **Threads**: `-j N` limits the search. (The suite's picture rendering uses every core regardless.)
- **No GPU**: silhouettes are always traced on the CPU (the GPU tracer is the editor's GL context). `semantic.semantic_gpu`
  stays off unless ONNX Runtime has CUDA.
- Same seed and settings give the same result.

## What changed in the existing code

Only the suite, and only additively: the editor behaves as before (`SuiteParams` defaults are the old behaviour).

- `suite.hpp/.cpp`: `SuiteParams::outDir`, `video`, `preview`, `previewSize`; with `video = false` ffmpeg is not needed and no
  frames are recorded; per mesh a `.report.txt` and `.png`, and a `summary.csv`; new `NormalizeMesh`, `WritePng` and
  `WritePreviewPng` (the last draws with the suite's own software rasterizer).
