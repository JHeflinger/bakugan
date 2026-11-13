#!/bin/sh
# Headless builds of the generator, without the editor's GUI: they need only raylib's headers.
#   RAYLIB_SRC=/path/to/raylib/src ./build.sh
#   ./autotest mesh.obj [steps] [seed] [outPrefix]     a carving, and its silhouettes as a picture
#   ./suitetest <folder> [steps] [videoSize] [searchSeconds]         the Generate tab's test suite (needs ffmpeg)
#   ./gradcheck mesh.obj      the loss's gradient against brute force differences of the whole loss
#   ./adamtest                Adam's memory as parameters come and go
#   ./checktimes              where the collision checks between open and closed go, for a few hinge timings
#   ./rulecheck mesh.obj      adding a cut changes nothing; closed, the pieces are always the sphere
# The semantic loss (DINOv3) is on in autotest and gradcheck with SEM=model.onnx ORT=path/to/libonnxruntime (see
# semantic_env.hpp); export_dinov3.py makes the model. Linking needs nothing more: ONNX Runtime is loaded at run time
# (on glibc older than 2.34, add -ldl).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
SRC="$HERE/../../src"
: "${RAYLIB_SRC:?set RAYLIB_SRC to raylib's src folder}"
FLAGS="-std=c++17 -O2 -Wall -Wextra -Wno-missing-field-initializers -I$SRC -I$RAYLIB_SRC"
CODE="$SRC/autodesign/*.cpp $SRC/data/design.cpp $SRC/data/field.cpp $SRC/data/superfrustum.cpp $SRC/data/templates.cpp $HERE/stubs.cpp"
g++ $FLAGS "$HERE/test.cpp" $CODE -lpthread -o "$HERE/autotest"
g++ $FLAGS "$HERE/suitetest.cpp" $CODE -lpthread -o "$HERE/suitetest"
g++ $FLAGS "$HERE/gradcheck.cpp" $CODE -lpthread -o "$HERE/gradcheck"
g++ $FLAGS "$HERE/adamtest.cpp" -o "$HERE/adamtest"
g++ $FLAGS "$HERE/checktimes.cpp" $CODE -lpthread -o "$HERE/checktimes"
g++ $FLAGS "$HERE/rulecheck.cpp" $CODE -lpthread -o "$HERE/rulecheck"
