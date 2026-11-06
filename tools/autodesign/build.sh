#!/bin/sh
# Headless builds of the generator, without the editor's GUI: they need only raylib's headers.
#   RAYLIB_SRC=/path/to/raylib/src ./build.sh
#   ./autotest mesh.obj [steps] [sphereBias] [seed] [outPrefix]     a design, and its silhouettes as a picture
#   ./suitetest <folder> [steps] [videoSize] [searchSeconds]         the Generate tab's test suite (needs ffmpeg)
#   ./gradcheck mesh.obj      the loss's gradient against brute force differences of the whole loss
#   ./adamtest                Adam's memory as parameters come and go
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
