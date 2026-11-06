#!/bin/sh
# Headless build of the generator for tuning, without the GUI: needs only raylib's headers (raylib.h, raymath.h).
#   RAYLIB_SRC=/path/to/raylib/src ./build.sh
#   ./autotest mesh.obj [iterations] [sphereBias] [quality] [seed] [outPrefix] [pieces] [maxBodies] [template] [cuts|features|maxShapes]
#   python3 render.py <outPrefix> mesh.obj out.png [cx cy cz r]     (needs numpy, scikit-image, matplotlib)
#   ./suitetest <folder> [iterations] [videoSize] [searchSeconds]   the Generate tab's test suite (needs ffmpeg)
# Writes <outPrefix>_design.txt (open it in the editor) and the open / closed fields for render.py.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
SRC="$HERE/../../src"
: "${RAYLIB_SRC:?set RAYLIB_SRC to raylib's src folder}"
g++ -std=c++17 -O2 -Wall -Wextra -Wno-missing-field-initializers -I"$SRC" -I"$RAYLIB_SRC" \
  "$HERE/test.cpp" "$HERE/stubs.cpp" "$SRC/data/autodesign.cpp" "$SRC/data/design.cpp" "$SRC/data/field.cpp" \
  "$SRC/data/superfrustum.cpp" "$SRC/data/templates.cpp" -lpthread -o "$HERE/autotest"
g++ -std=c++17 -O2 -Wall -Wextra -Wno-missing-field-initializers -I"$SRC" -I"$RAYLIB_SRC" \
  "$HERE/suitetest.cpp" "$HERE/stubs.cpp" "$SRC/data/suite.cpp" "$SRC/data/autodesign.cpp" "$SRC/data/design.cpp" \
  "$SRC/data/field.cpp" "$SRC/data/superfrustum.cpp" "$SRC/data/templates.cpp" -lpthread -o "$HERE/suitetest"
