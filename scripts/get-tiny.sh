#!/usr/bin/env bash
# Builds the tiny build tool into .tools/ (tiny is a single C file).
#   TINY_REF  git ref to build (default: main) - pin a tag/sha for reproducible CI
#   TINY_DIR  where to put it (default: .tools)
set -euo pipefail

TINY_REF="${TINY_REF:-main}"
DEST="${TINY_DIR:-.tools}"
SRC="$DEST/tiny-src"

mkdir -p "$DEST"
if [ ! -d "$SRC/.git" ]; then
    git init --quiet "$SRC"
    git -C "$SRC" remote add origin https://github.com/JHeflinger/tiny.git
fi
git -C "$SRC" fetch --quiet --depth 1 origin "$TINY_REF"
git -C "$SRC" checkout --quiet --force FETCH_HEAD

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) OUT="tiny.exe"; THREADS="" ;;   # no -pthread on Windows
    *)                    OUT="tiny";     THREADS="-pthread" ;;
esac

gcc "$SRC/tiny.c" -O2 $THREADS -o "$DEST/$OUT"
echo "Built $DEST/$OUT from tiny @ $(git -C "$SRC" rev-parse --short HEAD)"
