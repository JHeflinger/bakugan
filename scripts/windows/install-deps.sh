#!/usr/bin/env bash
# Installs the MinGW-w64 libraries this project links against.
# Run it from an "MSYS2 MinGW x64" shell (MSYSTEM=MINGW64), not plain MSYS2.
set -euo pipefail

if [ "${MSYSTEM:-}" != "MINGW64" ]; then
    echo "Run this from an MSYS2 MINGW64 shell (current MSYSTEM: '${MSYSTEM:-none}')." >&2
    exit 1
fi

# git: tiny uses it to fetch modules
pacman -S --needed --noconfirm \
    git \
    mingw-w64-x86_64-gcc \
    mingw-w64-x86_64-glfw \
    mingw-w64-x86_64-openvdb \
    mingw-w64-x86_64-tbb \
    mingw-w64-x86_64-imath
