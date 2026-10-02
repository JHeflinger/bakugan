#!/usr/bin/env bash
# Installs the system libraries this project links against (GLFW, OpenGL, OpenVDB).
# Supports Debian/Ubuntu (apt) and Arch (pacman).
set -euo pipefail

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

if command -v apt-get >/dev/null 2>&1; then
    $SUDO apt-get update
    $SUDO apt-get install -y --no-install-recommends \
        build-essential git \
        libglfw3-dev libgl1-mesa-dev \
        libopenvdb-dev libtbb-dev libimath-dev
elif command -v pacman >/dev/null 2>&1; then
    $SUDO pacman -S --needed --noconfirm \
        base-devel git \
        glfw mesa \
        openvdb onetbb imath
else
    echo "Unsupported distro: install g++, git, GLFW, OpenGL, OpenVDB, oneTBB and Imath dev packages manually." >&2
    exit 1
fi
