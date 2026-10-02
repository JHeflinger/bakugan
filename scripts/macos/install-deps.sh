#!/usr/bin/env bash
# Installs the system libraries this project links against via Homebrew.
set -euo pipefail

if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required: https://brew.sh" >&2
    exit 1
fi

# Compiler toolchain (provides clang, which is what `g++` resolves to on macOS)
xcode-select -p >/dev/null 2>&1 || xcode-select --install

# openvdb pulls in boost, blosc and jemalloc itself
brew install glfw openvdb tbb imath
