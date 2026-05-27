#!/bin/bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Configuration
LLAMA_DIR="${SCRIPT_DIR}/../llama.cpp"
TARGET="llama-server"
BUILD_JOBS=$(nproc)

# Change to llama.cpp directory
echo "Building llama.cpp from: $LLAMA_DIR"
cd "$LLAMA_DIR"

# Configure with CUDA support
echo "Configuring with CMake..."
cmake -B build \
  -G Ninja \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_UI=OFF \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release

# Build llama-server
echo "Building $TARGET with $BUILD_JOBS cores..."
cmake --build build -j$BUILD_JOBS --target $TARGET

echo "Build complete! Binary at: build/bin/llama-server"