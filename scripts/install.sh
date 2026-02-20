#!/usr/bin/env bash
set -euo pipefail

error_exit() {
    echo "$1" >&2
    exit 1
}

if ! command -v python &> /dev/null; then
    error_exit "python command not found. Please run inside pixi shell: pixi shell"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALL_DIR="$REPO_ROOT/libs/install"

export CC=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc
export CXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++

mkdir -p "$INSTALL_DIR"

# ---- DART ----------------------------------------------------------------
DART_SRC="$REPO_ROOT/libs/dart"
if [ ! -f "$DART_SRC/CMakeLists.txt" ]; then
    error_exit "libs/dart not found. Run: git submodule update --init --recursive"
fi

echo "==== Building DART from $DART_SRC ===="
mkdir -p "$DART_SRC/build"
pushd "$DART_SRC/build"
cmake -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
      -DCMAKE_INSTALL_RPATH="$INSTALL_DIR" \
      -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
      -DDART_BUILD_DARTPY=false \
      -DBUILD_SHARED_LIBS=true \
      -DDART_BUILD_GUI_OSG=false \
      -DDART_ENABLE_SIMD=true \
      ..
ninja install
popd

# ---- Ceres ---------------------------------------------------------------
CERES_SRC="$REPO_ROOT/libs/ceres-solver"
if [ ! -f "$CERES_SRC/CMakeLists.txt" ]; then
    error_exit "libs/ceres-solver not found. Run: git submodule update --init --recursive"
fi

echo "==== Building Ceres from $CERES_SRC ===="
mkdir -p "$CERES_SRC/build"
pushd "$CERES_SRC/build"
cmake -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DBUILD_TESTING=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_BENCHMARKS=OFF \
      -DBUILD_SHARED_LIBS=OFF \
      -DUSE_CUDA=OFF \
      ..
ninja install
popd

echo "==== Installation complete ===="
