#!/usr/bin/env bash
error_exit() {
    echo "$1" >&2
    exit 1
}

DARTSIM_VERSION=v6.13.2

if ! command -v python &> /dev/null; then
    error_exit "python command not found. Please activate conda env"
fi

# Get Python version (e.g., Python 3.8.10)
python_version=$(python3 --version 2>&1)

ENVDIR=${ENVDIR:-~/pkgenv}
SRCDIR=${SRCDIR:-~/pkgsrc}

mkdir -p $ENVDIR
mkdir -p $ENVDIR/include
mkdir -p $ENVDIR/lib
mkdir -p $ENVDIR/lib/cmake
mkdir -p $SRCDIR

export CC=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc
export CXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++

pushd $SRCDIR
git clone --depth 1 --branch $DARTSIM_VERSION https://github.com/dartsim/dart dart
echo "==== Installing dartsim($DARTSIM_VERSION) at $ENVDIR ===="
mkdir dart/build
pushd dart/build
cmake -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$ENVDIR \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
      -DCMAKE_INSTALL_RPATH=$ENVDIR \
      -DDART_BUILD_DARTPY=false \
      -DBUILD_SHARED_LIBS=true \
      -DDART_BUILD_GUI_OSG=false \
      -DDART_ENABLE_SIMD=true \
      ..
ninja install
popd

git clone --depth 1 https://github.com/ceres-solver/ceres-solver ceres
echo "==== Installing ceres at $ENVDIR ===="
mkdir -p ceres/build
pushd ceres/build
cmake -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$ENVDIR \
      -DBUILD_TESTING=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_BENCHMARKS=OFF \
      -DBUILD_SHARED_LIBS=OFF \
      -DUSE_CUDA=OFF \
      ..
ninja install
popd

popd
echo "==== Installation complete ===="