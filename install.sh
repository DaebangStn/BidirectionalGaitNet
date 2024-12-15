#!/usr/bin/env bash
# Remove redundant package (gui, external)

DARTSIM_VERSION=v6.13.2
ENVDIR=${ENVDIR:-~/pkgenv36}
SRCDIR=${SRCDIR:-~/pkgsrc36}

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
