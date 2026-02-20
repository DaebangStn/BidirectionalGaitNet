# Pixi Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate the build environment from micromamba to pixi with system-provided OpenGL, move DART into `libs/dart` as a git submodule, and consolidate all built-from-source deps under `libs/install/`.

**Architecture:** pixi replaces micromamba as the conda env manager (same conda packages, new `pixi.toml`). GL packages are dropped from conda and found via CMakePresets.json hints pointing at system paths. DART and Ceres are built from their `libs/` submodules and installed to `libs/install/`.

**Tech Stack:** pixi, CMake 3.28+, Ninja, DART v6.13.2, Ceres (from existing submodule), system OpenGL (GLVND)

---

### Task 1: Add `libs/dart` git submodule

**Files:**
- Modify: `.gitmodules`
- New directory: `libs/dart/`

**Step 1: Add the submodule pinned to v6.13.2**

```bash
git submodule add --depth 1 -b v6.13.2 https://github.com/dartsim/dart libs/dart
```

> Note: `--depth 1` keeps history shallow. If `add` does not support `-b` with `--depth`, omit `--depth 1`.

**Step 2: Verify the submodule was added**

```bash
git submodule status libs/dart
cat .gitmodules | grep -A3 "libs/dart"
```

Expected: entry for `libs/dart` with `url = https://github.com/dartsim/dart` and the submodule directory populated.

**Step 3: Add build and install dirs to .gitignore**

Add to `.gitignore`:
```
libs/dart/build/
libs/ceres-solver/build/
libs/install/
```

**Step 4: Commit**

```bash
git add .gitmodules libs/dart .gitignore
git commit -m "feat: add libs/dart submodule (v6.13.2) and ignore build artifacts"
```

---

### Task 2: Rewrite `scripts/install.sh`

**Files:**
- Modify: `scripts/install.sh`

**Step 1: Replace the script content**

The new script builds DART from `libs/dart` and Ceres from `libs/ceres-solver`, both installed to `libs/install/`. It no longer clones anything.

```bash
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
```

**Step 2: Make it executable**

```bash
chmod +x scripts/install.sh
```

**Step 3: Commit**

```bash
git add scripts/install.sh
git commit -m "feat: rewrite install.sh to build from libs/ submodules into libs/install/"
```

---

### Task 3: Update `CMakeLists.txt`

**Files:**
- Modify: `CMakeLists.txt:18` (OpenGL_GL_PREFERENCE)
- Modify: `CMakeLists.txt:33` (pkgenv → libs/install)

**Step 1: Remove the hardcoded LEGACY GL preference (line 18)**

Change:
```cmake
set(OpenGL_GL_PREFERENCE LEGACY)
```
To: *(delete this line entirely — preference is now set in CMakePresets.json)*

**Step 2: Replace `~/pkgenv` with `libs/install` (line 33)**

Change:
```cmake
set(CMAKE_PREFIX_PATH $ENV{HOME}/pkgenv ${CMAKE_PREFIX_PATH})
```
To:
```cmake
set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/libs/install ${CMAKE_PREFIX_PATH})
```

**Step 3: Verify the diff looks right**

```bash
git diff CMakeLists.txt
```

Expected: line 18 removed, line 33 path changed.

**Step 4: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: point CMAKE_PREFIX_PATH to libs/install, remove LEGACY GL preference"
```

---

### Task 4: Update `CMakePresets.json`

**Files:**
- Modify: `CMakePresets.json`

**Step 1: Add GL variables to the `release` preset**

The `release` preset is the base — `debug`, `a6000`, and `gait` all inherit from it, so GL settings propagate automatically.

Add to `release`'s `cacheVariables`:
```json
"OpenGL_GL_PREFERENCE": "GLVND",
"OPENGL_INCLUDE_DIR": "/usr/include",
"OPENGL_gl_LIBRARY": "/usr/lib/x86_64-linux-gnu/libGL.so"
```

The full updated `release` preset:
```json
{
  "name": "release",
  "hidden": false,
  "generator": "Ninja",
  "binaryDir": "${sourceDir}/build/release",
  "cacheVariables": {
      "CMAKE_BUILD_TYPE": "Release",
      "SERVER_BUILD": false,
      "CUDAToolkit_ROOT": "/usr/local/cuda",
      "CUDA_TOOLKIT_ROOT_DIR": "/usr/local/cuda",
      "CMAKE_CUDA_COMPILER": "/usr/local/cuda/bin/nvcc",
      "OpenGL_GL_PREFERENCE": "GLVND",
      "OPENGL_INCLUDE_DIR": "/usr/include",
      "OPENGL_gl_LIBRARY": "/usr/lib/x86_64-linux-gnu/libGL.so"
  }
}
```

**Step 2: Commit**

```bash
git add CMakePresets.json
git commit -m "build: add system GL paths to CMakePresets.json (GLVND)"
```

---

### Task 5: Create `pixi.toml`

**Files:**
- Create: `pixi.toml`

**Step 1: Write pixi.toml translating `data/environment.yml`**

Key changes from the conda env:
- Drop GL runtime packages: `freeglut`, `libgl-devel`, `libglu`, `libglvnd`, `libglx`, `libopengl`
- Keep `glad` and `glfw` (CMake config providers for the loader and window library)
- Pip deps move to `[pypi-dependencies]`
- `cuda-cudart_linux-64` is platform-specific → `[target.linux-64.dependencies]`
- `ray[rllib]` + `ray[client]` merge into one entry with extras

```toml
[project]
name = "bidirectional-gait-net"
version = "0.1.0"
description = "Bidirectional GaitNet with PPO and hierarchical muscle control"
channels = ["conda-forge", "pytorch"]
platforms = ["linux-64"]

[dependencies]
python = "3.10.*"
assimp = "5.4.2.*"
boost = "1.84.0.*"
boost-cpp = "1.84.0.*"
bullet = "3.25.*"
bullet-cpp = "3.25.*"
cuda-cudart = "12.4.*"
cuda-nvrtc = "12.4.*"
cuda-nvtx = "12.4.*"
cuda-version = "12.4.*"
eigen = "3.4.0.*"
fcl = "0.7.0.*"
fmt = "11.0.2.*"
gcc_impl_linux-64 = "14.1.0.*"
gcc_linux-64 = "14.1.0.*"
glad = "*"
glfw = "*"
gxx_impl_linux-64 = "14.1.0.*"
gxx_linux-64 = "14.1.0.*"
h5py = "*"
hdf5 = "*"
ezc3d = "*"
dill = "*"
tinyxml2 = "*"
kernel-headers_linux-64 = "3.10.0.*"
libboost = "1.84.0.*"
libboost-devel = "1.84.0.*"
libboost-headers = "1.84.0.*"
libboost-python = "1.84.0.*"
libboost-python-devel = "1.84.0.*"
libccd-double = "2.1.*"
libgcc = "14.1.0.*"
libgcc-devel_linux-64 = "14.1.0.*"
libgcc-ng = "14.1.0.*"
libgcrypt-lib = "1.11.1.*"
matplotlib = "*"
ncurses = "6.5.*"
numpy = "1.23.4.*"
pip = "*"
polars = "*"
pybind11 = "*"
pybind11-global = "*"
pybullet = "3.25.*"
pytest = "*"
pytorch = {version = "2.3.0", build = "cuda120_py310h2c91c31_301"}
pyyaml = "*"
scipy = "*"
setuptools = "*"
tqdm = "*"
umap-learn = "0.5.6.*"
xarray = "*"
h5netcdf = "*"
yaml-cpp = "*"
psutil = "*"
netCDF4 = "*"
yaml = "*"

[target.linux-64.dependencies]
cuda-cudart_linux-64 = "12.4.*"

[pypi-dependencies]
bvh = "*"
click = "==8.0.4"
c3d = "*"
dm-tree = "*"
scikit-image = "==0.23.0"
pyarrow = "==14.0.2"
ray = {version = "==2.12.0", extras = ["rllib", "client"]}
```

**Step 2: Verify pixi resolves the environment**

```bash
cd /home/geon/BidirectionalGaitNet
pixi install
```

Expected: environment solves and installs without errors. If a package version conflicts, relax `.*` pins (remove version constraint entirely) for the conflicting package.

**Step 3: Commit**

```bash
git add pixi.toml pixi.lock
git commit -m "feat: add pixi.toml (migrated from micromamba data/environment.yml)"
```

---

### Task 6: Create `INSTALL.md`

**Files:**
- Create: `INSTALL.md`

**Step 1: Write the install guide**

```markdown
# Installation

## System Prerequisites

Install system GL and dev tools (Ubuntu/Debian):

```bash
sudo apt install libgl-dev libglfw3-dev libglew-dev build-essential ninja-build
```

CUDA driver ≥ 12.4 must be installed separately (see [NVIDIA docs](https://docs.nvidia.com/cuda/)).

## 1. Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

## 2. Clone and initialize submodules

```bash
git clone <repo-url> BidirectionalGaitNet
cd BidirectionalGaitNet
git submodule update --init --recursive
```

## 3. Set up the conda environment

```bash
pixi install
```

## 4. Build DART and Ceres

Run inside the pixi environment so the conda compiler is on PATH:

```bash
pixi run bash scripts/install.sh
```

This builds DART (v6.13.2) and Ceres from source into `libs/install/`.
Subsequent runs are skipped if the build directory already exists — delete
`libs/dart/build/` or `libs/ceres-solver/build/` to force a rebuild.

## 5. Configure and build the project

```bash
cmake --preset release
ninja -C build/release -j16 -l12
```

Available presets: `release`, `debug`, `a6000` (server, local CUDA), `gait` (HPC cluster).

## Notes

- `libs/install/` is generated — do not commit it.
- To update DART or Ceres, update the submodule ref and re-run `scripts/install.sh`.
```

**Step 2: Commit**

```bash
git add INSTALL.md
git commit -m "docs: add INSTALL.md for pixi-based setup"
```

---

### Task 7: Smoke-test the full build

**Step 1: Enter the pixi environment**

```bash
pixi shell
```

**Step 2: Check submodules are populated**

```bash
ls libs/dart/CMakeLists.txt
ls libs/ceres-solver/CMakeLists.txt
```

Both should exist.

**Step 3: Run the install script**

```bash
bash scripts/install.sh
```

Expected: both DART and Ceres build and install to `libs/install/`. Check:
```bash
ls libs/install/lib/cmake/
```
Should contain `dart/` and `Ceres/` directories.

**Step 4: Configure CMake**

```bash
cmake --preset release
```

Expected: all `find_package` calls succeed, including `DART`, `Ceres`, `OpenGL`, `glfw3`, `glad`.

**Step 5: Build**

```bash
ninja -C build/release -j16 -l12
```

Expected: clean build, no GL or DART-related linker errors.

**Step 6: Commit any fixups, then tag**

```bash
git add -p   # stage any fixup changes
git commit -m "fix: <describe any fixup>"
```
