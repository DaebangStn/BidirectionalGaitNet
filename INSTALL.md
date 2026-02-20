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
