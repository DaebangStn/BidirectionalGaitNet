# Installation

## System Prerequisites

Install system GL and dev tools (Ubuntu/Debian):

```bash
sudo apt install libgl-dev libglu1-mesa-dev freeglut3-dev libglfw3-dev libglew-dev libc6-dev build-essential ninja-build
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

```bash
pixi run install-deps
```

Builds DART (v6.13.2) and Ceres from source into `libs/install/`.
To force a rebuild, delete `libs/dart/build/` or `libs/ceres-solver/build/` first.

## 5. Configure and build the project

```bash
pixi run setup-machine   # first time only — choose preset (release/debug/a6000/gait)
pixi run build
```

Available presets: `release`, `debug`, `a6000` (server, local CUDA), `gait` (HPC cluster).

## Notes

- `libs/install/` is generated — do not commit it.
- To update DART or Ceres, update the submodule ref and re-run `pixi run install-deps`.
