# Pixi Migration Design

Date: 2026-02-20

## Goal

Migrate the project from micromamba (`data/environment.yml`) to pixi (`pixi.toml`) with system-provided OpenGL. Move DART into `libs/dart` as a git submodule and consolidate all built-from-source deps under `libs/install/`.

## Section 1: Package Manager — pixi.toml

Translate `data/environment.yml` to `pixi.toml` at the repo root.

**GL packages to drop from conda** (provided by system instead):
- `freeglut`, `glad`, `glfw`, `libgl-devel`, `libglu`, `libglvnd`, `libglx`, `libopengl`

System prerequisites (apt): `libgl-dev`, `libglfw3-dev`, `libglew-dev`

All other conda packages are carried over as-is. Pip dependencies (`bvh`, `ray`, `c3d`, etc.) go into `[pypi-dependencies]`.

## Section 2: DART + Ceres as git submodules → libs/install/

- Add `libs/dart` submodule pinned to tag `v6.13.2` (`https://github.com/dartsim/dart`)
- `libs/ceres-solver` already exists as a submodule
- Update `scripts/install.sh`:
  - Remove the `$SRCDIR` clone steps (source is now in `libs/`)
  - Build dart from `libs/dart`, install to `libs/install/`
  - Build ceres from `libs/ceres-solver`, install to `libs/install/`
- `CMakeLists.txt`: replace `$ENV{HOME}/pkgenv` with `${CMAKE_SOURCE_DIR}/libs/install` in `CMAKE_PREFIX_PATH`
- Add `libs/install/` to `.gitignore`

## Section 3: System GL via CMakePresets.json

Add to each preset's `cacheVariables`:

```json
"CMAKE_PREFIX_PATH": "/usr/lib/x86_64-linux-gnu;/usr/include",
"OpenGL_GL_PREFERENCE": "GLVND"
```

No changes to `CMakeLists.txt` itself. `GLVND` selects the vendor-neutral dispatch layer present on both NVIDIA and Mesa systems.

## Section 4: INSTALL.md

Document the full setup sequence:
1. System prerequisites (apt packages for GL, CUDA driver)
2. `pixi install`
3. `git submodule update --init --recursive`
4. `pixi run bash scripts/install.sh` (builds dart + ceres → `libs/install/`)
5. `cmake --preset release && ninja -C build/release -j16`
