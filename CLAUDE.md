## Environment (first time only)
```bash
micromamba create -f data/environment.yml -y
CONDA_PREFIX=$HOME/micromamba/envs/bidir
ln -sf "$CONDA_PREFIX/lib/libnvToolsExt.so.1" "$CONDA_PREFIX/lib/libnvToolsExt.so"
ln -sf "$CONDA_PREFIX/lib/libGL.so.1"          "$CONDA_PREFIX/lib/libGL.so"
```

## Build
```bash
pixi run install-deps  # first time only — builds DART + Ceres into libs/install/
pixi run build         # cmake (if no cache) + ninja -j16
```
- Preset stored in `.machine` (default: `release`). To change: `pixi run setup-machine`
- Incremental builds: just `pixi run build`
- Direct ninja: `ninja -C build/$(cat .machine) -j16 -l12`

## Run viewers
```bash
pixi run render-ckpt <ckpt_dir>   # RenderCkpt (DART simulation viewer)
pixi run muscle-personalizer       # Muscle Personalizer
pixi run motion-editor             # Motion Editor
pixi run marker-editor             # Marker Editor
pixi run c3d-processor             # C3D Processor
pixi run surgery                   # Surgery tool
```

## Python scripts (ray / PPO)
Use the micromamba `bidir` env (NOT pixi):
```bash
micromamba run -n bidir python python/ray_rollout.py --checkpoint <ckpt> --config <cfg> --workers 1
```

## Agent rules
- Rebuild after any C++ source change before running: `pixi run build`
- Never sleep > 3 seconds. Use background tmux for long-running jobs.
