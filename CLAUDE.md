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
Use pixi for all Python scripts:
```bash
pixi run train <config_name>       # PPO training, e.g. pixi run train base_imit
pixi run rollout                   # rollout CLI
```

## Agent rules
- Use pixi for everything. Do NOT use micromamba.
- Rebuild after any C++ source change before running: `pixi run build`
- Never sleep > 3 seconds. Use background tmux for long-running jobs.

## Remote host: a6000
- SSH: `ssh a6000`
- Ubuntu 24.04, RTX A6000 GPU, IP 147.47.206.51
- Slurm scheduler: single partition `all`, single node `a6000`
- Project path: `~/BidirectionalGaitNet`
- **First-time setup on a6000:**
  ```bash
  git pull
  git submodule update --init --recursive   # dart, ceres-solver, etc.
  # Ensure GL dev symlinks exist (may need sudo):
  sudo ln -sf /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
  sudo ln -sf /usr/lib/x86_64-linux-gnu/libGLU.so.1 /usr/lib/x86_64-linux-gnu/libGLU.so
  sudo apt-get install -y freeglut3-dev
  echo a6000 > .machine
  pixi run install-deps   # builds DART + Ceres (~10-20 min)
  pixi run build
  ```

## Slurm commands
```bash
sinfo                        # show partitions and node states
squeue -a                    # all jobs in queue
squeue --me                  # your jobs only
sbatch <script.sh>           # submit a job
scancel <jobid>              # cancel a job
scontrol show job <jobid>    # job details
```

### Submit training job
```bash
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=base_imit
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd ~/BidirectionalGaitNet
pixi run train base_imit
EOF
```

## All pixi tasks
| Command | Description |
|---------|-------------|
| `pixi run install-deps` | Build DART + Ceres into `libs/install/` (first time) |
| `pixi run build` | cmake + ninja build (incremental) |
| `pixi run setup-machine` | Set `.machine` preset (release/debug/a6000/gait) |
| `pixi run train <config>` | PPO training, e.g. `pixi run train base_imit` |
| `pixi run rollout` | Rollout CLI |
| `pixi run render-ckpt <dir>` | DART simulation viewer |
| `pixi run muscle-personalizer` | Muscle Personalizer |
| `pixi run motion-editor` | Motion Editor |
| `pixi run marker-editor` | Marker Editor |
| `pixi run c3d-processor` | C3D Processor |
| `pixi run surgery` | Surgery tool |
| `pixi run convert-ckpt` | Convert checkpoint to TorchScript |
| `pixi run extract-cycle` | Extract cycle |
