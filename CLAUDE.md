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

## Remote host: gait (Slurm cluster)
- SSH: `ssh gait`
- CentOS 7, 15 nodes (n1-n15), partitions: `all`, `exo`, `etri`, `shoe`
- **pixi does NOT work** — CentOS 7 glibc 2.17 < required 2.28
- Uses micromamba env `bidir` at `/opt/ohpc/pub/micromamba/envs/bidir` (PyTorch 2.3.0)
- cmake 3.30.1 at `/opt/ohpc/pub/utils/cmake/3.30.1/bin/cmake`
- CUDA 12.4 at `/opt/ohpc/pub/cuda/cuda-12.4`
- Project path: `~/BidirectionalGaitNet`, `.machine = gait`
- **First-time build / rebuild after pull:**
  ```bash
  git pull && git submodule update --init --recursive
  echo gait > .machine
  bash /tmp/build_gait.sh   # see build script below
  ```
- **Build script** (`/tmp/build_gait.sh` — recreate if needed):
  ```bash
  #!/bin/bash
  export MAMBA_ROOT_PREFIX=/opt/ohpc/pub/micromamba
  eval "$(/opt/ohpc/pub/micromamba/bin/micromamba shell hook -s bash)"
  micromamba activate bidir
  set -eo pipefail
  export CONDA_PREFIX=/opt/ohpc/pub/micromamba/envs/bidir
  export PATH=/opt/ohpc/pub/utils/cmake/3.30.1/bin:$PATH
  cd ~/BidirectionalGaitNet
  P=$(pwd); I=$P/libs/install
  export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
  export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
  # ... (install-deps + cmake --preset gait + ninja -C build/gait -j16 -l12)
  ```
- **Incremental rebuild** (after C++ changes):
  ```bash
  export MAMBA_ROOT_PREFIX=/opt/ohpc/pub/micromamba
  eval "$(/opt/ohpc/pub/micromamba/bin/micromamba shell hook -s bash)"
  micromamba activate bidir
  export PATH=/opt/ohpc/pub/utils/cmake/3.30.1/bin:$PATH
  ninja -C ~/BidirectionalGaitNet/build/gait -j16 -l12
  ```
- **Submit training job:**
  ```bash
  sbatch << 'EOF'
  #!/bin/bash
  #SBATCH --job-name=base_imit_px
  #SBATCH --partition=all
  #SBATCH --nodes=1
  #SBATCH --ntasks=1
  #SBATCH --cpus-per-task=16
  #SBATCH --gres=gpu:1
  #SBATCH --output=%x-%j.out
  #SBATCH --error=%x-%j.err

  export MAMBA_ROOT_PREFIX=/opt/ohpc/pub/micromamba
  eval "$(/opt/ohpc/pub/micromamba/bin/micromamba shell hook -s bash)"
  micromamba activate bidir
  cd ~/BidirectionalGaitNet
  python -m ppo.learn --env_file data/env/base_imit_px.yaml
  EOF
  ```

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
squeue -u geon               # your jobs only (gait doesn't support --me)
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
