# BidirectionalGaitNet Scripts

All scripts use the micromamba `bidir` environment automatically. No need to activate environments manually.

## Training Scripts

- `./scripts/train-main` - Unified training interface
- `./scripts/train-fgn` - Train Feedforward Gait Network
- `./scripts/train-bgn` - Train Bidirectional Gait Network
- `./scripts/train-pipeline` - Train full pipeline

## Rollout Script

- `./scripts/rollout` - Rollout with preset support
  ```bash
  ./scripts/rollout list
  ./scripts/rollout preset1 -c <checkpoint>
  ./scripts/rollout gait-metabolic --checkpoint <path>
  ```

## C++ Binary Wrappers

- `./scripts/viewer` - Launch viewer
- `./scripts/physical-exam` - Physical examination tool
- `./scripts/surgery` - Surgery tool
- `./scripts/extract-cycle` - Extract cycle data

## Utility Scripts

- `./scripts/cleanup_old_checkpoints.py` - Clean old checkpoints
- `./scripts/move_and_rename_ray_results.py` - Organize ray results
- `./scripts/rollBvh.py` - Process BVH files
- `./scripts/remove_ray_lock.sh` - Remove ray lock files
- `./scripts/launch_ckpts.sh` - Launch checkpoints
- `./scripts/install.sh` - Installation script

## Usage Examples

```bash
# Training
./scripts/train-main --config config/train.yaml
./scripts/train-fgn
./scripts/train-bgn

# Rollout with presets
./scripts/rollout list                           # List presets
./scripts/rollout preset1 -c ./ray_results/...   # Use preset1
./scripts/rollout test -c ./ray_results/...      # Quick test

# Tools
./scripts/viewer
./scripts/physical-exam
./scripts/surgery
./scripts/extract-cycle
```

## Environment

All scripts automatically:
1. Set `PYTHONPATH` to include the project root
2. Change to project root directory for consistent relative paths
3. Run with micromamba `bidir` environment

No need to:
- Install uv or manage .venv
- Manually activate micromamba environment
- Set PYTHONPATH yourself

## Troubleshooting

If a script fails with import errors, ensure:
1. Micromamba environment is properly installed: `micromamba env list`
2. Dependencies are installed: `micromamba list -n bidir`
3. C++ binaries are built: Check for `.so` files in `python/`