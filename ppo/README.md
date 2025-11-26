## Performance Equivalence

For equivalent sample throughput:

**Ray ppo_small_pc**:
- 16 workers × 1 env each = 16 parallel environments
- train_batch_size = 8192 = 16 × 512 steps per worker
- Updates: ~305 iterations for 10M timesteps

**CleanRL**:
- 16 vectorized environments
- batch_size = 16 × 2048 = 32,768 steps per update
- Updates: ~305 iterations for 10M timesteps

**Note**: CleanRL collects 4× more steps per update (32,768 vs 8,192) but runs the same number of updates. This is a design difference - Ray's smaller batches may provide more frequent updates but require more communication overhead in distributed settings.

## Usage Examples

### Ray equivalent configuration
```bash
# Matches ppo_small_pc configuration
python ppo/ppo_hierarchical.py \
    --env-file data/env/base_lonly.xml \
    --num-envs 16 \
    --total-timesteps 10000000 \
    --learning-rate 1e-4 \
    --gamma 0.99 \
    --gae-lambda 0.99 \
    --update-epochs 4 \
    --muscle-num-epochs 10 \
    --muscle-batch-size 512
```

### Minimal configuration (for testing)
```bash
# Matches ppo_mini
python ppo/ppo_hierarchical.py \
    --env-file data/env/base_lonly.xml \
    --num-envs 1 \
    --total-timesteps 100000 \
    --muscle-batch-size 64
```

### Large scale (cluster equivalent)
```bash
# Matches ppo_large_pc concept (32 envs)
python ppo/ppo_hierarchical.py \
    --env-file data/env/base_lonly.xml \
    --num-envs 32 \
    --total-timesteps 50000000 \
    --muscle-batch-size 4096
```

## Checkpointing and Resume

### Checkpoint Types

| Mode | Files Saved | Use Case |
|------|-------------|----------|
| Default | agent.pt, muscle.pt, metadata.yaml | Inference/rendering only |
| `--save_optimizer` | + optimizer.pt, muscle.opt.pt, training_state.pt | Resume training |

### Resume Training

```bash
# Save resumable checkpoints during training
python ppo/learn.py --env_file data/env/A2.yaml --save_optimizer

# Resume from a checkpoint
python ppo/learn.py --env_file data/env/A2.yaml \
    --resume_from runs/A2/241126_120000/A2-01000-1126_130000 \
    --save_optimizer
```

### Checkpoint Contents

**Default checkpoint:**
- `agent.pt` - PPO policy network weights
- `muscle.pt` - Muscle network weights (if hierarchical)
- `metadata.yaml` - Environment config (with `resumed_from` lineage if applicable)

**With `--save_optimizer`:**
- `optimizer.pt` - PPO Adam optimizer state (momentum buffers)
- `muscle.opt.pt` - Muscle Adam optimizer state
- `training_state.pt` - iteration, global_step, args

### Notes
- Resume creates a **new TensorBoard run** (fresh timestamp)
- `metadata.yaml` includes `resumed_from` field for lineage tracking
- Args compatibility is validated on resume (warnings for mismatches)
