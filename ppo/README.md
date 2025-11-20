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
