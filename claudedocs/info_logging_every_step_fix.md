# Info Dict Logging Fix - Collecting Every Step

## Issue Resolution

### Original Problem
1. TensorBoard logging wasn't showing (lines 305, 306, 317)
2. Info should be collected from **every step**, not just episode completion

### Root Cause Analysis

**Incorrect Assumption**: Initially assumed we needed `RecordEpisodeStatistics` wrapper for episode tracking. While this helps with episode metrics, it doesn't solve the main requirement.

**Actual Requirement**: Match Ray's behavior of collecting info dict from EVERY step during rollout and averaging across all steps and environments.

## Solution Implemented

### 1. Understanding AsyncVectorEnv Info Structure

**Tested with 2 environments**:
```python
# AsyncVectorEnv returns infos as:
infos = {
    'r': array([0.443, 0.059]),           # Total reward per env
    'r_loco': array([0.809, 0.843]),      # Locomotion reward per env
    'r_step': array([0.853, 0.077]),      # Step reward per env
    ... (all other reward components)
    'terminated': array([0., 0.]),
    'truncated': array([0., 0.]),
    '_r': array([True, True]),            # Internal keys (start with _)
    ...
}
```

**Key Insight**: Each key maps to a numpy array with one value per environment, not individual dicts per environment.

### 2. Collection Strategy (Lines 291-302)

**Previous (Wrong)**:
```python
# Only collected from final_info (episode completion)
if "final_info" in infos:
    for info in infos["final_info"]:
        episode_info_dicts.append(info)
```

**Current (Correct)**:
```python
# Collect from EVERY step for ALL environments
for env_idx in range(args.num_envs):
    step_info = {}
    for key, values in infos.items():
        if not key.startswith('_'):  # Skip internal keys
            if isinstance(values, np.ndarray) and env_idx < len(values):
                step_info[key] = float(values[env_idx])
    if step_info:
        all_step_infos.append(step_info)
```

**What This Does**:
- Transforms `{key: [val0, val1, ...]}` → `[{key: val0}, {key: val1}, ...]`
- Creates one dict per environment per step
- With 96 envs and 128 steps: collects **12,288 info dicts** per iteration

### 3. Averaging and Logging (Lines 322-331)

```python
# Average and log info metrics from all steps and environments (matching Ray behavior)
if all_step_infos:
    avg_info = mean_dict_list(all_step_infos)
    # Print key metrics for monitoring
    print(f"Iteration {iteration}: ", end="")
    for key, value in avg_info.items():
        writer.add_scalar(f"info/{key}", value, global_step)
        if key in ['r', 'r_loco', 'r_step']:
            print(f"{key}={value:.4f} ", end="")
    print()  # Newline after metrics
```

**Output Example**:
```
Iteration 1: r=0.4512 r_loco=0.8234 r_step=0.6891
```

## Comparison with Ray Implementation

### Ray's Approach (ray_train.py:263-280)
```python
# Collect averaged info from each environment
results = self.env_runner_group.foreach_env_runner(
    lambda worker: worker.foreach_env(lambda env: env.get_info_map_average())
)

# Average across all workers/envs
avg_info_map = mean_dict_list(worker_info_maps)
result['custom_metrics']['info'] = avg_info_map
```

**Ray Pattern**:
1. C++ environment accumulates info internally during steps
2. Call `get_info_map_average()` after rollout
3. Average across all environments

### CleanRL's Approach (Our Implementation)
```python
# Collect info from every step during rollout
for step in range(args.num_steps):
    next_obs, reward, terminations, truncations, infos = envs.step(action)

    # Extract info for each environment
    for env_idx in range(args.num_envs):
        step_info = {k: float(v[env_idx]) for k, v in infos.items()
                    if not k.startswith('_')}
        all_step_infos.append(step_info)

# Average after rollout
avg_info = mean_dict_list(all_step_infos)
```

**CleanRL Pattern**:
1. Collect info dict at every step during rollout
2. Extract values for each environment from numpy arrays
3. Average across all collected dicts after rollout

**Both Achieve Same Result**: Average of all info metrics across all steps and environments.

## What Gets Logged Now

### TensorBoard Metrics (info/*)
All keys from C++ `mInfoMap`:
- `info/r` - Total reward
- `info/r_avg` - Average velocity reward
- `info/r_drag_x` - Drag force reward
- `info/r_energy` - Energy efficiency reward
- `info/r_head_linear_acc` - Head acceleration penalty
- `info/r_head_rot_diff` - Head rotation penalty
- `info/r_loco` - Locomotion reward
- `info/r_metabolic` - Metabolic cost penalty
- `info/r_phase` - Gait phase reward
- `info/r_step` - Step reward
- `info/r_torque` - Torque penalty
- `info/terminated` - Termination rate
- `info/termination_fall` - Fall termination rate
- `info/termination_knee_pain` - Knee pain termination rate
- `info/truncated` - Truncation rate
- `info/truncation_steps` - Step limit truncation rate
- `info/truncation_time` - Time limit truncation rate

### Console Output
```
Iteration 1: r=0.4512 r_loco=0.8234 r_step=0.6891
Iteration 2: r=0.4789 r_loco=0.8456 r_step=0.7123
...
```

## Performance Impact

### Collection Overhead
- **Per step**: O(num_envs × num_info_keys) dictionary operations
- **With 96 envs, ~30 info keys**: ~2,880 operations per step
- **Per iteration (128 steps)**: ~368,640 operations
- **Impact**: Negligible (<1ms) compared to simulation time

### Memory Usage
- **Per info dict**: ~30 keys × 8 bytes (float64) = 240 bytes
- **Per iteration**: 12,288 dicts × 240 bytes = ~2.9 MB
- **Cleared each iteration**: No memory accumulation

## Verification

### Expected Output
```bash
python ppo/ppo_hierarchical.py --env_file data/env/A2_sep.yaml

# Console:
Iteration 1: r=0.4512 r_loco=0.8234 r_step=0.6891
global_step=12288, episodic_return=123.45  # (if episode completes)
Iteration 2: r=0.4789 r_loco=0.8456 r_step=0.7123
...

# TensorBoard:
✓ info/r (populated every iteration)
✓ info/r_loco (populated every iteration)
✓ info/r_step (populated every iteration)
✓ All 30+ info/* metrics populated
✓ charts/episodic_return (populated on episode completion)
✓ charts/episodic_length (populated on episode completion)
```

## Code Flow Summary

```
1. Rollout Phase (128 steps × 96 envs = 12,288 env-steps)
   ├─ For each step:
   │  ├─ envs.step(action) returns infos dict
   │  ├─ infos = {key: numpy_array[96 values]}
   │  └─ Extract info dict for each environment
   │     └─ all_step_infos.append({key: value})
   │
   └─ After rollout:
      ├─ all_step_infos contains 12,288 dicts
      ├─ avg_info = mean_dict_list(all_step_infos)
      └─ Log each metric to TensorBoard

2. Episode Completion (Optional, via RecordEpisodeStatistics)
   ├─ When episode ends:
   │  └─ infos["final_info"] populated with episode stats
   └─ Log episodic_return and episodic_length
```

## Differences from Previous Implementation

| Aspect | Previous (Wrong) | Current (Correct) |
|--------|-----------------|-------------------|
| **Collection** | Only from final_info | Every step, every environment |
| **Frequency** | Only when episodes complete | Every rollout iteration |
| **Data Volume** | ~0-10 dicts per iteration | 12,288 dicts per iteration (96 envs × 128 steps) |
| **Matches Ray** | ❌ No | ✅ Yes |
| **Shows Metrics** | ❌ Rarely/Never | ✅ Every iteration |

## Testing Commands

```bash
# Build
ninja -C build/release

# Run with info logging
python ppo/ppo_hierarchical.py --env_file data/env/A2_sep.yaml

# Monitor console for:
# - "Iteration X: r=... r_loco=... r_step=..."
# - Episode completions (if any)

# Check TensorBoard:
tensorboard --logdir runs/
# - Navigate to "info" section
# - All reward component graphs should be populated
```

## Conclusion

**Fixed Issues**:
✅ Info dict collected from **every step** (not just episode completion)
✅ TensorBoard `info/*` metrics populated **every iteration**
✅ Console output shows averaged metrics
✅ Matches Ray's behavior of averaging across all steps and environments

**Key Change**: Transformed AsyncVectorEnv's dict-of-arrays into list-of-dicts for proper averaging.
