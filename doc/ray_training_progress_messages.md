# Ray Training Progress Messages Guide

This document explains the two types of progress messages shown during Ray/RLlib training and how to control their frequency.

## Overview

During training with `ray_train.py`, you'll see two distinct types of progress messages:

1. **Per-Iteration Results** - Detailed metrics after each training iteration
2. **Periodic Status Updates** - Overall trial status and resource usage

---

## 1. Per-Iteration Results (Every Epoch)

### Example Output
```
Trial MyTrainer_MyEnv_97280_00000 finished iteration 25 at 2025-10-26 10:48:55. Total running time: 2min 26s
╭───────────────────────────────────────────────────────╮
│ Trial MyTrainer_MyEnv_97280_00000 result              │
├───────────────────────────────────────────────────────┤
│ episodes_total                                   7042 │
│ num_env_steps_sampled                          204800 │
│ num_env_steps_trained                          204800 │
│ sampler_results/episode_len_mean              35.6565 │
│ sampler_results/episode_reward_mean          0.736581 │
╰───────────────────────────────────────────────────────╯
```

### What It Is
- Ray Tune's per-iteration callback that logs training metrics
- Appears after each `trainer.train()` call completes
- Shows detailed training statistics for the completed iteration

### Frequency
- **Trigger**: One message per training iteration
- **Cannot be easily suppressed** without modifying Ray's internal logging
- **Frequency = Training iteration frequency** (depends on your batch size and worker configuration)

### Control Options
This is standard Ray behavior and provides valuable training metrics. Options to modify:

1. **Accept it** (Recommended): Useful for monitoring training progress
2. **Redirect stderr**: Capture iteration reports separately from other output
3. **Custom callbacks**: Override Ray's default reporting (advanced, requires Ray internals modification)

---

## 2. Periodic Status Updates (Every ~30 seconds)

### Example Output
```
Trial status: 1 RUNNING
Current time: 2025-10-26 10:48:59. Total running time: 2min 30s
Logical resource usage: 96.0/96 CPUs, 1.0/1 GPUs (0.0/1.0 accelerator_type:RTX)
╭────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                    status       iter     total time (s)       ts     reward │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ MyTrainer_MyEnv_97280_00000   RUNNING        25            131.866   204800   0.736581 │
╰────────────────────────────────────────────────────────────────────────────────────────╯
```

### What It Is
- CLIReporter periodic status display
- Shows overall trial status, resource usage, and high-level progress
- Throttled by Ray to avoid overwhelming the console

### Frequency
- **Controlled by**: `CLIReporter(max_report_frequency=500)` in python/ray_train.py:479
- **Current setting**: 500 milliseconds (maximum 2 updates per second)
- **Actual frequency**: Depends on iteration duration
  - Ray throttles updates intelligently
  - Shows approximately every 30 seconds when iterations take ~30 seconds each

### Configuration Location
File: `python/ray_train.py`, line 479

```python
progress_reporter=CLIReporter(max_report_frequency=500)
```

---

## How to Control Periodic Status Updates

### Adjusting Update Frequency

Edit line 479 in `python/ray_train.py` with one of these options:

#### Less Frequent Updates (Every 5 seconds max)
```python
progress_reporter=CLIReporter(max_report_frequency=5000)
```

#### More Frequent Updates (Every 10 seconds max)
```python
progress_reporter=CLIReporter(max_report_frequency=10000)
```

#### Minimal Output (Only on iteration completion)
```python
progress_reporter=CLIReporter(max_report_frequency=60000)
```

#### Disable Periodic Updates Entirely
```python
progress_reporter=CLIReporter(max_report_frequency=float('inf'))
```
**Note**: You'll still see per-iteration results (first message type)

### Customizing Displayed Columns

Control which metrics are shown in the status table:

```python
progress_reporter=CLIReporter(
    max_report_frequency=500,
    metric_columns=["episodes_total", "sampler_results/episode_reward_mean"],
    parameter_columns=[]  # Hide parameters
)
```

**Available metric columns**:
- `episodes_total`
- `num_env_steps_sampled`
- `num_env_steps_trained`
- `sampler_results/episode_len_mean`
- `sampler_results/episode_reward_mean`
- `training_iteration`
- Any custom metrics you log

### Advanced: Verbosity Control

```python
from ray.tune import CLIReporter

progress_reporter=CLIReporter(
    max_report_frequency=500,
    verbosity=1  # 0 = minimal, 1 = normal, 2 = verbose, 3 = very verbose
)
```

---

## Recommendations

### For Most Users
- **Keep per-iteration reports**: Provides valuable training metrics
- **Current setting is reasonable**: 500ms works well for ~30 second iterations
- **Monitor both message types**: Each provides different insights

### For Long Training Runs
- **Increase max_report_frequency**: Set to 10000-30000 (10-30 seconds) to reduce console spam
- **Customize metric_columns**: Show only the most important metrics

### For Debugging
- **Keep default frequency**: More frequent updates help identify issues quickly
- **Enable verbose mode**: Set `verbosity=2` for detailed information

### For Production/Logging
- **Redirect output**: Capture to log files for post-analysis
- **Use custom callbacks**: Implement custom logging for specific needs

---

## Example Configurations

### Minimal Console Output
```python
progress_reporter=CLIReporter(
    max_report_frequency=float('inf'),  # Disable periodic updates
    metric_columns=["sampler_results/episode_reward_mean"],  # Only show reward
    parameter_columns=[]
)
```

### Detailed Monitoring
```python
progress_reporter=CLIReporter(
    max_report_frequency=1000,  # Update every second (max)
    verbosity=2,  # Verbose output
    metric_columns=[
        "episodes_total",
        "num_env_steps_sampled",
        "sampler_results/episode_reward_mean",
        "sampler_results/episode_len_mean"
    ]
)
```

### Production Training
```python
progress_reporter=CLIReporter(
    max_report_frequency=30000,  # Update every 30 seconds (max)
    metric_columns=["episodes_total", "sampler_results/episode_reward_mean"]
)
```

---

## Summary

| Message Type | Frequency | Control Method | Location |
|--------------|-----------|----------------|----------|
| Per-Iteration Results | Every training iteration | Limited (Ray internals) | N/A |
| Periodic Status | Controlled by max_report_frequency | CLIReporter parameter | ray_train.py:479 |

**Current Default**: `max_report_frequency=500` (500ms, ~2 updates/sec max)

**Actual Behavior**: Updates appear approximately every 30 seconds during normal training due to Ray's intelligent throttling based on iteration duration.
