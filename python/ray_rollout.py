import ray
import torch
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from io import BytesIO
import argparse
import pickle
import tempfile
from datetime import datetime
from tqdm import tqdm
from pyrollout import RolloutEnvironment, RolloutRecord, RecordConfig
from ray_model import SelectiveUnpickler
from uri_resolver import resolve_path
from rollout_worker import (
    PolicyWorker, EnvWorker, FileWorker,
    load_metadata_from_checkpoint, load_config_yaml, load_parameters_from_csv
)

def create_sample_directory(sample_top_dir: str,
                           checkpoint_path: str,
                           config_path: str) -> Path:
    """Create sample directory with format: [checkpoint_name]+[config_name]+on_[timestamp]"""

    # Extract names
    checkpoint_name = Path(checkpoint_path).stem
    config_name = Path(config_path).stem

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directory name
    dir_name = f"{checkpoint_name}+{config_name}+on_{timestamp}"

    # Create full path
    sample_dir = Path(sample_top_dir) / dir_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    print(f"Created sample directory: {sample_dir}")
    return sample_dir

def _run_with_parameters(env_workers, policy, file_worker, parameters, target_cycles):
    """
    Handle parameter sweep with dynamic worker assignment.
    Each worker processes parameters at its own pace.
    """
    num_workers = len(env_workers)
    param_queue = list(parameters)
    total_params = len(parameters)

    # Track worker states
    worker_states = [
        {'active': False, 'param_idx': None, 'state': None, 'step_count': 0}
        for _ in range(num_workers)
    ]

    # Track async done operations
    pending_done_futures = []
    completed_count = 0

    # Progress bar for parameter completion
    pbar = tqdm(total=total_params, desc="Parameter sweep", unit="param", ncols=100)

    # Main stepping loop
    while param_queue or any(ws['active'] for ws in worker_states):

        # 1. Assign new parameters to idle workers
        for worker_idx, ws in enumerate(worker_states):
            if not ws['active'] and param_queue:
                param_idx, param_dict = param_queue.pop(0)

                state = ray.get(env_workers[worker_idx].reset.remote(param_dict))
                ws['active'] = True
                ws['param_idx'] = param_idx
                ws['state'] = state
                ws['step_count'] = 0

        # 2. Get active workers
        active_indices = [i for i, ws in enumerate(worker_states) if ws['active']]
        if not active_indices:
            continue

        active_states = [worker_states[i]['state'] for i in active_indices]

        # 3. Batch action inference (GPU)
        actions = ray.get(policy.compute_actions.remote(active_states))

        # 4. Parallel stepping (CPU)
        step_futures = [
            env_workers[idx].step.remote(action)
            for idx, action in zip(active_indices, actions)
        ]
        results = ray.get(step_futures)

        # 5. Handle step results
        for worker_idx, (state, cycle, done, early_term) in zip(active_indices, results):
            ws = worker_states[worker_idx]
            ws['state'] = state
            ws['step_count'] += 1

            if done:
                param_idx = ws['param_idx']
                success = not early_term

                # Async done handling in EnvWorker process
                done_future = env_workers[worker_idx].handle_done.remote(
                    param_idx, success, file_worker
                )
                pending_done_futures.append((done_future, param_idx, success, cycle, ws['step_count']))

                # Mark worker idle immediately
                ws['active'] = False
                ws['param_idx'] = None
                ws['state'] = None

                # Update progress bar
                completed_count += 1
                pbar.update(1)
                pbar.set_postfix({'completed': completed_count, 'active': sum(1 for w in worker_states if w['active'])})

    pbar.close()

    # 6. Wait for all async done operations to complete
    print(f"\nWaiting for {len(pending_done_futures)} async write operations...")
    failed_count = 0
    success_count = 0
    for done_future, param_idx, success, cycle, steps in tqdm(pending_done_futures, desc="Writing results", unit="write", ncols=100):
        ray.get(done_future)
        if success:
            success_count += 1
        else:
            failed_count += 1
            print(f"  ✗ param_idx={param_idx} failed (cycles={cycle}/{target_cycles})")

    print(f"✓ Completed: {success_count} successful, {failed_count} failed")


def _run_with_random_sampling(env_workers, policy, file_worker, target_cycles, num_samples):
    """
    Sample random parameters for each rollout.
    reset(None) triggers random sampling in ray_env.py.

    Args:
        num_samples: Total number of random samples to generate
    """
    num_workers = len(env_workers)
    target_sample_count = num_samples
    assigned_count = 0
    completed_count = 0

    worker_states = [
        {'active': False, 'sample_idx': None, 'state': None, 'step_count': 0}
        for _ in range(num_workers)
    ]

    pending_done_futures = []

    # Progress bar for sample completion
    pbar = tqdm(total=target_sample_count, desc="Random sampling", unit="sample", ncols=100)

    while completed_count < target_sample_count or any(ws['active'] for ws in worker_states):

        # 1. Assign to idle workers
        for worker_idx, ws in enumerate(worker_states):
            if not ws['active'] and assigned_count < target_sample_count:
                sample_idx = assigned_count

                # reset(None) -> random parameter in ray_env.py:50-52
                state = ray.get(env_workers[worker_idx].reset.remote(None))
                ws['active'] = True
                ws['sample_idx'] = sample_idx
                ws['state'] = state
                ws['step_count'] = 0

                assigned_count += 1

        # 2. Get active workers
        active_indices = [i for i, ws in enumerate(worker_states) if ws['active']]
        if not active_indices:
            continue

        active_states = [worker_states[i]['state'] for i in active_indices]

        # 3. Batch inference
        actions = ray.get(policy.compute_actions.remote(active_states))

        # 4. Parallel stepping
        step_futures = [
            env_workers[idx].step.remote(action)
            for idx, action in zip(active_indices, actions)
        ]
        results = ray.get(step_futures)

        # 5. Handle results
        for worker_idx, (state, cycle, done, early_term) in zip(active_indices, results):
            ws = worker_states[worker_idx]
            ws['state'] = state
            ws['step_count'] += 1

            if done:
                sample_idx = ws['sample_idx']
                success = not early_term

                # Async done handling
                done_future = env_workers[worker_idx].handle_done.remote(
                    sample_idx, success, file_worker
                )
                pending_done_futures.append((done_future, sample_idx, success, cycle, ws['step_count']))

                ws['active'] = False
                completed_count += 1

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'completed': completed_count, 'active': sum(1 for w in worker_states if w['active'])})

    pbar.close()

    # 6. Wait for all async done operations to complete
    print(f"\nWaiting for {len(pending_done_futures)} async write operations...")
    for done_future, sample_idx, success, cycle, steps in tqdm(pending_done_futures, desc="Writing results", unit="write", ncols=100):
        ray.get(done_future)
    print(f"✓ All {target_sample_count} samples completed and written")


def run_rollout(checkpoint_path: str,
                record_config_path: str,
                output_path: str,
                sample_dir: Path,
                param_file: Optional[str] = None,
                num_workers: int = None,
                num_samples: int = None):
    """Run distributed rollout with Ray, optionally sweeping over parameters

    Args:
        checkpoint_path: Path to checkpoint directory
        record_config_path: Path to rollout config YAML
        output_path: Output HDF5 file path
        sample_dir: Sample directory for error logging
        param_file: Optional CSV file with parameter sweep
        num_workers: Number of workers (default: auto-detect)
        num_samples: Number of random samples when param_file is not provided (default: num_workers)
    """

    # Initialize Ray (Never modify this)
    ray.init(address="auto")

    # Resolve checkpoint path using URIResolver and convert to absolute path for Ray workers
    checkpoint_path = resolve_path(checkpoint_path)
    checkpoint_path = str(Path(checkpoint_path).resolve())
    print(f"[Python] Loading network from {checkpoint_path}")

    # Resolve record config path using URIResolver and convert to absolute path for Ray workers
    record_config_path = resolve_path(record_config_path)
    record_config_path = str(Path(record_config_path).resolve())

    # Load metadata from checkpoint
    print(f"Loading metadata from checkpoint: {checkpoint_path}")
    metadata_xml = load_metadata_from_checkpoint(checkpoint_path)

    # Load config to get target cycles
    print(f"Loading rollout configuration: {record_config_path}")
    config = load_config_yaml(record_config_path)
    target_cycles = config.get('sample', {}).get('cycle', 5)
    print(f"Target cycles: {target_cycles}")

    # Load parameter sweep if provided
    parameters = []
    if param_file:
        param_file = resolve_path(param_file)
        param_file = str(Path(param_file).resolve())
        print(f"Resolved parameter file path: {param_file}")
        parameters = load_parameters_from_csv(param_file)

    # Determine number of workers
    if num_workers is None:
        num_workers = int(ray.available_resources().get('CPU', 1))

    print(f"Starting rollout with {num_workers} environment workers")

    # Create temporary environment to get state/action dimensions
    temp_metadata_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
    temp_metadata_file.write(metadata_xml)
    temp_metadata_file.close()

    temp_env = RolloutEnvironment(temp_metadata_file.name)
    temp_env.reset()
    state_sample = temp_env.get_state()
    num_states = len(state_sample)

    # Get action dimension from checkpoint
    checkpoint_file = Path(checkpoint_path) if Path(checkpoint_path).is_file() else Path(checkpoint_path) / "max_checkpoint"
    with open(checkpoint_file, 'rb') as f:
        state = pickle.load(f)
        # Get action space from observation/action space info
        if "space" in state:
            num_actions = state["space"]["action_space"].shape[0]
        else:
            # Fallback: infer from policy weights
            worker_state = SelectiveUnpickler(BytesIO(state['worker'])).load()
            policy_state = worker_state["state"]['default_policy']['weights']
            # Get action dimension from last layer of p_fc
            num_actions = policy_state['p_fc.6.weight'].shape[0]

    print(f"State dimension: {num_states}, Action dimension: {num_actions}")

    # Extract parameter names for HDF5 metadata
    parameter_names = temp_env.get_parameter_names()

    # Clean up temp file
    os.unlink(temp_metadata_file.name)

    # Extract checkpoint name
    checkpoint_name = Path(checkpoint_path).name

    # Read config file content
    with open(record_config_path, 'r') as f:
        config_content = f.read()

    # Create single policy worker with correct dimensions
    policy = PolicyWorker.remote(checkpoint_path, num_states, num_actions)

    # Get muscle network weights from policy worker
    mcn_weights_ref = policy.get_mcn_weights.remote()

    # Create environment workers
    env_workers = [
        EnvWorker.remote(i, metadata_xml, record_config_path, target_cycles)
        for i in range(num_workers)
    ]

    # Set muscle network weights on all workers
    ray.get([w.set_mcn_weights.remote(mcn_weights_ref) for w in env_workers])

    # Create FileWorker for thread-safe I/O with global metadata
    file_worker = FileWorker.remote(
        output_path,
        sample_dir,
        parameter_names=parameter_names,
        checkpoint_name=checkpoint_name,
        metadata_xml=metadata_xml,
        config_content=config_content
    )

    # Run appropriate rollout method
    if parameters:
        print(f"Running parameter sweep with {len(parameters)} parameter sets")
        _run_with_parameters(env_workers, policy, file_worker, parameters, target_cycles)
    else:
        # Default to num_workers if num_samples not specified
        if num_samples is None:
            num_samples = num_workers
        print(f"Running random sampling with {num_samples} samples using {num_workers} workers")
        _run_with_random_sampling(env_workers, policy, file_worker, target_cycles, num_samples)

    # Finalize writes
    ray.get(file_worker.finalize.remote())
    print(f"✓ Rollout complete. Data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ray-based rollout for BidirectionalGaitNet")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory (contains metadata)")
    parser.add_argument("--config", required=True, help="Path to record config YAML (contains target cycles)")
    parser.add_argument("--param-file", default=None, help="CSV file with parameter sweep (optional)")
    parser.add_argument("--workers", type=int, default=16, help="Number of workers (default: auto-detect)")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of random samples (when --param-file not provided, default: same as workers)")
    parser.add_argument("--sample-dir", help="Top-level sample directory (subdirectory will be auto-created)", default="./sampled")

    args = parser.parse_args()

    # Create sample directory with format: [checkpoint]+[config]+on_[timestamp]
    sample_dir = create_sample_directory(
        args.sample_dir,
        args.checkpoint,
        args.config
    )

    # Generate output path inside the sample directory (HDF5)
    output_path = sample_dir / "rollout_data.h5"

    # Convert to absolute path for Ray workers
    output_path = output_path.resolve()
    sample_dir = sample_dir.resolve()

    run_rollout(
        checkpoint_path=args.checkpoint,
        record_config_path=args.config,
        output_path=str(output_path),
        sample_dir=sample_dir,
        param_file=args.param_file,
        num_workers=args.workers,
        num_samples=args.num_samples
    )

