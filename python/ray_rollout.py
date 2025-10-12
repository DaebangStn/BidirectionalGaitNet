import ray
import torch
import numpy as np
import polars as pl
import h5py
import yaml
import os
import dill
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from io import BytesIO
import argparse
import pickle
import tempfile
from datetime import datetime
from pyrollout import RolloutEnvironment, RolloutRecord, RecordConfig
from ray_model import SelectiveUnpickler, loading_network
from uri_resolver import resolve_path

def load_metadata_from_checkpoint(checkpoint_path: str) -> str:
    """Load metadata XML string from checkpoint file or directory

    Args:
        checkpoint_path: Path to checkpoint file or directory containing checkpoints

    Returns:
        Metadata XML string
    """
    checkpoint_path = Path(checkpoint_path)

    # Case 1: Direct checkpoint file
    if checkpoint_path.is_file():
        with open(checkpoint_path, 'rb') as f:
            state = pickle.load(f)
            if "metadata" not in state:
                raise RuntimeError(f"Checkpoint file {checkpoint_path} does not contain 'metadata' key")
            return state["metadata"]

    # Case 2: Directory containing checkpoints
    if not checkpoint_path.is_dir():
        raise RuntimeError(f"Checkpoint path does not exist: {checkpoint_path}")

    # Try to load max_checkpoint (best model)
    max_checkpoint = checkpoint_path / "max_checkpoint"
    if max_checkpoint.exists():
        with open(max_checkpoint, 'rb') as f:
            state = pickle.load(f)
            return state["metadata"]

    # Fall back to finding the latest checkpoint file
    checkpoint_files = list(checkpoint_path.glob("ckpt-*"))
    if not checkpoint_files:
        raise RuntimeError(f"No checkpoint files found in {checkpoint_path}")

    # Sort by modification time and get the latest
    latest_checkpoint = sorted(checkpoint_files, key=lambda p: p.stat().st_mtime)[-1]
    with open(latest_checkpoint, 'rb') as f:
        state = pickle.load(f)
        return state["metadata"]

def load_config_yaml(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_parameters_from_csv(csv_path: str) -> List[Tuple[int, Dict[str, float]]]:
    """Load parameter sweep from CSV file

    Expected CSV format:
        param_idx,cadence,stride,...
        0,0.4,0.4
        1,0.4,0.45
        ...

    Returns:
        List of (param_idx, {param_name: value}) tuples
    """
    df = pl.read_csv(csv_path)

    # Extract param_idx column
    if 'param_idx' not in df.columns:
        raise ValueError("CSV must contain 'param_idx' column")

    param_indices = df['param_idx'].to_list()

    # Get parameter column names (exclude param_idx)
    param_columns = [col for col in df.columns if col != 'param_idx']

    # Build list of (param_idx, param_dict) tuples
    parameters = []
    for i, param_idx in enumerate(param_indices):
        param_dict = {}
        for col in param_columns:
            param_dict[col] = float(df[col][i])
        parameters.append((param_idx, param_dict))

    print(f"Loaded {len(parameters)} parameter sets from {csv_path}")
    print(f"Parameter columns: {param_columns}")

    return parameters

@ray.remote(num_gpus=1)
class PolicyWorker:
    """Single GPU worker for batched policy inference"""

    def __init__(self, checkpoint_path: str, num_states: int, num_actions: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use loading_network to load Ray checkpoint - always load muscle network
        self.pdnet, self.mcn = loading_network(
            str(checkpoint_path),
            num_states=num_states,
            num_actions=num_actions,
            use_musclenet=True,
            device=str(self.device)
        )
        
        # Store muscle network weights on CPU for environment setup
        self.muscle_weights = None
        if self.mcn is not None:
            self.mcn = self.mcn.to('cpu')
            self.muscle_weights = self.mcn.state_dict()
    
    def get_muscle_weights(self):
        """Return muscle network weights for environment setup"""
        return self.muscle_weights
    
    def compute_actions(self, states: List[np.ndarray]) -> List[np.ndarray]:
        """Batch inference using PolicyNN.get_action()"""
        actions = []
        for state in states:
            action = self.pdnet.get_action(state)
            actions.append(action)
        return actions
    
    def get_mcn_weights(self):
        return self.mcn.state_dict()

@ray.remote(num_cpus=1)
class EnvWorker:
    """CPU worker for environment simulation and data collection"""

    def __init__(self, env_idx: int, metadata_xml: str, record_config_path: str, target_cycles: int,
                 param_assignments: Optional[List[Tuple[int, Dict[str, float]]]] = None):
        self.env_idx = env_idx
        self.target_cycles = target_cycles
        self.param_assignments = param_assignments or []

        # Write metadata XML to temporary file
        # RolloutEnvironment expects a file path, not XML string
        self.temp_metadata_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.temp_metadata_file.write(metadata_xml)
        self.temp_metadata_file.close()

        # Create rollout environment
        self.rollout_env = RolloutEnvironment(self.temp_metadata_file.name)
        self.rollout_env.load_config(record_config_path)

        # Get fields and create record buffer
        self.fields = self.rollout_env.get_record_fields()
        self.record = RolloutRecord(self.fields)

        # Storage for all parameter rollouts
        self.all_rollout_data = []  # List of (param_idx, data, fields)
    
    def reset(self, param_dict: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Reset environment and record buffer, optionally setting parameters"""
        # Initialize parameters BEFORE reset (matches training setup in ray_env.py)
        if param_dict is None:
            self.rollout_env.set_parameters({})  # Empty dict triggers default parameter sampling
        else:
            self.rollout_env.set_parameters(param_dict)
        self.rollout_env.reset()
        self.record.reset()
        return self.rollout_env.get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, int, bool, bool]:
        """Step environment and record data

        Returns:
            state: Current environment state
            cycle_count: Number of completed cycles
            is_done: Whether rollout is finished
            early_termination: True if ended due to EOE before reaching target cycles
        """
        self.rollout_env.set_action(action)
        self.rollout_env.step(self.record)

        state = self.rollout_env.get_state()
        cycle_count = self.rollout_env.get_cycle_count()
        is_eoe = (self.rollout_env.is_eoe() != 0)
        reached_target = (cycle_count >= self.target_cycles)

        is_done = is_eoe or reached_target
        early_termination = is_eoe and not reached_target

        return state, cycle_count, is_done, early_termination

    def save_current_rollout(self, param_idx: int, success: bool = True):
        """Save current rollout data to internal storage

        Args:
            param_idx: Parameter index for this rollout
            success: Whether rollout completed successfully (reached target cycles)
        """
        self.all_rollout_data.append((param_idx, self.record.data.copy(), self.record.matrix_data, self.fields, success))

    def get_all_data(self) -> List[Tuple[int, np.ndarray, Dict[str, np.ndarray], List[str], bool]]:
        """Get all recorded rollout data"""
        return self.all_rollout_data
    
    def set_mcn_weights(self, weights):
        self.rollout_env.set_mcn_weights(weights)

def save_to_hdf5(rollout_data: List[Tuple[int, np.ndarray, Dict[str, np.ndarray], List[str], bool]],
                 output_path: str,
                 sample_dir: Path):
    """Save collected rollout data to HDF5 with hierarchical structure

    Structure: /param_{idx}/cycle_{idx}/field_name datasets

    Args:
        rollout_data: List of (param_idx, data, matrix_data, fields, success) tuples
        output_path: Output HDF5 file path
        sample_dir: Sample directory for error log
    """
    total_rows = 0
    failed_params = []

    with h5py.File(output_path, 'w') as f:
        for param_idx, data, matrix_data, fields, success in rollout_data:
            if data.shape[0] == 0:
                continue  # Skip empty rollouts

            # Create group for this parameter index
            param_group_name = f"param_{param_idx}"
            param_grp = f.create_group(param_group_name)

            # Store metadata as attributes
            param_grp.attrs['param_idx'] = param_idx
            param_grp.attrs['num_steps'] = data.shape[0]
            param_grp.attrs['success'] = success

            # Group by cycle
            if 'cycle' in fields:
                cycle_col_idx = fields.index('cycle')
                cycles = data[:, cycle_col_idx].astype(np.int32)
                unique_cycles = np.unique(cycles)

                for cycle_idx in unique_cycles:
                    cycle_mask = (cycles == cycle_idx)
                    cycle_data = data[cycle_mask]

                    if cycle_data.shape[0] == 0:
                        continue

                    cycle_group_name = f"cycle_{cycle_idx}"
                    cycle_grp = param_grp.create_group(cycle_group_name)
                    cycle_grp.attrs['cycle_idx'] = cycle_idx
                    cycle_grp.attrs['num_steps'] = cycle_data.shape[0]

                    # Store each scalar field as a separate dataset
                    for field_idx, field_name in enumerate(fields):
                        if field_name in matrix_data or field_name == 'cycle':
                            continue

                        field_data = cycle_data[:, field_idx]

                        # Use appropriate dtype
                        if field_name in ['step', 'contact_left', 'contact_right']:
                            cycle_grp.create_dataset(field_name, data=field_data.astype(np.int32),
                                                     compression='gzip', compression_opts=4)
                        else:
                            cycle_grp.create_dataset(field_name, data=field_data.astype(np.float32),
                                                     compression='gzip', compression_opts=4)

                    # Store each matrix field as a separate dataset
                    for field_name, matrix in matrix_data.items():
                        cycle_matrix_data = matrix[cycle_mask]
                        cycle_grp.create_dataset(field_name, data=cycle_matrix_data.astype(np.float32),
                                                   compression='gzip', compression_opts=4)
            else:
                # Fallback for rollouts without cycle data
                print(f"Warning: 'cycle' field not found for param_idx={param_idx}. Saving data without cycle grouping.")
                # Store each scalar field as a separate dataset
                for field_idx, field_name in enumerate(fields):
                    if field_name in matrix_data:
                        continue
                    field_data = data[:, field_idx]
                    if field_name in ['step', 'cycle', 'contact_left', 'contact_right']:
                        param_grp.create_dataset(field_name, data=field_data.astype(np.int32),
                                             compression='gzip', compression_opts=4)
                    else:
                        param_grp.create_dataset(field_name, data=field_data.astype(np.float32),
                                             compression='gzip', compression_opts=4)
                # Store each matrix field
                for field_name, matrix in matrix_data.items():
                    param_grp.create_dataset(field_name, data=matrix.astype(np.float32),
                                         compression='gzip', compression_opts=4)

            total_rows += data.shape[0]

            # Track failed parameters
            if not success:
                failed_params.append(param_idx)

    print(f"Saved {total_rows} total rows across {len(rollout_data)} parameter sets to {output_path}")

    # Write error log if there were failures
    if failed_params:
        error_log_path = sample_dir / "FAILED_PARAMS.txt"
        with open(error_log_path, 'w') as f:
            f.write(f"Failed to sample target cycles for the following parameters:\n")
            for param_idx in failed_params:
                f.write(f"  param_idx={param_idx}\n")
        print(f"⚠️  WARNING: {len(failed_params)} parameter(s) failed to reach target cycles")
        print(f"⚠️  Failed parameters logged to: {error_log_path}")

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

def run_rollout(checkpoint_path: str,
                record_config_path: str,
                output_path: str,
                sample_dir: Path,
                param_file: Optional[str] = None,
                num_workers: int = None):
    """Run distributed rollout with Ray, optionally sweeping over parameters

    Args:
        checkpoint_path: Path to checkpoint directory
        record_config_path: Path to rollout config YAML
        output_path: Output HDF5 file path
        sample_dir: Sample directory for error logging
        param_file: Optional CSV file with parameter sweep
        num_workers: Number of workers (default: auto-detect)
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

    # Clean up temp file
    os.unlink(temp_metadata_file.name)

    # Create single policy worker with correct dimensions
    policy = PolicyWorker.remote(checkpoint_path, num_states, num_actions)

    # Get muscle network weights from policy worker
    mcn_weights_ref = policy.get_mcn_weights.remote()

    # If parameter sweep, distribute parameters across workers
    if parameters:
        # Distribute parameters roughly evenly across workers
        params_per_worker = len(parameters) // num_workers
        param_assignments = []
        for i in range(num_workers):
            start_idx = i * params_per_worker
            if i == num_workers - 1:
                # Last worker gets remainder
                end_idx = len(parameters)
            else:
                end_idx = (i + 1) * params_per_worker
            param_assignments.append(parameters[start_idx:end_idx])

        print(f"Distributed {len(parameters)} parameter sets across {num_workers} workers")

        # Create workers with parameter assignments
        env_workers = [
            EnvWorker.remote(i, metadata_xml, record_config_path, target_cycles, param_assignments[i])
            for i in range(num_workers)
        ]

        # Set muscle network weights on all workers
        ray.get([w.set_mcn_weights.remote(mcn_weights_ref) for w in env_workers])

        # Process all parameter assignments for each worker
        for worker_idx, worker in enumerate(env_workers):
            assigned_params = param_assignments[worker_idx]
            print(f"Worker {worker_idx} processing {len(assigned_params)} parameter sets")

            for param_idx, param_dict in assigned_params:
                # Reset with parameters
                state = ray.get(worker.reset.remote(param_dict))
                done = False
                step_count = 0
                early_term = False
                final_cycle = 0

                while not done:
                    # Get action from policy
                    action = ray.get(policy.compute_actions.remote([state]))[0]

                    # Step environment
                    state, cycle, done, early_term = ray.get(worker.step.remote(action))
                    final_cycle = cycle
                    step_count += 1

                # Check if rollout succeeded
                success = not early_term

                # Save this parameter's rollout data
                ray.get(worker.save_current_rollout.remote(param_idx, success))

                if success:
                    print(f"  ✓ Completed param_idx={param_idx} in {step_count} steps (cycles={final_cycle})")
                else:
                    print(f"  ✗ ERROR: param_idx={param_idx} terminated early at cycle {final_cycle}/{target_cycles} (steps={step_count})")

    else:
        # No parameter sweep - run single rollout per worker
        env_workers = [
            EnvWorker.remote(i, metadata_xml, record_config_path, target_cycles)
            for i in range(num_workers)
        ]

        # Set muscle network weights on all workers
        ray.get([w.set_mcn_weights.remote(mcn_weights_ref) for w in env_workers])

        # Reset all environments
        states = ray.get([w.reset.remote() for w in env_workers])
        dones = [False] * num_workers
        early_terminations = [False] * num_workers
        final_cycles = [0] * num_workers

        step_count = 0
        while not all(dones):
            # Centralized action computation (single GPU)
            active_states = [s for s, d in zip(states, dones) if not d]
            active_indices = [i for i, d in enumerate(dones) if not d]

            if not active_states:
                print(f"No active states, breaking")
                break

            actions = ray.get(policy.compute_actions.remote(active_states))

            # Distributed stepping (multiple CPUs)
            step_futures = []
            for idx, action in zip(active_indices, actions):
                step_futures.append(env_workers[idx].step.remote(action))

            results = ray.get(step_futures)

            # Update states and check for completion
            for local_idx, global_idx in enumerate(active_indices):
                state, cycle, done, early_term = results[local_idx]
                states[global_idx] = state
                dones[global_idx] = done
                early_terminations[global_idx] = early_term
                final_cycles[global_idx] = cycle

            step_count += 1
            if step_count % 100 == 0:
                active_count = sum(1 for d in dones if not d)
                print(f"Step {step_count}, {active_count} environments active")

        print(f"Rollout completed in {step_count} steps")

        # Save each worker's data with worker_id as param_idx
        for worker_idx, worker in enumerate(env_workers):
            success = not early_terminations[worker_idx]
            ray.get(worker.save_current_rollout.remote(worker_idx, success))

            if success:
                print(f"  ✓ Worker {worker_idx} completed successfully (cycles={final_cycles[worker_idx]})")
            else:
                print(f"  ✗ ERROR: Worker {worker_idx} terminated early at cycle {final_cycles[worker_idx]}/{target_cycles}")

    # Collect all data from all workers
    all_data = []
    for worker in env_workers:
        worker_data = ray.get(worker.get_all_data.remote())
        all_data.extend(worker_data)

    # Save to HDF5
    save_to_hdf5(all_data, output_path, sample_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ray-based rollout for BidirectionalGaitNet")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory (contains metadata)")
    parser.add_argument("--config", required=True, help="Path to record config YAML (contains target cycles)")
    parser.add_argument("--param-file", default=None, help="CSV file with parameter sweep (optional)")
    parser.add_argument("--workers", type=int, default=16, help="Number of workers (default: auto-detect)")
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

    run_rollout(
        checkpoint_path=args.checkpoint,
        record_config_path=args.config,
        output_path=str(output_path),
        sample_dir=sample_dir,
        param_file=args.param_file,
        num_workers=args.workers
    )

