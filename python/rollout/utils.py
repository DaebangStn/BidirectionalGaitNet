"""
Non-ray utility functions for rollout module.

These utilities can be used by both ray and non-ray rollout implementations.
"""

import numpy as np
import yaml
import pickle
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional, NamedTuple


class RolloutData(NamedTuple):
    """Structured rollout data for HDF5 storage"""
    param_idx: int
    data: np.ndarray
    matrix_data: Dict[str, np.ndarray]
    fields: List[str]
    success: bool
    param_state: Optional[np.ndarray]
    cycle_attributes: Optional[Dict]
    character_mass: float


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


def get_git_info(repo_path: Optional[str] = None) -> Dict[str, str]:
    """Capture current git commit information

    Args:
        repo_path: Path to git repository (default: current directory)

    Returns:
        Dictionary with 'commit_hash' and 'commit_message' keys
        Returns empty strings if not in a git repository or git command fails
    """
    git_info = {
        'commit_hash': '',
        'commit_message': ''
    }

    try:
        # Set working directory for git commands
        cwd = repo_path if repo_path else None

        # Get commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            git_info['commit_hash'] = result.stdout.strip()

        # Get commit message
        result = subprocess.run(
            ['git', 'log', '-1', '--pretty=%B'],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            git_info['commit_message'] = result.stdout.strip()

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        # Silently handle errors - git info is optional metadata
        pass

    return git_info


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
    import polars as pl
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


def save_to_hdf5(rollout_data: List[Tuple],
                 output_path: str,
                 sample_dir: Path,
                 mode: str = 'w',
                 config: Optional[Dict] = None):
    """Save collected rollout data to HDF5 with hierarchical structure

    Structure:
        / (root) - global attributes and datasets
        /metadata/ - rollout metadata group
            ├─ record_config (recording configuration YAML)
            ├─ simulation_xml (simulation metadata XML)
            ├─ rollout_time (timestamp when rollout started)
            ├─ checkpoint_path (path to neural network checkpoint)
            ├─ param_file (path to parameter CSV or "random")
            ├─ commit_hash (git commit hash)
            └─ commit_message (git commit message)
        /param_{idx}/cycle_{idx}/field_name datasets

    Args:
        rollout_data: List of (param_idx, data, matrix_data, fields, success, param_state, cycle_attributes, param_attributes) tuples
        output_path: Output HDF5 file path
        sample_dir: Sample directory for error log
        mode: File mode - 'w' for create/overwrite, 'a' for append
        config: Optional dict with configuration:
            - parameter_names: List of parameter names
            - checkpoint_name: Checkpoint name (deprecated, use checkpoint_path)
            - checkpoint_path: Path to checkpoint
            - metadata_xml: Simulation metadata XML string
            - config_content: Record configuration YAML string
            - rollout_time: Timestamp when rollout started
            - param_file: Path to parameter CSV file or "random"
            - commit_hash: Git commit hash
            - commit_message: Git commit message
    """
    import h5py  # Import here to avoid Ray worker import issues

    total_rows = 0
    failed_params = []
    empty_params = []

    with h5py.File(output_path, mode) as f:
        # Write global metadata on first write
        if config is not None and mode == 'w':
            # Create metadata group
            metadata_grp = f.create_group('metadata')

            # Store simulation XML
            if config.get('metadata_xml'):
                metadata_grp.create_dataset('simulation_xml',
                                           data=config['metadata_xml'].encode('utf-8'),
                                           dtype=h5py.string_dtype())

            # Store record config
            if config.get('config_content'):
                metadata_grp.create_dataset('record_config',
                                           data=config['config_content'].encode('utf-8'),
                                           dtype=h5py.string_dtype())

            # Store rollout time
            if config.get('rollout_time'):
                metadata_grp.attrs['rollout_time'] = config['rollout_time']

            # Store checkpoint path
            checkpoint_path = config.get('checkpoint_path') or config.get('checkpoint_name', '')
            if checkpoint_path:
                metadata_grp.attrs['checkpoint_path'] = checkpoint_path

            # Store param file path
            if config.get('param_file'):
                metadata_grp.attrs['param_file'] = config['param_file']

            # Store git information
            if config.get('commit_hash'):
                metadata_grp.attrs['commit_hash'] = config['commit_hash']
            if config.get('commit_message'):
                metadata_grp.attrs['commit_message'] = config['commit_message']

            # Store parameter names as dataset at root level
            if config.get('parameter_names'):
                param_names_str = [name.encode('utf-8') for name in config['parameter_names']]
                f.create_dataset('parameter_names', data=param_names_str, dtype=h5py.string_dtype())

            # Store muscle names as dataset at root level (when muscle recording is enabled)
            if config.get('muscle_names'):
                muscle_names_str = [name.encode('utf-8') for name in config['muscle_names']]
                f.create_dataset('muscle_names', data=muscle_names_str, dtype=h5py.string_dtype())

            # Initialize statistics (will be updated in finalize)
            f.attrs['total_samples'] = 0
            f.attrs['failed_samples'] = 0
            f.attrs['success_samples'] = 0

        for param_idx, data, matrix_data, fields, success, param_state, cycle_attributes, param_attributes in rollout_data:
            # Ensure param_idx is integer for HDF5 group naming
            param_idx = int(param_idx)

            # Create group for this parameter index
            param_group_name = f"param_{param_idx}"
            param_grp = f.create_group(param_group_name)

            # Check if cycle statistics are stored (computed before filtering)
            # This is set by CycleStatisticsFilter before RemoveCycleDataFilter removes data
            cycle_stats = matrix_data.get('_cycle_statistics', {})
            has_cycle_stats = isinstance(cycle_stats, dict) and len(cycle_stats) > 0

            # Handle empty data after filtering
            if data.shape[0] == 0:
                # Use cycle statistics if available (data was removed by filter but rollout succeeded)
                # Otherwise this is a truly empty/failed rollout
                if has_cycle_stats and cycle_stats.get('total_steps', 0) > 0:
                    # Data was removed by filter (e.g., RemoveCycleDataFilter) but rollout was successful
                    # Store pre-filter metadata to reflect actual rollout performance
                    param_grp.attrs['param_idx'] = param_idx
                    param_grp.attrs['num_steps'] = cycle_stats['total_steps']
                    param_grp.attrs['num_cycles'] = cycle_stats['num_cycles']
                    param_grp.attrs['success'] = success  # Preserve original success status
                    # Note: Not tracking as empty_params since this is intentional filtering
                else:
                    # Truly empty rollout (failed before filtering)
                    empty_params.append(param_idx)
                    param_grp.attrs['param_idx'] = param_idx
                    param_grp.attrs['num_steps'] = 0
                    param_grp.attrs['success'] = False
                    param_grp.attrs['error'] = 'empty_data'

                # If we have param-level data from filters, continue to write it
                # Otherwise skip to next parameter
                has_param_level_data = (
                    '_averaged_attributes' in matrix_data or
                    '_averaged_arrays' in matrix_data or
                    any(key.startswith('_metabolic_') for key in matrix_data.keys()) or
                    '_travel_distance' in matrix_data
                )
                if not has_param_level_data:
                    continue
            else:
                # Store standard metadata as attributes (data not empty)
                param_grp.attrs['param_idx'] = param_idx
                param_grp.attrs['num_steps'] = data.shape[0]
                param_grp.attrs['success'] = success

            # Write param attributes generically (flexible - any new attribute auto-written!)
            for attr_name, attr_value in (param_attributes or {}).items():
                if isinstance(attr_value, (int, bool, str)):
                    param_grp.attrs[attr_name] = attr_value
                elif isinstance(attr_value, float):
                    param_grp.attrs[attr_name] = np.float32(attr_value)
                elif isinstance(attr_value, np.ndarray) and attr_value.size == 1:
                    param_grp.attrs[attr_name] = np.float32(attr_value)

            # Store parameter state if available
            if param_state is not None:
                param_grp.create_dataset('param_state', data=param_state.astype(np.float32),
                                        compression='gzip', compression_opts=4)

            # Write averaged attributes (from StatisticsFilter)
            if '_averaged_attributes' in matrix_data:
                averaged_attrs = matrix_data['_averaged_attributes']
                # averaged_attrs should be a dict, but check in case of incorrect type
                if isinstance(averaged_attrs, dict):
                    for attr_name, attr_value in averaged_attrs.items():
                        if isinstance(attr_value, (int, bool)):
                            param_grp.attrs[attr_name] = attr_value
                        elif isinstance(attr_value, float):
                            param_grp.attrs[attr_name] = np.float32(attr_value)
                        elif isinstance(attr_value, np.ndarray) and attr_value.size == 1:
                            param_grp.attrs[attr_name] = np.float32(attr_value)

            # Write averaged arrays (from AverageFilter)
            if '_averaged_arrays' in matrix_data:
                averaged_arrays = matrix_data['_averaged_arrays']
                for array_path, array_data in averaged_arrays.items():
                    # array_path like 'angle/AnkleR' - h5py creates nested groups automatically
                    param_grp.create_dataset(array_path, data=array_data.astype(np.float32),
                                             compression='gzip', compression_opts=4)

            # Group by cycle (skip if data is empty after filtering)
            if data.shape[0] > 0 and 'cycle' in fields:
                cycle_col_idx = fields.index('cycle')
                cycles = data[:, cycle_col_idx].astype(np.int32)
                unique_cycles = np.unique(cycles)

                for cycle_idx in unique_cycles:
                    # Ensure cycle_idx is integer for HDF5 group naming
                    cycle_idx = int(cycle_idx)

                    cycle_mask = (cycles == cycle_idx)
                    cycle_data = data[cycle_mask]

                    if cycle_data.shape[0] == 0:
                        continue

                    cycle_group_name = f"cycle_{cycle_idx}"
                    cycle_grp = param_grp.create_group(cycle_group_name)
                    cycle_grp.attrs['cycle_idx'] = cycle_idx
                    cycle_grp.attrs['num_steps'] = cycle_data.shape[0]

                    # Store travel_distance if computed
                    if '_travel_distance' in matrix_data and cycle_idx in matrix_data['_travel_distance']:
                        cycle_grp.attrs['travel_distance'] = np.float32(matrix_data['_travel_distance'][cycle_idx])

                    # Store metabolic cumulative energy and CoT if computed
                    # Check for any _metabolic_* keys in matrix_data
                    for key, value_dict in matrix_data.items():
                        if key.startswith('_metabolic_cumulative_'):
                            # Extract metabolic type from key (e.g., _metabolic_cumulative_A -> A)
                            metabolic_type = key.replace('_metabolic_cumulative_', '')
                            if cycle_idx in value_dict:
                                attr_name = f'metabolic/cumulative/{metabolic_type}'
                                cycle_grp.attrs[attr_name] = np.float32(value_dict[cycle_idx])
                        elif key.startswith('_metabolic_cot_'):
                            # Extract metabolic type from key (e.g., _metabolic_cot_A -> A)
                            metabolic_type = key.replace('_metabolic_cot_', '')
                            if cycle_idx in value_dict:
                                attr_name = f'metabolic/cot/{metabolic_type}'
                                cycle_grp.attrs[attr_name] = np.float32(value_dict[cycle_idx])

                    # Store cycle-level attributes from C++ (if any)
                    if cycle_attributes and cycle_idx in cycle_attributes:
                        for attr_name, attr_value in cycle_attributes[cycle_idx].items():
                            cycle_grp.attrs[attr_name] = np.float32(attr_value)

                    # Store each scalar field as a separate dataset
                    for field_idx, field_name in enumerate(fields):
                        # Skip matrix data and cycle field
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

                    # Store each matrix field as a separate dataset (skip special metadata keys)
                    for field_name, matrix in matrix_data.items():
                        if field_name.startswith('_'):  # Skip metadata like '_travel_distance'
                            continue
                        cycle_matrix_data = matrix[cycle_mask]
                        cycle_grp.create_dataset(field_name, data=cycle_matrix_data.astype(np.float32),
                                                   compression='gzip', compression_opts=4)
            elif data.shape[0] > 0:
                # Fallback for rollouts without cycle data (only if data exists)
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

    print(f"\nSaved {total_rows} total rows across {len(rollout_data)} parameter sets to {output_path}")

    # Write error log if there were failures or empty data
    if failed_params or empty_params:
        error_log_path = sample_dir / "FAILED_PARAMS.txt"
        with open(error_log_path, 'w') as f:
            if failed_params:
                f.write(f"Failed to reach target cycles:\n")
                for param_idx in failed_params:
                    f.write(f"  param_idx={param_idx}\n")
                f.write(f"\n")
            if empty_params:
                f.write(f"Empty data (no steps recorded):\n")
                for param_idx in empty_params:
                    f.write(f"  param_idx={param_idx}\n")

        if failed_params:
            print(f"WARNING: {len(failed_params)} parameter(s) failed to reach target cycles")
        if empty_params:
            print(f"WARNING: {len(empty_params)} parameter(s) had empty data (no steps recorded)")
        print(f"WARNING: Failed parameters logged to: {error_log_path}")
