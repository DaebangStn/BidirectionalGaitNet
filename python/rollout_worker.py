import ray
import torch
import numpy as np
import yaml
import pickle
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional, NamedTuple
from pyrollout import RolloutEnvironment, RolloutRecord
from ray_model import SelectiveUnpickler, loading_network
from io import BytesIO


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
                 global_metadata: Optional[Dict] = None):
    """Save collected rollout data to HDF5 with hierarchical structure

    Structure:
        / (root) - global attributes and datasets
        /param_{idx}/cycle_{idx}/field_name datasets

    Args:
        rollout_data: List of (param_idx, data, matrix_data, fields, success, param_state, cycle_attributes, param_attributes) tuples
        output_path: Output HDF5 file path
        sample_dir: Sample directory for error log
        mode: File mode - 'w' for create/overwrite, 'a' for append
        global_metadata: Optional dict with global metadata (parameter_names, checkpoint_name, etc.)
    """
    import h5py  # Import here to avoid Ray worker import issues

    total_rows = 0
    failed_params = []
    empty_params = []

    with h5py.File(output_path, mode) as f:
        # Write global metadata on first write
        if global_metadata is not None and mode == 'w':
            # Store parameter names as dataset
            if global_metadata.get('parameter_names'):
                param_names_str = [name.encode('utf-8') for name in global_metadata['parameter_names']]
                f.create_dataset('parameter_names', data=param_names_str, dtype=h5py.string_dtype())

            # Store checkpoint name as attribute
            if global_metadata.get('checkpoint_name'):
                f.attrs['checkpoint_name'] = global_metadata['checkpoint_name']

            # Store metadata XML as dataset (can be large)
            if global_metadata.get('metadata_xml'):
                f.create_dataset('metadata_xml', data=global_metadata['metadata_xml'].encode('utf-8'),
                                dtype=h5py.string_dtype())

            # Store config content as dataset
            if global_metadata.get('config_content'):
                f.create_dataset('config_content', data=global_metadata['config_content'].encode('utf-8'),
                                dtype=h5py.string_dtype())

            # Initialize statistics (will be updated in finalize)
            f.attrs['total_samples'] = 0
            f.attrs['failed_samples'] = 0
            f.attrs['success_samples'] = 0

        for param_idx, data, matrix_data, fields, success, param_state, cycle_attributes, param_attributes in rollout_data:
            # Create group for this parameter index
            param_group_name = f"param_{param_idx}"
            param_grp = f.create_group(param_group_name)

            # Track empty rollouts
            if data.shape[0] == 0:
                empty_params.append(param_idx)
                # Store minimal metadata for empty rollouts
                param_grp.attrs['param_idx'] = param_idx
                param_grp.attrs['num_steps'] = 0
                param_grp.attrs['success'] = False
                param_grp.attrs['error'] = 'empty_data'
                # Write param attributes generically
                for attr_name, attr_value in (param_attributes or {}).items():
                    if isinstance(attr_value, (int, float, bool, str)):
                        param_grp.attrs[attr_name] = attr_value
                    elif isinstance(attr_value, np.ndarray) and attr_value.size == 1:
                        param_grp.attrs[attr_name] = float(attr_value)
                continue

            # Store standard metadata as attributes
            param_grp.attrs['param_idx'] = param_idx
            param_grp.attrs['num_steps'] = data.shape[0]
            param_grp.attrs['success'] = success

            # Write param attributes generically (flexible - any new attribute auto-written!)
            for attr_name, attr_value in (param_attributes or {}).items():
                if isinstance(attr_value, (int, float, bool, str)):
                    param_grp.attrs[attr_name] = attr_value
                elif isinstance(attr_value, np.ndarray) and attr_value.size == 1:
                    param_grp.attrs[attr_name] = float(attr_value)

            # Store parameter state if available
            if param_state is not None:
                param_grp.create_dataset('param_state', data=param_state.astype(np.float32),
                                        compression='gzip', compression_opts=4)

            # Write averaged attributes (from StatisticsFilter)
            if '_averaged_attributes' in matrix_data:
                averaged_attrs = matrix_data['_averaged_attributes']
                for attr_name, attr_value in averaged_attrs.items():
                    if isinstance(attr_value, (int, float, bool)):
                        param_grp.attrs[attr_name] = attr_value
                    elif isinstance(attr_value, np.ndarray) and attr_value.size == 1:
                        param_grp.attrs[attr_name] = float(attr_value)

            # Write averaged arrays (from AverageFilter)
            if '_averaged_arrays' in matrix_data:
                averaged_arrays = matrix_data['_averaged_arrays']
                for array_path, array_data in averaged_arrays.items():
                    # array_path like 'angle/AnkleR' - h5py creates nested groups automatically
                    param_grp.create_dataset(array_path, data=array_data.astype(np.float32),
                                             compression='gzip', compression_opts=4)

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

                    # Store travel_distance if computed
                    if '_travel_distance' in matrix_data and cycle_idx in matrix_data['_travel_distance']:
                        cycle_grp.attrs['travel_distance'] = float(matrix_data['_travel_distance'][cycle_idx])

                    # Store metabolic cumulative energy and CoT if computed
                    # Check for any _metabolic_* keys in matrix_data
                    for key, value_dict in matrix_data.items():
                        if key.startswith('_metabolic_cumulative_'):
                            # Extract metabolic type from key (e.g., _metabolic_cumulative_A -> A)
                            metabolic_type = key.replace('_metabolic_cumulative_', '')
                            if cycle_idx in value_dict:
                                attr_name = f'metabolic/cumulative/{metabolic_type}'
                                cycle_grp.attrs[attr_name] = float(value_dict[cycle_idx])
                        elif key.startswith('_metabolic_cot_'):
                            # Extract metabolic type from key (e.g., _metabolic_cot_A -> A)
                            metabolic_type = key.replace('_metabolic_cot_', '')
                            if cycle_idx in value_dict:
                                attr_name = f'metabolic/cot/{metabolic_type}'
                                cycle_grp.attrs[attr_name] = float(value_dict[cycle_idx])

                    # Store cycle-level attributes from C++ (if any)
                    if cycle_attributes and cycle_idx in cycle_attributes:
                        for attr_name, attr_value in cycle_attributes[cycle_idx].items():
                            cycle_grp.attrs[attr_name] = float(attr_value)

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
            print(f"⚠️  WARNING: {len(failed_params)} parameter(s) failed to reach target cycles")
        if empty_params:
            print(f"⚠️  WARNING: {len(empty_params)} parameter(s) had empty data (no steps recorded)")
        print(f"⚠️  Failed parameters logged to: {error_log_path}")


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
            use_mcn=True,
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
                 filter_config: Dict = None, record_config: Dict = None):
        self.env_idx = env_idx
        self.target_cycles = target_cycles

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

        # Collect param-level attributes based on config
        self.param_attributes = {}

        # Add environment attributes if configured
        env_config = (record_config or {}).get('environment', {})
        if env_config.get('character_mass', False):
            self.param_attributes['character_mass'] = self.rollout_env.get_mass()

        # Add new attributes here - easy to extend based on config!
        # if env_config.get('skeleton_dof', False):
        #     self.param_attributes['skeleton_dof'] = self.rollout_env.get_skeleton_dof()
        # if env_config.get('simulation_hz', False):
        #     self.param_attributes['simulation_hz'] = self.rollout_env.get_simulation_hz()

        # Initialize filter pipeline for parallel filtering
        # Filters get metabolic_type and mass directly from env when needed
        from data_filters import FilterPipeline
        self.filter_pipeline = FilterPipeline.from_config(
            filter_config or {}, record_config, env=self.rollout_env
        )

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

        # Only treat fall (isEOE == 1) as early termination
        # Ignore horizon limit (isEOE == 3) to allow rollout beyond mHorizon
        eoe_value = self.rollout_env.is_eoe()
        is_fall = (eoe_value == 1)
        reached_target = (cycle_count >= self.target_cycles)

        is_done = is_fall or reached_target
        early_termination = is_fall and not reached_target

        return state, cycle_count, is_done, early_termination

    def handle_done(self, param_idx: int, success: bool, file_worker):
        """
        Called asynchronously when simulation completes.
        Applies filters in parallel, then sends filtered data to FileWorker.
        Executes in EnvWorker's process.

        Args:
            param_idx: Parameter index for this rollout
            success: Whether rollout completed successfully
            file_worker: Ray actor reference to FileWorker

        Returns:
            Tuple of (param_idx, success) for tracking
        """
        data = self.record.data.copy()
        matrix_data = self.record.matrix_data
        cycle_attributes = self.record.cycle_attributes
        fields = self.fields

        # Extract parameter state
        param_state = self.rollout_env.get_param_state()

        # Apply filters in parallel on this worker
        param_idx, data, matrix_data, fields, success, param_state = self.filter_pipeline.apply(
            param_idx, data, matrix_data, fields, success, param_state
        )

        # Wait for FileWorker to receive and buffer the filtered data
        ray.get(file_worker.write_rollout.remote(
            param_idx=param_idx,
            data=data,
            matrix_data=matrix_data,
            fields=fields,
            success=success,
            param_state=param_state,
            cycle_attributes=cycle_attributes,
            param_attributes=self.param_attributes
        ))

        return param_idx, success

    def set_mcn_weights(self, weights):
        self.rollout_env.set_mcn_weights(weights)


@ray.remote(num_cpus=1)
class FileWorker:
    """Dedicated HDF5 writer to prevent race conditions"""

    def __init__(self, output_path: str, sample_dir: Path, buffer_size: int = 10,
                 parameter_names: List[str] = None, checkpoint_name: str = None,
                 metadata_xml: str = None, config_content: str = None):
        self.output_path = output_path
        self.sample_dir = sample_dir
        self.buffer = []
        self.buffer_size = buffer_size
        self.first_write = True

        # Store global metadata
        self.parameter_names = parameter_names or []
        self.checkpoint_name = checkpoint_name or ""
        self.metadata_xml = metadata_xml or ""
        self.config_content = config_content or ""

        # Track statistics
        self.total_samples = 0
        self.failed_samples = 0

        # Ensure output directory exists on this Ray worker node
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Also ensure sample_dir exists for error logging
        if isinstance(sample_dir, str):
            sample_dir = Path(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)

    def write_rollout(self, param_idx: int, data: np.ndarray, matrix_data: Dict[str, np.ndarray],
                     fields: List[str], success: bool, param_state: np.ndarray = None,
                     cycle_attributes: Dict = None, param_attributes: Dict = None):
        """Buffer rollout data and flush if buffer is full"""
        self.buffer.append((param_idx, data, matrix_data, fields, success, param_state, cycle_attributes, param_attributes or {}))

        # Track statistics
        self.total_samples += 1
        if not success:
            self.failed_samples += 1

        if len(self.buffer) >= self.buffer_size:
            self._flush()

    def _flush(self):
        """Write buffered data to HDF5 (data already filtered by EnvWorkers)"""
        if not self.buffer:
            return

        # First write creates file, subsequent writes append
        mode = 'w' if self.first_write else 'a'

        # Pass global metadata on first write
        global_metadata = None
        if self.first_write:
            global_metadata = {
                'parameter_names': self.parameter_names,
                'checkpoint_name': self.checkpoint_name,
                'metadata_xml': self.metadata_xml,
                'config_content': self.config_content
            }

        save_to_hdf5(self.buffer, self.output_path, self.sample_dir, mode=mode,
                     global_metadata=global_metadata)
        self.first_write = False
        self.buffer.clear()

    def finalize(self):
        """Final flush and update statistics"""
        self._flush()

        # Update statistics in HDF5 file
        import h5py
        with h5py.File(self.output_path, 'a') as f:
            f.attrs['total_samples'] = self.total_samples
            f.attrs['failed_samples'] = self.failed_samples
            f.attrs['success_samples'] = self.total_samples - self.failed_samples

        return {"status": "complete", "total": self.total_samples, "failed": self.failed_samples}
