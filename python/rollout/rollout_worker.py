"""
Ray-based rollout workers for distributed simulation.

For utilities, import from python.rollout.utils directly.
"""

import ray
import torch
import numpy as np
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from python.rollout.pyrollout import RolloutEnvironment, RolloutRecord
from ppo.model import loading_network

# Import shared utilities
from python.rollout.utils import (
    RolloutData,
    load_metadata_from_checkpoint,
    load_config_yaml,
    get_git_info,
    load_parameters_from_csv,
    save_to_hdf5,
)


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
        from python.data_filters import FilterPipeline
        self.filter_pipeline = FilterPipeline.from_config(
            filter_config or {}, record_config, env=self.rollout_env
        )

    def reset(self, param_dict: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Reset environment and record buffer, optionally setting parameters"""
        # Initialize parameters BEFORE reset (matches training setup)
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

        # Check for termination (fall/failure)
        # Rollout ignores truncation to allow continuing beyond mHorizon
        is_fall = self.rollout_env.is_terminated()
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

    def __init__(self, output_path: str, sample_dir: Path, config: Dict = None, buffer_size: int = 10):
        """Initialize FileWorker with config dictionary

        Args:
            output_path: Path to output HDF5 file
            sample_dir: Directory for error logs
            config: Configuration dictionary containing:
                - parameter_names: List of parameter names
                - checkpoint_path: Path to checkpoint (or checkpoint_name for legacy)
                - metadata_xml: Simulation metadata XML string
                - config_content: Record configuration YAML string
                - rollout_time: Timestamp when rollout started
                - param_file: Path to parameter CSV or "random"
            buffer_size: Number of rollouts to buffer before flushing to disk
        """
        self.output_path = output_path
        self.sample_dir = sample_dir
        self.buffer = []
        self.buffer_size = buffer_size
        self.first_write = True

        # Store config dictionary
        self.config = config or {}

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

        # Pass config on first write
        config = self.config if self.first_write else None

        save_to_hdf5(self.buffer, self.output_path, self.sample_dir, mode=mode,
                     config=config)
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
