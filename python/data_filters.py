"""
Data filtering and processing pipeline for rollout data.

Provides scalable, composable filters for cycle-level data conditioning.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np


class DataFilter(ABC):
    """Base class for rollout data filters

    Filters operate on cycle-grouped data and can:
    - Remove cycles (filtering)
    - Modify cycle data (transformation)
    - Add derived fields (augmentation)
    """

    @abstractmethod
    def filter_cycles(
        self,
        cycles_dict: Dict[int, np.ndarray],
        matrix_cycles_dict: Dict[str, Dict[int, np.ndarray]],
        fields: List[str]
    ) -> Tuple[Dict[int, np.ndarray], Dict[str, Dict[int, np.ndarray]], List[str]]:
        """Apply filter to cycle-grouped data

        Args:
            cycles_dict: {cycle_idx: scalar_data[num_steps, num_fields]}
            matrix_cycles_dict: {field_name: {cycle_idx: matrix_data[num_steps, dim]}}
            fields: List of scalar field names

        Returns:
            Filtered cycles_dict, matrix_cycles_dict, updated fields list
        """
        pass


class DropShortCyclesFilter(DataFilter):
    """Remove cycles with fewer than min_steps"""

    def __init__(self, min_steps: int):
        self.min_steps = min_steps

    def filter_cycles(
        self,
        cycles_dict: Dict[int, np.ndarray],
        matrix_cycles_dict: Dict[str, Dict[int, np.ndarray]],
        fields: List[str]
    ) -> Tuple[Dict[int, np.ndarray], Dict[str, Dict[int, np.ndarray]], List[str]]:
        """Keep only cycles with >= min_steps"""

        filtered_cycles = {}
        filtered_matrix_cycles = {field: {} for field in matrix_cycles_dict}

        for cycle_idx, cycle_data in cycles_dict.items():
            if cycle_data.shape[0] >= self.min_steps:
                filtered_cycles[cycle_idx] = cycle_data
                for field_name, matrix_dict in matrix_cycles_dict.items():
                    filtered_matrix_cycles[field_name][cycle_idx] = matrix_dict[cycle_idx]

        return filtered_cycles, filtered_matrix_cycles, fields


class DropFirstNCyclesFilter(DataFilter):
    """Remove first N cycles from rollout"""

    def __init__(self, n: int):
        self.n = n

    def filter_cycles(
        self,
        cycles_dict: Dict[int, np.ndarray],
        matrix_cycles_dict: Dict[str, Dict[int, np.ndarray]],
        fields: List[str]
    ) -> Tuple[Dict[int, np.ndarray], Dict[str, Dict[int, np.ndarray]], List[str]]:
        """Drop first N cycles"""

        if not cycles_dict:
            return cycles_dict, matrix_cycles_dict, fields

        sorted_cycle_indices = sorted(cycles_dict.keys())
        cycles_to_drop = sorted_cycle_indices[:self.n]

        filtered_cycles = {k: v for k, v in cycles_dict.items() if k not in cycles_to_drop}
        filtered_matrix_cycles = {
            field: {k: v for k, v in matrix_dict.items() if k not in cycles_to_drop}
            for field, matrix_dict in matrix_cycles_dict.items()
        }

        return filtered_cycles, filtered_matrix_cycles, fields


class DropLastMCyclesFilter(DataFilter):
    """Remove last M cycles from rollout"""

    def __init__(self, m: int):
        self.m = m

    def filter_cycles(
        self,
        cycles_dict: Dict[int, np.ndarray],
        matrix_cycles_dict: Dict[str, Dict[int, np.ndarray]],
        fields: List[str]
    ) -> Tuple[Dict[int, np.ndarray], Dict[str, Dict[int, np.ndarray]], List[str]]:
        """Drop last M cycles"""

        if not cycles_dict:
            return cycles_dict, matrix_cycles_dict, fields

        sorted_cycle_indices = sorted(cycles_dict.keys())
        cycles_to_drop = sorted_cycle_indices[-self.m:] if self.m > 0 else []

        filtered_cycles = {k: v for k, v in cycles_dict.items() if k not in cycles_to_drop}
        filtered_matrix_cycles = {
            field: {k: v for k, v in matrix_dict.items() if k not in cycles_to_drop}
            for field, matrix_dict in matrix_cycles_dict.items()
        }

        return filtered_cycles, filtered_matrix_cycles, fields


class ComputeTravelDistanceFilter(DataFilter):
    """Compute travel distance from root position data

    Adds 'travel_distance' field computed from cumulative displacement.
    Requires 'root' in matrix_cycles_dict with shape [num_steps, 3].
    """

    def __init__(self, record_config: Optional[Dict] = None):
        """Initialize and validate config

        Args:
            record_config: Optional full record config dict for validation
        """
        # Check config if provided
        if record_config:
            kinematics = record_config.get('record', {}).get('kinematics', {})
            kinematics_enabled = kinematics.get('enabled', False)
            root_enabled = kinematics.get('root', False)

            if not kinematics_enabled or not root_enabled:
                print(f"⚠️  WARNING: compute_travel_distance enabled but root recording is disabled")
                print(f"    record.kinematics.enabled = {kinematics_enabled}")
                print(f"    record.kinematics.root = {root_enabled}")
                print(f"    Enable both in config YAML to compute travel_distance")

    def filter_cycles(
        self,
        cycles_dict: Dict[int, np.ndarray],
        matrix_cycles_dict: Dict[str, Dict[int, np.ndarray]],
        fields: List[str]
    ) -> Tuple[Dict[int, np.ndarray], Dict[str, Dict[int, np.ndarray]], List[str]]:
        """Compute and add travel_distance as cycle metadata"""

        # Check if root position components exist in scalar fields
        has_root_x = 'root/x' in fields
        has_root_z = 'root/z' in fields

        if not (has_root_x and has_root_z):
            # Skip silently - warning already printed in __init__
            return cycles_dict, matrix_cycles_dict, fields

        # Get column indices for root/x and root/z
        root_x_idx = fields.index('root/x')
        root_z_idx = fields.index('root/z')

        # Compute total travel distance for each cycle (first to last frame)
        # Store in special matrix_cycles_dict key for metadata
        travel_distances = {}
        for cycle_idx, cycle_data in cycles_dict.items():
            if cycle_data.shape[0] > 1:
                # Extract x and z positions from scalar data
                root_x = cycle_data[:, root_x_idx]
                root_z = cycle_data[:, root_z_idx]

                # Compute total distance: sum of all displacements
                positions_xz = np.column_stack([root_x, root_z])
                displacements = np.diff(positions_xz, axis=0)
                distances = np.linalg.norm(displacements, axis=1)
                total_distance = np.sum(distances)

                travel_distances[cycle_idx] = total_distance
            else:
                # Empty or single-frame cycle
                travel_distances[cycle_idx] = 0.0

        # Store travel distances in matrix_cycles_dict with special key
        matrix_cycles_dict['_travel_distance'] = travel_distances

        return cycles_dict, matrix_cycles_dict, fields


class FilterPipeline:
    """Composable pipeline of data filters

    Applies filters sequentially to rollout data before HDF5 write.
    """

    def __init__(self, filters: Optional[List[DataFilter]] = None):
        self.filters = filters or []
        self._cycle_col_idx = None  # Track cycle column index for renumbering

    def add_filter(self, filter: DataFilter):
        """Add filter to pipeline"""
        self.filters.append(filter)

    def print_pipeline(self):
        """Print the filter pipeline execution order"""
        if not self.filters:
            print("Filter Pipeline: [empty]")
            return

        filter_names = []
        for f in self.filters:
            if isinstance(f, DropShortCyclesFilter):
                filter_names.append(f"drop_short_cycles(min={f.min_steps})")
            elif isinstance(f, DropFirstNCyclesFilter):
                filter_names.append(f"drop_first_n_cycles(n={f.n})")
            elif isinstance(f, DropLastMCyclesFilter):
                filter_names.append(f"drop_last_m_cycles(m={f.m})")
            elif isinstance(f, ComputeTravelDistanceFilter):
                filter_names.append("compute_travel_distance")
            else:
                filter_names.append(f.__class__.__name__)

        print("Filter Pipeline: " + " -> ".join(filter_names))

    def apply(
        self,
        param_idx: int,
        data: np.ndarray,
        matrix_data: Dict[str, np.ndarray],
        fields: List[str],
        success: bool,
        param_state: Optional[np.ndarray] = None
    ) -> Tuple[int, np.ndarray, Dict[str, np.ndarray], List[str], bool, Optional[np.ndarray]]:
        """Apply all filters to rollout data

        Args:
            param_idx: Parameter index
            data: Scalar data [num_steps, num_fields]
            matrix_data: Matrix data {field_name: [num_steps, dim]}
            fields: Field names
            success: Whether rollout succeeded
            param_state: Parameter state array

        Returns:
            Filtered (param_idx, data, matrix_data, fields, success, param_state)
        """

        # Early exit if no filters or empty data
        if not self.filters or data.shape[0] == 0:
            return param_idx, data, matrix_data, fields, success, param_state

        # Group data by cycle
        cycles_dict, matrix_cycles_dict = self._group_by_cycle(data, matrix_data, fields)

        # Apply each filter sequentially
        for filter in self.filters:
            cycles_dict, matrix_cycles_dict, fields = filter.filter_cycles(
                cycles_dict, matrix_cycles_dict, fields
            )

        # Reconstruct flat arrays
        data, matrix_data = self._flatten_cycles(cycles_dict, matrix_cycles_dict)

        return param_idx, data, matrix_data, fields, success, param_state

    def _group_by_cycle(
        self,
        data: np.ndarray,
        matrix_data: Dict[str, np.ndarray],
        fields: List[str]
    ) -> Tuple[Dict[int, np.ndarray], Dict[str, Dict[int, np.ndarray]]]:
        """Group data arrays by cycle index"""

        if 'cycle' not in fields:
            # No cycle field, treat as single cycle
            self._cycle_col_idx = None
            return {0: data}, {field: {0: matrix} for field, matrix in matrix_data.items()}

        cycle_col_idx = fields.index('cycle')
        self._cycle_col_idx = cycle_col_idx  # Store for renumbering later
        cycles = data[:, cycle_col_idx].astype(np.int32)
        unique_cycles = np.unique(cycles)

        # Group scalar data
        cycles_dict = {}
        for cycle_idx in unique_cycles:
            cycle_mask = (cycles == cycle_idx)
            cycles_dict[int(cycle_idx)] = data[cycle_mask]

        # Group matrix data
        matrix_cycles_dict = {}
        for field_name, matrix in matrix_data.items():
            matrix_cycles_dict[field_name] = {}
            for cycle_idx in unique_cycles:
                cycle_mask = (cycles == cycle_idx)
                matrix_cycles_dict[field_name][int(cycle_idx)] = matrix[cycle_mask]

        return cycles_dict, matrix_cycles_dict

    def _flatten_cycles(
        self,
        cycles_dict: Dict[int, np.ndarray],
        matrix_cycles_dict: Dict[str, Dict[int, np.ndarray]]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Flatten cycle-grouped data back to arrays with renumbered cycle indices"""

        if not cycles_dict:
            # Return empty arrays
            return np.empty((0, 0)), {field: np.empty((0, 0)) for field in matrix_cycles_dict}

        # Sort cycles by original index for consistent ordering
        sorted_cycle_indices = sorted(cycles_dict.keys())

        # Renumber cycles from 0 sequentially
        renumbered_cycles = []
        for new_cycle_idx, old_cycle_idx in enumerate(sorted_cycle_indices):
            cycle_data = cycles_dict[old_cycle_idx].copy()

            # Update cycle column if it exists
            if self._cycle_col_idx is not None and cycle_data.size > 0:
                # Set all cycle values to new sequential index
                cycle_data[:, self._cycle_col_idx] = new_cycle_idx

            renumbered_cycles.append(cycle_data)

        # Concatenate scalar data
        data = np.concatenate(renumbered_cycles, axis=0)

        # Concatenate matrix data (skip metadata keys starting with '_')
        matrix_data = {}
        for field_name, matrix_dict in matrix_cycles_dict.items():
            if field_name.startswith('_'):
                # Renumber metadata dict keys to match renumbered cycle indices
                renumbered_metadata = {}
                for new_cycle_idx, old_cycle_idx in enumerate(sorted_cycle_indices):
                    if old_cycle_idx in matrix_dict:
                        renumbered_metadata[new_cycle_idx] = matrix_dict[old_cycle_idx]
                matrix_data[field_name] = renumbered_metadata
            else:
                # Concatenate array data
                matrix_data[field_name] = np.concatenate(
                    [matrix_dict[idx] for idx in sorted_cycle_indices], axis=0
                )

        return data, matrix_data

    @staticmethod
    def from_config(filter_config: Dict, record_config: Optional[Dict] = None) -> 'FilterPipeline':
        """Create FilterPipeline from configuration dict

        Args:
            filter_config: Filter configuration dictionary
            record_config: Optional full record config for validation

        Returns:
            Configured FilterPipeline

        Example filter_config:
            {
                'drop_short_cycles': {'enabled': True, 'min_steps': 10},
                'drop_first_n_cycles': {'enabled': True, 'n': 1},
                'drop_last_m_cycles': {'enabled': True, 'm': 1},
                'compute_travel_distance': {'enabled': True}
            }
        """
        pipeline = FilterPipeline()


        # Drop first N cycles
        if filter_config.get('drop_first_n_cycles', {}).get('enabled', False):
            n = filter_config['drop_first_n_cycles'].get('n', 1)
            pipeline.add_filter(DropFirstNCyclesFilter(n))

        # Drop last M cycles
        if filter_config.get('drop_last_m_cycles', {}).get('enabled', False):
            m = filter_config['drop_last_m_cycles'].get('m', 1)
            pipeline.add_filter(DropLastMCyclesFilter(m))

        # Drop short cycles
        if filter_config.get('drop_short_cycles', {}).get('enabled', False):
            min_steps = filter_config['drop_short_cycles'].get('min_steps', 10)
            pipeline.add_filter(DropShortCyclesFilter(min_steps))

        # Compute travel distance
        if filter_config.get('compute_travel_distance', {}).get('enabled', False):
            pipeline.add_filter(ComputeTravelDistanceFilter(record_config))

        return pipeline
