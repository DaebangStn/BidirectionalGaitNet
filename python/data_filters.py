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


class MetabolicCumulativeFilter(DataFilter):
    """Compute cumulative metabolic energy per cycle

    Adds cycle-level attribute 'metabolic/cumulative/{TYPE}' where TYPE is the metabolic energy type.
    Requires 'metabolic/step_energy' in fields.
    """

    def __init__(self, metabolic_type: str, record_config: Optional[Dict] = None):
        """Initialize and validate config

        Args:
            metabolic_type: Energy type (A, A2, MA, MA2, LEGACY)
            record_config: Optional full record config dict for validation
        """
        self.metabolic_type = metabolic_type

        # Check config if provided
        if record_config:
            metabolic = record_config.get('record', {}).get('metabolic', {})
            metabolic_enabled = metabolic.get('enabled', False)

            if not metabolic_enabled:
                print(f"⚠️  WARNING: MetabolicCumulativeFilter enabled but step_energy recording is disabled")
                print(f"    record.metabolic.enabled = {metabolic_enabled}")

    def filter_cycles(
        self,
        cycles_dict: Dict[int, np.ndarray],
        matrix_cycles_dict: Dict[str, Dict[int, np.ndarray]],
        fields: List[str]
    ) -> Tuple[Dict[int, np.ndarray], Dict[str, Dict[int, np.ndarray]], List[str]]:
        """Compute cumulative metabolic energy for each cycle"""

        # Check if metabolic/step_energy exists in scalar fields
        if 'metabolic/step_energy' not in fields:
            # Skip silently - metabolic recording not enabled
            return cycles_dict, matrix_cycles_dict, fields

        # Get column index for metabolic/step_energy
        step_energy_idx = fields.index('metabolic/step_energy')

        # Compute cumulative energy for each cycle
        # Store as cycle-level metadata with metabolic type in key
        cumulative_key = f'_metabolic_cumulative_{self.metabolic_type}'
        cumulative_energies = {}

        for cycle_idx, cycle_data in cycles_dict.items():
            if cycle_data.shape[0] > 0:
                # Extract step energies and sum
                step_energies = cycle_data[:, step_energy_idx]
                cumulative_energy = np.sum(step_energies)
                cumulative_energies[cycle_idx] = cumulative_energy
            else:
                cumulative_energies[cycle_idx] = 0.0

        # Store cumulative energies in matrix_cycles_dict with special key
        matrix_cycles_dict[cumulative_key] = cumulative_energies

        return cycles_dict, matrix_cycles_dict, fields


class ComputeCoTFilter(DataFilter):
    """Compute Cost of Transport (CoT) per cycle

    CoT = cumulative_energy / (character_mass × travel_distance)

    Adds cycle-level attribute 'metabolic/cot/{TYPE}' where TYPE is the metabolic energy type.
    Requires both '_metabolic_cumulative_{TYPE}' and '_travel_distance' in matrix_cycles_dict.
    """

    def __init__(self, metabolic_type: str, character_mass: float, record_config: Optional[Dict] = None,
                 filter_config: Optional[Dict] = None):
        """Initialize and validate config

        Args:
            metabolic_type: Energy type (A, A2, MA, MA2, LEGACY)
            character_mass: Character mass in kg
            record_config: Optional full record config dict for validation
            filter_config: Optional filter config dict for dependency validation
        """
        self.metabolic_type = metabolic_type
        self.character_mass = character_mass

        if character_mass <= 0:
            raise ValueError(f"character_mass must be positive, got {character_mass}")

        # Check record config if provided
        if record_config:
            metabolic = record_config.get('record', {}).get('metabolic', {})
            kinematics = record_config.get('record', {}).get('kinematics', {})

            metabolic_enabled = metabolic.get('enabled', False)
            kinematics_enabled = kinematics.get('enabled', False)
            root_enabled = kinematics.get('root', False)

            if not metabolic_enabled:
                print(f"⚠️  WARNING: ComputeCoTFilter requires metabolic energy recording")
                print(f"    record.metabolic.enabled = {metabolic_enabled}")

            if not (kinematics_enabled and root_enabled):
                print(f"⚠️  WARNING: ComputeCoTFilter requires travel distance (root position)")
                print(f"    record.kinematics.enabled = {kinematics_enabled}")
                print(f"    record.kinematics.root = {root_enabled}")

        # Check filter dependencies
        if filter_config:
            cumulative_enabled = filter_config.get('metabolic_cumulative', {}).get('enabled', False)
            travel_enabled = filter_config.get('compute_travel_distance', {}).get('enabled', False)

            if not cumulative_enabled:
                print(f"⚠️  WARNING: ComputeCoTFilter requires metabolic_cumulative filter")
                print(f"    filters.metabolic_cumulative.enabled = {cumulative_enabled}")

            if not travel_enabled:
                print(f"⚠️  WARNING: ComputeCoTFilter requires compute_travel_distance filter")
                print(f"    filters.compute_travel_distance.enabled = {travel_enabled}")

    def filter_cycles(
        self,
        cycles_dict: Dict[int, np.ndarray],
        matrix_cycles_dict: Dict[str, Dict[int, np.ndarray]],
        fields: List[str]
    ) -> Tuple[Dict[int, np.ndarray], Dict[str, Dict[int, np.ndarray]], List[str]]:
        """Compute Cost of Transport for each cycle"""

        cumulative_key = f'_metabolic_cumulative_{self.metabolic_type}'
        travel_key = '_travel_distance'

        # Check if both dependencies exist
        if cumulative_key not in matrix_cycles_dict:
            print(f"⚠️  WARNING: {cumulative_key} not found, skipping CoT computation")
            return cycles_dict, matrix_cycles_dict, fields

        if travel_key not in matrix_cycles_dict:
            print(f"⚠️  WARNING: {travel_key} not found, skipping CoT computation")
            return cycles_dict, matrix_cycles_dict, fields

        # Compute CoT for each cycle
        cot_key = f'_metabolic_cot_{self.metabolic_type}'
        cot_values = {}

        cumulative_energies = matrix_cycles_dict[cumulative_key]
        travel_distances = matrix_cycles_dict[travel_key]

        for cycle_idx in cycles_dict.keys():
            if cycle_idx in cumulative_energies and cycle_idx in travel_distances:
                cumulative = cumulative_energies[cycle_idx]
                distance = travel_distances[cycle_idx]

                # CoT = Energy / (Mass × Distance)
                # Avoid division by zero
                if distance > 0 and self.character_mass > 0:
                    cot = cumulative / (self.character_mass * distance)
                    cot_values[cycle_idx] = cot
                else:
                    cot_values[cycle_idx] = 0.0
            else:
                cot_values[cycle_idx] = 0.0

        # Store CoT values in matrix_cycles_dict with special key
        matrix_cycles_dict[cot_key] = cot_values

        return cycles_dict, matrix_cycles_dict, fields


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


class StatisticsFilter(DataFilter):
    """Compute statistics (mean, std) of cycle-level attributes and store as param-level attributes

    Takes specified attribute keys (e.g., 'metabolic/cot/MA', 'metabolic/cumulative/MA')
    and computes their mean and standard deviation across all cycles. Stores results in a special
    '_averaged_attributes' dictionary that the HDF5 writer recognizes and writes
    to param-level group instead of cycle-level.

    Example:
        If you have 'metabolic/cot/MA' for cycles [0, 1, 2] with values [1.5, 1.6, 1.4],
        this filter will compute:
        - 'metabolic/cot/MA/mean' = 1.5
        - 'metabolic/cot/MA/std' = 0.0816...

    Config:
        stat_filter:
          enabled: true
          keys: ['metabolic/cot/MA', 'metabolic/cumulative/MA']
    """

    def __init__(self, keys: List[str]):
        """Initialize with list of attribute keys for statistics computation

        Args:
            keys: List of attribute keys to compute statistics for (e.g., ['metabolic/cot/MA'])
        """
        self.keys = keys
        if not keys:
            print("⚠️  WARNING: StatisticsFilter initialized with empty keys list")

    def filter_cycles(
        self,
        cycles_dict: Dict[int, np.ndarray],
        matrix_cycles_dict: Dict[str, Dict[int, np.ndarray]],
        fields: List[str]
    ) -> Tuple[Dict[int, np.ndarray], Dict[str, Dict[int, np.ndarray]], List[str]]:
        """Compute statistics (mean, std) for specified attributes across all cycles"""

        if not self.keys:
            return cycles_dict, matrix_cycles_dict, fields

        # Initialize statistics attributes dictionary
        stat_attributes = {}

        # Process each key
        for key in self.keys:
            # Convert external key to internal key format (add _ prefix for internal keys)
            internal_key = f'_{key.replace("/", "_")}'

            # Check if this key exists in matrix_cycles_dict
            if internal_key not in matrix_cycles_dict:
                print(f"⚠️  WARNING: StatisticsFilter key '{key}' not found in matrix_cycles_dict (looking for '{internal_key}')")
                continue

            # Get all values for this attribute across cycles
            attribute_dict = matrix_cycles_dict[internal_key]

            if not attribute_dict:
                print(f"⚠️  WARNING: StatisticsFilter key '{key}' has no cycle data")
                continue

            # Compute statistics
            values = list(attribute_dict.values())
            if values:
                mean_value = np.mean(values)
                std_value = np.std(values)

                # Store with /mean and /std suffixes
                stat_attributes[f'{key}/mean'] = mean_value
                stat_attributes[f'{key}/std'] = std_value

        # Store statistics attributes in special key for HDF5 writer
        if stat_attributes:
            matrix_cycles_dict['_averaged_attributes'] = stat_attributes

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
        """Print the filter pipeline execution order in a visible box format"""
        if not self.filters:
            print("\n╔════════════════════════════════════╗")
            print("║     FILTER PIPELINE: [EMPTY]      ║")
            print("╚════════════════════════════════════╝\n")
            return

        filter_names = []
        for f in self.filters:
            if isinstance(f, DropShortCyclesFilter):
                filter_names.append(f"drop_short_cycles(min={f.min_steps})")
            elif isinstance(f, DropFirstNCyclesFilter):
                filter_names.append(f"drop_first_n_cycles(n={f.n})")
            elif isinstance(f, DropLastMCyclesFilter):
                filter_names.append(f"drop_last_m_cycles(m={f.m})")
            elif isinstance(f, MetabolicCumulativeFilter):
                filter_names.append("metabolic_cumulative")
            elif isinstance(f, ComputeTravelDistanceFilter):
                filter_names.append("compute_travel_distance")
            elif isinstance(f, ComputeCoTFilter):
                filter_names.append(f"compute_cot(type={f.metabolic_type})")
            elif isinstance(f, StatisticsFilter):
                keys_str = ", ".join(f.keys[:3])  # Show first 3 keys
                if len(f.keys) > 3:
                    keys_str += f" (+{len(f.keys)-3} more)"
                filter_names.append(f"statistics({keys_str})")
            else:
                filter_names.append(f.__class__.__name__)

        # Find max width for box
        max_width = max(len(name) for name in filter_names)
        box_width = max(max_width + 4, 40)

        print("\n╔" + "═" * box_width + "╗")
        print("║" + " FILTER PIPELINE".center(box_width) + "║")
        print("╠" + "═" * box_width + "╣")

        for i, name in enumerate(filter_names):
            if i == 0:
                print("║  " + name.ljust(box_width - 3) + " ║")
            else:
                print("║  " + "↓".ljust(box_width - 3) + " ║")
                print("║  " + name.ljust(box_width - 3) + " ║")

        print("╚" + "═" * box_width + "╝\n")

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
                # Special case: _averaged_attributes is NOT cycle-indexed
                if field_name == '_averaged_attributes':
                    # Pass through unchanged - it's a simple {attr_name: value} dict
                    matrix_data[field_name] = matrix_dict
                else:
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
    def from_config(filter_config: Dict, record_config: Optional[Dict] = None,
                   metabolic_type: str = 'LEGACY', character_mass: float = 0.0) -> 'FilterPipeline':
        """Create FilterPipeline from configuration dict

        Args:
            filter_config: Filter configuration dictionary
            record_config: Optional full record config for validation
            metabolic_type: Metabolic energy type (A, A2, MA, MA2, LEGACY)
            character_mass: Character mass in kg (required for CoT filter)

        Returns:
            Configured FilterPipeline

        Example filter_config:
            {
                'drop_short_cycles': {'enabled': True, 'min_steps': 10},
                'drop_first_n_cycles': {'enabled': True, 'n': 1},
                'drop_last_m_cycles': {'enabled': True, 'm': 1},
                'metabolic_cumulative': {'enabled': True},
                'compute_travel_distance': {'enabled': True},
                'metabolic_cot': {'enabled': True}  # Auto-enables dependencies
            }
        """
        pipeline = FilterPipeline()

        # Check if CoT is enabled - if so, auto-enable dependencies
        # CoT requires: metabolic_cumulative AND compute_travel_distance
        cot_enabled = filter_config.get('metabolic_cot', {}).get('enabled', False)

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

        # Metabolic cumulative energy (auto-enabled by CoT if needed)
        metabolic_enabled = filter_config.get('metabolic_cumulative', {}).get('enabled', False)
        if cot_enabled:
            metabolic_enabled = True  # Auto-enable for CoT

        if metabolic_enabled:
            pipeline.add_filter(MetabolicCumulativeFilter(metabolic_type, record_config))

        # Compute travel distance (auto-enabled by CoT if needed)
        travel_distance_enabled = filter_config.get('compute_travel_distance', {}).get('enabled', False)
        if cot_enabled:
            travel_distance_enabled = True  # Auto-enable for CoT

        if travel_distance_enabled:
            pipeline.add_filter(ComputeTravelDistanceFilter(record_config))

        # Compute Cost of Transport (requires cumulative and travel_distance)
        if cot_enabled:
            if character_mass > 0:
                pipeline.add_filter(ComputeCoTFilter(metabolic_type, character_mass, record_config, filter_config))
            else:
                print(f"⚠️  WARNING: metabolic_cot enabled but character_mass not provided or invalid")

        # Compute statistics across cycles (must be last - operates on all cycle data)
        if filter_config.get('stat_filter', {}).get('enabled', False):
            keys = filter_config['stat_filter'].get('keys', [])
            if keys:
                pipeline.add_filter(StatisticsFilter(keys))
            else:
                print(f"⚠️  WARNING: stat_filter enabled but no keys specified")

        return pipeline
