"""
Single-core checkpoint rollout without Ray dependency.

Loads checkpoint via cleanrl_model.py, runs rollouts sequentially using C++ libtorch
inference, saves results to HDF5.

Usage:
    python -m rollouts.sample_rollout --checkpoint <path>
    python -m rollouts.sample_rollout --checkpoint <path> --param-file <csv>

Example:
    python -m rollouts.sample_rollout --checkpoint runs/latest --num-samples 10
"""

import argparse
import h5py
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

# Import from local sample/ subdirectory
from rollouts.sample.pysamplerollout import RolloutSampleEnv, RecordConfig

# Use cleanrl_model.py for checkpoint loading (NOT ray_model.py)
from python.cleanrl_model import loading_network, loading_metadata

# Reuse utilities (non-ray)
from python.rollout.utils import (
    load_config_yaml,
    load_parameters_from_csv,
    get_git_info,
    save_to_hdf5,
)
from python.data_filters import FilterPipeline
from python.uri_resolver import resolve_path


def _extract_agent_state_dict(policy_wrapper) -> dict:
    """
    Extract raw state_dict from CleanRLPolicyWrapper for C++ loading.

    The CleanRLPolicyWrapper stores actor_mean, actor_logstd, critic separately.
    We need to reconstruct a state_dict that matches the C++ PolicyNet expectations.

    Args:
        policy_wrapper: CleanRLPolicyWrapper instance from cleanrl_model.py

    Returns:
        dict: State dict with keys matching C++ PolicyNet layer names
    """
    state_dict = {}

    # Actor mean layers (actor_mean is nn.Sequential with 4 Linear layers)
    # Indices: 0, 2, 4, 6 for linear layers (1, 3, 5 are ReLU)
    actor_mean = policy_wrapper.actor_mean
    state_dict['actor_mean.0.weight'] = actor_mean[0].weight.data
    state_dict['actor_mean.0.bias'] = actor_mean[0].bias.data
    state_dict['actor_mean.2.weight'] = actor_mean[2].weight.data
    state_dict['actor_mean.2.bias'] = actor_mean[2].bias.data
    state_dict['actor_mean.4.weight'] = actor_mean[4].weight.data
    state_dict['actor_mean.4.bias'] = actor_mean[4].bias.data
    state_dict['actor_mean.6.weight'] = actor_mean[6].weight.data
    state_dict['actor_mean.6.bias'] = actor_mean[6].bias.data

    # Actor log std
    state_dict['actor_logstd'] = policy_wrapper.actor_logstd.data

    # Critic layers (critic is nn.Sequential with 4 Linear layers)
    critic = policy_wrapper.critic
    state_dict['critic.0.weight'] = critic[0].weight.data
    state_dict['critic.0.bias'] = critic[0].bias.data
    state_dict['critic.2.weight'] = critic[2].weight.data
    state_dict['critic.2.bias'] = critic[2].bias.data
    state_dict['critic.4.weight'] = critic[4].weight.data
    state_dict['critic.4.bias'] = critic[4].bias.data
    state_dict['critic.6.weight'] = critic[6].weight.data
    state_dict['critic.6.bias'] = critic[6].bias.data

    return state_dict


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


def run_sample_rollout(
    checkpoint_path: str,
    record_config_path: str,
    output_path: str,
    sample_dir: Path,
    param_file: Optional[str] = None,
    num_samples: int = 1,
):
    """Run sequential rollouts and save to HDF5

    Args:
        checkpoint_path: Path to checkpoint directory
        record_config_path: Path to rollout config YAML
        output_path: Output HDF5 file path
        sample_dir: Sample directory for error logging
        param_file: Optional CSV file with parameter sweep
        num_samples: Number of random samples when param_file is not provided
    """

    # 1. Resolve paths
    checkpoint_path = resolve_path(checkpoint_path)
    checkpoint_path = str(Path(checkpoint_path).resolve())
    record_config_path = resolve_path(record_config_path)
    record_config_path = str(Path(record_config_path).resolve())

    # 2. Load checkpoint metadata using cleanrl_model.py
    print(f"Loading metadata from checkpoint: {checkpoint_path}")
    metadata_xml = loading_metadata(checkpoint_path)

    print(f"Loading rollout configuration: {record_config_path}")
    config = load_config_yaml(record_config_path)
    target_cycles = config.get('sample', {}).get('cycle', 5)
    filter_config = config.get('filters', {})

    print(f"Target cycles: {target_cycles}")

    # 3. Create environment
    print("Creating RolloutSampleEnv...")
    env = RolloutSampleEnv(metadata_xml)
    env.load_config(record_config_path)
    env.set_target_cycles(target_cycles)

    num_states = env.get_state_dim()
    num_actions = env.get_action_dim()
    use_mcn = env.is_hierarchical()

    print(f"State dimension: {num_states}, Action dimension: {num_actions}")
    print(f"Hierarchical control: {use_mcn}")

    # 4. Load policy/muscle weights using cleanrl_model.py
    print("Loading network weights...")
    policy_wrapper, muscle_state_dict = loading_network(
        checkpoint_path,
        num_states=num_states,
        num_actions=num_actions,
        use_mcn=use_mcn,
        device="cpu"
    )

    # 5. Pass weights to C++ for libtorch inference
    agent_state_dict = _extract_agent_state_dict(policy_wrapper)
    env.load_policy_weights(agent_state_dict)

    if muscle_state_dict:
        env.load_muscle_weights(muscle_state_dict)

    # 6. Load parameters
    if param_file:
        param_file = resolve_path(param_file)
        param_file = str(Path(param_file).resolve())
        print(f"Loading parameters from: {param_file}")
        parameters = load_parameters_from_csv(param_file)
    else:
        print(f"Using random parameter sampling ({num_samples} samples)")
        parameters = [(i, None) for i in range(num_samples)]

    # 7. Create filter pipeline (pass env for motion interpolation)
    filter_pipeline = FilterPipeline.from_config(filter_config, config, env=env)
    filter_pipeline.print_pipeline()

    # 8. Prepare config for HDF5 metadata
    git_info = get_git_info()
    parameter_names = env.get_parameter_names()

    with open(record_config_path, 'r') as f:
        config_content = f.read()

    file_config = {
        'parameter_names': parameter_names,
        'checkpoint_path': checkpoint_path,
        'metadata_xml': metadata_xml,
        'config_content': config_content,
        'rollout_time': datetime.now().isoformat(),
        'param_file': param_file if param_file else "random",
        'commit_hash': git_info.get('commit_hash', ''),
        'commit_message': git_info.get('commit_message', '')
    }

    # 9. Run rollouts and collect data
    rollout_data = []
    success_count = 0
    failed_count = 0

    for param_idx, param_dict in tqdm(parameters, desc="Rollouts", unit="sample", ncols=100):
        try:
            # Run rollout
            result = env.collect_rollout(param_dict)

            # Extract data from result
            data = np.array(result['data'])
            matrix_data = dict(result['matrix_data'])
            fields = list(result['fields'])
            success = result['success']
            param_state = np.array(result['param_state'])
            cycle_attributes = dict(result['cycle_attributes'])
            metrics = dict(result['metrics'])

            # Apply filters
            if data.shape[0] > 0:
                _, filtered_data, filtered_matrix_data, filtered_fields, _, _ = filter_pipeline.apply(
                    param_idx, data, matrix_data, fields, success, param_state
                )
                # Extract param_attributes from matrix_data if present
                param_attributes = filtered_matrix_data.pop('_averaged_attributes', {})
            else:
                filtered_data = data
                filtered_matrix_data = matrix_data
                filtered_fields = fields
                param_attributes = {}

            # Add to rollout data
            rollout_data.append((
                param_idx,
                filtered_data,
                filtered_matrix_data,
                filtered_fields,
                success,
                param_state,
                cycle_attributes,
                param_attributes
            ))

            if success:
                success_count += 1
            else:
                failed_count += 1

        except Exception as e:
            print(f"\n  ✗ param_idx={param_idx} error: {e}")
            failed_count += 1
            # Add empty entry for failed rollout
            rollout_data.append((
                param_idx,
                np.array([]),  # empty data
                {},  # empty matrix_data
                [],  # empty fields
                False,  # not success
                None,  # no param_state
                {},  # no cycle_attributes
                {}   # no param_attributes
            ))

    # 10. Save to HDF5
    print(f"\nSaving results to: {output_path}")
    save_to_hdf5(rollout_data, output_path, sample_dir, mode='w', config=file_config)

    # Update final statistics
    with h5py.File(output_path, 'a') as f:
        f.attrs['total_samples'] = len(parameters)
        f.attrs['success_samples'] = success_count
        f.attrs['failed_samples'] = failed_count

    print(f"\n✓ Rollout complete: {success_count} successful, {failed_count} failed")
    print(f"✓ Data saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Single-core checkpoint rollout (non-Ray)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Random sampling with 10 samples
  python -m rollouts.sample_rollout --checkpoint ckpts/latest --num-samples 10

  # Parameter sweep from CSV
  python -m rollouts.sample_rollout --checkpoint ckpts/latest --param-file params.csv

  # Custom config
  python -m rollouts.sample_rollout --checkpoint ckpts/latest --config data/rollout/config/metabolic.yaml
"""
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--config", default="data/rollout/config/angle.yaml",
                        help="Path to record config YAML (default: data/rollout/config/angle.yaml)")
    parser.add_argument("--param-file", default=None,
                        help="CSV file with parameter sweep (optional)")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of random samples (when --param-file not provided, default: 1)")
    parser.add_argument("--sample-dir", default="./sampled",
                        help="Top-level sample directory (subdirectory will be auto-created)")
    parser.add_argument("--output", default=None,
                        help="Output HDF5 file path (default: auto-generated in sample-dir)")

    args = parser.parse_args()

    # Create sample directory with format: [checkpoint]+[config]+on_[timestamp]
    sample_dir = create_sample_directory(
        args.sample_dir,
        args.checkpoint,
        args.config
    )

    # Generate output path inside the sample directory (HDF5)
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = sample_dir / "rollout_data.h5"
        output_path = output_path.resolve()

    sample_dir = sample_dir.resolve()

    run_sample_rollout(
        checkpoint_path=args.checkpoint,
        record_config_path=args.config,
        output_path=str(output_path),
        sample_dir=sample_dir,
        param_file=args.param_file,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
