#!/usr/bin/env python3
"""
Rollout CLI with preset support for BidirectionalGaitNet

Usage:
    uv run rollout <preset> --checkpoint <path>
    uv run rollout gait-metabolic --checkpoint ./ray_results/base_anchor_all-026000-1014_092619
    uv run rollout preset1 -c ./ray_results/base_anchor_all-026000-1014_092619

Presets are defined in data/rollout/presets.yaml and provide default values for:
    - config: YAML config file path
    - workers: Number of Ray workers
    - param-file: CSV parameter file
    - sample-dir: Output directory

You can override any preset value:
    uv run rollout gait-metabolic -c <checkpoint> --workers 32 --config custom.yaml
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

def load_presets(presets_path: Path) -> Dict[str, Any]:
    """Load presets from YAML configuration file"""
    if not presets_path.exists():
        print(f"Error: Presets file not found: {presets_path}")
        sys.exit(1)

    with open(presets_path, 'r') as f:
        config = yaml.safe_load(f)

    return config.get('presets', {}), config.get('aliases', {})

def resolve_preset_name(name: str, aliases: Dict[str, str]) -> str:
    """Resolve preset name through aliases"""
    return aliases.get(name, name)

def get_preset(name: str, presets: Dict[str, Any], aliases: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Get preset configuration by name (with alias resolution)"""
    resolved_name = resolve_preset_name(name, aliases)
    return presets.get(resolved_name)

def list_presets(presets: Dict[str, Any], aliases: Dict[str, str]) -> None:
    """List all available presets with descriptions"""
    print("\n=== Available Rollout Presets ===\n")

    for name, config in presets.items():
        desc = config.get('description', 'No description')
        print(f"  {name:20s} - {desc}")
        print(f"    config:     {config.get('config', 'N/A')}")
        print(f"    workers:    {config.get('workers', 'N/A')}")
        print(f"    param-file: {config.get('param-file', 'N/A')}")
        print()

    if aliases:
        print("=== Preset Aliases ===\n")
        for alias, target in aliases.items():
            print(f"  {alias:15s} -> {target}")
        print()

def main():
    """Main CLI entry point"""
    # Find presets file (from python/rollout/ go up to project root)
    project_root = Path(__file__).parent.parent.parent
    presets_path = project_root / "data" / "rollout" / "presets.yaml"

    # Load presets
    presets, aliases = load_presets(presets_path)

    # Create parser
    parser = argparse.ArgumentParser(
        description="Rollout CLI with preset support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("preset",
                       help="Preset name (or 'list' to show available presets)")
    parser.add_argument("-c", "--checkpoint",
                       help="Path to checkpoint directory (REQUIRED unless listing presets)")
    parser.add_argument("--config",
                       help="Override preset config YAML path")
    parser.add_argument("--workers", type=int,
                       help="Override preset worker count")
    parser.add_argument("--param-file",
                       help="Override preset parameter CSV file")
    parser.add_argument("--num-samples", type=int,
                       help="Number of random samples (when --param-file not provided)")
    parser.add_argument("--sample-dir",
                       help="Override preset sample directory")

    args = parser.parse_args()

    # Handle special 'list' command
    if args.preset == "list":
        list_presets(presets, aliases)
        return 0

    # Get preset configuration
    preset_config = get_preset(args.preset, presets, aliases)

    if preset_config is None:
        print(f"Error: Unknown preset '{args.preset}'")
        print(f"\nRun 'uv run rollout list' to see available presets.")
        return 1

    # Checkpoint is required for actual rollout
    if not args.checkpoint:
        print("Error: --checkpoint (-c) is required")
        print(f"\nUsage: uv run rollout {args.preset} --checkpoint <path>")
        return 1

    # Build arguments for ray_rollout.py
    rollout_args = [
        "python", "-m", "python.rollout.ray_rollout",
        "--checkpoint", args.checkpoint,
        "--config", args.config or preset_config.get('config'),
        "--workers", str(args.workers or preset_config.get('workers', 16)),
    ]

    # Add optional parameter file
    param_file = args.param_file or preset_config.get('param-file')
    if param_file:
        rollout_args.extend(["--param-file", param_file])

    # Add optional num-samples
    if args.num_samples:
        rollout_args.extend(["--num-samples", str(args.num_samples)])

    # Add sample directory
    sample_dir = args.sample_dir or preset_config.get('sample-dir', './sampled')
    rollout_args.extend(["--sample-dir", sample_dir])

    # Print preset being used
    print(f"\n=== Using preset: {args.preset} ===")
    if args.preset != resolve_preset_name(args.preset, aliases):
        print(f"    (alias for: {resolve_preset_name(args.preset, aliases)})")
    print(f"    Description: {preset_config.get('description', 'N/A')}")
    print(f"\n=== Rollout Configuration ===")
    print(f"    checkpoint:  {args.checkpoint}")
    print(f"    config:      {args.config or preset_config.get('config')}")
    print(f"    workers:     {args.workers or preset_config.get('workers')}")
    print(f"    param-file:  {param_file or '(random sampling)'}")
    print(f"    sample-dir:  {sample_dir}")
    print()

    # Execute ray_rollout.py
    import subprocess
    try:
        result = subprocess.run(rollout_args, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\nError: Rollout failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n\nRollout interrupted by user")
        return 130

if __name__ == "__main__":
    sys.exit(main())
