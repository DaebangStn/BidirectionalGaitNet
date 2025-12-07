#!/usr/bin/env python3
"""
Example: Dataclass presets + argparse subcommands with override support.

Usage:
    python args_preset_cli.py                          # Use default values
    python args_preset_cli.py gait                     # Use gait preset
    python args_preset_cli.py a6000 --seed 123         # A6000 preset, override seed
    python args_preset_cli.py --num_envs 64            # Default + override
"""

import sys
from pathlib import Path
from dataclasses import dataclass, fields
from typing import Callable, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ppo.utils import parse_args_with_presets


@dataclass
class Args:
    """Training arguments with preset support."""

    # Core training params
    seed: int = 1
    torch_deterministic: bool = False
    checkpoint_interval: int = 1000

    # Environment params
    num_envs: int = 32
    num_steps: int = 4
    muscle_batch_size: int = 64
    num_minibatches: int = 4
    env_file: str = "data/env/A2.yaml"

    # --- Preset classmethods matching shell scripts ---

    @classmethod
    def gait(cls) -> "Args":
        """Preset matching scripts/train/gait.sh (CPU cluster, 128 cores)."""
        return cls(
            num_envs=128,
            num_steps=128,
            muscle_batch_size=512,
            num_minibatches=16,
        )

    @classmethod
    def a6000(cls) -> "Args":
        """Preset matching scripts/train/a6000.sh (GPU node, 96 cores)."""
        return cls(
            num_envs=96,
            num_steps=128,
            muscle_batch_size=512,
            num_minibatches=16,
        )


# Map preset names to their classmethods
# None key = default constructor (dataclass defaults)
PRESET_FNS: dict[Optional[str], Callable[[], Args]] = {
    None: Args,  # Default: use dataclass defaults
    "gait": Args.gait,
    "a6000": Args.a6000,
}


if __name__ == "__main__":
    args = parse_args_with_presets(Args, PRESET_FNS, "Training with preset configurations")

    print("=" * 50)
    print("Resolved configuration:")
    print("=" * 50)
    for field in fields(args):
        print(f"  {field.name}: {getattr(args, field.name)}")
    print("=" * 50)
