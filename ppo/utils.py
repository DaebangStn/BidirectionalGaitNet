"""
PPO utility functions: argument parsing with preset support.

Supports three patterns:
    python learn.py                              # Default dataclass values
    python learn.py gait                         # gait preset (CPU cluster)
    python learn.py a6000 --num_envs 64          # a6000 preset + override
"""

from dataclasses import dataclass, fields, asdict
from typing import Callable, Optional, TypeVar, Type
import argparse

T = TypeVar('T')


def _str_to_bool(v: str) -> bool:
    """Parse boolean from string for argparse."""
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def _add_override_args(parser: argparse.ArgumentParser, dataclass_type: Type[T]) -> None:
    """Add CLI flags for overriding any dataclass field."""
    for field in fields(dataclass_type):
        flag_name = f"--{field.name}"

        # Skip computed fields (default=0 runtime fields)
        if field.name in ('batch_size', 'minibatch_size', 'num_iterations'):
            continue

        if field.type == bool:
            parser.add_argument(
                flag_name,
                type=_str_to_bool,
                default=None,
                metavar="BOOL",
                help=f"Override {field.name}",
            )
        elif field.type == Optional[float]:
            parser.add_argument(
                flag_name,
                type=lambda x: None if x.lower() == 'none' else float(x),
                default=None,
                help=f"Override {field.name}",
            )
        elif field.type == Optional[str]:
            parser.add_argument(
                flag_name,
                type=lambda x: None if x.lower() == 'none' else str(x),
                default=None,
                help=f"Override {field.name}",
            )
        else:
            # Get base type for Optional types
            base_type = field.type
            if hasattr(field.type, '__origin__') and field.type.__origin__ is type(None):
                base_type = field.type.__args__[0]

            parser.add_argument(
                flag_name,
                type=base_type if base_type in (int, float, str) else str,
                default=None,
                help=f"Override {field.name}",
            )


def build_preset_parser(
    dataclass_type: Type[T],
    preset_fns: dict[Optional[str], Callable[[], T]],
    description: str = "Training with preset configurations",
) -> argparse.ArgumentParser:
    """
    Build argparse with optional preset subcommands.

    Args:
        dataclass_type: The dataclass type (e.g., Args)
        preset_fns: Map of preset names to classmethods. Use None key for default.
        description: Parser description

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Create subparsers for presets (optional - can run without preset)
    subparsers = parser.add_subparsers(
        dest="preset",
        required=False,
        help="Select a configuration preset (optional, uses defaults if omitted)",
    )

    # Add override args to main parser (for default case: no preset given)
    _add_override_args(parser, dataclass_type)

    # Add a subcommand for each named preset
    for preset_name in preset_fns:
        if preset_name is None:
            continue  # Skip the default entry
        sub = subparsers.add_parser(
            preset_name,
            help=f"Use {preset_name} preset",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        _add_override_args(sub, dataclass_type)

    return parser


def parse_args_with_presets(
    dataclass_type: Type[T],
    preset_fns: dict[Optional[str], Callable[[], T]],
    description: str = "Training with preset configurations",
) -> T:
    """
    Parse CLI arguments with preset + override support.

    Precedence: preset values -> CLI overrides

    Args:
        dataclass_type: The dataclass type (e.g., Args)
        preset_fns: Map of preset names to classmethods. Use None key for default.
        description: Parser description

    Returns:
        Configured dataclass instance
    """
    parser = build_preset_parser(dataclass_type, preset_fns, description)
    cli_args = parser.parse_args()

    # Step 1: Get base config from selected preset (or default constructor)
    preset_fn = preset_fns.get(cli_args.preset, preset_fns.get(None, dataclass_type))
    args = preset_fn()

    # Step 2: Apply any explicitly provided CLI overrides
    args_dict = asdict(args)
    for field in fields(dataclass_type):
        cli_value = getattr(cli_args, field.name, None)
        if cli_value is not None:  # Only override if explicitly set
            args_dict[field.name] = cli_value

    return dataclass_type(**args_dict)
