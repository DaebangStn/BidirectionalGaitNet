"""Quick training scripts and C++ binary wrappers for UV integration."""

import sys
import subprocess
from pathlib import Path


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def viewer():
    """
    Run the viewer binary with checkpoint path
    Usage: uv run viewer [checkpoint_path]
    Example: uv run viewer data/trained_nn/base_0928
    """
    project_root = get_project_root()
    binary_path = project_root / "build/release/viewer/viewer"

    if not binary_path.exists():
        print(f"Error: Binary not found at {binary_path}", file=sys.stderr)
        print("Please build the project first with: ninja -C build/release", file=sys.stderr)
        sys.exit(1)

    # Get checkpoint path from arguments, default to base checkpoint
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    else:
        ckpt_path = "data/trained_nn/base_0928"
        print(f"No checkpoint path provided, using default: {ckpt_path}")

    # Run the binary with micromamba environment (required for pybind11 dependencies)
    # cmd = ["/opt/miniconda3/bin/micromamba", "run", "-n", "bidir", str(binary_path), ckpt_path]
    cmd = ["micromamba", "run", "-n", "bidir", str(binary_path), ckpt_path]
    try:
        subprocess.run(cmd, cwd=project_root, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


def physical_exam():
    """
    Run the physical_exam binary with config path
    Usage: uv run physical_exam [config_path]
    Example: uv run physical_exam data/config/physical_exam_example2.yaml
    """
    project_root = get_project_root()
    binary_path = project_root / "build/release/viewer/physical_exam"

    if not binary_path.exists():
        print(f"Error: Binary not found at {binary_path}", file=sys.stderr)
        print("Please build the project first with: ninja -C build/release", file=sys.stderr)
        sys.exit(1)

    # Get config path from arguments
    if len(sys.argv) < 2:
        print("Error: Config path required", file=sys.stderr)
        print("Usage: uv run physical_exam <config_path>", file=sys.stderr)
        print("Example: uv run physical_exam data/config/physical_exam_example2.yaml", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]

    # Run the binary with micromamba environment (required for pybind11 dependencies)
    # cmd = ["/opt/miniconda3/bin/micromamba", "run", "-n", "bidir", str(binary_path), config_path]
    cmd = ["micromamba", "run", "-n", "bidir", str(binary_path), config_path]
    try:
        subprocess.run(cmd, cwd=project_root, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


def train_fgn():
    """Quick Forward GaitNet training."""
    sys.argv = ["train.py", "--config", "data/config/fgn_default.yaml"]
    from python.train import main
    main()

def train_bgn():
    """Quick Backward GaitNet training - requires FGN checkpoint."""
    if len(sys.argv) < 2:
        print("Usage: uv run train-bgn <fgn_checkpoint_path>")
        print("Example: uv run train-bgn distillation/fgn/best_checkpoint")
        sys.exit(1)
        
    fgn_checkpoint = sys.argv[1]
    sys.argv = ["train.py", "--config", "data/config/bgn_default.yaml", "--fgn", fgn_checkpoint]
    from python.train import main
    main()

def train_pipeline():
    """Quick full pipeline training."""
    sys.argv = ["train.py", "--config", "data/config/pipeline.yaml"]  
    from python.train import main
    main()