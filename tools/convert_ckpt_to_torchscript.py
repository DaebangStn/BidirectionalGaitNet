"""Convert old pickle-format checkpoints (agent.pt / muscle.pt) to TorchScript format.

Old format (pre-Feb-2026): torch.save(state_dict, path)  → pickle OrderedDict
New format: torch.jit.script module with tensors as buffers  → loadable by C++ loadStateDict()

Usage:
    micromamba run -n bidir python tools/convert_ckpt_to_torchscript.py <ckpt_dir>
    micromamba run -n bidir python tools/convert_ckpt_to_torchscript.py <ckpt_dir> --dry-run
"""

import sys
import shutil
import argparse
from pathlib import Path

import torch
import torch.nn as nn


def is_torchscript(path: Path) -> bool:
    """Return True if file is already TorchScript (torch::jit::load compatible)."""
    try:
        torch.jit.load(str(path), map_location="cpu")
        return True
    except Exception:
        return False


def save_as_torchscript(state_dict: dict, path: Path) -> None:
    """Replicate ppo/learn.py:save_state_dict_as_torchscript exactly."""
    module = nn.Module()
    for key, tensor in state_dict.items():
        safe_key = key.replace(".", "_")
        module.register_buffer(safe_key, tensor.clone().detach().cpu())
    scripted = torch.jit.script(module)
    scripted.save(str(path))


def convert_file(pt_path: Path, dry_run: bool) -> bool:
    """Convert a single .pt file in-place. Returns True if converted."""
    if not pt_path.exists():
        print(f"  skip  {pt_path.name}  (not found)")
        return False

    if is_torchscript(pt_path):
        print(f"  ok    {pt_path.name}  (already TorchScript)")
        return False

    try:
        state_dict = torch.load(str(pt_path), map_location="cpu")
    except Exception as e:
        print(f"  ERROR {pt_path.name}  cannot load: {e}")
        return False

    if not isinstance(state_dict, dict):
        print(f"  ERROR {pt_path.name}  unexpected type: {type(state_dict)}")
        return False

    print(f"  convert {pt_path.name}  ({len(state_dict)} keys) ...", end="", flush=True)
    if dry_run:
        print("  [dry-run, skipped]")
        return False

    backup = pt_path.with_suffix(".pt.bak")
    shutil.copy2(pt_path, backup)
    try:
        save_as_torchscript(state_dict, pt_path)
        print(f"  done  (backup → {backup.name})")
        return True
    except Exception as e:
        shutil.copy2(backup, pt_path)
        print(f"  FAILED: {e}  (restored from backup)")
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ckpt_dir", help="Checkpoint directory containing agent.pt / muscle.pt")
    parser.add_argument("--dry-run", action="store_true", help="Check only, do not write")
    args = parser.parse_args()

    ckpt = Path(args.ckpt_dir)
    if not ckpt.is_dir():
        print(f"Error: {ckpt} is not a directory")
        sys.exit(1)

    print(f"Checkpoint: {ckpt}")
    converted = 0
    for name in ("agent.pt", "muscle.pt"):
        if convert_file(ckpt / name, args.dry_run):
            converted += 1

    if args.dry_run:
        print("(dry-run: no files written)")
    else:
        print(f"Done — {converted} file(s) converted.")


if __name__ == "__main__":
    main()
