"""
Checkpoint utilities for hierarchical PPO.

Handles saving and loading of policy agent and muscle network states
with CleanRL checkpoint format compatibility.
"""

import torch
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np


def save_checkpoint(
    run_name: str,
    global_step: int,
    agent,
    muscle_learner=None,
    optimizer_state: Optional[Dict] = None,
    metadata: Optional[Dict] = None
) -> str:
    """
    Save training checkpoint.

    Args:
        run_name: Name of the training run
        global_step: Current global step count
        agent: Policy agent (Actor-Critic network)
        muscle_learner: Optional MuscleLearner instance
        optimizer_state: Optional optimizer state dict
        metadata: Optional training metadata (hyperparameters, etc.)

    Returns:
        Path to saved checkpoint directory
    """
    checkpoint_dir = Path("runs") / run_name / "checkpoints" / f"step_{global_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save policy agent
    agent_path = checkpoint_dir / "agent.pt"
    torch.save(agent.state_dict(), agent_path)
    print(f"💾 Saved policy agent: {agent_path}")

    # Save optimizer if provided
    if optimizer_state is not None:
        optimizer_path = checkpoint_dir / "optimizer.pt"
        torch.save(optimizer_state, optimizer_path)
        print(f"💾 Saved optimizer: {optimizer_path}")

    # Save muscle network if present
    if muscle_learner is not None:
        muscle_path = checkpoint_dir / "muscle_network.pt"
        muscle_learner.save(str(muscle_path))
        print(f"💾 Saved muscle network: {muscle_path}")

    # Save metadata
    if metadata is None:
        metadata = {}
    metadata["global_step"] = global_step
    metadata["run_name"] = run_name

    metadata_path = checkpoint_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        # Convert numpy types to Python types for JSON serialization
        serializable_metadata = {
            k: (v.item() if isinstance(v, np.generic) else v)
            for k, v in metadata.items()
        }
        json.dump(serializable_metadata, f, indent=2)
    print(f"💾 Saved metadata: {metadata_path}")

    return str(checkpoint_dir)


def load_checkpoint(
    checkpoint_dir: str,
    agent,
    muscle_learner=None,
    load_optimizer: bool = False,
    device: str = "cuda"
) -> Tuple[int, Optional[Dict], Dict]:
    """
    Load training checkpoint.

    Args:
        checkpoint_dir: Path to checkpoint directory
        agent: Policy agent to load weights into
        muscle_learner: Optional MuscleLearner to load weights into
        load_optimizer: Whether to load optimizer state
        device: Device for loading tensors

    Returns:
        Tuple of (global_step, optimizer_state, metadata)
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Load metadata
    metadata_path = checkpoint_dir / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    global_step = metadata.get("global_step", 0)
    print(f"📂 Loading checkpoint from step {global_step}")

    # Load policy agent
    agent_path = checkpoint_dir / "agent.pt"
    agent.load_state_dict(
        torch.load(agent_path, map_location=device)
    )
    print(f"✅ Loaded policy agent: {agent_path}")

    # Load optimizer if requested
    optimizer_state = None
    if load_optimizer:
        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            optimizer_state = torch.load(optimizer_path, map_location=device)
            print(f"✅ Loaded optimizer: {optimizer_path}")

    # Load muscle network if present
    if muscle_learner is not None:
        muscle_path = checkpoint_dir / "muscle_network.pt"
        if muscle_path.exists():
            muscle_learner.load(str(muscle_path))
            print(f"✅ Loaded muscle network: {muscle_path}")

    return global_step, optimizer_state, metadata


def find_latest_checkpoint(run_name: str) -> Optional[str]:
    """
    Find the latest checkpoint for a run.

    Args:
        run_name: Name of the training run

    Returns:
        Path to latest checkpoint directory, or None if no checkpoints exist
    """
    checkpoint_base = Path("runs") / run_name / "checkpoints"
    if not checkpoint_base.exists():
        return None

    # Find all checkpoint directories
    checkpoints = list(checkpoint_base.glob("step_*"))
    if not checkpoints:
        return None

    # Sort by step number
    checkpoints.sort(key=lambda p: int(p.name.split("_")[1]))

    latest = checkpoints[-1]
    print(f"🔍 Found latest checkpoint: {latest}")
    return str(latest)


def list_checkpoints(run_name: str) -> list:
    """
    List all checkpoints for a run.

    Args:
        run_name: Name of the training run

    Returns:
        List of checkpoint directory paths sorted by step
    """
    checkpoint_base = Path("runs") / run_name / "checkpoints"
    if not checkpoint_base.exists():
        return []

    checkpoints = list(checkpoint_base.glob("step_*"))
    checkpoints.sort(key=lambda p: int(p.name.split("_")[1]))

    return [str(cp) for cp in checkpoints]
