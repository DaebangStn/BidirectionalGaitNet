"""
Checkpoint Format Detector

Lightweight module to detect checkpoint format without importing heavy dependencies.
This allows the viewer to route to the appropriate loader (CleanRL vs Ray) without
loading unnecessary modules.
"""

import os
from python.uri_resolver import resolve_path


def detect_checkpoint_type(path):
    """
    Detect checkpoint format from path without importing loaders.

    Args:
        path: Path to checkpoint (file or directory)

    Returns:
        str: One of 'cleanrl', 'ray_2.0.1', 'ray_2.12.0', or 'unknown'
    """
    # Resolve path (handles URIs, relative paths, etc.)
    resolved_path = resolve_path(path)

    if not os.path.exists(resolved_path):
        return 'unknown'

    # Check for CleanRL format (directory with agent.pt)
    if os.path.isdir(resolved_path):
        agent_pt = os.path.join(resolved_path, "agent.pt")
        if os.path.exists(agent_pt):
            return 'cleanrl'

        # Check for Ray 2.12.0 format (directory with algorithm_state.pkl or policies/)
        algo_state = os.path.join(resolved_path, "algorithm_state.pkl")
        policies_dir = os.path.join(resolved_path, "policies")
        if os.path.exists(algo_state) or os.path.isdir(policies_dir):
            return 'ray_2.12.0'

        return 'unknown'

    # Check for Ray 2.0.1 format (single file)
    if os.path.isfile(resolved_path):
        return 'ray_2.0.1'

    return 'unknown'
