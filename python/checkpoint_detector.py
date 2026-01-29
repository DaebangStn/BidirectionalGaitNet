"""
Checkpoint Format Detector

Lightweight module to detect checkpoint format without importing heavy dependencies.
"""

import os
from python.uri_resolver import resolve_path


def detect_checkpoint_type(path):
    """
    Detect checkpoint format from path without importing loaders.

    Args:
        path: Path to checkpoint (file or directory)

    Returns:
        str: One of 'cleanrl' or 'unknown'
    """
    resolved_path = resolve_path(path)

    if not os.path.exists(resolved_path):
        return 'unknown'

    if os.path.isdir(resolved_path):
        if os.path.exists(os.path.join(resolved_path, "agent.pt")):
            return 'cleanrl'

    return 'unknown'
