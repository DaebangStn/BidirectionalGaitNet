"""Skeleton data loading utilities for global transforms."""
import sys
sys.path.insert(0, '/home/geon/BidirectionalGaitNet')

import yaml
import streamlit as st
from pathlib import Path
from rm import rm_mgr


def _get_skeleton_dir(pid: str, visit: str) -> Path | None:
    """Get skeleton directory path by resolving a known file."""
    try:
        # List items in skeleton directory to find a yaml file
        items = rm_mgr.list(f"@pid:{pid}/{visit}/skeleton")
        yaml_files = [i for i in items if i.endswith('.yaml')]
        if not yaml_files:
            return None
        # Resolve one file to get the parent directory
        path = rm_mgr.resolve(f"@pid:{pid}/{visit}/skeleton/{yaml_files[0]}")
        return Path(path).parent
    except Exception:
        return None


@st.cache_data
def list_global_files(pid: str, visit: str) -> list[str]:
    """List available YAML files in global/ directory.

    Returns list of skeleton names (without .yaml extension).
    """
    skel_dir = _get_skeleton_dir(pid, visit)
    if skel_dir is None:
        return []

    global_dir = skel_dir / "global"
    if not global_dir.exists():
        return []

    yamls = list(global_dir.glob("*.yaml"))
    return sorted([f.stem for f in yamls])


@st.cache_data(ttl=60)
def load_global_transforms(pid: str, visit: str, name: str) -> dict:
    """Load pre-computed global transforms from global/ directory."""
    skel_dir = _get_skeleton_dir(pid, visit)
    if skel_dir is None:
        raise FileNotFoundError(f"Skeleton directory not found for {pid}/{visit}")
    path = skel_dir / "global" / f"{name}.yaml"
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_node_names(global_data: dict) -> list[str]:
    """Extract node names from global transform data."""
    if 'nodes' not in global_data:
        return []
    return [node['name'] for node in global_data['nodes']]


def get_node_data(global_data: dict, node_name: str) -> dict | None:
    """Get data for a specific node."""
    if 'nodes' not in global_data:
        return None
    for node in global_data['nodes']:
        if node['name'] == node_name:
            return node
    return None
