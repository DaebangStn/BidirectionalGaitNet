"""Normative ROM data loader from config files."""
import sys
sys.path.insert(0, '/home/geon/BidirectionalGaitNet')

import glob
import os
import streamlit as st
import yaml

# Config directory path
CONFIG_DIR = '/home/geon/BidirectionalGaitNet/data/config/rom'


@st.cache_data
def load_normative_data() -> dict:
    """Load all ROM normative configs from data/config/rom/*.yaml.

    Returns:
        dict mapping (side, joint, field) -> {
            'name': str,
            'description': str,
            'normative': float,
            'neg': bool,
            'alias': str
        }
    """
    result = {}
    config_dir = CONFIG_DIR

    for f in glob.glob(f"{config_dir}/*.yaml"):
        try:
            with open(f) as fp:
                cfg = yaml.safe_load(fp)

            cd = cfg.get("clinical_data", {})
            if not cd:
                continue

            side = cd.get("side")
            joint = cd.get("joint")
            field = cd.get("field")

            if not all([side, joint, field]):
                continue

            key = (side, joint, field)
            result[key] = {
                "name": cfg.get("name", ""),
                "description": cfg.get("description", ""),
                "normative": cfg.get("exam", {}).get("normative"),
                "neg": cd.get("neg", False),
                "alias": cfg.get("exam", {}).get("alias", "")
            }
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue

    return result


def get_dof_list() -> list[dict]:
    """Get list of DOFs organized by joint for UI display.

    Returns:
        List of dicts with: joint, field, display_name, normative, neg
    """
    normative = load_normative_data()

    # Name-to-display mapping for better names
    name_mapping = {
        'abd_k90': 'Abduction (k90)',
        'abd_k0': 'Abduction (k0)',
        'add': 'Adduction',
        'intRot': 'Internal Rotation',
        'extRot': 'External Rotation',
        'staheliExtension': 'Extension (Staheli)',
        'popliteal': 'Popliteal',
        'dorsi_k0': 'Dorsiflexion (k0)',
        'dorsi_k90': 'Dorsiflexion (k90)',
        'plantar': 'Plantarflexion',
    }

    # Group by joint
    dofs = []
    for (side, joint, field), info in normative.items():
        # Only process left side (we'll pair with right)
        if side != "left":
            continue

        # Create display name from config name
        config_name = info.get("name", "")
        # Remove side suffix (_L or _R)
        base_name = config_name.rsplit('_', 1)[0] if config_name.endswith(('_L', '_R')) else config_name

        # Look up display name or fall back to field
        display_name = name_mapping.get(base_name, field)

        dofs.append({
            "joint": joint.capitalize(),
            "field": field,
            "display_name": display_name,
            "normative": info.get("normative"),
            "neg": info.get("neg", False)
        })

    # Sort by joint order (Hip, Knee, Ankle) then by display name
    joint_order = {"Hip": 0, "Knee": 1, "Ankle": 2}
    dofs.sort(key=lambda x: (joint_order.get(x["joint"], 99), x["display_name"]))

    return dofs


def get_normative_value(side: str, joint: str, field: str) -> tuple[float, bool]:
    """Get normative value and negation flag for a specific measurement.

    Args:
        side: 'left' or 'right'
        joint: 'hip', 'knee', or 'ankle'
        field: ROM field name

    Returns:
        (normative_value, neg_flag) or (None, False) if not found
    """
    normative = load_normative_data()
    key = (side, joint, field)

    if key in normative:
        info = normative[key]
        return info.get("normative"), info.get("neg", False)

    return None, False
