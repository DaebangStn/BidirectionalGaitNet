"""Skeleton Variability Visualization App.

Visualizes skeleton calibration variability across multiple motion trials.
Shows how body scales, positions, and orientations vary between trials
with trimmed_unified as the reference baseline.

Run with:
    streamlit run plot/streamlit_skeleton/app.py
"""
import sys
sys.path.insert(0, '/home/geon/BidirectionalGaitNet')
sys.path.insert(0, '/home/geon/BidirectionalGaitNet/plot/streamlit_skeleton')

import streamlit as st
from pathlib import Path
from rm import rm_mgr
from core.skeleton import list_global_files, load_global_transforms, get_node_names, _get_skeleton_dir
from views import variability, summary

st.set_page_config(
    page_title="Skeleton Variability",
    page_icon="🦴",
    layout="wide"
)

st.title("🦴 Skeleton Calibration Variability")


@st.cache_data(ttl=300)
def get_available_pids_with_data():
    """Get PIDs that have skeleton/global data for at least one visit."""
    try:
        all_pids = rm_mgr.list("@pid")
        all_pids = sorted([p for p in all_pids if p.isdigit()])
    except Exception:
        return [], {}

    available_pids = []
    pid_visits = {}  # pid -> [visits with data]

    for pid in all_pids:
        visits_with_data = []
        for visit in ["pre", "op1", "op2"]:
            try:
                skel_dir = _get_skeleton_dir(pid, visit)
                if skel_dir and (skel_dir / "global").exists():
                    # Check if there are actual files
                    global_files = list((skel_dir / "global").glob("*.yaml"))
                    if global_files:
                        visits_with_data.append(visit)
            except Exception:
                pass

        if visits_with_data:
            available_pids.append(pid)
            pid_visits[pid] = visits_with_data

    return available_pids, pid_visits


@st.cache_data(ttl=300)
def get_pid_metadata(pid: str) -> dict:
    """Get patient metadata (name, gmfcs)."""
    metadata = {'name': '', 'gmfcs': ''}
    try:
        handle = rm_mgr.fetch(f"@pid:{pid}/name")
        metadata['name'] = handle.as_string()
    except Exception:
        pass
    try:
        handle = rm_mgr.fetch(f"@pid:{pid}/gmfcs")
        metadata['gmfcs'] = handle.as_string()
    except Exception:
        pass
    return metadata


# Get available PIDs with data
pids, pid_visits = get_available_pids_with_data()

if not pids:
    st.error("No skeleton/global data found for any patient")
    st.info("Run skeleton calibration and skeleton_global_export first.")
    st.stop()

# Sidebar for patient/visit selection
st.sidebar.header("Settings")

# View selection
view_mode = st.sidebar.radio("View", ["Summary", "Per Node"], horizontal=True)

# Build PID display strings with metadata
pid_display = []
for pid in pids:
    meta = get_pid_metadata(pid)
    parts = [pid]
    if meta['name']:
        parts.append(meta['name'])
    if meta['gmfcs']:
        parts.append(f"{meta['gmfcs']}")
    pid_display.append(" | ".join(parts))

# PID selection
default_idx = 0
for i, p in enumerate(pids):
    if p == "29792292":
        default_idx = i
        break

selected_idx = st.sidebar.selectbox(
    "Patient",
    range(len(pids)),
    index=default_idx,
    format_func=lambda i: pid_display[i]
)
pid = pids[selected_idx]

# Visit selection (only show visits with data)
available_visits = pid_visits.get(pid, [])
if not available_visits:
    st.warning(f"No skeleton/global data found for PID {pid}")
    st.stop()

visit = st.sidebar.selectbox("Visit", available_visits)

# Show data summary in sidebar
st.sidebar.markdown("---")
names = list_global_files(pid, visit)
st.sidebar.markdown(f"**Trials**: {len(names) - 1}")  # -1 for trimmed_unified

if not names:
    st.warning("No global transform files found")
    st.stop()

# Render based on view mode
if view_mode == "Summary":
    summary.render(pid, visit)
else:
    # Get node names from first file
    first_data = load_global_transforms(pid, visit, names[0])
    node_names = get_node_names(first_data)

    if not node_names:
        st.error("No body nodes found in skeleton data")
        st.stop()

    # Node selection with keyboard navigation
    node_idx = st.sidebar.number_input(
        "Body Node Index",
        min_value=0,
        max_value=len(node_names) - 1,
        value=0,
        step=1,
        help="Use ↑↓ arrow keys to browse nodes"
    )
    node_name = node_names[node_idx]
    st.sidebar.markdown(f"**→ {node_name}**")

    variability.render(pid, visit, node_name)
