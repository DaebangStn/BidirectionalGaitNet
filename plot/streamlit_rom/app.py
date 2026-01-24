"""ROM Browser - Streamlit app for browsing ROM data."""
import sys
sys.path.insert(0, '/home/geon/BidirectionalGaitNet')
sys.path.insert(0, '/home/geon/BidirectionalGaitNet/plot/streamlit_rom')

import streamlit as st

# Page config
st.set_page_config(
    page_title="ROM Browser",
    page_icon=":bone:",
    layout="wide"
)

# Import after path setup
from core.data import list_pids, list_visits, list_all_surgeries
from core.normative import get_dof_list
from views import per_patient, distribution


def render_patient_sidebar():
    """Render sidebar controls for patient mode. Returns (patient, visit, compare_mode)."""
    pids = list_pids()

    if not pids:
        st.warning("No patients found")
        return None, None, False

    # Patient selector
    options = [f"{p['pid']} ({p['name']}, {p['gmfcs']})" if p['name'] else f"{p['pid']} ({p['gmfcs']})" for p in pids]
    selected_option = st.selectbox("Patient", options)

    selected_pid = selected_option.split(' ')[0]
    selected_patient = next((p for p in pids if p['pid'] == selected_pid), None)

    if not selected_patient:
        st.error("Patient not found")
        return None, None, False

    # Get available visits
    visits = list_visits(selected_pid)
    if not visits:
        st.warning(f"No ROM data found for {selected_pid}")
        return None, None, False

    # Compare mode checkbox
    compare_mode = st.checkbox(
        "Compare visits",
        value=False,
        help="Show pre/op1/op2 values side by side for comparison"
    )

    if compare_mode:
        # In compare mode, return all visits
        return selected_patient, visits, True
    else:
        # Single visit selector
        visit_labels = {'pre': 'Pre-op', 'op1': 'Post-op 1', 'op2': 'Post-op 2'}
        visit_options = [visit_labels.get(v, v) for v in visits]
        selected_visit_label = st.selectbox("Visit", visit_options)
        selected_visit = visits[visit_options.index(selected_visit_label)]
        return selected_patient, selected_visit, False


def render_distribution_sidebar():
    """Render sidebar controls for distribution mode. Returns (gmfcs, dofs, anonymous, surgery, color_by)."""
    st.subheader("Filters")

    gmfcs_options = ['All', '1', '2', '3'] # 4 and 5 are not included in the data
    selected_gmfcs = st.selectbox(
        "GMFCS Level",
        options=gmfcs_options,
        index=0,
        help="Filter patients by GMFCS level."
    )

    # Surgery filter
    all_surgeries = list_all_surgeries()
    surgery_options = ['All'] + all_surgeries
    selected_surgery = st.selectbox(
        "Operation",
        options=surgery_options,
        index=0,
        help="Filter patients by surgery type"
    )

    # Surgery side filter
    selected_side = st.radio(
        "Side",
        options=["All", "Both", "Left", "Right"],
        index=0,
        horizontal=True,
        help="All=any, Both=bilateral, Left=left only, Right=right only"
    )

    # Color/legend grouping
    color_by = st.radio(
        "Color by",
        options=["Visit", "GMFCS"],
        index=0,
        horizontal=True,
        help="Group legend by visit (Pre/Op1) or GMFCS level"
    )

    anonymous = st.checkbox(
        "Show PID only",
        value=True,
        help="If checked, shows patient ID. Otherwise shows patient name."
    )

    st.divider()
    st.subheader("Select DOFs")

    dofs = get_dof_list()
    current_joint = None
    selected_dofs = []

    for dof in dofs:
        if dof['joint'] != current_joint:
            current_joint = dof['joint']
            st.caption(f"**{current_joint}**")

        key = f"dof_{dof['field']}"
        default = dof['field'] in [
        ]

        if st.checkbox(dof['display_name'], value=default, key=key):
            selected_dofs.append(dof)

    return selected_gmfcs, selected_dofs, anonymous, selected_surgery, selected_side, color_by


# Main app
st.title("ROM Browser")

# Sidebar
with st.sidebar:
    st.header("Settings")

    mode = st.radio(
        "Mode",
        options=["Patient", "Distribution"],
        index=0,
        help="Patient: View individual patient ROM data\nDistribution: Compare across patients"
    )

    st.divider()

    if mode == "Patient":
        patient, visit_or_visits, compare_mode = render_patient_sidebar()
    else:
        gmfcs, dofs, anonymous, surgery, side, color_by = render_distribution_sidebar()

# Main view - render based on mode selected above
if mode == "Patient":
    if patient and visit_or_visits:
        if compare_mode:
            per_patient.render_compare(patient, visit_or_visits)
        else:
            per_patient.render(patient, visit_or_visits)
    else:
        st.info("Select a patient from the sidebar")
else:
    distribution.render(gmfcs, dofs, anonymous, surgery, side, color_by)
