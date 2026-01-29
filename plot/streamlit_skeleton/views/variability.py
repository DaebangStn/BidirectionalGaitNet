"""Skeleton variability visualization with dot plots."""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from core.skeleton import list_global_files, load_global_transforms, get_node_data


def create_dot_plot(df: pd.DataFrame, value_col: str, ref_value: float, title: str,
                    value_width: float | None = None) -> go.Figure:
    """Create horizontal dot plot with reference line centered.

    Args:
        df: DataFrame with 'motion' column and value column
        value_col: Column name for values to plot
        ref_value: Reference value (trimmed_unified) for vertical line
        title: Y-axis label (e.g., 'X', 'Y', 'Z')
        value_width: Half-width of x-axis range (centered on ref_value). If None, auto-compute.
    """
    fig = go.Figure()

    # Compute x-axis range centered on ref_value
    if value_width is not None:
        x_range = [ref_value - value_width, ref_value + value_width]
    else:
        # Auto-compute: symmetric range with 10% padding
        all_values = list(df[value_col]) + [ref_value]
        min_val, max_val = min(all_values), max(all_values)
        max_dist = max(abs(ref_value - min_val), abs(ref_value - max_val))
        padding = max_dist * 0.1 if max_dist > 0 else 0.01
        x_range = [ref_value - max_dist - padding, ref_value + max_dist + padding]

    # Reference line (trimmed_unified) - dotted vertical line
    fig.add_vline(
        x=ref_value,
        line_dash="dot",
        line_color="red",
        line_width=2
    )

    # Dots (all other motions)
    fig.add_trace(go.Scatter(
        x=df[value_col],
        y=[title] * len(df),
        mode='markers',
        marker=dict(size=12, color='steelblue', opacity=0.7),
        text=df['motion'],
        hovertemplate='%{text}<br>Value: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        height=100,
        margin=dict(l=70, r=20, t=10, b=25),
        showlegend=False,
        yaxis=dict(visible=True, tickfont=dict(size=12)),
        xaxis=dict(title=None, tickfont=dict(size=10), range=x_range)
    )
    return fig


def render(pid: str, visit: str, node_name: str):
    """Render variability visualization for a body node.

    Args:
        pid: Patient ID
        visit: Visit (pre/op1/op2)
        node_name: Body node name to visualize
    """
    st.header(f"Skeleton Variability: {node_name}")

    # Load all global files
    names = list_global_files(pid, visit)
    if not names:
        st.warning("No global transform files found. Run skeleton_global_export first.")
        return

    # Separate reference (trimmed_unified) from others
    ref_name = "trimmed_unified"
    if ref_name not in names:
        st.error(f"Reference skeleton '{ref_name}' not found in global/ directory")
        st.info(f"Available files: {', '.join(names)}")
        return

    trial_names = [n for n in names if n != ref_name]

    # Load reference data
    ref_data = load_global_transforms(pid, visit, ref_name)
    ref_node = get_node_data(ref_data, node_name)
    if not ref_node:
        st.error(f"Node '{node_name}' not found in reference skeleton")
        return

    # Load trial data
    data = []
    for name in trial_names:
        global_data = load_global_transforms(pid, visit, name)
        node = get_node_data(global_data, node_name)
        if node:
            local_trans = node.get('local_translation', [0, 0, 0])
            data.append({
                'motion': name,
                'size_x': node['size_meters'][0],
                'size_y': node['size_meters'][1],
                'size_z': node['size_meters'][2],
                'pos_x': node['global_position'][0],
                'pos_y': node['global_position'][1],
                'pos_z': node['global_position'][2],
                'local_x': local_trans[0],
                'local_y': local_trans[1],
                'local_z': local_trans[2],
                'axis_x': node['global_rotation']['axis'][0],
                'axis_y': node['global_rotation']['axis'][1],
                'axis_z': node['global_rotation']['axis'][2],
                'angle': node['global_rotation']['angle_deg']
            })

    if not data:
        st.warning("No trial data found for this node")
        return

    df = pd.DataFrame(data)

    # Sidebar: X-axis range controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("X-Axis Range (±)")

    auto_scale = st.sidebar.checkbox("Auto Scale", value=True, key="auto_scale")

    if auto_scale:
        size_width = None
        position_width = None
        local_width = None
        orientation_width = None
    else:
        size_width = st.sidebar.number_input(
            "Size (m)", min_value=0.001, max_value=1.0, value=0.05, step=0.01,
            format="%.3f", key="size_width"
        )
        position_width = st.sidebar.number_input(
            "Position (m)", min_value=0.001, max_value=1.0, value=0.1, step=0.01,
            format="%.3f", key="position_width"
        )
        local_width = st.sidebar.number_input(
            "Local Trans (m)", min_value=0.001, max_value=1.0, value=0.05, step=0.01,
            format="%.3f", key="local_width"
        )
        orientation_width = st.sidebar.number_input(
            "Orientation (axis)", min_value=0.01, max_value=2.0, value=0.5, step=0.1,
            format="%.2f", key="orientation_width"
        )
        angle_width = st.sidebar.number_input(
            "Angle (deg)", min_value=1.0, max_value=180.0, value=30.0, step=5.0,
            format="%.1f", key="angle_width"
        )

    # Display stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Reference", ref_name)
    with col2:
        st.metric("Trials", len(df))

    # Scale plots
    st.subheader("Body Size (meters)")
    st.plotly_chart(create_dot_plot(df, 'size_x', ref_node['size_meters'][0], 'X', size_width),
                    key=f"scale_x_{node_name}")
    st.plotly_chart(create_dot_plot(df, 'size_y', ref_node['size_meters'][1], 'Y', size_width),
                    key=f"scale_y_{node_name}")
    st.plotly_chart(create_dot_plot(df, 'size_z', ref_node['size_meters'][2], 'Z', size_width),
                    key=f"scale_z_{node_name}")

    # Position plots
    st.subheader("Global Position (T-pose)")
    st.plotly_chart(create_dot_plot(df, 'pos_x', ref_node['global_position'][0], 'X', position_width),
                    key=f"pos_x_{node_name}")
    st.plotly_chart(create_dot_plot(df, 'pos_y', ref_node['global_position'][1], 'Y', position_width),
                    key=f"pos_y_{node_name}")
    st.plotly_chart(create_dot_plot(df, 'pos_z', ref_node['global_position'][2], 'Z', position_width),
                    key=f"pos_z_{node_name}")

    # Local translation plots (w.r.t. parent body node)
    ref_local = ref_node.get('local_translation', [0, 0, 0])
    local_w = local_width if not auto_scale else None
    st.subheader("Local Translation (w.r.t. Parent)")
    st.plotly_chart(create_dot_plot(df, 'local_x', ref_local[0], 'X', local_w),
                    key=f"local_x_{node_name}")
    st.plotly_chart(create_dot_plot(df, 'local_y', ref_local[1], 'Y', local_w),
                    key=f"local_y_{node_name}")
    st.plotly_chart(create_dot_plot(df, 'local_z', ref_local[2], 'Z', local_w),
                    key=f"local_z_{node_name}")

    # Orientation plots
    st.subheader("Global Orientation (Axis-Angle)")
    orient_w = orientation_width if not auto_scale else None
    angle_w = angle_width if not auto_scale else None
    st.plotly_chart(create_dot_plot(df, 'axis_x', ref_node['global_rotation']['axis'][0], 'Axis X', orient_w),
                    key=f"axis_x_{node_name}")
    st.plotly_chart(create_dot_plot(df, 'axis_y', ref_node['global_rotation']['axis'][1], 'Axis Y', orient_w),
                    key=f"axis_y_{node_name}")
    st.plotly_chart(create_dot_plot(df, 'axis_z', ref_node['global_rotation']['axis'][2], 'Axis Z', orient_w),
                    key=f"axis_z_{node_name}")
    st.plotly_chart(create_dot_plot(df, 'angle', ref_node['global_rotation']['angle_deg'], 'Angle (deg)', angle_w),
                    key=f"angle_{node_name}")

    # Raw data table
    with st.expander("Raw Data"):
        st.dataframe(df, width="stretch")
