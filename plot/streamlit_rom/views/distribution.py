"""Distribution view - scatter plots across patients."""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.data import list_pids, load_rom_data, get_rom_value, load_surgery_info, load_metadata
from core.normative import get_dof_list


def render(selected_gmfcs: str, selected_dofs: list, anonymous: bool, selected_surgery: str, selected_side: str = "Both", color_by: str = "Visit", show_anthro: bool = False):
    """Render distribution scatter plot view."""
    # Display current filter info
    filter_text = f"**GMFCS:** {selected_gmfcs} | **Operation:** {selected_surgery} | **Side:** {selected_side}"
    st.markdown(filter_text)

    if not selected_dofs and not show_anthro:
        st.info("Select at least one DOF to display")
        return

    # Fixed visits (pre and op1 only)
    selected_visits = ['pre', 'op1']

    # Load all patients
    pids = list_pids()

    # Filter by GMFCS
    if selected_gmfcs != 'All':
        pids = [p for p in pids if p['gmfcs'] == selected_gmfcs]

    # Filter by surgery and side (if not 'All')
    if selected_surgery != 'All':
        filtered_pids = []
        for p in pids:
            for visit in ['op1', 'op2']:
                surgery_list = load_surgery_info(p['pid'], visit)

                # Check for surgery with correct side
                has_left = False
                has_right = False
                for surg in surgery_list:
                    base = surg[:-3] if surg.endswith(('_Rt', '_Lt')) else surg
                    if base == selected_surgery:
                        if surg.endswith('_Lt'):
                            has_left = True
                        elif surg.endswith('_Rt'):
                            has_right = True
                        else:
                            # No side suffix - count as both
                            has_left = True
                            has_right = True

                # Check if matches side filter
                match = False
                if selected_side == "All" and (has_left or has_right):
                    match = True
                elif selected_side == "Both" and has_left and has_right:
                    match = True
                elif selected_side == "Left" and has_left and not has_right:
                    match = True
                elif selected_side == "Right" and has_right and not has_left:
                    match = True

                if match:
                    filtered_pids.append(p)
                    break
        pids = filtered_pids

    # Store filtered pids for checkbox display at bottom
    all_filtered_pids = pids.copy()

    if not all_filtered_pids:
        st.warning("No patients match the selected filters")
        return

    # Filter by checkbox selection (use session_state)
    pids = [p for p in pids if st.session_state.get(f"pid_checkbox_{p['pid']}", True)]

    # Show plot only if there are selected patients
    if not pids:
        st.info("No patients selected. Check patients below to display data.")
    else:
        if selected_dofs:
            _render_plot(pids, selected_dofs, selected_visits, anonymous, color_by)
        if show_anthro:
            st.divider()
            st.markdown("### Anthropometrics")
            _render_anthro_plot(pids, selected_visits, anonymous, color_by)

    # Patient selection checkboxes at bottom
    st.markdown("---")
    st.markdown("**Select Patients:**")

    # Select All / None buttons
    btn_cols = st.columns([1, 1, 6])
    with btn_cols[0]:
        if st.button("Select All"):
            for p in all_filtered_pids:
                st.session_state[f"pid_checkbox_{p['pid']}"] = True
            st.rerun()
    with btn_cols[1]:
        if st.button("Select None"):
            for p in all_filtered_pids:
                st.session_state[f"pid_checkbox_{p['pid']}"] = False
            st.rerun()

    cols = st.columns(5)
    for i, p in enumerate(all_filtered_pids):
        pid = p['pid']
        name = p['name'] or ''
        # Display: "PID (Name)" or just "PID" if no name
        label = f"{pid} ({name})" if name else pid

        col_idx = i % 5
        with cols[col_idx]:
            st.checkbox(label, value=True, key=f"pid_checkbox_{pid}")


def _render_plot(pids: list, selected_dofs: list, selected_visits: list, anonymous: bool, color_by: str):
    """Render the distribution plot and data table."""
    # Create subplots
    num_dofs = len(selected_dofs)
    fig = make_subplots(
        rows=num_dofs, cols=1,
        subplot_titles=[d['display_name'] for d in selected_dofs],
        vertical_spacing=0.15 if num_dofs > 1 else 0.1
    )

    # Color palettes
    visit_colors = {
        'pre': '#1f77b4',
        'op1': '#2ca02c',
        'op2': '#ff7f0e'
    }

    gmfcs_colors = {
        '1': '#1f77b4',  # blue
        '2': '#2ca02c',  # green
        '3': '#ff7f0e',  # orange
    }

    # Determine coloring mode
    color_by_gmfcs = (color_by == "GMFCS")

    # Build x-axis positions and labels for visits
    # Layout: L-Pre, L-Op1, gap, R-Pre, R-Op1
    visit_labels_map = {'pre': 'Pre', 'op1': 'Op1'}
    num_visits = len(selected_visits)
    gap = 1.0  # gap between Left and Right groups

    # X positions for each (side, visit) combination
    x_positions = {}
    tick_vals = []
    tick_texts = []

    for i, visit in enumerate(selected_visits):
        x_positions[('left', visit)] = i
        tick_vals.append(i)
        tick_texts.append(f"L-{visit_labels_map.get(visit, visit)}")

    for i, visit in enumerate(selected_visits):
        x_positions[('right', visit)] = num_visits + gap + i
        tick_vals.append(num_visits + gap + i)
        tick_texts.append(f"R-{visit_labels_map.get(visit, visit)}")

    x_range = [-0.5, num_visits + gap + num_visits - 0.5]

    # Process each DOF
    for dof_idx, dof in enumerate(selected_dofs):
        field = dof['field']
        joint = dof['joint'].lower()
        normative = dof['normative']

        # Collect data per patient for connecting lines
        patient_data = {}
        for p in pids:
            pid = p['pid']
            label = p['pid'] if anonymous else (p['name'] or p['pid'])
            gmfcs = p['gmfcs']
            patient_data[pid] = {'label': label, 'gmfcs': gmfcs, 'left': {}, 'right': {}}

            for visit in selected_visits:
                rom_data = load_rom_data(pid, visit)
                if not rom_data:
                    continue

                left_val = get_rom_value(rom_data, 'left', joint, field)
                right_val = get_rom_value(rom_data, 'right', joint, field)

                if left_val is not None:
                    patient_data[pid]['left'][visit] = left_val
                if right_val is not None:
                    patient_data[pid]['right'][visit] = right_val

        # Colors for points and lines
        pre_color = visit_colors['pre']    # blue
        op1_color = visit_colors['op1']    # green
        line_color = '#888888'             # gray for connecting lines

        # Build hover text for a single point
        def build_point_hover(label, val, visit, side, norm, pre_val=None):
            """Build hover text for a single point."""
            side_label = 'L' if side == 'left' else 'R'
            visit_label = 'Pre' if visit == 'pre' else 'Op1'
            lines = [f"<b>{label}</b> {side_label}-{visit_label}"]
            lines.append(f"Value: {val:.1f}")
            if visit == 'op1' and pre_val is not None:
                lines.append(f"Δ: {val - pre_val:+.1f}")
            if norm is not None:
                lines.append(f"vs Norm: {val - norm:+.1f}")
            return "<br>".join(lines)

        # Create legend entries based on color mode
        if dof_idx == 0:
            if color_by_gmfcs:
                # Legend for GMFCS levels
                for gmfcs_level in ['1', '2', '3']:
                    fig.add_trace(
                        go.Scatter(
                            x=[None], y=[None],
                            mode='markers',
                            marker=dict(size=10, color=gmfcs_colors.get(gmfcs_level, '#333')),
                            name=f'GMFCS {gmfcs_level}',
                            showlegend=True,
                            legendgroup=f'gmfcs_{gmfcs_level}'
                        ),
                        row=1, col=1
                    )
            else:
                # Legend for visits (Pre/Op1)
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(size=10, color=pre_color),
                        name='Pre',
                        showlegend=True,
                        legendgroup='pre'
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(size=10, color=op1_color),
                        name='Op1',
                        showlegend=True,
                        legendgroup='op1'
                    ),
                    row=1, col=1
                )

        for pid, data in patient_data.items():
            label = data['label']
            gmfcs = data['gmfcs']
            left_pre = data['left'].get('pre')
            left_op1 = data['left'].get('op1')
            right_pre = data['right'].get('pre')
            right_op1 = data['right'].get('op1')

            # Skip if no data
            if all(v is None for v in [left_pre, left_op1, right_pre, right_op1]):
                continue

            # Determine marker color based on color mode
            if color_by_gmfcs:
                patient_color = gmfcs_colors.get(gmfcs, '#333')
            else:
                patient_color = None  # Will use per-point colors

            # Build arrays for this patient's trace
            # Order: L-Pre, L-Op1, None (gap), R-Pre, R-Op1
            x_vals = []
            y_vals = []
            colors = []
            hovertexts = []  # Hover info

            # Left side
            if left_pre is not None:
                x_vals.append(x_positions[('left', 'pre')])
                y_vals.append(left_pre)
                colors.append(patient_color if color_by_gmfcs else pre_color)
                hovertexts.append(build_point_hover(label, left_pre, 'pre', 'left', normative))

            if left_op1 is not None:
                x_vals.append(x_positions[('left', 'op1')])
                y_vals.append(left_op1)
                colors.append(patient_color if color_by_gmfcs else op1_color)
                hovertexts.append(build_point_hover(label, left_op1, 'op1', 'left', normative, left_pre))

            # Add gap between left and right (so line doesn't connect across)
            has_left = left_pre is not None or left_op1 is not None
            has_right = right_pre is not None or right_op1 is not None
            if has_left and has_right:
                x_vals.append(None)
                y_vals.append(None)
                colors.append(line_color)
                hovertexts.append("")

            # Right side
            if right_pre is not None:
                x_vals.append(x_positions[('right', 'pre')])
                y_vals.append(right_pre)
                colors.append(patient_color if color_by_gmfcs else pre_color)
                hovertexts.append(build_point_hover(label, right_pre, 'pre', 'right', normative))

            if right_op1 is not None:
                x_vals.append(x_positions[('right', 'op1')])
                y_vals.append(right_op1)
                colors.append(patient_color if color_by_gmfcs else op1_color)
                hovertexts.append(build_point_hover(label, right_op1, 'op1', 'right', normative, right_pre))

            # Create marker trace for this patient (no text labels)
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines+markers',
                    line=dict(color=patient_color if color_by_gmfcs else line_color, width=1),
                    marker=dict(size=10, color=colors),
                    hovertext=hovertexts,
                    hovertemplate='%{hovertext}<extra></extra>',
                    showlegend=False,
                    name=label
                ),
                row=dof_idx + 1, col=1
            )

        # Calculate mean deltas for left and right
        left_deltas = []
        right_deltas = []
        for pid, data in patient_data.items():
            left_pre = data['left'].get('pre')
            left_op1 = data['left'].get('op1')
            right_pre = data['right'].get('pre')
            right_op1 = data['right'].get('op1')

            if left_pre is not None and left_op1 is not None:
                left_deltas.append(left_op1 - left_pre)
            if right_pre is not None and right_op1 is not None:
                right_deltas.append(right_op1 - right_pre)

        # Add mean delta annotations below the plot
        left_mean_x = (x_positions[('left', 'pre')] + x_positions[('left', 'op1')]) / 2
        right_mean_x = (x_positions[('right', 'pre')] + x_positions[('right', 'op1')]) / 2

        if left_deltas:
            left_mean = sum(left_deltas) / len(left_deltas)
            fig.add_annotation(
                x=left_mean_x,
                y=0,
                yref=f'y{dof_idx + 1}' if dof_idx > 0 else 'y',
                xref=f'x{dof_idx + 1}' if dof_idx > 0 else 'x',
                text=f"<b>Mean Δ: {left_mean:+.1f}</b>",
                showarrow=False,
                yanchor='top',
                yshift=-55,
                font=dict(size=14, color='white')
            )

        if right_deltas:
            right_mean = sum(right_deltas) / len(right_deltas)
            fig.add_annotation(
                x=right_mean_x,
                y=0,
                yref=f'y{dof_idx + 1}' if dof_idx > 0 else 'y',
                xref=f'x{dof_idx + 1}' if dof_idx > 0 else 'x',
                text=f"<b>Mean Δ: {right_mean:+.1f}</b>",
                showarrow=False,
                yanchor='top',
                yshift=-55,
                font=dict(size=14, color='white')
            )

        # Normative line
        if normative is not None:
            fig.add_hline(
                y=normative, line_dash="dash", line_color="gray",
                annotation_text=f"Norm: {normative}",
                annotation_position="top right",
                row=dof_idx + 1, col=1
            )

        fig.update_xaxes(
            tickvals=tick_vals, ticktext=tick_texts,
            range=x_range, row=dof_idx + 1, col=1
        )
        fig.update_yaxes(title_text="Degrees", row=dof_idx + 1, col=1)

    fig.update_layout(
        height=350 * num_dofs,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, b=60)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Data table expander
    with st.expander("View data table"):
        render_data_table(pids, selected_visits, selected_dofs, anonymous)


def render_data_table(pids: list, selected_visits: list, selected_dofs: list, anonymous: bool):
    """Render data table using streamlit columns."""
    # Build data rows
    rows = []
    for p in pids:
        for visit in selected_visits:
            rom_data = load_rom_data(p['pid'], visit)
            if not rom_data:
                continue

            patient_label = p['pid'] if anonymous else (p['name'] or p['pid'])
            row = [patient_label, p['gmfcs'], visit]

            for dof in selected_dofs:
                field = dof['field']
                joint = dof['joint'].lower()

                left_val = get_rom_value(rom_data, 'left', joint, field)
                right_val = get_rom_value(rom_data, 'right', joint, field)

                if left_val is not None:
                    row.append(f"{left_val:.1f}")
                else:
                    row.append("-")

                if right_val is not None:
                    row.append(f"{right_val:.1f}")
                else:
                    row.append("-")

            rows.append(row)

    if not rows:
        st.info("No data available")
        return

    # Build header
    header = ['Patient', 'GMFCS', 'Visit']
    for dof in selected_dofs:
        header.append(f"{dof['display_name']} (L)")
        header.append(f"{dof['display_name']} (R)")

    # Display using st.table format
    # Create a simple text table
    col_widths = [max(len(str(row[i])) for row in [header] + rows) for i in range(len(header))]

    # Header
    header_str = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(header))
    st.text(header_str)
    st.text("-" * len(header_str))

    # Rows
    for row in rows:
        row_str = " | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row))
        st.text(row_str)


def _render_anthro_plot(pids: list, selected_visits: list, anonymous: bool, color_by: str):
    """Render height/weight scatter plots across patients comparing pre/op1."""
    measures = [
        ('Height (cm)', 'height', lambda m: m.get('height')),
        ('Weight (kg)', 'weight', lambda m: m.get('weight')),
    ]

    fig = make_subplots(
        rows=len(measures), cols=1,
        subplot_titles=[m[0] for m in measures],
        vertical_spacing=0.2
    )

    visit_colors = {'pre': '#1f77b4', 'op1': '#2ca02c', 'op2': '#ff7f0e'}
    gmfcs_colors = {'1': '#1f77b4', '2': '#2ca02c', '3': '#ff7f0e'}
    visit_labels_map = {'pre': 'Pre', 'op1': 'Op1', 'op2': 'Op2'}
    color_by_gmfcs = (color_by == "GMFCS")

    # X positions: Pre, Op1
    x_positions = {visit: i for i, visit in enumerate(selected_visits)}
    tick_vals = list(range(len(selected_visits)))
    tick_texts = [visit_labels_map.get(v, v) for v in selected_visits]
    x_range = [-0.5, len(selected_visits) - 0.5]

    # Legend (only once)
    if color_by_gmfcs:
        for gmfcs_level in ['1', '2', '3']:
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode='markers',
                    marker=dict(size=10, color=gmfcs_colors.get(gmfcs_level, '#333')),
                    name=f'GMFCS {gmfcs_level}', showlegend=True,
                ),
                row=1, col=1
            )
    else:
        for visit in selected_visits:
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode='markers',
                    marker=dict(size=10, color=visit_colors.get(visit, '#333')),
                    name=visit_labels_map.get(visit, visit), showlegend=True,
                ),
                row=1, col=1
            )

    for m_idx, (m_label, m_key, extractor) in enumerate(measures):
        deltas = []

        for p in pids:
            pid = p['pid']
            label = p['pid'] if anonymous else (p['name'] or p['pid'])
            gmfcs = p['gmfcs']

            vals = {}
            for visit in selected_visits:
                meta = load_metadata(pid, visit)
                if meta:
                    val = extractor(meta)
                    if val is not None:
                        vals[visit] = val

            if not vals:
                continue

            x_vals = []
            y_vals = []
            colors = []
            hovertexts = []

            for visit in selected_visits:
                if visit not in vals:
                    continue
                val = vals[visit]
                x_vals.append(x_positions[visit])
                y_vals.append(val)

                if color_by_gmfcs:
                    colors.append(gmfcs_colors.get(gmfcs, '#333'))
                else:
                    colors.append(visit_colors.get(visit, '#333'))

                hover_lines = [f"<b>{label}</b> {visit_labels_map.get(visit, visit)}"]
                hover_lines.append(f"{m_label}: {val:.1f}")
                pre_val = vals.get('pre')
                if visit != 'pre' and pre_val is not None:
                    hover_lines.append(f"Δ: {val - pre_val:+.1f}")
                hovertexts.append("<br>".join(hover_lines))

            # Track deltas
            pre_val = vals.get('pre')
            op1_val = vals.get('op1')
            if pre_val is not None and op1_val is not None:
                deltas.append(op1_val - pre_val)

            line_color = gmfcs_colors.get(gmfcs, '#888') if color_by_gmfcs else '#888888'
            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines+markers',
                    line=dict(color=line_color, width=1),
                    marker=dict(size=10, color=colors),
                    hovertext=hovertexts,
                    hovertemplate='%{hovertext}<extra></extra>',
                    showlegend=False, name=label,
                ),
                row=m_idx + 1, col=1
            )

        # Mean delta annotation
        if deltas and len(selected_visits) >= 2:
            mean_delta = sum(deltas) / len(deltas)
            mid_x = (x_positions[selected_visits[0]] + x_positions[selected_visits[1]]) / 2
            fig.add_annotation(
                x=mid_x, y=0,
                yref=f'y{m_idx + 1}' if m_idx > 0 else 'y',
                xref=f'x{m_idx + 1}' if m_idx > 0 else 'x',
                text=f"<b>Mean Δ: {mean_delta:+.1f}</b>",
                showarrow=False, yanchor='top', yshift=-55,
                font=dict(size=14, color='white')
            )

        fig.update_xaxes(
            tickvals=tick_vals, ticktext=tick_texts,
            range=x_range, row=m_idx + 1, col=1
        )
        fig.update_yaxes(title_text=m_label, row=m_idx + 1, col=1)

    fig.update_layout(
        height=350 * len(measures),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, b=60)
    )

    st.plotly_chart(fig, use_container_width=True)
