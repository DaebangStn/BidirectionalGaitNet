"""Summary table view for all body nodes with styled HTML tables."""
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from core.skeleton import list_global_files, load_global_transforms, get_node_names, get_node_data


def get_std_color(std_val: float, threshold: float) -> str:
    """Get color based on std deviation threshold."""
    if std_val is None:
        return None
    ratio = std_val / threshold if threshold > 0 else 0
    if ratio <= 0.5:
        return '#4CAF50'  # Green - low variability
    elif ratio <= 1.0:
        return '#FFC107'  # Yellow - moderate
    else:
        return '#F44336'  # Red - high variability


def render(pid: str, visit: str):
    """Render summary tables for all body nodes.

    Shows unified reference values and mean±std across all trials.
    Split into Size, Position, and Rotation sections.
    """
    st.header("Skeleton Summary")

    # Sidebar: Std threshold controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Std Warning Thresholds")
    size_threshold = st.sidebar.number_input(
        "Size (m)", min_value=0.001, max_value=0.1, value=0.005, step=0.001,
        format="%.3f", key="size_threshold",
        help="Highlight std > threshold in red"
    )
    pos_threshold = st.sidebar.number_input(
        "Position (m)", min_value=0.001, max_value=0.1, value=0.01, step=0.005,
        format="%.3f", key="pos_threshold"
    )
    local_threshold = st.sidebar.number_input(
        "Local Trans (m)", min_value=0.001, max_value=0.1, value=0.01, step=0.005,
        format="%.3f", key="local_threshold"
    )
    angle_threshold = st.sidebar.number_input(
        "Angle (deg)", min_value=0.1, max_value=30.0, value=5.0, step=1.0,
        format="%.1f", key="angle_threshold"
    )

    # Load all global files
    names = list_global_files(pid, visit)
    if not names:
        st.warning("No global transform files found.")
        return

    ref_name = "trimmed_unified"
    if ref_name not in names:
        st.error(f"Reference '{ref_name}' not found")
        return

    trial_names = [n for n in names if n != ref_name]
    ref_data = load_global_transforms(pid, visit, ref_name)
    node_names = get_node_names(ref_data)

    # Collect all trial data
    all_trials = []
    for name in trial_names:
        all_trials.append(load_global_transforms(pid, visit, name))

    # Prepare data for each node
    node_data = []
    for node_name in node_names:
        ref_node = get_node_data(ref_data, node_name)
        if not ref_node:
            continue

        # Collect trial values
        trial_values = {
            'size_x': [], 'size_y': [], 'size_z': [],
            'pos_x': [], 'pos_y': [], 'pos_z': [],
            'local_x': [], 'local_y': [], 'local_z': [],
            'axis_x': [], 'axis_y': [], 'axis_z': [], 'angle': []
        }

        for trial in all_trials:
            node = get_node_data(trial, node_name)
            if node:
                trial_values['size_x'].append(node['size_meters'][0])
                trial_values['size_y'].append(node['size_meters'][1])
                trial_values['size_z'].append(node['size_meters'][2])
                trial_values['pos_x'].append(node['global_position'][0])
                trial_values['pos_y'].append(node['global_position'][1])
                trial_values['pos_z'].append(node['global_position'][2])
                local_trans = node.get('local_translation', [0, 0, 0])
                trial_values['local_x'].append(local_trans[0])
                trial_values['local_y'].append(local_trans[1])
                trial_values['local_z'].append(local_trans[2])
                trial_values['axis_x'].append(node['global_rotation']['axis'][0])
                trial_values['axis_y'].append(node['global_rotation']['axis'][1])
                trial_values['axis_z'].append(node['global_rotation']['axis'][2])
                trial_values['angle'].append(node['global_rotation']['angle_deg'])

        # Compute stats
        stats = {}
        for key, vals in trial_values.items():
            if vals:
                stats[key] = {'mean': np.mean(vals), 'std': np.std(vals)}
            else:
                stats[key] = {'mean': 0, 'std': 0}

        node_data.append({
            'name': node_name,
            'ref': ref_node,
            'stats': stats
        })

    st.markdown(f"**Reference**: `{ref_name}` | **Trials**: {len(trial_names)}")

    # Common CSS styles
    css = '''
    <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: transparent; color: #fff; margin: 0; padding: 0; }
    .summary-table { width: 100%; margin-bottom: 20px; }
    .summary-header {
        display: grid;
        gap: 8px;
        padding: 10px 8px;
        border-bottom: 2px solid rgba(255,255,255,0.2);
        font-weight: 600;
        color: rgba(255,255,255,0.8);
        font-size: 0.9em;
    }
    .summary-row {
        display: grid;
        gap: 8px;
        padding: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        border-radius: 4px;
        align-items: center;
        font-size: 0.85em;
    }
    .summary-row:hover { background: rgba(255,255,255,0.08); }
    .dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
    }
    .node-name { font-weight: 500; }
    .value-cell { text-align: right; font-family: monospace; }
    .ref-val { color: rgba(255,255,255,0.6); }
    .mean-val { color: rgba(255,255,255,0.9); }
    .std-val { color: rgba(255,255,255,0.5); font-size: 0.9em; }
    .section-title {
        font-size: 1.1em;
        font-weight: 600;
        margin: 20px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 2px solid rgba(255,255,255,0.3);
    }
    .separator-row {
        border-top: 3px solid rgba(255,255,255,0.4);
        margin-top: 4px;
    }
    </style>
    '''

    # Nodes that should have a thick separator BEFORE them
    separator_nodes = {'Spine', 'ArmR'}

    # SIZE TABLE
    size_html = f'''
    <!DOCTYPE html><html><head>{css}</head><body>
    <div class="section-title">BODY SIZE (meters)</div>
    <div class="summary-table">
    <div class="summary-header" style="grid-template-columns: 2fr 1fr 1fr 1fr 1fr 1fr 1fr;">
        <div>Node</div>
        <div style="text-align:right">Ref X</div>
        <div style="text-align:right">Mean±Std X</div>
        <div style="text-align:right">Ref Y</div>
        <div style="text-align:right">Mean±Std Y</div>
        <div style="text-align:right">Ref Z</div>
        <div style="text-align:right">Mean±Std Z</div>
    </div>
    '''

    for nd in node_data:
        ref = nd['ref']
        stats = nd['stats']

        def format_size_cell(axis_idx, key):
            ref_val = ref['size_meters'][axis_idx]
            mean_val = stats[key]['mean']
            std_val = stats[key]['std']
            color = get_std_color(std_val, size_threshold)
            dot = f'<span class="dot" style="background:{color}"></span>' if color else ''
            return (
                f'<div class="value-cell ref-val">{ref_val:.4f}</div>',
                f'<div class="value-cell">{dot}<span class="mean-val">{mean_val:.4f}</span><span class="std-val">±{std_val:.4f}</span></div>'
            )

        ref_x, cell_x = format_size_cell(0, 'size_x')
        ref_y, cell_y = format_size_cell(1, 'size_y')
        ref_z, cell_z = format_size_cell(2, 'size_z')

        sep_class = ' separator-row' if nd['name'] in separator_nodes else ''
        size_html += f'''
        <div class="summary-row{sep_class}" style="grid-template-columns: 2fr 1fr 1fr 1fr 1fr 1fr 1fr;">
            <div class="node-name">{nd['name']}</div>
            {ref_x}{cell_x}{ref_y}{cell_y}{ref_z}{cell_z}
        </div>
        '''

    size_html += '</div></body></html>'

    # POSITION TABLE
    pos_html = f'''
    <!DOCTYPE html><html><head>{css}</head><body>
    <div class="section-title">GLOBAL POSITION (meters)</div>
    <div class="summary-table">
    <div class="summary-header" style="grid-template-columns: 2fr 1fr 1fr 1fr 1fr 1fr 1fr;">
        <div>Node</div>
        <div style="text-align:right">Ref X</div>
        <div style="text-align:right">Mean±Std X</div>
        <div style="text-align:right">Ref Y</div>
        <div style="text-align:right">Mean±Std Y</div>
        <div style="text-align:right">Ref Z</div>
        <div style="text-align:right">Mean±Std Z</div>
    </div>
    '''

    for nd in node_data:
        ref = nd['ref']
        stats = nd['stats']

        def format_pos_cell(axis_idx, key):
            ref_val = ref['global_position'][axis_idx]
            mean_val = stats[key]['mean']
            std_val = stats[key]['std']
            color = get_std_color(std_val, pos_threshold)
            dot = f'<span class="dot" style="background:{color}"></span>' if color else ''
            return (
                f'<div class="value-cell ref-val">{ref_val:.4f}</div>',
                f'<div class="value-cell">{dot}<span class="mean-val">{mean_val:.4f}</span><span class="std-val">±{std_val:.4f}</span></div>'
            )

        ref_x, cell_x = format_pos_cell(0, 'pos_x')
        ref_y, cell_y = format_pos_cell(1, 'pos_y')
        ref_z, cell_z = format_pos_cell(2, 'pos_z')

        sep_class = ' separator-row' if nd['name'] in separator_nodes else ''
        pos_html += f'''
        <div class="summary-row{sep_class}" style="grid-template-columns: 2fr 1fr 1fr 1fr 1fr 1fr 1fr;">
            <div class="node-name">{nd['name']}</div>
            {ref_x}{cell_x}{ref_y}{cell_y}{ref_z}{cell_z}
        </div>
        '''

    pos_html += '</div></body></html>'

    # LOCAL TRANSLATION TABLE
    local_html = f'''
    <!DOCTYPE html><html><head>{css}</head><body>
    <div class="section-title">LOCAL TRANSLATION (w.r.t. Parent, meters)</div>
    <div class="summary-table">
    <div class="summary-header" style="grid-template-columns: 2fr 1fr 1fr 1fr 1fr 1fr 1fr;">
        <div>Node</div>
        <div style="text-align:right">Ref X</div>
        <div style="text-align:right">Mean±Std X</div>
        <div style="text-align:right">Ref Y</div>
        <div style="text-align:right">Mean±Std Y</div>
        <div style="text-align:right">Ref Z</div>
        <div style="text-align:right">Mean±Std Z</div>
    </div>
    '''

    for nd in node_data:
        ref = nd['ref']
        stats = nd['stats']
        ref_local = ref.get('local_translation', [0, 0, 0])

        def format_local_cell(axis_idx, key):
            ref_val = ref_local[axis_idx]
            mean_val = stats[key]['mean']
            std_val = stats[key]['std']
            color = get_std_color(std_val, local_threshold)
            dot = f'<span class="dot" style="background:{color}"></span>' if color else ''
            return (
                f'<div class="value-cell ref-val">{ref_val:.4f}</div>',
                f'<div class="value-cell">{dot}<span class="mean-val">{mean_val:.4f}</span><span class="std-val">±{std_val:.4f}</span></div>'
            )

        ref_x, cell_x = format_local_cell(0, 'local_x')
        ref_y, cell_y = format_local_cell(1, 'local_y')
        ref_z, cell_z = format_local_cell(2, 'local_z')

        sep_class = ' separator-row' if nd['name'] in separator_nodes else ''
        local_html += f'''
        <div class="summary-row{sep_class}" style="grid-template-columns: 2fr 1fr 1fr 1fr 1fr 1fr 1fr;">
            <div class="node-name">{nd['name']}</div>
            {ref_x}{cell_x}{ref_y}{cell_y}{ref_z}{cell_z}
        </div>
        '''

    local_html += '</div></body></html>'

    # ROTATION TABLE
    rot_html = f'''
    <!DOCTYPE html><html><head>{css}</head><body>
    <div class="section-title">GLOBAL ROTATION (Axis-Angle)</div>
    <div class="summary-table">
    <div class="summary-header" style="grid-template-columns: 2fr 1fr 1fr 1fr 1fr 1fr 1fr 1fr 1fr;">
        <div>Node</div>
        <div style="text-align:right">Ref aX</div>
        <div style="text-align:right">Mean±Std aX</div>
        <div style="text-align:right">Ref aY</div>
        <div style="text-align:right">Mean±Std aY</div>
        <div style="text-align:right">Ref aZ</div>
        <div style="text-align:right">Mean±Std aZ</div>
        <div style="text-align:right">Ref Angle</div>
        <div style="text-align:right">Mean±Std Angle</div>
    </div>
    '''

    for nd in node_data:
        ref = nd['ref']
        stats = nd['stats']

        ref_axis = ref['global_rotation']['axis']
        ref_angle = ref['global_rotation']['angle_deg']

        def format_axis_cell(axis_idx, key):
            ref_val = ref_axis[axis_idx]
            mean_val = stats[key]['mean']
            std_val = stats[key]['std']
            color = get_std_color(std_val, angle_threshold / 10.0)
            dot = f'<span class="dot" style="background:{color}"></span>' if color else ''
            return (
                f'<div class="value-cell ref-val">{ref_val:.3f}</div>',
                f'<div class="value-cell">{dot}<span class="mean-val">{mean_val:.3f}</span><span class="std-val">±{std_val:.3f}</span></div>'
            )

        ref_ax, cell_ax = format_axis_cell(0, 'axis_x')
        ref_ay, cell_ay = format_axis_cell(1, 'axis_y')
        ref_az, cell_az = format_axis_cell(2, 'axis_z')

        mean_angle = stats['angle']['mean']
        std_angle = stats['angle']['std']
        color = get_std_color(std_angle, angle_threshold)
        dot = f'<span class="dot" style="background:{color}"></span>' if color else ''

        sep_class = ' separator-row' if nd['name'] in separator_nodes else ''
        rot_html += f'''
        <div class="summary-row{sep_class}" style="grid-template-columns: 2fr 1fr 1fr 1fr 1fr 1fr 1fr 1fr 1fr;">
            <div class="node-name">{nd['name']}</div>
            {ref_ax}{cell_ax}{ref_ay}{cell_ay}{ref_az}{cell_az}
            <div class="value-cell ref-val">{ref_angle:.1f}°</div>
            <div class="value-cell">{dot}<span class="mean-val">{mean_angle:.1f}°</span><span class="std-val">±{std_angle:.1f}</span></div>
        </div>
        '''

    rot_html += '</div></body></html>'

    # Render all tables
    num_nodes = len(node_data)
    row_height = 35
    header_height = 80

    components.html(size_html, height=header_height + num_nodes * row_height, scrolling=False)
    components.html(pos_html, height=header_height + num_nodes * row_height, scrolling=False)
    components.html(local_html, height=header_height + num_nodes * row_height, scrolling=False)
    components.html(rot_html, height=header_height + num_nodes * row_height, scrolling=False)

    # CSV download
    with st.expander("Download CSV"):
        import pandas as pd
        rows = []
        for nd in node_data:
            ref = nd['ref']
            stats = nd['stats']
            ref_local = ref.get('local_translation', [0, 0, 0])
            rows.append({
                'Node': nd['name'],
                'Size_X_ref': ref['size_meters'][0],
                'Size_X_mean': stats['size_x']['mean'],
                'Size_X_std': stats['size_x']['std'],
                'Size_Y_ref': ref['size_meters'][1],
                'Size_Y_mean': stats['size_y']['mean'],
                'Size_Y_std': stats['size_y']['std'],
                'Size_Z_ref': ref['size_meters'][2],
                'Size_Z_mean': stats['size_z']['mean'],
                'Size_Z_std': stats['size_z']['std'],
                'Pos_X_ref': ref['global_position'][0],
                'Pos_X_mean': stats['pos_x']['mean'],
                'Pos_X_std': stats['pos_x']['std'],
                'Pos_Y_ref': ref['global_position'][1],
                'Pos_Y_mean': stats['pos_y']['mean'],
                'Pos_Y_std': stats['pos_y']['std'],
                'Pos_Z_ref': ref['global_position'][2],
                'Pos_Z_mean': stats['pos_z']['mean'],
                'Pos_Z_std': stats['pos_z']['std'],
                'Local_X_ref': ref_local[0],
                'Local_X_mean': stats['local_x']['mean'],
                'Local_X_std': stats['local_x']['std'],
                'Local_Y_ref': ref_local[1],
                'Local_Y_mean': stats['local_y']['mean'],
                'Local_Y_std': stats['local_y']['std'],
                'Local_Z_ref': ref_local[2],
                'Local_Z_mean': stats['local_z']['mean'],
                'Local_Z_std': stats['local_z']['std'],
                'Angle_ref': ref['global_rotation']['angle_deg'],
                'Angle_mean': stats['angle']['mean'],
                'Angle_std': stats['angle']['std'],
            })
        df = pd.DataFrame(rows)
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, "skeleton_summary.csv", "text/csv")
