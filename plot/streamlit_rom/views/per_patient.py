"""Per-patient ROM tabular view."""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from core.data import load_rom_data, get_rom_value, load_surgery_info, load_kinematics_data, load_metadata
from core.normative import get_dof_list, get_normative_value


# Ordered list of joints for display
JOINT_ORDER = [
    'Pelvis[0]', 'Pelvis[1]', 'Pelvis[2]',
    'FemurR[0]', 'FemurR[1]', 'FemurR[2]',
    'TibiaR[0]',
    'TalusR[0]',
    'FemurL[0]', 'FemurL[1]', 'FemurL[2]',
    'TibiaL[0]',
    'TalusL[0]',
]

JOINT_DISPLAY_NAMES = {
    'Pelvis[0]': 'Pelvis Tilt',
    'Pelvis[1]': 'Pelvis Rotation',
    'Pelvis[2]': 'Pelvis Obliquity',
    'FemurR[0]': 'Hip Flex/Ext (R)',
    'FemurR[1]': 'Hip Abduction (R)',
    'FemurR[2]': 'Hip Int Rotation (R)',
    'TibiaR[0]': 'Knee Flexion (R)',
    'TalusR[0]': 'Ankle Dorsiflex (R)',
    'FemurL[0]': 'Hip Flex/Ext (L)',
    'FemurL[1]': 'Hip Abduction (L)',
    'FemurL[2]': 'Hip Int Rotation (L)',
    'TibiaL[0]': 'Knee Flexion (L)',
    'TalusL[0]': 'Ankle Dorsiflex (L)',
}


def get_color(value: float, normative: float) -> str:
    if value is None or normative is None:
        return None
    diff_abs = abs(value - normative)
    if diff_abs <= 10:
        return '#4CAF50'
    elif diff_abs <= 20:
        return '#FFC107'
    else:
        return '#F44336'


def format_diff(value: float, normative: float) -> str:
    if value is None or normative is None:
        return '-'
    diff = value - normative
    return f"+{diff:.0f}" if diff >= 0 else f"{diff:.0f}"


def get_bar_width(value: float, normative: float) -> float:
    if value is None or normative is None:
        return 0
    diff = abs(value - normative)
    return max(0, 100 - (diff * 2))


def render_kinematics_table(kin_data: dict):
    """Render kinematics range table as a single table with all joints."""
    if not kin_data:
        return

    # Build table data in order
    rows = []
    for joint in JOINT_ORDER:
        if joint in kin_data:
            data = kin_data[joint]
            rows.append({
                'Joint': joint,
                'Min': f"{data['range_min']:.1f}",
                'Max': f"{data['range_max']:.1f}",
                'Range': f"{data['range']:.1f}",
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_kinematics_plot(kin_data: dict):
    """Render gait cycle plot with mean ± std shaded regions."""
    if not kin_data:
        return

    # Filter to ordered joints that exist in data
    joints_to_plot = [j for j in JOINT_ORDER if j in kin_data]
    if not joints_to_plot:
        return

    n_joints = len(joints_to_plot)
    n_cols = 2
    n_rows = (n_joints + 1) // 2

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[JOINT_DISPLAY_NAMES.get(j, j) for j in joints_to_plot],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    x = np.linspace(0, 100, 100)  # Gait cycle 0-100%

    for idx, joint in enumerate(joints_to_plot):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        data = kin_data[joint]
        mean = data['mean']
        std = data['std']

        # Shaded region (mean ± std)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([mean + std, (mean - std)[::-1]]),
                fill='toself',
                fillcolor='rgba(99, 110, 250, 0.3)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip',
            ),
            row=row, col=col
        )

        # Mean line
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean,
                mode='lines',
                line=dict(color='rgb(99, 110, 250)', width=2),
                showlegend=False,
                name=joint,
            ),
            row=row, col=col
        )

        # Update axes
        fig.update_xaxes(
            title_text='Gait Cycle (%)' if row == n_rows else '',
            range=[0, 100],
            row=row, col=col
        )
        fig.update_yaxes(
            title_text='Angle (°)',
            row=row, col=col
        )

    fig.update_layout(
        height=250 * n_rows,
        showlegend=False,
        margin=dict(t=40, b=40, l=60, r=20),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_surgery_table(surgery_list: list):
    """Render surgery info as a table with Rt and Lt rows."""
    # Separate surgeries by side
    rt_surgeries = []
    lt_surgeries = []

    for surg in surgery_list:
        if surg.endswith('_Rt'):
            rt_surgeries.append(surg[:-3])  # Remove _Rt suffix
        elif surg.endswith('_Lt'):
            lt_surgeries.append(surg[:-3])  # Remove _Lt suffix
        else:
            # No side suffix, add to both or neither
            rt_surgeries.append(surg)
            lt_surgeries.append(surg)

    # Build HTML table
    rt_str = ', '.join(rt_surgeries) if rt_surgeries else '-'
    lt_str = ', '.join(lt_surgeries) if lt_surgeries else '-'

    html = f'''
    <table style="margin: 10px 0; border-collapse: collapse; font-size: 0.9em;">
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
            <td style="padding: 5px 15px 5px 0; font-weight: bold; color: rgba(255,255,255,0.7);">Rt</td>
            <td style="padding: 5px 0;">{rt_str}</td>
        </tr>
        <tr>
            <td style="padding: 5px 15px 5px 0; font-weight: bold; color: rgba(255,255,255,0.7);">Lt</td>
            <td style="padding: 5px 0;">{lt_str}</td>
        </tr>
    </table>
    '''
    st.markdown(html, unsafe_allow_html=True)


def render_metadata_comparison(pid: str, visits: list):
    """Render height/weight/age/foot comparison across visits."""
    visit_labels = {'pre': 'Pre', 'op1': 'Op1', 'op2': 'Op2'}

    # Load metadata for all visits
    meta_by_visit = {}
    for visit in visits:
        meta = load_metadata(pid, visit)
        if meta:
            meta_by_visit[visit] = meta

    if not meta_by_visit:
        st.info("No metadata available")
        return

    fields = [
        ('Age (yr)', lambda m: m.get('age')),
        ('Height (cm)', lambda m: m.get('height')),
        ('Weight (kg)', lambda m: m.get('weight')),
        ('Foot L length (cm)', lambda m: m.get('foot', {}).get('left', {}).get('length')),
        ('Foot L width (cm)', lambda m: m.get('foot', {}).get('left', {}).get('width')),
        ('Foot R length (cm)', lambda m: m.get('foot', {}).get('right', {}).get('length')),
        ('Foot R width (cm)', lambda m: m.get('foot', {}).get('right', {}).get('width')),
    ]

    pre_meta = meta_by_visit.get('pre')

    rows = []
    for label, extractor in fields:
        row = {'Measure': label}
        row_vals = []
        for v in visits:
            meta = meta_by_visit.get(v)
            val = extractor(meta) if meta else None
            row_vals.append(val)
            row[visit_labels.get(v, v)] = f"{val:.1f}" if val is not None else "-"

        # Skip row if all None
        if all(v is None for v in row_vals):
            continue

        # Change columns relative to pre
        pre_val = extractor(pre_meta) if pre_meta else None
        for i, v in enumerate(visits):
            if v == 'pre':
                continue
            post_val = row_vals[i]
            if pre_val is not None and post_val is not None:
                change = post_val - pre_val
                row[f"Chg {visit_labels.get(v, v)}"] = f"{change:+.1f}"
            else:
                row[f"Chg {visit_labels.get(v, v)}"] = "-"

        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)


def render(pid_info: dict, visit: str, motion_file: str = None):
    pid = pid_info['pid']
    name = pid_info['name']
    gmfcs = pid_info['gmfcs']

    st.markdown(f"### Patient: {pid} ({name}) | GMFCS: {gmfcs} | Visit: {visit}")

    # Display surgery info for op1/op2
    if visit in ['op1', 'op2']:
        surgery_list = load_surgery_info(pid, visit)
        if surgery_list:
            render_surgery_table(surgery_list)

    rom_data = load_rom_data(pid, visit)
    if not rom_data:
        st.warning(f"No ROM data found for {pid}/{visit}")
        return

    dofs = get_dof_list()
    joints = ['Hip', 'Knee', 'Ankle']

    # Build complete HTML with embedded CSS
    full_html = '''
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: transparent; color: #fff; margin: 0; padding: 0; }
    .rom-table { width: 100%; margin-bottom: 30px; }
    .rom-header {
        display: grid;
        grid-template-columns: 2fr 1fr 1fr 0.8fr 0.8fr 0.8fr;
        gap: 10px;
        padding: 12px 10px;
        border-bottom: 2px solid rgba(255,255,255,0.2);
        font-weight: 600;
        color: rgba(255,255,255,0.8);
    }
    .rom-row {
        display: grid;
        grid-template-columns: 2fr 1fr 1fr 0.8fr 0.8fr 0.8fr;
        gap: 10px;
        padding: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        border-radius: 6px;
        align-items: center;
    }
    .rom-row:hover { background: rgba(255,255,255,0.08); }
    .dot {
        display: inline-block;
        width: 12px; height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        vertical-align: middle;
    }
    .bar-container {
        background: rgba(255,255,255,0.1);
        border-radius: 4px;
        height: 8px;
        width: 100%;
        margin-top: 4px;
    }
    .bar { height: 8px; border-radius: 4px; }
    .bar-green { background: rgba(76, 175, 80, 0.6); }
    .bar-yellow { background: rgba(255, 193, 7, 0.6); }
    .bar-red { background: rgba(244, 67, 54, 0.6); }
    .value-cell { display: flex; flex-direction: column; }
    .value-text { display: flex; align-items: center; }
    .joint-title {
        font-size: 1.1em;
        font-weight: 600;
        margin: 25px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(255,255,255,0.3);
    }
    .diff-cell { text-align: center; color: rgba(255,255,255,0.7); }
    .norm-cell { text-align: center; color: rgba(255,255,255,0.5); }
    </style>
    </head>
    <body>
    '''

    total_rows = 0
    for joint in joints:
        joint_dofs = [d for d in dofs if d['joint'] == joint]
        if not joint_dofs:
            continue

        rows = []
        for dof in joint_dofs:
            field = dof['field']
            joint_lower = joint.lower()
            display_name = dof['display_name']

            left_val = get_rom_value(rom_data, 'left', joint_lower, field)
            right_val = get_rom_value(rom_data, 'right', joint_lower, field)

            if left_val is None and right_val is None:
                continue

            # Get normative value (same for both sides)
            norm_val, _ = get_normative_value('left', joint_lower, field)
            if norm_val is None:
                norm_val, _ = get_normative_value('right', joint_lower, field)

            rows.append({
                'name': display_name,
                'left': left_val,
                'right': right_val,
                'norm': norm_val,
                'left_color': get_color(left_val, norm_val),
                'right_color': get_color(right_val, norm_val),
                'left_diff': format_diff(left_val, norm_val),
                'right_diff': format_diff(right_val, norm_val),
                'left_bar': get_bar_width(left_val, norm_val),
                'right_bar': get_bar_width(right_val, norm_val),
            })

        if not rows:
            continue

        total_rows += len(rows)

        full_html += f'<div class="joint-title">{joint.upper()}</div>'
        full_html += '<div class="rom-table">'
        full_html += '''
            <div class="rom-header">
                <div>Measurement</div>
                <div>L</div>
                <div>R</div>
                <div style="text-align:center">Norm</div>
                <div style="text-align:center">Diff L</div>
                <div style="text-align:center">Diff R</div>
            </div>
        '''

        for r in rows:
            left_str = f"{r['left']:.0f}" if r['left'] is not None else "-"
            right_str = f"{r['right']:.0f}" if r['right'] is not None else "-"
            norm_str = f"{r['norm']:.1f}" if r['norm'] else "-"

            left_dot = f'<span class="dot" style="background:{r["left_color"]}"></span>' if r['left_color'] else ''
            right_dot = f'<span class="dot" style="background:{r["right_color"]}"></span>' if r['right_color'] else ''

            lbc = 'bar-green' if r['left_color'] == '#4CAF50' else ('bar-yellow' if r['left_color'] == '#FFC107' else 'bar-red')
            rbc = 'bar-green' if r['right_color'] == '#4CAF50' else ('bar-yellow' if r['right_color'] == '#FFC107' else 'bar-red')

            full_html += f'''
                <div class="rom-row">
                    <div>{r['name']}</div>
                    <div class="value-cell">
                        <div class="value-text">{left_dot}{left_str}</div>
                        <div class="bar-container"><div class="bar {lbc}" style="width:{r['left_bar']:.0f}%"></div></div>
                    </div>
                    <div class="value-cell">
                        <div class="value-text">{right_dot}{right_str}</div>
                        <div class="bar-container"><div class="bar {rbc}" style="width:{r['right_bar']:.0f}%"></div></div>
                    </div>
                    <div class="norm-cell">{norm_str}</div>
                    <div class="diff-cell">{r['left_diff']}</div>
                    <div class="diff-cell">{r['right_diff']}</div>
                </div>
            '''

        full_html += '</div>'

    full_html += '</body></html>'

    # Calculate height based on content
    height = 150 + (total_rows * 60) + (len(joints) * 80)
    components.html(full_html, height=height, scrolling=True)

    # Display kinematics section if motion file provided
    if motion_file:
        kin_data = load_kinematics_data(pid, visit, motion_file)
        if kin_data:
            st.divider()
            st.markdown("### Kinematics (Gait Cycle)")
            st.caption(f"Motion: {motion_file}")

            # Table with joint ranges
            render_kinematics_table(kin_data)

            # Gait cycle plot
            render_kinematics_plot(kin_data)


def render_compare(pid_info: dict, visits: list, motion_file: str = None):
    """Render comparison view showing all visits side by side."""
    pid = pid_info['pid']
    name = pid_info['name']
    gmfcs = pid_info['gmfcs']

    visit_labels = {'pre': 'Pre', 'op1': 'Op1', 'op2': 'Op2'}
    visits_str = ' / '.join([visit_labels.get(v, v) for v in visits])
    st.markdown(f"### Patient: {pid} ({name}) | GMFCS: {gmfcs} | Compare: {visits_str}")

    # Display surgery info for each op visit
    for visit in visits:
        if visit in ['op1', 'op2']:
            surgery_list = load_surgery_info(pid, visit)
            if surgery_list:
                st.markdown(f"**{visit_labels.get(visit, visit)} Surgery**")
                render_surgery_table(surgery_list)

    # Load ROM data for all visits
    rom_data_by_visit = {}
    for visit in visits:
        rom_data_by_visit[visit] = load_rom_data(pid, visit)

    dofs = get_dof_list()
    joints = ['Hip', 'Knee', 'Ankle']

    # Build complete HTML with embedded CSS
    full_html = '''
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: transparent; color: #fff; margin: 0; padding: 0; }
    .rom-table { width: 100%; margin-bottom: 30px; }
    .rom-header {
        display: grid;
        grid-template-columns: 2fr 1.3fr 0.8fr 0.8fr 1.3fr 0.8fr 0.8fr 0.6fr;
        gap: 8px;
        padding: 12px 10px;
        border-bottom: 2px solid rgba(255,255,255,0.2);
        font-weight: 600;
        color: rgba(255,255,255,0.8);
    }
    .rom-row {
        display: grid;
        grid-template-columns: 2fr 1.3fr 0.8fr 0.8fr 1.3fr 0.8fr 0.8fr 0.6fr;
        gap: 8px;
        padding: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        border-radius: 6px;
        align-items: center;
    }
    .rom-row:hover { background: rgba(255,255,255,0.08); }
    .dot {
        display: inline-block;
        width: 10px; height: 10px;
        border-radius: 50%;
        margin-right: 4px;
        vertical-align: middle;
    }
    .joint-title {
        font-size: 1.1em;
        font-weight: 600;
        margin: 25px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(255,255,255,0.3);
    }
    .norm-cell { text-align: center; color: rgba(255,255,255,0.5); }
    .compare-cell { display: flex; flex-direction: column; gap: 2px; }
    .diff-cell { display: flex; flex-direction: column; gap: 2px; text-align: center; }
    .visit-row { display: flex; align-items: center; font-size: 0.9em; }
    .visit-label { color: rgba(255,255,255,0.5); width: 35px; font-size: 0.8em; }
    .visit-value { margin-left: 4px; }
    .diff-row { font-size: 0.9em; color: rgba(255,255,255,0.7); }
    .improved { color: #4CAF50; }
    .worsened { color: #F44336; }
    </style>
    </head>
    <body>
    '''

    total_rows = 0
    for joint in joints:
        joint_dofs = [d for d in dofs if d['joint'] == joint]
        if not joint_dofs:
            continue

        rows = []
        for dof in joint_dofs:
            field = dof['field']
            joint_lower = joint.lower()
            display_name = dof['display_name']

            # Get normative value
            norm_val, _ = get_normative_value('left', joint_lower, field)
            if norm_val is None:
                norm_val, _ = get_normative_value('right', joint_lower, field)

            # Collect values for all visits
            left_values = {}
            right_values = {}
            has_any_value = False

            for visit in visits:
                rom_data = rom_data_by_visit.get(visit)
                if rom_data:
                    left_val = get_rom_value(rom_data, 'left', joint_lower, field)
                    right_val = get_rom_value(rom_data, 'right', joint_lower, field)
                    left_values[visit] = left_val
                    right_values[visit] = right_val
                    if left_val is not None or right_val is not None:
                        has_any_value = True
                else:
                    left_values[visit] = None
                    right_values[visit] = None

            if not has_any_value:
                continue

            rows.append({
                'name': display_name,
                'left_values': left_values,
                'right_values': right_values,
                'norm': norm_val,
            })

        if not rows:
            continue

        total_rows += len(rows)

        full_html += f'<div class="joint-title">{joint.upper()}</div>'
        full_html += '<div class="rom-table">'
        full_html += '''
            <div class="rom-header">
                <div>Measurement</div>
                <div>Left</div>
                <div style="text-align:center">Diff L</div>
                <div style="text-align:center">Chg L</div>
                <div>Right</div>
                <div style="text-align:center">Diff R</div>
                <div style="text-align:center">Chg R</div>
                <div style="text-align:center">Norm</div>
            </div>
        '''

        for r in rows:
            norm_str = f"{r['norm']:.1f}" if r['norm'] else "-"

            # Get pre values as baseline
            pre_left = r['left_values'].get('pre')
            pre_right = r['right_values'].get('pre')

            # Build left cell with all visits
            left_html = '<div class="compare-cell">'
            left_diff_html = '<div class="diff-cell">'
            left_chg_html = '<div class="diff-cell">'
            for visit in visits:
                val = r['left_values'].get(visit)
                color = get_color(val, r['norm'])
                dot = f'<span class="dot" style="background:{color}"></span>' if color else ''
                val_str = f"{val:.0f}" if val is not None else "-"
                diff_str = format_diff(val, r['norm'])

                # Change from pre (surgery enhancement)
                if visit == 'pre':
                    chg_str = '-'
                    chg_class = ''
                elif pre_left is not None and val is not None and r['norm'] is not None:
                    pre_diff_from_norm = abs(pre_left - r['norm'])
                    curr_diff_from_norm = abs(val - r['norm'])
                    change = curr_diff_from_norm - pre_diff_from_norm  # negative = improved
                    if change < 0:
                        chg_class = 'improved'
                        chg_str = f"{change:.0f}"  # already negative
                    elif change > 0:
                        chg_class = 'worsened'
                        chg_str = f"+{change:.0f}"
                    else:
                        chg_class = ''
                        chg_str = "0"
                else:
                    chg_str = '-'
                    chg_class = ''

                left_html += f'<div class="visit-row"><span class="visit-label">{visit_labels.get(visit, visit)}:</span>{dot}<span class="visit-value">{val_str}</span></div>'
                left_diff_html += f'<div class="diff-row">{diff_str}</div>'
                left_chg_html += f'<div class="diff-row {chg_class}">{chg_str}</div>'
            left_html += '</div>'
            left_diff_html += '</div>'
            left_chg_html += '</div>'

            # Build right cell with all visits
            right_html = '<div class="compare-cell">'
            right_diff_html = '<div class="diff-cell">'
            right_chg_html = '<div class="diff-cell">'
            for visit in visits:
                val = r['right_values'].get(visit)
                color = get_color(val, r['norm'])
                dot = f'<span class="dot" style="background:{color}"></span>' if color else ''
                val_str = f"{val:.0f}" if val is not None else "-"
                diff_str = format_diff(val, r['norm'])

                # Change from pre (surgery enhancement)
                if visit == 'pre':
                    chg_str = '-'
                    chg_class = ''
                elif pre_right is not None and val is not None and r['norm'] is not None:
                    pre_diff_from_norm = abs(pre_right - r['norm'])
                    curr_diff_from_norm = abs(val - r['norm'])
                    change = curr_diff_from_norm - pre_diff_from_norm  # negative = improved
                    if change < 0:
                        chg_class = 'improved'
                        chg_str = f"{change:.0f}"  # already negative
                    elif change > 0:
                        chg_class = 'worsened'
                        chg_str = f"+{change:.0f}"
                    else:
                        chg_class = ''
                        chg_str = "0"
                else:
                    chg_str = '-'
                    chg_class = ''

                right_html += f'<div class="visit-row"><span class="visit-label">{visit_labels.get(visit, visit)}:</span>{dot}<span class="visit-value">{val_str}</span></div>'
                right_diff_html += f'<div class="diff-row">{diff_str}</div>'
                right_chg_html += f'<div class="diff-row {chg_class}">{chg_str}</div>'
            right_html += '</div>'
            right_diff_html += '</div>'
            right_chg_html += '</div>'

            full_html += f'''
                <div class="rom-row">
                    <div>{r['name']}</div>
                    {left_html}
                    {left_diff_html}
                    {left_chg_html}
                    {right_html}
                    {right_diff_html}
                    {right_chg_html}
                    <div class="norm-cell">{norm_str}</div>
                </div>
            '''

        full_html += '</div>'

    full_html += '</body></html>'

    # Calculate height based on content
    height = 150 + (total_rows * 80) + (len(joints) * 80)
    components.html(full_html, height=height, scrolling=True)

    # Display kinematics section if motion file provided (use first visit)
    if motion_file and visits:
        first_visit = visits[0]
        kin_data = load_kinematics_data(pid, first_visit, motion_file)
        if kin_data:
            st.divider()
            st.markdown(f"### Kinematics (Gait Cycle) - {first_visit}")
            st.caption(f"Motion: {motion_file}")

            # Table with joint ranges
            render_kinematics_table(kin_data)

            # Gait cycle plot
            render_kinematics_plot(kin_data)
