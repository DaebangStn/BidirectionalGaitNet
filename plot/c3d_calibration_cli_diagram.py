#!/usr/bin/env python3
"""Generate calibration algorithm diagram for presentation slide (16:9, no empty space)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

# ══════════════════════════════════════════════════════════════
# TEXT CONTENT — edit these to change diagram labels
# ══════════════════════════════════════════════════════════════

STATIC_LABEL = "Static Calibration\n(single frame)"
STATIC_TEXTS = [
    "Hip joint center\n(Harrington regression)",
    "Bone fitting\nscale, transformation\n(1 frame, per bone)",
    "Talus marker offset\n(find marker height)",
    "Marker offset for\ntorso, shank\n(back-project from data)",
    "Bone scale and\nMarker offset",
]

DYNAMIC_LABEL = "Dynamic Calibration (~5k frames)"

STAGE_A_LABEL = "A: Bone Fitting"
STAGE_A_TEXTS = [
    "Hip joint center\n(Harrington regression)",
    "Bone fitting\nscale, transformation\n(all frames, per bone)",
    "Arm scaling\n(two marker body part)",
    "Dependent scales\nspine → torso\nforearm → hand\n(copy from neighbor)",
    "Symmetry\nenforcement",
]

STAGE_B_LABEL = "B: Build skeleton and motion"
STAGE_B_TEXTS = [
    "Relative transform\n(parent-to-child)",
    "Build free joint skeleton\n(rotation for each DOFs)",
    "Joint offset estimation\n(least-squares fitting)",
    "Project joint angles to\nconstrained skeleton",
]

STAGE_C_LABEL = "C: IK Refinement"
STAGE_C_TEXTS = [
    "Foot lock detection\nbased on the bone velocity",
    "Leg IK for locked phases\nDLS + line search",
    "Arm IK\n2-link analytic",
]

# ══════════════════════════════════════════════════════════════
# LAYOUT CONFIG — edit these to adjust positions and sizes
# ══════════════════════════════════════════════════════════════

# Canvas
W, H = 16, 9                   # slide dimensions (16:9)
MARGIN = 0.30                   # outer margin around everything
DPI = 200                       # output resolution

# Gaps
STATIC_DYN_GAP = 0.40          # horizontal gap: Static ↔ Dynamic
ROW_GAP = 0.35                  # uniform vertical gap between A↔B and B↔C

# Dynamic area
DYN_HEADER_H = 0.40            # height for "Dynamic Calibration" header text
DYN_PAD = 0.10                  # inner padding of dynamic outer box

# Static column
STATIC_W = 2.55                 # width of static calibration column

# Row height split (3 rows, must sum to 1.0)
ROW_A_RATIO = 0.38              # Stage A height fraction
ROW_B_RATIO = 0.28              # Stage B height fraction
ROW_C_RATIO = 0.34              # Stage C height fraction

# Node heights
NODE_H_2LINE = 0.80             # node height for 2-line text
NODE_H_3LINE = 1.05             # node height for 3-line text

# Cluster padding (shared)
CLUSTER_BG_PAD = 0.20           # padding of cluster background around nodes
CLUSTER_LABEL_H = 0.30          # height reserved for cluster label

# Stage A
A_NODE_PAD_X = 0.22            # horizontal padding inside A cluster
A_NODE_GAP = 0.30              # horizontal gap between A nodes
A_CY_OFFSET = -0.10            # vertical offset of A nodes from cluster center

# Stage B (horizontal row)
B_X_OFFSET = 0.00              # extra horizontal offset (positive = right)
B_WIDTH_RATIO = 1.00            # width as fraction of DYN_W
B_NODE_PAD = 0.20               # horizontal padding inside B cluster
B_NODE_GAP = 0.30               # horizontal gap between B nodes

# Stage C (horizontal row)
C_X_OFFSET = 0.00              # extra horizontal offset (positive = right)
C_WIDTH_RATIO = 1.00            # width as fraction of DYN_W
C_NODE_PAD = 0.20               # horizontal padding inside C cluster
C_NODE_GAP = 0.30               # horizontal gap between C nodes

# Font sizes
FS_DYN_HEADER = 10             # "Dynamic Calibration" header
FS_STATIC_LABEL = 8            # Static cluster label
FS_STATIC_NODE = 8             # Static nodes
FS_A_LABEL = 9                 # A cluster label
FS_A_NODE = 8.5                # A nodes
FS_B_LABEL = 8                 # B cluster label
FS_B_NODE = 8.5                # B nodes
FS_C_LABEL = 8                 # C cluster label
FS_C_NODE = 8.5                # C nodes

# ══════════════════════════════════════════════════════════════
# RENDERING (derived values + drawing — generally no need to edit)
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# ── helpers ──────────────────────────────────────────────────

def cluster(x, y, w, h, label, bg, ec, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
        fc=bg, ec=ec, lw=1.8, zorder=2))
    ax.text(x + w/2, y + h - 0.15, label, fontsize=fs, fontweight='bold',
        color=ec, ha='center', va='top', zorder=5)

def node(cx, cy, w, h, txt, fc, fs=9, ec='#444444'):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
        boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=0.8, zorder=3))
    ax.text(cx, cy, txt, fontsize=fs, ha='center', va='center',
        zorder=5, linespacing=1.2, family='sans-serif')
    return cx, cy

def harrow(x1, y1, x2, y2, c='#555', lw=1.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=c, lw=lw), zorder=4)

def larrow(pts, c='#555', lw=1.8):
    for i in range(len(pts)-2):
        ax.plot([pts[i][0], pts[i+1][0]], [pts[i][1], pts[i+1][1]],
            color=c, lw=lw, zorder=4, solid_capstyle='butt')
    ax.annotate('', xy=pts[-1], xytext=pts[-2],
        arrowprops=dict(arrowstyle='->', color=c, lw=lw), zorder=4)

# ── derived layout ──────────────────────────────────────────

DYN_X = MARGIN + STATIC_W + STATIC_DYN_GAP
DYN_W = W - DYN_X - MARGIN

avail_h = H - 2*MARGIN - 2*ROW_GAP - DYN_HEADER_H
ROW_A_H = avail_h * ROW_A_RATIO
ROW_B_H = avail_h * ROW_B_RATIO
ROW_C_H = avail_h * ROW_C_RATIO

ROW_C_Y = MARGIN
ROW_B_Y = MARGIN + ROW_C_H + ROW_GAP
ROW_A_Y = MARGIN + ROW_C_H + ROW_GAP + ROW_B_H + ROW_GAP
MID_AB = ROW_A_Y - ROW_GAP/2
MID_BC = ROW_B_Y - ROW_GAP/2

# ── Dynamic Calibration outer background ─────────────────────

dyn_box_x = DYN_X - DYN_PAD
dyn_box_y = MARGIN - DYN_PAD
dyn_box_w = DYN_W + 2*DYN_PAD
dyn_box_h = H - 2*MARGIN + 2*DYN_PAD
ax.add_patch(FancyBboxPatch((dyn_box_x, dyn_box_y), dyn_box_w, dyn_box_h,
    boxstyle="round,pad=0.08", fc='#FAFAFA', ec='#616161', lw=2.2, zorder=0))
ax.text(DYN_X + DYN_W/2, dyn_box_y + dyn_box_h - 0.12, DYNAMIC_LABEL,
    fontsize=FS_DYN_HEADER, fontweight='bold', color='#424242',
    ha='center', va='top', zorder=5)

# ── STATIC column (left, full height) ────────────────────────

sx, sy, sw, sh = MARGIN, MARGIN, STATIC_W, H - 2*MARGIN
cluster(sx, sy, sw, sh, STATIC_LABEL, '#E3F2FD', '#1565C0', fs=FS_STATIC_LABEL)

sn = len(STATIC_TEXTS)
sn_w = sw - 0.36
sn_h = NODE_H_2LINE
sn_top = sy + sh - 0.80
sn_bot = sy + 0.12
sn_spacing = (sn_top - sn_bot - sn*sn_h) / max(sn-1, 1)
scx = sx + sw/2
sc = []
for i in range(sn):
    cy = sn_top - sn_h/2 - i*(sn_h + sn_spacing)
    fc = '#90CAF9' if i == 2 else '#BBDEFB'
    node(scx, cy, sn_w, sn_h, STATIC_TEXTS[i], fc, fs=FS_STATIC_NODE)
    sc.append((scx, cy))

for i in range(sn-1):
    harrow(sc[i][0], sc[i][1]-sn_h/2, sc[i+1][0], sc[i+1][1]+sn_h/2, '#1565C0', 2)

# ── Stage A (top row) ─────────────────────────────────────────

cluster(DYN_X, ROW_A_Y, DYN_W, ROW_A_H,
    STAGE_A_LABEL, '#FFF3E0', '#E65100', fs=FS_A_LABEL)

an = len(STAGE_A_TEXTS)
an_area_w = DYN_W - 2*A_NODE_PAD_X
an_w = (an_area_w - (an-1)*A_NODE_GAP) / an
an_h = NODE_H_3LINE
an_cy = ROW_A_Y + ROW_A_H/2 + A_CY_OFFSET
ac = []
for i in range(an):
    cx = DYN_X + A_NODE_PAD_X + an_w/2 + i*(an_w + A_NODE_GAP)
    node(cx, an_cy, an_w, an_h, STAGE_A_TEXTS[i], '#FFE0B2', fs=FS_A_NODE)
    ac.append((cx, an_cy))

for i in range(an-1):
    harrow(ac[i][0]+an_w/2, ac[i][1], ac[i+1][0]-an_w/2, ac[i+1][1], '#E65100', 2)

# ── Stage B (middle row) ─────────────────────────────────────

B_X = DYN_X + B_X_OFFSET
B_W = DYN_W * B_WIDTH_RATIO
cluster(B_X, ROW_B_Y, B_W, ROW_B_H,
    STAGE_B_LABEL, '#E8F5E9', '#2E7D32', fs=FS_B_LABEL)

bn = len(STAGE_B_TEXTS)
bt_h = NODE_H_2LINE
bt_w = (B_W - 2*B_NODE_PAD - (bn-1)*B_NODE_GAP) / bn
b_cy = ROW_B_Y + ROW_B_H/2 - 0.10
btc = []
for i in range(bn):
    cx = B_X + B_NODE_PAD + bt_w/2 + i*(bt_w + B_NODE_GAP)
    node(cx, b_cy, bt_w, bt_h, STAGE_B_TEXTS[i], '#C8E6C9', fs=FS_B_NODE)
    btc.append((cx, b_cy))
for i in range(bn-1):
    harrow(btc[i][0]+bt_w/2, btc[i][1], btc[i+1][0]-bt_w/2, btc[i+1][1], '#2E7D32', 1.8)

# ── Stage C (bottom row) ─────────────────────────────────────

C_X = DYN_X + C_X_OFFSET
C_W = DYN_W * C_WIDTH_RATIO
cluster(C_X, ROW_C_Y, C_W, ROW_C_H,
    STAGE_C_LABEL, '#FFEBEE', '#C62828', fs=FS_C_LABEL)

cn = len(STAGE_C_TEXTS)
cn_h = NODE_H_3LINE
cn_w = (C_W - 2*C_NODE_PAD - (cn-1)*C_NODE_GAP) / cn
c_cy = ROW_C_Y + ROW_C_H/2 - 0.10
cc = []
for i in range(cn):
    ccx = C_X + C_NODE_PAD + cn_w/2 + i*(cn_w + C_NODE_GAP)
    node(ccx, c_cy, cn_w, cn_h, STAGE_C_TEXTS[i], '#FFCDD2', fs=FS_C_NODE)
    cc.append((ccx, c_cy))
for i in range(cn-1):
    harrow(cc[i][0]+cn_w/2, cc[i][1], cc[i+1][0]-cn_w/2, cc[i+1][1], '#C62828', 1.8)

# ── Inter-stage arrows ───────────────────────────────────────

# A → B (L-shaped)
a_out_x = ac[-1][0]
a_out_y = ac[-1][1] - an_h/2
b_in_x = btc[0][0]
b_in_y = b_cy + bt_h/2
larrow([(a_out_x, a_out_y), (a_out_x, MID_AB),
        (b_in_x, MID_AB), (b_in_x, b_in_y)],
    '#2E7D32', 2.0)

# B → C (L-shaped)
b_out_x = btc[-1][0]
b_out_y = b_cy - bt_h/2
c_in_x = cc[0][0]
c_in_y = c_cy + cn_h/2
larrow([(b_out_x, b_out_y), (b_out_x, MID_BC),
        (c_in_x, MID_BC), (c_in_x, c_in_y)],
    '#C62828', 2.0)

# ── Save ─────────────────────────────────────────────────────

out = os.path.join(os.path.dirname(__file__), 'diagram', 'c3d_calibration_cli.png')
fig.savefig(out, bbox_inches='tight', facecolor='white', dpi=DPI, pad_inches=0.02)
plt.close()
print(f"Saved: {out}")
