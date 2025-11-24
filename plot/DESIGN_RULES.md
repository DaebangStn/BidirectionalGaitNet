# Diagram Design Rules

Core principle: **Visual hierarchy guides the eye to the main story**

## Problem Analysis

### Why Current Diagrams Look Cluttered

1. **Mixed visual language**: Data flow, function calls, learning signals, weight sync, CPU↔GPU boundaries all use similar line styles → eye loses track

2. **Overloaded labels**: "Policy NN libtorch CPU Actor-Critic" cramming too much into boxes → visual density kills clarity

3. **Flat hierarchy**: C++ rollout vs Python learning zones lack clear dominance → main flow not obvious at first glance

4. **Crossing spaghetti**: Default splines allow free curves/crossings → complexity becomes maze

5. **No alignment axes**: Auto-layout creates "random-ish grid" → human eye needs horizontal/vertical alignment for clean perception

## Design Strategy

### 1. Main Story First

The core narrative:
```
Rollout (C++) → Trajectory/Muscle → Learner (Python) → Weight Sync → Rollout
```

**Visual encoding:**
- Main flow: **thick solid black lines**
- Supporting data: **thin gray lines**
- Sync/loops: **dashed + single accent color**

### 2. Two-Layer Text Hierarchy

**Box labels:** Short names only
- Policy NN
- ThreadPool
- Environment
- Trajectory
- Muscle Net
- Agent Net
- PPO
- Weight Sync

**Technical details:** Small sublabels (DOT `xlabel`)
- "libtorch CPU"
- "PyTorch GPU"
- "Actor-Critic"
- "Hierarchical"

### 3. Column Pipeline Layout

**Left column (C++):** Vertical stack
```
Policy NN
    ↓
ThreadPool
    ↓
Environment
    ↓
Trajectory
```

**Right column (Python):** Parallel stack
```
PPO → Agent Net
Muscle → Muscle Net
    ↓
Weight Sync
```

Parallel axes → 70% cleaner perception

### 4. Orthogonal Routing

```dot
splines=ortho;  // Circuit-board style routing
```

Benefits:
- Reduces crossings dramatically
- Creates clear horizontal/vertical lanes
- Aligns with human spatial perception

### 5. Rank Control

**Same level alignment:**
```dot
{rank=same; node1; node2;}
```

**Layout guidance:**
```dot
invisible_edge [style=invis];  // Axis anchoring
```

### 6. Subtle Clusters

Background only - content is king:
- Very pale background colors
- Thin borders
- Small top-left labels

## DOT Base Template

```dot
digraph G {
  rankdir=LR;
  splines=ortho;
  nodesep=0.5;
  ranksep=0.8;
  fontname="Helvetica";

  node [
    shape=box,
    style="rounded,filled",
    fillcolor="white",
    color="gray30",
    fontname="Helvetica",
    fontsize=11,
    margin="0.15,0.10"
  ];

  edge [
    color="gray30",
    penwidth=1.2,
    arrowsize=0.8,
    fontname="Helvetica",
    fontsize=9
  ];

  // Define clusters with subtle backgrounds
  subgraph cluster_X {
    style=filled;
    fillcolor="gray95";
    color="gray70";
    penwidth=0.5;
    label=<<font point-size="9">Section</font>>;
  }
}
```

## Edge Visual Language

| Flow Type | Style | Usage |
|-----------|-------|-------|
| Main data flow | `penwidth=2.5, color=black` | Core pipeline |
| Supporting data | `penwidth=1.0, color=gray50` | Helper flows |
| Sync/feedback | `penwidth=1.5, color=orange, style=dashed` | Loops only |
| Internal call | `penwidth=1.0, color=gray70, style=dotted` | Inside boxes |

## Color Palette

**Functional zones:**
- C++ components: `#E3F2FD` (very pale blue)
- Python components: `#F1F8E9` (very pale green)
- Sync/bridge: `#FFF8E1` (very pale yellow)

**Node fills:**
- Default: `white`
- Emphasis: `gray10` text on white
- Critical path: `gray20` border

**Avoid:** Saturated colors for large areas

## Checklist Before Rendering

- [ ] Main flow uses thickest lines
- [ ] Box labels ≤ 2 words
- [ ] Technical details in xlabel or small font
- [ ] Clusters have pale backgrounds
- [ ] Same-level nodes use {rank=same}
- [ ] splines=ortho enabled
- [ ] No more than 3 edge styles used
- [ ] White space preserved between clusters

## Anti-Patterns

❌ **Don't:**
- Mix 5+ different edge styles
- Put paragraphs in box labels
- Use bright saturated cluster backgrounds
- Let auto-layout create random spacing
- Ignore rank alignment

✅ **Do:**
- Establish visual hierarchy (main vs supporting)
- Keep labels minimal (name only)
- Use subtle backgrounds (gray95)
- Control layout with rank/constraint
- Test readability at presentation scale
