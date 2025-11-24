# PPO Architecture Diagrams

Clean, presentation-ready diagrams with clear visual hierarchy and professional styling.

## Output Files

### PNG (300 DPI for presentations)
```
plot/diagram/ppo_hierarchical_clean.png     (152KB)
plot/diagram/ppo_rollout_learner_clean.png  (200KB)
```

### Source Files (Graphviz DOT)
```
plot/dot/ppo_hierarchical.dot
plot/dot/ppo_rollout_learner.dot
```

## Diagram Contents

### ppo_hierarchical.dot
**Dual backend architecture with Python-side inference**
- AsyncVectorEnv (Python multiprocessing)
- BatchEnv (C++ ThreadPool)
- Python GPU: Agent + Muscle networks
- Data flow: Rollout → GAE → PPO learning

### ppo_rollout_learner.dot
**Autonomous C++ rollout with libtorch inference**
- BatchRolloutEnv with C++ ThreadPool
- libtorch CPU policy inference (key difference)
- GPU training + CPU inference split
- Weight sync: Python ↔ C++
- Performance: ~2-3x speedup

## Build

```bash
# Render all diagrams
./plot/render.sh

# Or manually
dot -Tpng -Gdpi=300 plot/dot/ppo_hierarchical.dot -o plot/diagram/ppo_hierarchical_clean.png
dot -Tpng -Gdpi=300 plot/dot/ppo_rollout_learner.dot -o plot/diagram/ppo_rollout_learner_clean.png
```

## Editing DOT Files

```dot
// Node definition
nodename [label=<<b>Title</b><br/>Details>, shape=box, fillcolor="#4CAF50"];

// Edge with label
node1 -> node2 [label="description", fontsize=10];

// Cluster (subgraph)
subgraph cluster_name {
    label=<<b>Section Title</b>>;
    style=filled;
    fillcolor="#E8F5E9";
    node1; node2;
}
```

### Color Scheme
- **#4CAF50** (Green): Python GPU components
- **#2196F3** (Blue): C++ CPU components
- **#00BCD4** (Cyan): C++ simulation
- **#03A9F4** (Light Blue): Python multiprocessing
- **#FF9800** (Orange): Weight sync
- **#FFF9C4** (Yellow): Notes

## Design Principles

See `DESIGN_RULES.md` for detailed visual design guidelines.

**Key improvements:**
- **Visual hierarchy**: Main flow (thick black) vs supporting flows (thin gray) vs sync loops (dashed orange)
- **Minimal labels**: Short names in boxes, technical details in small xlabels
- **Orthogonal routing**: Circuit-board style with `splines=ortho` reduces crossings
- **Subtle backgrounds**: Pale colors (gray95, #E3F2FD, #F1F8E9) don't compete with content
- **Rank alignment**: Parallel components aligned horizontally for cleaner perception

**File sizes reduced by 55-60%** through cleaner layout and reduced visual complexity.

## File Organization

```
plot/
├── dot/                    # Source files (edit these)
│   ├── ppo_hierarchical.dot
│   └── ppo_rollout_learner.dot
├── diagram/               # Generated PNG files
│   ├── ppo_hierarchical_clean.png
│   └── ppo_rollout_learner_clean.png
├── render.sh             # Build script
├── DESIGN_RULES.md      # Visual design guidelines
└── README.md            # This file
```
