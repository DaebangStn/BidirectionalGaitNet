"""Quick training scripts for UV integration."""

import sys
from pathlib import Path

def train_fgn():
    """Quick Forward GaitNet training."""
    sys.argv = ["train.py", "--config", "data/config/fgn_default.yaml"]
    from python.train import main
    main()

def train_bgn():
    """Quick Backward GaitNet training - requires FGN checkpoint."""
    if len(sys.argv) < 2:
        print("Usage: uv run train-bgn <fgn_checkpoint_path>")
        print("Example: uv run train-bgn distillation/fgn/best_checkpoint")
        sys.exit(1)
        
    fgn_checkpoint = sys.argv[1]
    sys.argv = ["train.py", "--config", "data/config/bgn_default.yaml", "--fgn", fgn_checkpoint]
    from python.train import main
    main()

def train_pipeline():
    """Quick full pipeline training."""
    sys.argv = ["train.py", "--config", "data/config/pipeline.yaml"]  
    from python.train import main
    main()