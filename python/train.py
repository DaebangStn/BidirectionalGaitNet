#!/usr/bin/env python3
"""
Unified Training Interface for BidirectionalGaitNet

Quick Commands:
  Forward GaitNet:  python python/train.py --config data/config/fgn_default.yaml
  Backward GaitNet: python python/train.py --config data/config/bgn_default.yaml --fgn distillation/fgn/best_checkpoint
  Full Pipeline:    python python/train.py --config data/config/pipeline.yaml
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from python.config import load_config
from python.config.schema import TrainingConfig, PipelineConfig

def main():
    parser = argparse.ArgumentParser(
        description="Unified training interface for BidirectionalGaitNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick Commands:
  Forward GaitNet:
    python python/train.py --config data/config/fgn_default.yaml
    
  Backward GaitNet:  
    python python/train.py --config data/config/bgn_default.yaml --fgn distillation/fgn/best_checkpoint
    
  Full Pipeline:
    python python/train.py --config data/config/pipeline.yaml
    
UV Commands:
  uv run python/train.py --config data/config/fgn_default.yaml
  uv run python/train.py --config data/config/bgn_default.yaml --fgn distillation/fgn/best_checkpoint  
  uv run python/train.py --config data/config/pipeline.yaml
        """
    )
    
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML configuration file")
    parser.add_argument("--fgn", type=str, 
                       help="Path to Forward GaitNet checkpoint (for BGN training)")
    parser.add_argument("--name", type=str,
                       help="Override experiment name")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show configuration and exit without training")
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Handle different configuration types
    if isinstance(config, PipelineConfig):
        print(f"🔄 Running pipeline: {config.experiment.name}")
        if args.dry_run:
            print("Pipeline stages:")
            for stage in config.stages:
                print(f"  - {stage.name}: {stage.config}")
        else:
            run_pipeline(config)
            
    elif isinstance(config, TrainingConfig):
        # Override FGN checkpoint if provided
        if args.fgn:
            config.data.fgn_checkpoint = args.fgn
            
        # Override experiment name if provided
        if args.name:
            config.experiment.name = args.name
            
        print(f"🚀 Starting training: {config.experiment.name}")
        print(f"   Model: {config.model.architecture}")
        print(f"   Data: {config.data.augmented_data_path}")
        
        if args.dry_run:
            print_config_summary(config)
        else:
            run_training(config)
    else:
        raise ValueError(f"Unknown configuration type: {type(config)}")

def print_config_summary(config: TrainingConfig):
    """Print configuration summary for dry run."""
    print(f"""
Configuration Summary:
  Experiment: {config.experiment.name}
  Description: {config.experiment.description}
  
  Model: {config.model.architecture}
  Learning Rate: {config.model.learning_rate}
  Batch Size: {config.model.batch_size} 
  Epochs: {config.model.num_epochs}
  
  Data:
    Data Path: {config.data.augmented_data_path}
    Environment: {config.data.env_file}
    Batch Size: {config.data.batch_size}
    
  Checkpointing:
    Save Interval: {config.checkpointing.save_interval}
    Top-K: {config.checkpointing.top_k}
    Uniform Interval: {config.checkpointing.uniform_interval}
    
  Trainer:
    Max Iterations: {config.trainer.max_iterations}
    Device: {config.trainer.device}
    Gradient Clipping: {config.trainer.gradient_clipping}
""")

def run_training(config: TrainingConfig):
    """Run single model training."""
    if config.model.architecture == "RefNN":
        print("🎯 Training Forward GaitNet...")
        from python.train_forward_gaitnet import main as train_fgn
        # Convert config to legacy format and run
        run_forward_training(config, train_fgn)
        
    elif config.model.architecture == "AdvancedVAE":
        print("🎨 Training Backward GaitNet...")
        if not config.data.fgn_checkpoint:
            raise ValueError("BGN training requires --fgn checkpoint path")
        from python.train_backward_gaitnet import main as train_bgn
        run_backward_training(config, train_bgn)
        
    else:
        raise ValueError(f"Unknown architecture: {config.model.architecture}")

def run_forward_training(config: TrainingConfig, train_fgn):
    """Run Forward GaitNet training with configuration."""
    # Import here to avoid circular imports
    import sys
    import torch
    from pathlib import Path
    from datetime import datetime
    import hashlib
    import json
    
    # Add original training imports
    from forward_gaitnet import RefNN
    from pysim import RayEnvManager
    from ray.rllib.utils.torch_utils import convert_to_torch_tensor
    
    # Set up similar to original train_forward_gaitnet.py but with config
    device = torch.device("cuda" if torch.cuda.is_available() and config.trainer.device == "auto" else config.trainer.device)
    
    # Create experiment ID and directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_str = json.dumps(config.__dict__, sort_keys=True, default=str)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    exp_id = f"{config.experiment.name}_{timestamp}_{config_hash}"
    
    exp_dir = Path(config.directories.output) / config.experiment.name / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✨ Experiment ID: {exp_id}")
    print(f"📁 Output directory: {exp_dir}")
    
    # Call the actual training function with config parameters
    print("🚀 Starting Forward GaitNet training...")
    # Use the imported train_fgn function with config parameters
    train_fgn(
        motion_file=config.data.augmented_data_path,
        env_file=config.data.env_file, 
        name=config.experiment.name,
        max_iterations=config.trainer.max_iterations,
        exp_dir=str(exp_dir)
    )

def run_backward_training(config: TrainingConfig, train_bgn):
    """Run Backward GaitNet training with configuration."""
    # Import here to avoid circular imports
    import sys
    import torch
    from pathlib import Path
    from datetime import datetime
    import hashlib
    import json
    
    # Create experiment ID and directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_str = json.dumps(config.__dict__, sort_keys=True, default=str)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    exp_id = f"{config.experiment.name}_{timestamp}_{config_hash}"
    
    exp_dir = Path(config.directories.output) / config.experiment.name / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✨ Experiment ID: {exp_id}")
    print(f"📁 Output directory: {exp_dir}")
    
    # Call the actual training function with config parameters
    print("🚀 Starting Backward GaitNet training...")
    # Use the imported train_bgn function with config parameters
    train_bgn(
        fgn_checkpoint=config.data.fgn_checkpoint,
        motion_file=config.data.augmented_data_path,
        name=config.experiment.name,
        max_iterations=config.trainer.max_iterations,
        exp_dir=str(exp_dir)
    )

def run_pipeline(config: PipelineConfig):
    """Run full training pipeline."""
    print("🔄 Pipeline execution:")
    for i, stage in enumerate(config.stages, 1):
        print(f"   {i}. {stage.name}")
        
    print("⚠️  Pipeline execution pending - individual stages work")
    print("   Run stages individually:")
    for stage in config.stages:
        print(f"     python python/train.py --config {stage.config}")

if __name__ == "__main__":
    main()