"""Configuration schema definitions using dataclasses."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

@dataclass
class ExperimentConfig:
    """Experiment metadata configuration."""
    name: str
    description: str = ""

@dataclass
class DataConfig:
    """Data loading configuration."""
    augmented_data_path: str  # Direct path to augmented motion data directory
    env_file: str = "data/env_merge.xml"
    batch_size: int = 65536
    large_batch_scale: int = 100
    buffer_scale: int = 100
    validation_split: float = 0.1
    fgn_checkpoint: Optional[str] = None

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    architecture: str
    learning_rate: float = 1e-5
    num_epochs: int = 5
    batch_size: int = 128
    hidden_layers: List[int] = field(default_factory=lambda: [512, 512, 512])
    pose_dof: str = "auto"
    frame_num: int = 60

@dataclass
class LossWeights:
    """Loss function weights configuration."""
    # FGN weights
    root_position: float = 2.0
    arm_motion: float = 0.5
    base_mse: float = 5.0
    
    # BGN weights  
    velocity: float = 0.1
    arm: float = 0.01
    toe: float = 0.01
    mse: float = 50.0
    regularization: float = 1.0
    kl_divergence: float = 0.001
    weakness: float = 0.5

@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""
    save_interval: int = 50
    top_k: int = 5
    uniform_interval: int = 500
    validation_interval: int = 100

@dataclass
class TrainerConfig:
    """Trainer configuration."""
    max_iterations: int = 10000
    log_every_n_steps: int = 1
    gradient_clipping: float = 1.0
    device: str = "auto"

@dataclass
class DirectoryConfig:
    """Directory structure configuration."""
    output: str = "distillation"
    experiments: str = "experiments"

@dataclass
class TrainingConfig:
    """Complete training configuration."""
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    loss_weights: LossWeights
    checkpointing: CheckpointConfig
    trainer: TrainerConfig
    directories: DirectoryConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create TrainingConfig from dictionary."""
        return cls(
            experiment=ExperimentConfig(**config_dict.get('experiment', {})),
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            loss_weights=LossWeights(**config_dict.get('loss_weights', {})),
            checkpointing=CheckpointConfig(**config_dict.get('checkpointing', {})),
            trainer=TrainerConfig(**config_dict.get('trainer', {})),
            directories=DirectoryConfig(**config_dict.get('directories', {}))
        )

@dataclass
class PipelineStage:
    """Pipeline stage configuration."""
    name: str
    config: str
    output_path: str
    depends_on: Optional[str] = None
    fgn_checkpoint: Optional[str] = None

@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    experiment: ExperimentConfig
    stages: List[PipelineStage] = field(default_factory=list)
    overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)