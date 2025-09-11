"""Configuration loading and validation."""

import yaml
from pathlib import Path
from typing import Dict, Any, Union
from .schema import TrainingConfig, PipelineConfig

class ConfigLoader:
    """YAML configuration loader with validation."""
    
    @staticmethod
    def load_yaml(config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML configuration file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        if config_dict is None:
            raise ValueError(f"Empty configuration file: {config_path}")
            
        return config_dict
    
    @staticmethod
    def load_training_config(config_path: Union[str, Path]) -> TrainingConfig:
        """Load training configuration from YAML file."""
        config_dict = ConfigLoader.load_yaml(config_path)
        return TrainingConfig.from_dict(config_dict)
    
    @staticmethod 
    def load_pipeline_config(config_path: Union[str, Path]) -> PipelineConfig:
        """Load pipeline configuration from YAML file."""
        config_dict = ConfigLoader.load_yaml(config_path)
        
        # Parse pipeline stages
        stages = []
        for stage_dict in config_dict.get('pipeline', {}).get('stages', []):
            from .schema import PipelineStage
            stages.append(PipelineStage(**stage_dict))
            
        # Create pipeline config
        from .schema import ExperimentConfig
        return PipelineConfig(
            experiment=ExperimentConfig(**config_dict.get('experiment', {})),
            stages=stages,
            overrides=config_dict.get('overrides', {})
        )
    
    @staticmethod
    def validate_config(config: TrainingConfig) -> None:
        """Validate configuration parameters."""
        # Validate paths exist
        if not Path(config.data.augmented_data_path).exists():
            raise FileNotFoundError(f"Augmented data path not found: {config.data.augmented_data_path}")
            
        if not Path(config.data.env_file).exists():
            raise FileNotFoundError(f"Environment file not found: {config.data.env_file}")
            
        # Validate model parameters
        if config.model.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive: {config.model.learning_rate}")
            
        if config.model.batch_size <= 0:
            raise ValueError(f"Batch size must be positive: {config.model.batch_size}")
            
        # Validate trainer parameters
        if config.trainer.max_iterations <= 0:
            raise ValueError(f"Max iterations must be positive: {config.trainer.max_iterations}")

def load_config(config_path: Union[str, Path]) -> Union[TrainingConfig, PipelineConfig]:
    """Load configuration from YAML file - auto-detect type."""
    config_dict = ConfigLoader.load_yaml(config_path)
    
    # Detect configuration type
    if 'pipeline' in config_dict:
        return ConfigLoader.load_pipeline_config(config_path)
    else:
        return ConfigLoader.load_training_config(config_path)