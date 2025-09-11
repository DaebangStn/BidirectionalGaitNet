"""Configuration management for BidirectionalGaitNet training."""

from .loader import load_config, ConfigLoader
from .schema import ExperimentConfig, TrainingConfig

__all__ = ['load_config', 'ConfigLoader', 'ExperimentConfig', 'TrainingConfig']