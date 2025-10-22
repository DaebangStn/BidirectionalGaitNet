import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class SimpleLogger:
    """Simple logger for tracking training metrics with TensorBoard support."""

    def __init__(self, log_dir: Path, experiment_name: str = "regression", use_tensorboard: bool = True):
        """
        Args:
            log_dir: Directory to save log files
            experiment_name: Name of the experiment
            use_tensorboard: Whether to enable TensorBoard logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{experiment_name}_{timestamp}"
        self.log_file = self.log_dir / f"{self.experiment_name}.json"

        # Initialize log storage
        self.metrics_history = []

        # Initialize TensorBoard writer
        self.tb_writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir))
            print(f"[SimpleLogger] TensorBoard logging to {self.log_dir}")
        elif use_tensorboard and not TENSORBOARD_AVAILABLE:
            print(f"[SimpleLogger] Warning: TensorBoard not available, skipping tfevents logging")

        print(f"[SimpleLogger] JSON logging to {self.log_file}")

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log metrics for a given step.

        Args:
            metrics: Dictionary of metric name to value
            step: Current training step/epoch
        """
        entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics_history.append(entry)

        # Write to JSON file immediately (safer for long runs)
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        # Log to TensorBoard
        if self.tb_writer is not None:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)
            self.tb_writer.flush()

    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """
        Log hyperparameters.

        Args:
            hparams: Dictionary of hyperparameter names to values
        """
        hparams_file = self.log_dir / f"{self.experiment_name}_hparams.json"
        with open(hparams_file, 'w') as f:
            json.dump(hparams, f, indent=2)
        print(f"[SimpleLogger] Hyperparameters saved to {hparams_file}")

    def get_latest_metrics(self, metric_name: str, n: int = 1) -> list:
        """
        Get the latest N values of a specific metric.

        Args:
            metric_name: Name of the metric
            n: Number of latest values to retrieve

        Returns:
            List of latest metric values
        """
        values = []
        for entry in reversed(self.metrics_history):
            if metric_name in entry:
                values.append(entry[metric_name])
                if len(values) >= n:
                    break
        return list(reversed(values))

    def print_summary(self):
        """Print summary of logged metrics."""
        if not self.metrics_history:
            print("[SimpleLogger] No metrics logged yet.")
            return

        latest = self.metrics_history[-1]
        print(f"\n[SimpleLogger] Latest metrics (step {latest['step']}):")
        for key, value in latest.items():
            if key not in ['step', 'timestamp']:
                print(f"  {key}: {value:.6f}")

    def close(self):
        """Close logger and TensorBoard writer."""
        if self.tb_writer is not None:
            self.tb_writer.close()
            print(f"[SimpleLogger] TensorBoard writer closed")
