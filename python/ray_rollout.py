import ray
import torch
import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Tuple
import argparse
from pyrollout import RolloutEnvironment, RolloutRecord, RecordConfig

@ray.remote(num_gpus=1)
class PolicyWorker:
    """Single GPU worker for batched policy inference"""
    
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda")
        
        # Load TorchScript policy
        policy_path = Path(checkpoint_path) / "sim_nn.pt"
        self.policy = torch.jit.load(str(policy_path))
        self.policy.to(self.device)
        self.policy.eval()
    
    def compute_actions(self, states: List[np.ndarray]) -> List[np.ndarray]:
        """Batch inference for all environment workers"""
        with torch.no_grad():
            # Stack states and move to GPU
            states_tensor = torch.from_numpy(np.stack(states)).float().to(self.device)
            
            # Batched forward pass
            actions_tensor = self.policy(states_tensor)
            
            # Move back to CPU and split
            actions = actions_tensor.cpu().numpy()
            return [actions[i] for i in range(len(states))]

@ray.remote(num_cpus=1)
class EnvWorker:
    """CPU worker for environment simulation and data collection"""
    
    def __init__(self, env_idx: int, metadata_path: str, record_config_path: str):
        self.env_idx = env_idx
        
        # Create rollout environment
        self.rollout_env = RolloutEnvironment(metadata_path)
        self.rollout_env.load_config(record_config_path)
        
        # Get fields and create record buffer
        self.fields = self.rollout_env.get_record_fields()
        self.record = RolloutRecord(self.fields)
        
        self.target_cycles = 5  # Default, can be loaded from config
    
    def reset(self) -> np.ndarray:
        """Reset environment and record buffer"""
        self.rollout_env.reset()
        self.record.reset()
        return self.rollout_env.get_state()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, int, bool]:
        """Step environment and record data"""
        self.rollout_env.set_action(action)
        self.rollout_env.step(self.record)
        
        state = self.rollout_env.get_state()
        cycle_count = self.rollout_env.get_cycle_count()
        is_done = (self.rollout_env.is_eoe() != 0) or (cycle_count >= self.target_cycles)
        
        return state, cycle_count, is_done
    
    def get_data(self) -> Tuple[np.ndarray, List[str], int]:
        """Get recorded data"""
        return self.record.data, self.fields, self.env_idx

def save_to_parquet(rollout_data: List[Tuple[np.ndarray, List[str], int]], 
                    output_path: str):
    """Save collected rollout data to parquet"""
    
    dfs = []
    for data, fields, worker_id in rollout_data:
        # Create dataframe from numpy array
        df_dict = {}
        for idx, field in enumerate(fields):
            # Use appropriate dtype
            if field in ['step', 'cycle', 'contact_left', 'contact_right']:
                df_dict[field] = data[:, idx].astype(np.int32)
            else:
                df_dict[field] = data[:, idx].astype(np.float32)
        
        df = pl.DataFrame(df_dict)
        df = df.with_columns(pl.lit(worker_id).cast(pl.Int32).alias('worker_id'))
        dfs.append(df)
    
    # Concatenate and save
    combined = pl.concat(dfs)
    combined.write_parquet(output_path, compression='zstd')
    print(f"Saved {len(combined)} rows to {output_path}")

def run_rollout(metadata_path: str, 
                checkpoint_path: str,
                record_config_path: str,
                output_path: str,
                num_workers: int = None,
                target_cycles: int = 5):
    """Run distributed rollout with Ray"""
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Determine number of workers
    if num_workers is None:
        num_workers = int(ray.available_resources().get('CPU', 1))
    
    print(f"Starting rollout with {num_workers} environment workers")
    
    # Create workers
    policy = PolicyWorker.remote(checkpoint_path)
    env_workers = [
        EnvWorker.remote(i, metadata_path, record_config_path)
        for i in range(num_workers)
    ]
    
    # Reset all environments
    states = ray.get([w.reset.remote() for w in env_workers])
    dones = [False] * num_workers
    
    step_count = 0
    while not all(dones):
        # Centralized action computation (single GPU)
        active_states = [s for s, d in zip(states, dones) if not d]
        active_indices = [i for i, d in enumerate(dones) if not d]
        
        if not active_states:
            break
        
        actions = ray.get(policy.compute_actions.remote(active_states))
        
        # Distributed stepping (multiple CPUs)
        step_futures = []
        for idx, action in zip(active_indices, actions):
            step_futures.append(env_workers[idx].step.remote(action))
        
        results = ray.get(step_futures)
        
        # Update states and check for completion
        for local_idx, global_idx in enumerate(active_indices):
            state, cycle, done = results[local_idx]
            states[global_idx] = state
            dones[global_idx] = done
        
        step_count += 1
        if step_count % 100 == 0:
            active_count = sum(1 for d in dones if not d)
            print(f"Step {step_count}, {active_count} environments active")
    
    print(f"Rollout completed in {step_count} steps")
    
    # Collect data from all workers
    data_futures = [w.get_data.remote() for w in env_workers]
    all_data = ray.get(data_futures)
    
    # Save to parquet
    save_to_parquet(all_data, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ray-based rollout for BidirectionalGaitNet")
    parser.add_argument("--metadata", required=True, help="Path to environment metadata XML")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--config", required=True, help="Path to record config YAML")
    parser.add_argument("--output", required=True, help="Output parquet file path")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--cycles", type=int, default=5, help="Target gait cycles")
    
    args = parser.parse_args()
    
    run_rollout(
        metadata_path=args.metadata,
        checkpoint_path=args.checkpoint,
        record_config_path=args.config,
        output_path=args.output,
        num_workers=args.workers,
        target_cycles=args.cycles
    )

