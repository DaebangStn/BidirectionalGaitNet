import argparse
import pickle
from pysim import EnvManager
import numpy as np
import torch
import os
import math
from python.forward_gaitnet import RefNN

import random
from torch.utils.tensorboard import SummaryWriter
from typing import List, Dict, Tuple
import heapq
from datetime import datetime
import hashlib
import json

import time
import torch.optim as optim
from pathlib import Path

w_root_pos = 2.0
w_arm = 0.5

class HybridCheckpointer:
    """Hybrid checkpointing system with top-K best + uniform interval saving."""
    
    def __init__(self, save_dir: str, top_k: int = 5, uniform_interval: int = 500):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.uniform_interval = uniform_interval
        
        # Track best K checkpoints by validation loss
        self.best_checkpoints = []  # [(loss, iteration, filepath), ...]
        self.uniform_checkpoints = []  # [filepath, ...]
        
    def should_save_checkpoint(self, iteration: int, validation_loss: float) -> Tuple[bool, str]:
        """Determine if checkpoint should be saved and return save type."""
        uniform_save = iteration % self.uniform_interval == 0
        
        # Check if this belongs in top-K
        is_top_k = (len(self.best_checkpoints) < self.top_k or 
                   validation_loss < self.best_checkpoints[0][0])  # heap max is at [0]
        
        save_type = []
        if uniform_save:
            save_type.append("uniform")
        if is_top_k:
            save_type.append("top_k")
            
        return len(save_type) > 0, "_".join(save_type)
    
    def save_checkpoint(self, state: dict, iteration: int, validation_loss: float, 
                       exp_name: str) -> str:
        """Save checkpoint and manage storage."""
        should_save, save_type = self.should_save_checkpoint(iteration, validation_loss)
        
        if not should_save:
            return None
            
        # Generate checkpoint filename
        checkpoint_name = f"{exp_name}_{iteration:06d}_{save_type}_{validation_loss:.6f}.fgn.pt"
        checkpoint_path = self.save_dir / checkpoint_name
        
        # Save checkpoint
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
            
        # Update top-K tracking
        if "top_k" in save_type:
            if len(self.best_checkpoints) >= self.top_k:
                # Remove worst checkpoint if at capacity
                old_loss, old_iter, old_path = heapq.heappop(self.best_checkpoints)
                if os.path.exists(old_path) and "uniform" not in old_path:
                    os.remove(old_path)  # Only remove if not also uniform checkpoint
                    
            heapq.heappush(self.best_checkpoints, (validation_loss, iteration, str(checkpoint_path)))
            
        # Update uniform tracking  
        if "uniform" in save_type:
            self.uniform_checkpoints.append(str(checkpoint_path))
            
        print(f"Checkpoint saved: {checkpoint_path} (type: {save_type}, val_loss: {validation_loss:.6f})")
        return str(checkpoint_path)
    
    def get_best_checkpoint(self) -> str:
        """Get path to best checkpoint."""
        if not self.best_checkpoints:
            return None
        # Best is the one with minimum loss
        return min(self.best_checkpoints, key=lambda x: x[0])[2]

def get_gpu_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0

def get_experiment_id(name: str, config: dict) -> str:
    """Generate unique experiment ID with timestamp and config hash."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    return f"{name}_{timestamp}_{config_hash}"

def log_comprehensive_metrics(writer: SummaryWriter, iteration: int, 
                            train_stats: dict, validation_loss: float = None,
                            learning_rate: float = None):
    """Log comprehensive training metrics to TensorBoard."""
    # Training metrics
    if 'loss_distillation' in train_stats:
        writer.add_scalar('Loss/Train', train_stats['loss_distillation'], iteration)
    
    # Performance metrics (restore timing info)
    if 'converting_time_ms' in train_stats:
        writer.add_scalar('Performance/Converting_Time_Ms', train_stats['converting_time_ms'], iteration)
    if 'learning_time_ms' in train_stats:
        writer.add_scalar('Performance/Learning_Time_Ms', train_stats['learning_time_ms'], iteration)
    
    # Data metrics
    if 'num_tuples' in train_stats:
        writer.add_scalar('Data/Num_Tuples', train_stats['num_tuples'], iteration)
    
    # Validation metrics
    if validation_loss is not None:
        writer.add_scalar('Loss/Validation', validation_loss, iteration)
        if 'loss_distillation' in train_stats:
            train_val_ratio = train_stats['loss_distillation'] / validation_loss
            writer.add_scalar('Metrics/Train_Val_Ratio', train_val_ratio, iteration)
    
    # System metrics
    writer.add_scalar('System/GPU_Memory_MB', get_gpu_memory_usage(), iteration)
    
    if learning_rate is not None:
        writer.add_scalar('System/Learning_Rate', learning_rate, iteration)

class RefLearner:
    def __init__(self, device, num_paramstate, ref_dof, phase_dof=1,
                 buffer_size=30000, learning_rate=1e-4, num_epochs=10, batch_size=128, model=None):
        self.device = device
        self.num_paramstate = num_paramstate
        self.ref_dof = ref_dof
        self.num_epochs = num_epochs
        self.ref_batch_size = batch_size
        self.default_learning_rate = learning_rate
        self.learning_rate = self.default_learning_rate
        print('learning rate : ', self.learning_rate,
              'num epochs : ', self.num_epochs)

        if model:
            self.model = model
        else:
            self.model = RefNN(self.num_paramstate + phase_dof,
                               self.ref_dof, self.device).to(self.device)

        parameters = self.model.parameters()
        self.optimizer = optim.Adam(parameters, lr=self.learning_rate)

        for param in parameters:
            param.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))

        self.stats = {}
        self.model.train()

    def get_weights(self) -> Dict:
        return {
            k: v.cpu().detach().numpy()
            for k, v in self.model.state_dict().items()
        }

    def set_weights(self, weights) -> None:
        weights = {k: v.to(self.device) for k, v in weights.items()}
        self.model.load_state_dict(weights)

    def get_optimizer_weights(self) -> Dict:
        return self.optimizer.state_dict()

    def set_optimizer_weights(self, weights) -> None:
        self.optimizer.load_state_dict(weights)

    def get_model_weights(self, device=None) -> Dict:
        if device:
            return {k: v.to(device) for k, v in self.model.state_dict().items()}
        else:
            return self.model.state_dict()

    def save(self, name):
        path = Path(name)
        torch.save(self.model.state_dict(), path)
        torch.save(self.optimizer.state_dict(),
                   path.with_suffix(".opt" + path.suffix))

    def load(self, name):
        path = Path(name)
        self.model.load_state_dict(torch.load(path))
        self.optimizer.load_state_dict(torch.load(
            path.with_suffix(".opt" + path.suffix)))

    def validation(self, param_all, d_all) -> Dict:
        with torch.no_grad():
            d = self.model(param_all)
            validation_loss = 5.0 * (d_all - d).pow(2).mean()
            return validation_loss.cpu()

    def learn(self, param_all, d_all) -> Dict:

        converting_time = 0.0
        learning_time = 0.0
        start_time = time.perf_counter()

        assert (len(param_all) == len(d_all))
        idx_all = np.asarray(range(len(param_all)))

        converting_time = (time.perf_counter() - start_time) * 1000
        start_time = time.perf_counter()
        loss_avg = 0.

        for _ in range(self.num_epochs):
            np.random.shuffle(idx_all)
            loss_avg = 0
            for i in range(len(param_all) // self.ref_batch_size):
                mini_batch_idx = torch.from_numpy(
                    idx_all[i*self.ref_batch_size: (i+1)*self.ref_batch_size]).cuda()
                param = torch.index_select(param_all, 0, mini_batch_idx)
                d = torch.index_select(d_all, 0, mini_batch_idx)
                d_out = self.model(param)
                diff = d - d_out
                diff[:, 6:9] *= w_root_pos
                diff[:, 57:] *= w_arm

                loss = (5.0 * diff.pow(2)).mean()
                self.optimizer.zero_grad()
                loss.backward()
                
                for param in self.model.parameters():
                    if param.grad != None:
                        param.grad.data.clamp_(-0.1, 0.1)

                self.optimizer.step()
                loss_avg += loss.cpu().detach().numpy().tolist()

        loss_ref = loss_avg / (len(param_all) // self.ref_batch_size)
        learning_time = (time.perf_counter() - start_time) * 1000

        return {
            'num_tuples': len(param_all),
            'loss_distillation': loss_ref,
            'converting_time_ms': converting_time,
            'learning_time_ms': learning_time
        }


parser = argparse.ArgumentParser()

# Raw Motion Path
parser.add_argument("--motion", type=str, default="python/motion.txt")
parser.add_argument("--env", type=str, default="/home/gait/BidirectionalGaitNet_Data/GridSampling/3rd_rollout/env.xml")
parser.add_argument("--name", type=str, default="distillation")
parser.add_argument("--validation", action='store_true')

# Enhanced training arguments
parser.add_argument("--top_k_checkpoints", type=int, default=5, help="Number of best checkpoints to keep")
parser.add_argument("--uniform_checkpoint_interval", type=int, default=500, help="Interval for uniform checkpoints")
parser.add_argument("--validation_interval", type=int, default=100, help="Validation frequency")
parser.add_argument("--max_iterations", type=int, default=10000, help="Maximum number of training iterations")

def main(motion_file=None, env_file=None, name=None, max_iterations=None, exp_dir=None):
    """Main training function that can be called from unified training interface."""
    import sys
    from pathlib import Path
    
    # Add project root to Python path for imports
    sys.path.append(str(Path(__file__).parent.parent))
    
    # Set up arguments for the training
    sys.argv = ['train_forward_gaitnet.py']
    if motion_file:
        sys.argv.extend(['--motion', motion_file])
    if env_file:
        sys.argv.extend(['--env', env_file])  
    if name:
        sys.argv.extend(['--name', name])
    if max_iterations:
        sys.argv.extend(['--max_iterations', str(max_iterations)])
        
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args = parser.parse_args()
    
    # Create experiment configuration for unique ID generation
    experiment_config = {
        'motion': args.motion,
        'env': args.env,
        'learning_rate': 1e-5,
        'batch_size': 128,
        'num_epochs': 5,
        'top_k_checkpoints': args.top_k_checkpoints,
        'uniform_checkpoint_interval': args.uniform_checkpoint_interval,
        'validation_interval': args.validation_interval
    }
    
    # Use provided exp_dir or create default
    if exp_dir:
        exp_dir = Path(exp_dir)
        exp_id = exp_dir.name
        print(f"Using unified interface directory: {exp_dir}")
    else:
        # Generate unique experiment ID
        exp_id = get_experiment_id(args.name, experiment_config)
        print(f"Starting experiment: {exp_id}")
        
        # Create experiment directory structure
        exp_dir = Path("experiments") / args.name / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment configuration
    with open(exp_dir / "config.json", 'w') as f:
        json.dump(experiment_config, f, indent=2)
    
    f = open(args.motion, 'r')
    motions = f.readlines()
    train_filenames = []
    for motion in motions:
        motion = motion.replace('\n', '')
        train_filenames = train_filenames + [motion + '/' + fn for fn in os.listdir(motion)]
    
    train_filenames.sort()

    file_idx = 0

    # Environment Loading
    env = EnvManager(args.env)

    # Loading all motion from file Data PreProcessing
    buffers = [[], []]  # {Param, Phi}, {Pose}
    raw_buffers = [[], []]
    training_sets = [None, None]

    batch_size = 65536
    large_batch_scale = 100
    
    # Adaptive buffer sizing for small datasets
    estimated_total_samples = len(train_filenames) * 1000  # Rough estimate
    target_buffer_size = batch_size * large_batch_scale
    if estimated_total_samples < target_buffer_size:
        # Use smaller scale for small datasets
        large_batch_scale = max(1, estimated_total_samples // batch_size)
        target_buffer_size = batch_size * large_batch_scale
        print(f"📊 Adaptive buffer: Using scale {large_batch_scale} for small dataset ({estimated_total_samples} est. samples)")
    pos_dof = 0
    ref_learner = \
        RefLearner(torch.device("cuda"), len(env.getNormalizedParamState()), len(env.posToSixDof(env.getPositions())), 2, learning_rate=1e-5, batch_size=128, num_epochs=5)
    iter = 0

    # Initialize enhanced training infrastructure
    writer = SummaryWriter(exp_dir / "tensorboard")
    checkpointer = HybridCheckpointer(
        save_dir=exp_dir / "checkpoints",
        top_k=args.top_k_checkpoints,
        uniform_interval=args.uniform_checkpoint_interval
    )
    
    # Create validation data split (use 10% of data for validation)
    validation_filenames = train_filenames[:len(train_filenames)//10]
    train_filenames = train_filenames[len(train_filenames)//10:]

    tuple_size = 0
    print(len(train_filenames), ' files are loaded ....... ')

    phi = np.array([[math.sin(i * (1.0/30) * 2 * math.pi), math.cos(i * (1.0/30) * 2 * math.pi)] for i in range(60)])
    num_knownparam = env.getNumKnownParam()
    print(f"Starting training loop: iter={iter}, max_iterations={args.max_iterations}")
    while iter < args.max_iterations:
        random.shuffle(train_filenames)     
        num_tuple = 0
        while True:
            if raw_buffers[0] != None and (len(raw_buffers[0]) > batch_size * large_batch_scale):
                break
            else:
                f = train_filenames[file_idx % len(train_filenames)]
                file_idx += 1
                if file_idx > len(train_filenames):
                    print('All files are used')
                    file_idx %= len(train_filenames)
                if f[-4:] != ".npz":
                    continue

                path = f 
                print(path)
                loaded_file = np.load(path, allow_pickle=True)
                loaded_motions = loaded_file["motions"]
                loaded_params = loaded_file["params"]
                loaded_idx = 0
                
                i = 0
                              
                for loaded_idx in range(len(loaded_motions)):
                    param_matrix = np.repeat(env.getNormalizedParamStateFromParam(loaded_params[loaded_idx]), 60).reshape(-1, 60).transpose()
                    data_in = np.concatenate((param_matrix, phi), axis=1)
                    data_out = loaded_motions[loaded_idx][:,:]
                    raw_buffers[0] += list(data_in)  
                    raw_buffers[1] += list(data_out)  


        buffers[0] = torch.tensor(
            np.array(raw_buffers[0][:batch_size * large_batch_scale], dtype=np.float32))
        buffers[1] = torch.tensor(
            np.array(raw_buffers[1][:batch_size * large_batch_scale], dtype=np.float32))

        raw_buffers[0] = raw_buffers[0][batch_size * large_batch_scale:]
        raw_buffers[1] = raw_buffers[1][batch_size * large_batch_scale:]

        if True:
            idx_all = np.asarray(range(len(buffers[0])))
            np.random.shuffle(idx_all)

            for i in range(len(idx_all) // batch_size):
                mini_batch_idx = torch.from_numpy(
                    idx_all[i*batch_size: (i+1)*batch_size])
                training_sets[0] = torch.index_select(
                    buffers[0], 0, mini_batch_idx).cuda()
                training_sets[1] = torch.index_select(
                    buffers[1], 0, mini_batch_idx).cuda()
                tuple_size += len(training_sets[0])

                v_stat = ref_learner.learn(training_sets[0], training_sets[1])

                # Enhanced logging and checkpointing
                print('Iteration : ', iter, '\tRaw Buffer : ', len(raw_buffers[0]), '\tBuffer : ', len(buffers[0]), v_stat, '\tTuple Size : ', tuple_size)

                # Validation and comprehensive logging
                validation_loss = None
                if iter % args.validation_interval == 0 and len(validation_filenames) > 0:
                    # Compute validation loss on a sample
                    val_loss_sum = 0
                    val_samples = 0
                    for val_file in validation_filenames[:5]:  # Use first 5 validation files
                        try:
                            val_data = np.load(val_file, allow_pickle=True)
                            val_motions = val_data["motions"]
                            val_params = val_data["params"]
                            
                            for val_idx in range(min(10, len(val_motions))):  # Sample 10 sequences
                                param_matrix = np.repeat(env.getNormalizedParamStateFromParam(val_params[val_idx]), 60).reshape(-1, 60).transpose()
                                val_data_in = np.concatenate((param_matrix, phi), axis=1)
                                val_data_out = val_motions[val_idx][:,:]
                                
                                val_input = torch.tensor(val_data_in, dtype=torch.float32).cuda()
                                val_target = torch.tensor(val_data_out, dtype=torch.float32).cuda()
                                
                                val_loss = ref_learner.validation(val_input, val_target)
                                val_loss_sum += val_loss.item()
                                val_samples += 1
                        except:
                            continue
                    
                    if val_samples > 0:
                        validation_loss = val_loss_sum / val_samples
                        print(f'Validation Loss: {validation_loss:.6f}')

                # Log comprehensive metrics to TensorBoard
                current_lr = ref_learner.optimizer.param_groups[0]['lr']
                log_comprehensive_metrics(writer, iter, v_stat, validation_loss, current_lr)

                # Enhanced checkpointing with hybrid system
                should_save_checkpoint = validation_loss is not None or (iter + 1) >= args.max_iterations
                
                if should_save_checkpoint:
                    # Use training loss if no validation loss available
                    checkpoint_loss = validation_loss if validation_loss is not None else v_stat.get('loss_distillation', 0.0)
                    
                    checkpoint_state = {
                        "metadata": env.getMetadata(),
                        "is_cascaded": True,
                        "ref": ref_learner.get_weights(),
                        "optimizer": ref_learner.get_optimizer_weights(),
                        "iteration": iter,
                        "validation_loss": checkpoint_loss,
                        "config": experiment_config,
                        "augmented_data_path": args.motion  # Save motion path for BGN validation
                    }
                    
                    saved_path = checkpointer.save_checkpoint(
                        checkpoint_state, iter, checkpoint_loss, args.name
                    )
                    print(f"💾 Checkpoint saved: {saved_path}")

                iter += 1
                print(f"Completed iteration {iter}/{args.max_iterations}")
                
                # Check if we've reached max iterations and break
                if iter >= args.max_iterations:
                    print(f"✅ Training completed: reached max_iterations {args.max_iterations}")
                    break

if __name__ == "__main__":
    main()
