import time
from pathlib import Path
from typing import List, Dict, Tuple
import torch.optim as optim
from symbol import parameters
from forward_gaitnet import RefNN
import argparse

import torch.nn.utils as torch_utils
from pysim import RayEnvManager
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
import numpy as np
import torch
import torch.nn as nn
import math
import random
from advanced_vae import AdvancedVAE
from torch.utils.tensorboard import SummaryWriter
import os
import pickle
import heapq
from datetime import datetime
import hashlib
import json

# Network Loading Function, f : (path) -> pi

w_v = 0.1
w_arm = 0.01
w_toe = 0.01
w_mse = 50.0
w_regul = 1
w_kl = 1E-3
w_weakness = 0.5

class HybridCheckpointer:
    """Hybrid checkpointing system with top-K best + uniform interval saving for BGN."""
    
    def __init__(self, save_dir: str, top_k: int = 5, uniform_interval: int = 500):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.uniform_interval = uniform_interval
        
        # Track best K checkpoints by total loss (lower is better)
        self.best_checkpoints = []  # [(loss, iteration, filepath), ...]
        self.uniform_checkpoints = []  # [filepath, ...]
        
    def should_save_checkpoint(self, iteration: int, total_loss: float) -> Tuple[bool, str]:
        """Determine if checkpoint should be saved and return save type."""
        uniform_save = iteration % self.uniform_interval == 0
        
        # Check if this belongs in top-K (lower loss is better)
        is_top_k = (len(self.best_checkpoints) < self.top_k or 
                   total_loss < self.best_checkpoints[0][0])  # heap max is at [0]
        
        save_type = []
        if uniform_save:
            save_type.append("uniform")
        if is_top_k:
            save_type.append("top_k")
            
        return len(save_type) > 0, "_".join(save_type)
    
    def save_checkpoint(self, state: dict, iteration: int, total_loss: float, 
                       exp_name: str) -> str:
        """Save checkpoint and manage storage."""
        should_save, save_type = self.should_save_checkpoint(iteration, total_loss)
        
        if not should_save:
            return None
            
        # Generate checkpoint filename
        checkpoint_name = f"{exp_name}_{iteration:06d}_{save_type}_{total_loss:.6f}.bgn.pt"
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
                    
            heapq.heappush(self.best_checkpoints, (total_loss, iteration, str(checkpoint_path)))
            
        # Update uniform tracking  
        if "uniform" in save_type:
            self.uniform_checkpoints.append(str(checkpoint_path))
            
        print(f"Checkpoint saved: {checkpoint_path} (type: {save_type}, total_loss: {total_loss:.6f})")
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

def log_comprehensive_vae_metrics(writer: SummaryWriter, iteration: int, 
                                train_stats: dict, learning_rate: float = None):
    """Log comprehensive VAE training metrics to TensorBoard."""
    # VAE-specific training metrics
    if 'loss_total' in train_stats:
        writer.add_scalar('Loss/Total', train_stats['loss_total'], iteration)
    if 'loss_recon' in train_stats:
        writer.add_scalar('Loss/Reconstruction', train_stats['loss_recon'], iteration)
    if 'loss_kld' in train_stats:
        writer.add_scalar('Loss/KL_Divergence', train_stats['loss_kld'], iteration)
    if 'regul_loss' in train_stats:
        writer.add_scalar('Loss/Regularization', train_stats['regul_loss'], iteration)
    
    # Data metrics
    if 'num_tuples' in train_stats:
        writer.add_scalar('Data/Num_Tuples', train_stats['num_tuples'], iteration)
    
    # System metrics
    writer.add_scalar('System/GPU_Memory_MB', get_gpu_memory_usage(), iteration)
    
    if learning_rate is not None:
        writer.add_scalar('System/Learning_Rate', learning_rate, iteration)


def loading_distilled_network(path, device):
    print('loading distilled network from', path)
    state = pickle.load(open(path, "rb"))
    env = RayEnvManager(state['metadata'])
    ref = RefNN(len(env.getNormalizedParamState()) + 2,
                len(env.posToSixDof(env.getPositions())), device)
    ref.load_state_dict(convert_to_torch_tensor(state['ref']))
    return ref, env

def loading_training_motion_fast(f):

    if True:
        print("loading ", f)
        loaded_file = np.load(f)

        loaded_param = loaded_file["params"]
        loaded_motion = loaded_file["motions"]

    return loaded_motion[:, :6073]


class VAELearner:
    def __init__(self, device,
                 pose_dof,
                 frame_num,
                 num_paramstate,
                 num_known_param,
                 Forward_GaitNet,
                 buffer_size=30000,
                 learning_rate=5e-5,
                 num_epochs=3,
                 batch_size=128,
                 encoder_hidden_dims=None,
                 decoder_hidden_dims=None,
                 model=None):
        self.device = device
        self.pose_dof = pose_dof
        self.frame_num = frame_num
        self.num_paramstate = num_paramstate

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.default_learning_rate = learning_rate
        self.learning_rate = self.default_learning_rate

        self.num_known_param = num_known_param
        if model:
            self.model = model
        else:
            self.model = AdvancedVAE(
                self.pose_dof, self.frame_num, self.num_known_param, self.num_paramstate, Forward_GaitNet)

        parameters = self.model.parameters()

        self.optimizer = optim.Adam(parameters, lr=self.learning_rate)

        self.stats = {}
        self.model.train()

        self.diff_w = None

        # Weight initialization
        init_w = np.ones(self.pose_dof)

        # For Root Velocity
        init_w[6] *= w_v
        init_w[8] *= w_v

        init_w[63:] *= w_arm

        init_w[22:24] *= w_toe
        init_w[35:37] *= w_toe

        self.init_w = torch.tensor(
            np.repeat(init_w, 60), dtype=torch.float32, device=self.device)

    def get_weights(self) -> Dict:
        return {
            k: v.cpu().detach().numpy()
            for k, v in self.model.state_dict().items()
        }

    def set_weights(self, weights) -> None:
        weights = convert_to_torch_tensor(weights, device=self.device)
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

    def learn(self, motion_all) -> Dict:

        converting_time = 0.0
        learning_time = 0.0
        start_time = time.perf_counter()

        idx_all = np.asarray(range(len(motion_all)))

        converting_time = (time.perf_counter() - start_time) * 1000
        start_time = time.perf_counter()
        loss_total = 0
        loss_recon = 0
        loss_kld = 0
        loss_regul = 0
        for _ in range(self.num_epochs):
            np.random.shuffle(idx_all)
            loss_total = 0
            loss_recon = 0
            loss_kld = 0
            loss_regul = 0
            for i in range(len(motion_all) // self.batch_size):
                mini_batch_idx = torch.from_numpy(
                    idx_all[i*self.batch_size: (i+1)*self.batch_size]).cuda()
                motion = torch.index_select(motion_all, 0, mini_batch_idx)
                input, recon, mu, log_var, conditions = self.model(motion)
                motion_diff = (input - recon)

                mse_loss = (motion_diff * self.init_w).pow(2).mean()

                kld_loss = torch.mean(-0.5 * torch.sum(1 +
                                      log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

                # condition diff
                condition_diff = torch.ones(
                    conditions.shape, device="cuda") - (conditions).cuda()
                weakness_num = (self.num_paramstate -
                                self.num_known_param) // 2
                condition_diff[:, weakness_num:] *= w_weakness
                regul_loss = condition_diff.pow(2).mean()

                # Multiply the weight

                loss = w_mse * mse_loss + w_regul * regul_loss + w_kl * kld_loss
                self.optimizer.zero_grad()
                loss.backward()
                
                # for param in self.model.parameters():
                #     if param.grad != None:
                #         param.grad.data.clamp_(-0.05, 0.05)
                torch_utils.clip_grad_norm_(self.model.parameters(), 0.5)

                self.optimizer.step()
                loss_total += loss.cpu().detach().numpy().tolist()
                loss_recon += mse_loss.cpu().detach().numpy().tolist()
                loss_kld += kld_loss.cpu().detach().numpy().tolist()
                loss_regul += regul_loss.cpu().detach().numpy().tolist()

        loss_total = loss_total / (len(motion_all) // self.batch_size)
        loss_recon = loss_recon / (len(motion_all) // self.batch_size)
        loss_kld = loss_kld / (len(motion_all) // self.batch_size)
        loss_regul = loss_regul / (len(motion_all) // self.batch_size)

        learning_time = (time.perf_counter() - start_time) * 1000

        time_stat = {'converting_time_ms': converting_time,
                     'learning_time_ms': learning_time}
        return {
            'num_tuples': len(motion_all),
            'loss_total': loss_total,
            'loss_recon': loss_recon,
            'loss_kld': loss_kld,
            'regul_loss': loss_regul,
        }


parser = argparse.ArgumentParser()
parser.add_argument("--fgn", type=str, required=True, help="Path to forward GaitNet checkpoint")
parser.add_argument("--motion", type=str, default="python/motion.txt", help="Motion file list")
parser.add_argument("--name", type=str, default="gvae_training", help="Experiment name")

# Enhanced BGN training arguments
parser.add_argument("--top_k_checkpoints", type=int, default=5, help="Number of best checkpoints to keep")
parser.add_argument("--uniform_checkpoint_interval", type=int, default=500, help="Interval for uniform checkpoints")
parser.add_argument("--checkpoint_interval", type=int, default=50, help="Base checkpointing frequency")

# Main
def main(fgn_checkpoint=None, motion_file=None, name=None, max_iterations=None, exp_dir=None):
    """Main training function that can be called from unified training interface."""
    import sys
    from pathlib import Path
    
    # Add project root to Python path for imports
    sys.path.append(str(Path(__file__).parent.parent))
    
    # Set up arguments for the training
    sys.argv = ['train_backward_gaitnet.py']
    if fgn_checkpoint:
        sys.argv.extend(['--fgn', fgn_checkpoint])
    if motion_file:
        sys.argv.extend(['--motion', motion_file])
    if name:
        sys.argv.extend(['--name', name])
    # Note: BGN doesn't use max_iterations the same way, it uses epochs
        
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args = parser.parse_args()
    
    # Create experiment configuration for unique ID generation
    experiment_config = {
        'fgn_checkpoint': args.fgn,
        'motion_file': args.motion,
        'learning_rate': 1E-5,
        'batch_size_vae': 32,
        'batch_size_data': 4096,
        'num_epochs': 10,
        'top_k_checkpoints': args.top_k_checkpoints,
        'uniform_checkpoint_interval': args.uniform_checkpoint_interval,
        'checkpoint_interval': args.checkpoint_interval,
        'weights': {
            'w_v': w_v, 'w_arm': w_arm, 'w_toe': w_toe,
            'w_mse': w_mse, 'w_regul': w_regul, 'w_kl': w_kl, 'w_weakness': w_weakness
        }
    }
    
    # Generate unique experiment ID and setup distillation directory
    exp_id = get_experiment_id(args.name, experiment_config)
    print(f"Starting BGN experiment: {exp_id}")
    
    # Use provided exp_dir or create default
    if exp_dir:
        exp_dir = Path(exp_dir)
        exp_id = exp_dir.name
        print(f"Using unified interface directory: {exp_dir}")
    else:
        # Use distillation directory structure as requested
        exp_dir = Path("distillation") / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment configuration
    with open(exp_dir / "config.json", 'w') as f:
        json.dump(experiment_config, f, indent=2)

    fgn, env = loading_distilled_network(args.fgn, device)
    
    # Validate motion path compatibility between FGN checkpoint and BGN config
    try:
        fgn_state = pickle.load(open(args.fgn, "rb"))
        fgn_motion_path = fgn_state.get("augmented_data_path", "unknown")
        bgn_motion_path = args.motion
        
        if fgn_motion_path != bgn_motion_path:
            print(f"⚠️  Motion Path Mismatch Warning:")
            print(f"   FGN trained on: {fgn_motion_path}")
            print(f"   BGN using:      {bgn_motion_path}")
            print(f"   This may affect training quality and consistency")
        else:
            print(f"✅ Motion path validated: {fgn_motion_path}")
    except Exception as e:
        print(f"⚠️  Could not validate motion paths: {e}")

    # (1) Declare VAE Learner
    vae_learner = VAELearner(device, len(env.posToSixDof(env.getPositions())), 60, len(
        env.getNormalizedParamState()), env.getNumKnownParam(), fgn.cpu().state_dict(), learning_rate=1E-5, batch_size=32, num_epochs=10)

    # (2) Training Dataset Loading
    f = open(args.motion, 'r')
    motions = f.readlines()
    train_filenames = []
    for motion in motions:
        motion = motion.replace('\n', '')
        train_filenames = train_filenames + \
            [motion + '/' + fn for fn in os.listdir(motion)]
    
    print(f"🔧 Debug: Processing {len(train_filenames)} training files")

    # (3) Enhanced Training Infrastructure
    writer = SummaryWriter(exp_dir / "tensorboard")
    checkpointer = HybridCheckpointer(
        save_dir=exp_dir / "checkpoints",
        top_k=args.top_k_checkpoints,
        uniform_interval=args.uniform_checkpoint_interval
    )
    batch_size = 4096
    buffer_scale = 100
    
    # Adaptive buffer sizing for small datasets
    estimated_total_samples = len(train_filenames) * 1000  # Rough estimate
    target_buffer_size = batch_size * buffer_scale
    if estimated_total_samples < target_buffer_size:
        # Use smaller scale for small datasets
        buffer_scale = max(1, estimated_total_samples // batch_size)
        target_buffer_size = batch_size * buffer_scale
        print(f"📊 BGN Adaptive buffer: Using scale {buffer_scale} for small dataset ({estimated_total_samples} est. samples)")
    
    training_iter = 0
    used_episode = 0
    num_known_param = env.getNumKnownParam()
    buffers = []
    file_idx = 0
    random.shuffle(train_filenames)
    while True:

        # Collect Training Data from the files until the batch size is full
        if len(buffers) < batch_size:
            while True:

                if file_idx >= len(train_filenames):
                    file_idx %= len(train_filenames)
                    print('All file used', file_idx)
                    random.shuffle(train_filenames)

                if (len(train_filenames[file_idx]) < 5 or train_filenames[file_idx][-4:] != '.npz'):
                    file_idx += 1
                    continue
                f = np.load(train_filenames[file_idx], 'r')
                loaded_motions = f["motions"]
                loaded_params = f["params"]

                # Converting
                for i in range(len(loaded_params)):
                    buffers.append(np.concatenate((loaded_motions[i][:,:].flatten(
                    ), env.getNormalizedParamStateFromParam(loaded_params[i])[:num_known_param])))

                file_idx += 1

                if len(buffers) > target_buffer_size:
                    random.shuffle(buffers)
                    break

        # Training
        training_data = torch.tensor(
            np.array(buffers[:batch_size]), device="cuda")
        buffers = buffers[batch_size:]
        used_episode += len(training_data)
        stat = vae_learner.learn(training_data)
        print("Buffer Size : ", len(buffers), "\tIteration : ",
              training_iter, "\tUsed Episode : ", used_episode, '\t', stat)

        # Enhanced logging with comprehensive VAE metrics
        current_lr = vae_learner.optimizer.param_groups[0]['lr']
        log_comprehensive_vae_metrics(writer, training_iter, stat, current_lr)

        # Enhanced checkpointing with hybrid system
        if training_iter % args.checkpoint_interval == 0:
            total_loss = stat.get('loss_total', float('inf'))
            
            checkpoint_state = {
                "metadata": env.getMetadata(),
                "gvae": vae_learner.get_weights(),
                "optimizer": vae_learner.get_optimizer_weights(),
                "iteration": training_iter,
                "total_loss": total_loss,
                "config": experiment_config,
                "training_stats": {
                    "used_episode": used_episode,
                    "buffer_size": len(buffers)
                }
            }
            
            saved_path = checkpointer.save_checkpoint(
                checkpoint_state, training_iter, total_loss, args.name
            )
            
            if saved_path:
                print(f"Best checkpoint path: {checkpointer.get_best_checkpoint()}")

        training_iter += 1

if __name__ == "__main__":
    main()
