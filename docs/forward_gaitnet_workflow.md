# Forward GaitNet Training Workflow

## Overview

Forward GaitNet (`train_forward_gaitnet.py`) implements a neural network distillation process that learns to predict human motion from parametric state inputs. This system is designed to create a lightweight neural network that can generate realistic human locomotion patterns based on gait parameters.

## System Architecture

### Core Components

1. **RefNN Network**: Multi-layer perceptron for motion prediction
2. **RefLearner**: Training framework with batch processing and optimization
3. **Data Pipeline**: Continuous streaming data loader with preprocessing
4. **Environment Interface**: RayEnvManager for physics simulation integration

## Network Structure

### RefNN Architecture
```
Input Layer:    num_paramstate + phase_dof (normalized_params + phi)
Hidden Layer 1: 512 neurons + ReLU activation
Hidden Layer 2: 512 neurons + ReLU activation  
Hidden Layer 3: 512 neurons + ReLU activation
Output Layer:   ref_dof (pure motion vectors - 101 dimensions)
```

**Key Features:**
- Xavier uniform weight initialization
- Gradient clipping (-1.0 to 1.0 for stability)
- CUDA/CPU adaptive device handling
- State dictionary serialization for checkpointing

### Input/Output Specifications

**Input Dimensions:**
- `num_paramstate`: Length of normalized parameter state from environment
- `phase_dof`: 2 (sin/cos phase encoding for temporal consistency)
- **Total Input**: `num_paramstate + 2`

**Output Dimensions:**
- `ref_dof`: Pure motion vector from `env.posToSixDof(env.getPositions())` 
- **Output Size**: 101 dimensions (6DOF pose + joint positions)

## Data Preprocessing Pipeline

### Phase 1: File Discovery and Loading
```python
# Motion directory scanning
train_filenames = [motion_dir + '/' + fn for fn in os.listdir(motion_dir)]
train_filenames.sort()  # Deterministic ordering
```

### Phase 2: Temporal Phase Encoding
```python
# Create 60-frame cyclic phase representation (30 Hz for 2 seconds)
phi = np.array([[math.sin(i * (1.0/30) * 2 * math.pi), 
                 math.cos(i * (1.0/30) * 2 * math.pi)] for i in range(60)])
```

**Purpose**: Provides temporal context for motion prediction across gait cycles

### Phase 3: Parameter State Processing
```python
# Convert parameters to normalized state and repeat for temporal sequence
param_matrix = np.repeat(env.getNormalizedParamStateFromParam(loaded_params[idx]), 60)
param_matrix = param_matrix.reshape(-1, 60).transpose()

# Combine parameters with phase encoding
data_in = np.concatenate((param_matrix, phi), axis=1)
```

**Data Flow:**
1. Raw parameters → Environment normalization
2. Expand to 60 timesteps (2-second sequences)
3. Concatenate with phase encoding
4. **Input Shape**: `(60, num_paramstate + 2)`

### Phase 4: Motion Target Processing
```python
data_out = loaded_motions[loaded_idx][:, :]  # Pure motion data (101 dims)
```

**Target Format:**
- **Shape**: `(60, 101)` - 60 timesteps of 101-dimensional motion vectors
- **Content**: Pure motion data from augmented preprocessing (no parameter concatenation)

## Learning Framework

### RefLearner Training System

#### Hyperparameters
```python
learning_rate = 1e-5       # Conservative learning rate
batch_size = 128          # Mini-batch size for gradient updates
num_epochs = 5            # Epochs per training iteration
buffer_size = 30000       # Experience replay buffer capacity
```

#### Loss Function with Weighted Components
```python
# Multi-component loss with domain-specific weighting
diff = target_motion - predicted_motion

# Apply domain-specific weights
diff[:, 6:9] *= w_root_pos    # Root position weight = 2.0
diff[:, 57:] *= w_arm         # Arm motion weight = 0.5

loss = (5.0 * diff.pow(2)).mean()  # Scaled MSE loss
```

**Weighting Strategy:**
- **Root Position** (dims 6-9): 2.0x weight - Critical for locomotion stability
- **Arm Motion** (dims 57+): 0.5x weight - Less critical for gait quality
- **Base MSE**: 5.0x scaling factor for numerical stability

#### Optimization Strategy
```python
# Adam optimizer with gradient clipping
optimizer = optim.Adam(parameters, lr=learning_rate)

# Gradient clipping for training stability
for param in model.parameters():
    if param.grad is not None:
        param.grad.data.clamp_(-0.1, 0.1)
```

### Training Loop Architecture

#### Continuous Data Streaming
```python
# Large-scale batch processing
batch_size = 65536           # Large batch for GPU efficiency
large_batch_scale = 100     # Buffer scaling factor

while True:  # Continuous training loop
    # Data accumulation phase
    while len(raw_buffers[0]) < batch_size * large_batch_scale:
        # Load and process motion files
        
    # Training phase  
    for mini_batch in create_mini_batches(batch_size):
        loss = ref_learner.learn(mini_batch)
```

#### Checkpointing System
```python
# Periodic model saving (every 50 iterations)
if iter % 50 == 0:
    state = {
        "metadata": env.getMetadata(),
        "is_cascaded": True,
        "ref": ref_learner.get_weights()
    }
    pickle.dump(state, checkpoint_file)
```

## Data Flow Diagram

```
Input Data Files (*.npz)
         ↓
    File Loading
         ↓
    Parameter Normalization → param_matrix (60 x num_paramstate)
         ↓
    Phase Encoding → phi (60 x 2)
         ↓
    Concatenation → data_in (60 x [num_paramstate + 2])
         ↓
    Motion Targets → data_out (60 x 101)
         ↓
    Batch Processing → Training Batches
         ↓
    RefNN Forward Pass
         ↓
    Weighted Loss Computation
         ↓
    Gradient Clipping & Optimization
         ↓
    Model Checkpointing
```

## Integration Points

### Environment Interface
- **RayEnvManager**: Physics simulation environment
- **Parameter Normalization**: `env.getNormalizedParamStateFromParam()`
- **Motion Space**: `env.posToSixDof(env.getPositions())`
- **Known Parameters**: `env.getNumKnownParam()`

### Data Pipeline Integration
- **Source**: Augmented motion data from `augument_raw_data.py`
- **Format**: NPZ files with 'motions' and 'params' arrays
- **Preprocessing**: Automatic parameter normalization and phase encoding

### Output Integration
- **Model Checkpoints**: Pickle format with metadata preservation
- **Tensorboard Logging**: Training metrics and loss visualization
- **State Persistence**: Compatible with cascaded subsumption networks

## Key Design Decisions

1. **Temporal Consistency**: 2-second (60-frame) motion sequences with phase encoding
2. **Parameter Generalization**: Normalized parameter space for robust learning
3. **Weighted Loss**: Domain-specific loss weighting for motion quality
4. **Gradient Stability**: Comprehensive gradient clipping at multiple levels
5. **Continuous Learning**: Infinite training loop with periodic checkpointing
6. **Memory Efficiency**: Large batch processing with buffer management

## Performance Characteristics

- **Training Stability**: Multi-level gradient clipping prevents training collapse
- **Memory Usage**: Efficient batch processing with configurable buffer sizes  
- **Convergence**: Conservative learning rate ensures stable convergence
- **Quality Control**: Weighted loss function prioritizes locomotion-critical features
- **Scalability**: Continuous data streaming supports large-scale datasets
