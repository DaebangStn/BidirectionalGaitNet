# Cascaded Subsumption Network (CSN) Action Computation

## Overview

The Cascaded Subsumption Network (CSN) is a hierarchical neural network architecture that enables learning control policies with 618 adjustable parameters (10 body + 608 muscle parameters) by progressively adding knowledge layers while preserving previously learned behaviors.

## Key Concepts

### 1. Progressive Learning Architecture
- **Base Network (π₀)**: Learns healthy individual control (body + gait parameters only)
- **Level 1 Networks (π₁ᵢ)**: Learn specific muscle disorder effects (hip vs knee/ankle)
- **Level 2 Network (π₂)**: Subsumes lower body muscle coordination
- **Level 3 Network (π₃)**: Manages upper/lower body co-occurrence relations

### 2. Parameter Partitioning
```
Cmuscle = Cweakness ∪ Ccontracture = ⋃ᵢ cʲᵢ
```
Where {cʲᵢ} represents muscle parameter partitions at hierarchical levels.

## Core Algorithm: Action Computation in `Environment::setAction`

### Phase 1: State Projection and Preparation

```cpp
if (mUseCascading) {
    // Clear previous computation states
    mProjStates.clear();
    mProjJointStates.clear();
    
    // Generate projected states for each network level
    for (Network nn : mPrevNetworks) {
        std::pair<Eigen::VectorXd, Eigen::VectorXd> prev_states = 
            getProjState(nn.minV, nn.maxV);
        mProjStates.push_back(prev_states.first);
        mProjJointStates.push_back(prev_states.second);
    }
    
    // Add current state
    mProjStates.push_back(mState);
    mProjJointStates.push_back(mJointState);
}
```

**Key Function**: `getProjState(minV, maxV)` clips current parameter state to network-specific ranges:
```cpp
projState[i] = dart::math::clip(curParamState[i], minV[i], maxV[i])
```

### Phase 2: Confidence Calculation

The confidence mechanism determines how much each network level contributes to the final action:

```cpp
// Initialize confidence arrays
mDmins.clear();
mWeights.clear();  
mBetas.clear();

// Base network always has full confidence
if (mPrevNetworks.size() > 0) {
    mDmins[0] = 0.0;
    mWeights[0] = 1.0;
    mBetas[0] = 0.0;
}

// Calculate impact distances for hierarchical networks
for (Eigen::Vector2i edge : mEdges) {
    double d = (mProjJointStates[edge[1]] - mProjJointStates[edge[0]]).norm() * 0.008;
    if (mDmins[edge[1]] > d)
        mDmins[edge[1]] = d;
}
```

**Confidence Formula** (Algorithm 1 from paper):
```
α = sigmoid(dmin - β)
```

Where:
- `dmin`: Minimum impact distance on joint force capacity
- `β`: Learned threshold parameter (0.2 + 0.1 * network_output)
- `α`: Final confidence weight

### Phase 3: Hierarchical Action Integration

#### Base Network (π₀) - Always Applied First
```cpp
if (i == 0) {
    // Direct application for healthy baseline
    mAction.head(mNumActuatorAction) = 
        mActionScale * (mUseWeights[0] ? 1 : 0) * prev_action.head(mNumActuatorAction);
    
    // Muscle activation summing  
    mAction.segment(mNumActuatorAction, (mAction.rows()-1) - mNumActuatorAction) += 
        (mUseWeights[0] ? 1 : 0) * prev_action.segment(...);
        
    // Phase displacement
    mPhaseDisplacement += mPhaseDisplacementScale * prev_action[mNumActuatorAction];
}
```

#### Higher Level Networks (π₁, π₂, π₃) - Confidence-Weighted Integration
```cpp
// Calculate dynamic confidence threshold
double beta = 0.2 + 0.1 * prev_action[prev_action.rows() - 1];
mBetas[i] = beta;
mWeights[i] = mPrevNetworks.front().joint.attr("weight_filter")(mDmins[i], beta).cast<double>();

// Joint angle integration using addPositions (spatial displacement)
mAction.head(mNumActuatorAction) = mCharacters[0]->addPositions(
    mAction.head(mNumActuatorAction),
    (mUseWeights[i*(mUseMuscle?2:1)] ? 1 : 0) * mWeights[i] * mActionScale * 
    prev_action.head(mNumActuatorAction), 
    false
);

// Muscle activation integration (temporal displacement) 
mAction.segment(mNumActuatorAction, (mAction.rows()-1) - mNumActuatorAction) += 
    (mUseWeights[i*(mUseMuscle?2:1)] ? 1 : 0) * mWeights[i] * 
    prev_action.segment(mNumActuatorAction, (mAction.rows()-1) - mNumActuatorAction);

// Phase displacement integration
mPhaseDisplacement += mWeights[i] * mPhaseDisplacementScale * prev_action[mNumActuatorAction];
```

### Phase 4: Current Network Integration

The currently training network (highest level) is integrated similarly to higher-level networks:

```cpp
if (mLoadedMuscleNN) {
    double beta = 0.2 + 0.1 * _action[_action.rows() - 1];
    mBetas[mBetas.size() - 1] = beta;
    mWeights[mWeights.size() - 1] = 
        mPrevNetworks.front().joint.attr("weight_filter")(mDmins.back(), beta).cast<double>();
    
    // Apply confidence-weighted action integration
    mAction.head(mNumActuatorAction) = mCharacters[0]->addPositions(
        mAction.head(mNumActuatorAction),
        (mUseWeights[mUseWeights.size()-(mUseMuscle?2:1)] ? 1 : 0) * mWeights.back() * 
        mActionScale * _action.head(mNumActuatorAction), 
        false
    );
    
    mAction.segment(mNumActuatorAction, (mAction.rows()-1) - mNumActuatorAction) += 
        (mUseWeights[mUseWeights.size()-(mUseMuscle?2:1)] ? 1 : 0) * mWeights.back() * 
        _action.segment(mNumActuatorAction, (mAction.rows()-1) - mNumActuatorAction);
}
```

## Mathematical Framework

### Equation 13 Implementation
The paper's core equation:
```
M̂ = M₀(φ + Δφ₀ + α₁Δφ₁) ⊕ ΔM₀ ⊕ α₁ΔM₁
```

Maps to code implementation:
- **M₀**: Base muscle model
- **φ + Δφ₀**: Base joint angles + base network displacement
- **α₁Δφ₁**: Confidence-weighted spatial displacement from level 1 network
- **ΔM₀ ⊕ α₁ΔM₁**: Additive muscle activation with confidence weighting

### Confidence Algorithm Details

```cpp
// For each network level n with children n-1
for (int i = 0; i < m; i++) {
    Δci = cn \ cin-1;  // Parameter difference
    Δsjoint = computeJointStateChange(Δci);  // Joint impact
    d = ||Δsjoint^T * Wjoint * Δsjoint||;  // Weighted norm
    if (dmin > d) dmin = d;
}
α = sigmoid(dmin - β);  // Final confidence
```

## Key Features

### 1. **Progressive Knowledge Preservation**
- Lower-level networks remain unchanged during higher-level training
- Confidence mechanism prevents over-learning and knowledge forgetting
- Hierarchical structure enables scalability to 618 parameters

### 2. **Adaptive Confidence Weighting**
- Networks with high impact on joint forces get higher confidence
- Healthy muscle states reduce higher-level network intervention
- Dynamic threshold β learned as part of policy action

### 3. **Muscle Activation Bounds**
Special handling for muscle regression networks:
```cpp
// Take unnormalized output, weighted sum, then sigmoid
A_unnormalized = Σ(αᵢ * Aᵢ_unnormalized)  
A_final = sigmoid(A_unnormalized)
```

### 4. **Hierarchical Network Structure**
- **Level 0**: Base healthy control (no muscle parameters)
- **Level 1**: Hip muscles vs Knee/Ankle muscles  
- **Level 2**: Complete lower body integration
- **Level 3**: Upper/lower body co-occurrence learning

## Implementation Architecture

### Network Structure (`struct Network`)
```cpp
struct Network {
    std::string name;        // Network path/identifier
    py::object joint;        // Joint control network
    py::object muscle;       // Muscle activation network  
    Eigen::VectorXd minV;    // Parameter range minimum
    Eigen::VectorXd maxV;    // Parameter range maximum
};
```

### Key Member Variables
- `mPrevNetworks`: Vector of hierarchical networks
- `mEdges`: Network hierarchy connections (parent-child relationships)
- `mWeights`: Confidence values for each network level
- `mDmins`: Impact distances for confidence calculation
- `mBetas`: Dynamic thresholds for each network
- `mUseCascading`: Flag to enable/disable cascaded computation

## Integration with Training Pipeline

### Ray Environment (`python/ray_env.py`)
The cascaded action computation integrates seamlessly with Ray RLlib training:
- Environment receives high-level network actions
- `setAction` processes them through the cascaded hierarchy
- Lower-level networks provide stable baseline behaviors
- Higher-level networks learn incremental improvements

### Configuration Control
```cpp
bool mUseCascading;  // Enable cascaded computation
std::vector<bool> mUseWeights;  // Per-network enable flags
```

This architecture enables training complex muscle-driven character control with hundreds of parameters while maintaining learning stability and preventing catastrophic forgetting.