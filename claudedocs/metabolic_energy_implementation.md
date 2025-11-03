# Metabolic Energy Implementation

## Overview
Added metabolic energy computation capabilities to the `Character` class based on the reference implementation in `prev_sources/Energy.cpp`. The implementation maintains backward compatibility through a mode-based system.

## Implementation Details

### Enum: MetabolicRewardType
Located in `sim/Character.h`:
```cpp
enum MetabolicRewardType
{
    LEGACY,  // No metabolic computation (backward compatible)
    A,       // abs(activation)
    A2,      // activation^2
    MA       // mass * abs(activation)
};
```

### New Member Variables
- `MetabolicRewardType mMetabolicRewardType` - Current metabolic mode
- `Eigen::VectorXd mMuscleMassCache` - Cached muscle masses
- `double mMetabolicEnergyAccum` - Accumulated metabolic energy
- `double mMetabolicAccumDivisor` - Accumulation count for averaging

### New Methods

#### `void cacheMuscleMass()`
Caches muscle masses by calling `GetMass()` on each muscle. **Must be called after `setMuscles()`** for MA mode to work correctly.

```cpp
character->setMuscles(muscle_path);
character->cacheMuscleMass();  // Required for MA mode
```

#### `void setMetabolicRewardType(MetabolicRewardType type)`
Sets the metabolic computation mode.

```cpp
character->setMetabolicRewardType(A2);   // Use activation^2
character->setMetabolicRewardType(MA);   // Use mass * abs(activation)
character->setMetabolicRewardType(LEGACY); // Disable (default)
```

#### `MetabolicRewardType getMetabolicRewardType()`
Returns the current metabolic mode.

#### `double getMetabolicReward()`
Returns the average accumulated metabolic energy: `mMetabolicEnergyAccum / mMetabolicAccumDivisor`

Returns 0.0 if divisor < 1e-6.

#### `void resetMetabolicEnergy()`
Resets accumulation variables to 0. Also called by `clearLogs()`.

## Usage Example

```cpp
// Initialize character
Character* character = new Character(skeleton_path);
character->setMuscles(muscle_path);
character->cacheMuscleMass();  // Cache muscle masses

// Set metabolic mode
character->setMetabolicRewardType(MA);  // Use mass-weighted activation

// During simulation loop
for (int i = 0; i < num_steps; i++) {
    // Set activations - this automatically accumulates metabolic energy
    character->setActivations(activation_vector);

    // Step simulation
    character->step();
}

// Get metabolic reward
double metabolic_reward = character->getMetabolicReward();
std::cout << "Metabolic reward: " << metabolic_reward << std::endl;

// Reset for next episode
character->resetMetabolicEnergy();
```

## Metabolic Modes

### LEGACY (Default)
No metabolic computation. Maintains backward compatibility.

### A Mode
Energy = sum of absolute activations
```cpp
energy = |a₁| + |a₂| + ... + |aₙ|
```
Implemented as: `mActivations.array().abs().sum()`

### A2 Mode
Energy = sum of squared activations
```cpp
energy = a₁² + a₂² + ... + aₙ²
```
Implemented as: `(mActivations.array() * mActivations.array()).sum()`

### MA Mode
Energy = sum of mass-weighted absolute activations
```cpp
energy = m₁|a₁| + m₂|a₂| + ... + mₙ|aₙ|
```
Implemented as: `(mMuscleMassCache.array() * mActivations.array().abs()).sum()`

**Note**: Requires `cacheMuscleMass()` to be called after `setMuscles()`.

## Backward Compatibility

The default mode is `LEGACY`, which performs no metabolic computation. This ensures existing code continues to work without modification. Only when explicitly setting a different mode will metabolic energy be computed.

## Integration Points

- **setActivations()**: Automatically accumulates metabolic energy when mode != LEGACY
- **clearLogs()**: Automatically resets metabolic energy accumulation
- **Constructor**: Initializes metabolic mode to LEGACY with zero accumulation

## Implementation Reference

Based on `prev_sources/Energy.cpp`:
- Uses Eigen array operations for efficient computation
- Accumulates energy step-by-step as activations are set
- Maintains divisor for averaging over time
- Pattern matches Energy::AccumActivation() method
