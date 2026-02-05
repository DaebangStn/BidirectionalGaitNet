# MABA SPD Implementation Plan

## Overview

This document outlines the plan to implement true MABA (Modified Articulated Body Algorithm) for SPD control in DART. The current implementation uses dense matrix operations O(n³), while true MABA achieves O(n) complexity.

**Reference Paper**: "}Stable Proportional-Derivative Controllers" (fastSPD)

---

## Current Status

### What We Have Now

1. **`computeSPDForces()`** - Original DART-style SPD using dense matrix inverse:
   ```cpp
   M_inv = (M + Kv*dt).inverse();  // O(n³)
   ddq = M_inv * (-cg + p_diff + v_diff + ext);
   tau = p_diff + v_diff - dt * Kv * ddq;
   ```

2. **`computeMABAForces()`** - Improved Q computation with proper BallJoint quaternion handling, but still uses dense matrix inverse.

### Why Current Approach Works
- For n ≈ 50 DOFs, dense inverse is ~125K operations
- At 180 Hz simulation, this is ~22M ops/second (acceptable on modern CPUs)
- BallJoint quaternion handling is now correct

### Why True MABA Would Be Better
- O(n) vs O(n³) complexity
- Better numerical stability for large articulated systems
- Consistent with physics engine internals (PhysX uses embedded MABA)

---

## DART ABA Architecture Analysis

### Key Files in DART Source (`~/pkgsrc/dart/dart/dynamics/`)

| File | Purpose |
|------|---------|
| `Skeleton.cpp:3735` | `computeForwardDynamics()` - Entry point |
| `BodyNode.cpp:1748` | `updateBiasForce()` - Pass 2 backward recursion |
| `BodyNode.cpp:1831` | `updateAccelerationFD()` - Pass 3 forward recursion |
| `detail/GenericJoint.hpp:1923` | `updateInvProjArtInertiaImplicitDynamic()` - D⁻¹ computation |
| `detail/GenericJoint.hpp:2140` | `updateTotalForceDynamic()` - u computation |
| `detail/GenericJoint.hpp:2242` | `updateAccelerationDynamic()` - q̈ computation |

### DART ABA Flow

```
computeForwardDynamics()
│
├── Pass 2: Backward (leaves → root)
│   └── BodyNode::updateBiasForce()
│       ├── mBiasForce = -dad(V, I*V) - Fext - Fgravity
│       ├── For each child joint:
│       │   ├── addChildArtInertiaImplicitTo()  → I_i^a accumulation
│       │   └── addChildBiasForceTo()           → p_i^a accumulation
│       └── Joint::updateTotalForce()           → u_i computation
│
└── Pass 3: Forward (root → leaves)
    └── BodyNode::updateAccelerationFD()
        └── Joint::updateAcceleration()         → q̈_i computation
```

### Key DART Internal Variables (Private)

| Variable | Type | Description | Location |
|----------|------|-------------|----------|
| `mArtInertia` | `Matrix6d` | Articulated body inertia I^A | BodyNode |
| `mArtInertiaImplicit` | `Matrix6d` | Implicit I^A (with dt terms) | BodyNode |
| `mBiasForce` | `Vector6d` | Bias force p^A | BodyNode |
| `mInvProjArtInertia` | `Matrix` | D⁻¹ (DOF × DOF) | Joint |
| `mInvProjArtInertiaImplicit` | `Matrix` | Implicit D⁻¹ | Joint |
| `mTotalForce` | `Vector` | Generalized force u | Joint |

---

## SPD vs DART Spring/Damper Comparison

### SPD Formulation
```
D_i = S^T * I^A * S + Kv_i * dt
Q_i = -Kp_i * (q_i + dt*v_i - target_i) - Kv_i * v_i
u_i = Q_i - S^T * p^A + tau_ext
q̈_i = D_i^{-1} * (u_i - H^T * a_parent)
τ_i = Q_i - Kv_i * dt * q̈_i
```

### DART Built-in Spring/Damper
```cpp
// In updateInvProjArtInertiaImplicitDynamic():
projAI = S^T * I^A * S + (Kd*dt + Kp*dt²).asDiagonal()  // Note: Kp*dt² term!

// In updateTotalForceDynamic():
springForce = -Kp * (q - rest + v*dt)   // Same as SPD's P term
dampingForce = -Kv * v                   // Same as SPD's D term
mTotalForce = tau + springForce + dampingForce - S^T * p^A
```

### Key Difference
DART adds `Kp*dt²` to D for implicit spring stability. SPD only adds `Kv*dt`.

---

## Implementation Plan: Option 3 (DART Fork)

### Phase 1: Add SPD Interface to Joint

**File**: `dart/dynamics/Joint.hpp`

```cpp
// Add new virtual methods
virtual void setSPDGains(const Eigen::VectorXd& kp, const Eigen::VectorXd& kv) = 0;
virtual void setSPDTarget(const Eigen::VectorXd& target) = 0;
virtual void enableSPDMode(bool enable) = 0;
```

**File**: `dart/dynamics/detail/GenericJoint.hpp`

```cpp
// Add member variables
Vector mSPD_Kp;
Vector mSPD_Kv;
Vector mSPD_Target;
bool mSPDEnabled = false;
```

### Phase 2: Modify D Computation

**File**: `dart/dynamics/detail/GenericJoint.hpp`
**Function**: `updateInvProjArtInertiaImplicitDynamic()`

```cpp
void GenericJoint<ConfigSpaceT>::updateInvProjArtInertiaImplicitDynamic(
    const Eigen::Matrix6d& artInertia, double timeStep)
{
  const JacobianMatrix& Jacobian = getRelativeJacobianStatic();
  Matrix projAI = Jacobian.transpose() * artInertia * Jacobian;

  if (mSPDEnabled) {
    // SPD: D = S^T * I^A * S + Kv * dt
    projAI += (mSPD_Kv * timeStep).asDiagonal();
  } else {
    // Original DART: D = S^T * I^A * S + Kd*dt + Kp*dt²
    projAI += (timeStep * Base::mAspectProperties.mDampingCoefficients
               + timeStep * timeStep * Base::mAspectProperties.mSpringStiffnesses)
                  .asDiagonal();
  }

  mInvProjArtInertiaImplicit = math::inverse<ConfigSpaceT>(projAI);
}
```

### Phase 3: Modify u Computation

**File**: `dart/dynamics/detail/GenericJoint.hpp`
**Function**: `updateTotalForceDynamic()`

```cpp
void GenericJoint<ConfigSpaceT>::updateTotalForceDynamic(
    const Eigen::Vector6d& bodyForce, double timeStep)
{
  Vector Q;

  if (mSPDEnabled) {
    // SPD: Q = -Kp*(q + dt*v - target) - Kv*v
    Q = -mSPD_Kp.cwiseProduct(
            getPositionsStatic() + getVelocitiesStatic() * timeStep - mSPD_Target)
        - mSPD_Kv.cwiseProduct(getVelocitiesStatic());
  } else {
    // Original DART spring/damper
    const Vector springForce = -Base::mAspectProperties.mSpringStiffnesses.cwiseProduct(
        getPositionsStatic() - Base::mAspectProperties.mRestPositions
        + getVelocitiesStatic() * timeStep);
    const Vector dampingForce = -Base::mAspectProperties.mDampingCoefficients.cwiseProduct(
        getVelocitiesStatic());
    Q = springForce + dampingForce;
  }

  mTotalForce = this->mAspectState.mForces + Q
                - getRelativeJacobianStatic().transpose() * bodyForce;
}
```

### Phase 4: Add BallJoint Quaternion Handling

For BallJoint, the position difference needs proper quaternion math:

**File**: `dart/dynamics/BallJoint.cpp`

```cpp
// Override or modify getPositionDifferencesStatic for SPD mode
Eigen::Vector3d BallJoint::getSPDPositionError(
    const Eigen::Vector3d& target) const
{
  const Eigen::Matrix3d R_target = convertToRotation(target);
  const Eigen::Matrix3d R_current = convertToRotation(getPositionsStatic());

  // Quaternion difference for shortest path
  Eigen::Quaterniond q_target(R_target);
  Eigen::Quaterniond q_current(R_current);
  Eigen::Quaterniond q_diff = q_target * q_current.inverse();

  if (q_diff.w() < 0) q_diff.coeffs() *= -1;  // Shortest path

  Eigen::AngleAxisd aa(q_diff);
  return aa.axis() * aa.angle();
}
```

### Phase 5: Add Skeleton-Level API

**File**: `dart/dynamics/Skeleton.hpp`

```cpp
/// Enable SPD mode for all joints
void enableSPDMode(bool enable);

/// Set SPD gains for all DOFs
void setSPDGains(const Eigen::VectorXd& kp, const Eigen::VectorXd& kv);

/// Set SPD target positions
void setSPDTarget(const Eigen::VectorXd& target);

/// Compute forward dynamics with SPD and return resulting torques
Eigen::VectorXd computeForwardDynamicsSPD(
    const Eigen::VectorXd& kp,
    const Eigen::VectorXd& kv,
    const Eigen::VectorXd& target,
    const Eigen::VectorXd& externalTorque = Eigen::VectorXd());
```

### Phase 6: Torque Recovery

After `computeForwardDynamics()` with SPD mode:

```cpp
Eigen::VectorXd Skeleton::getSPDTorque(double timeStep) const
{
  Eigen::VectorXd tau = Eigen::VectorXd::Zero(getNumDofs());

  for (size_t i = 0; i < getNumJoints(); i++) {
    Joint* joint = getJoint(i);
    if (!joint->isSPDEnabled()) continue;

    int idx = joint->getIndexInSkeleton(0);
    int dof = joint->getNumDofs();

    // τ = Q - Kv * dt * q̈
    Eigen::VectorXd Q = joint->getSPDForce();  // -Kp*(q+dt*v-target) - Kv*v
    Eigen::VectorXd ddq = joint->getAccelerations();
    Eigen::VectorXd Kv = joint->getSPDKv();

    tau.segment(idx, dof) = Q - Kv.cwiseProduct(ddq) * timeStep;
  }

  return tau;
}
```

---

## Integration with BidirectionalGaitNet

### Modified Controller.cpp

```cpp
Eigen::VectorXd Controller::computeMABAForces(const Eigen::VectorXd& pdTarget,
                                               const Eigen::VectorXd& ext)
{
    double dt = mSkeleton->getTimeStep() * mInferencePerSim;

    // 1. Enable SPD mode
    mSkeleton->enableSPDMode(true);
    mSkeleton->setSPDGains(mKp, mKv);
    mSkeleton->setSPDTarget(pdTarget);

    // 2. Set external forces
    if (ext.size() > 0) {
        mSkeleton->setForces(ext);
    }

    // 3. Run ABA with embedded SPD (O(n) complexity!)
    mSkeleton->computeForwardDynamics();

    // 4. Recover torque for logging/clipping
    Eigen::VectorXd tau = mSkeleton->getSPDTorque(dt);

    // 5. Disable SPD mode (restore normal operation)
    mSkeleton->enableSPDMode(false);

    // 6. Apply clipping
    if (mMaxTorque.size() > 0) {
        for (int i = 6; i < tau.size(); i++) {
            tau[i] = std::clamp(tau[i], -mMaxTorque[i], mMaxTorque[i]);
        }
    }

    tau.head<6>().setZero();
    return tau;
}
```

---

## Testing Plan

### Unit Tests

1. **Equivalence Test**: Compare `computeSPDForces()` vs `computeMABAForces()` outputs
   - Same Kp, Kv, target, q, v
   - Outputs should match within numerical tolerance

2. **Joint Type Tests**: Verify correct behavior for:
   - FreeJoint (6-DOF root)
   - BallJoint (3-DOF with quaternion)
   - RevoluteJoint (1-DOF)

3. **Performance Benchmark**: Measure actual speedup
   - Time dense inverse approach vs MABA
   - Profile at different DOF counts (20, 50, 100, 200)

### Integration Tests

1. Run simulation with MABA, compare trajectory to dense SPD
2. Visual inspection in viewer
3. Training convergence comparison

---

## Files to Modify in DART Fork

| File | Changes |
|------|---------|
| `dynamics/Joint.hpp` | Add SPD virtual interface |
| `dynamics/Joint.cpp` | Add default implementations |
| `dynamics/detail/GenericJoint.hpp` | Add SPD members, modify D and u computation |
| `dynamics/BallJoint.hpp` | Add quaternion SPD method |
| `dynamics/BallJoint.cpp` | Implement quaternion position error |
| `dynamics/Skeleton.hpp` | Add skeleton-level SPD API |
| `dynamics/Skeleton.cpp` | Implement SPD methods, torque recovery |

---

## Estimated Effort

| Phase | Effort | Priority |
|-------|--------|----------|
| Phase 1: Joint interface | 2 hours | High |
| Phase 2: D computation | 1 hour | High |
| Phase 3: u computation | 1 hour | High |
| Phase 4: BallJoint quaternion | 2 hours | High |
| Phase 5: Skeleton API | 2 hours | Medium |
| Phase 6: Torque recovery | 1 hour | Medium |
| Testing | 4 hours | High |
| **Total** | **~13 hours** | |

---

## References

1. **fastSPD Paper**: "Stable Proportional-Derivative Controllers"
2. **DART Source**: `~/pkgsrc/dart/dart/dynamics/`
3. **MABA Reference**: `/home/geon/MABA/src/articulation/spd/spd_ABA.cpp`
4. **Current Implementation**: `sim/Controller.cpp:computeMABAForces()`

---

## Notes

- Current `computeMABAForces()` has correct BallJoint quaternion handling but still uses O(n³) dense inverse
- DART's built-in spring/damper cannot be used directly due to `Kp*dt²` term mismatch
- True MABA requires DART source modification (Option 3)
- Performance gain may be marginal for n ≈ 50 DOFs but significant for larger articulations
