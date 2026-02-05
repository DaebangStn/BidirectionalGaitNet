#include "Controller.h"
#include "Character.h"
#include "Log.h"
#include <random>

// Thread-safe random number generation
namespace {
    thread_local std::mt19937 controller_rng(std::random_device{}());

    inline double thread_safe_uniform(double min, double max) {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(controller_rng);
    }
}

// Thread-safe ActuatorType lookup
ActuatorType getActuatorType(std::string type) {
    if (type == "torque") return tor;
    if (type == "pd") return pd;
    if (type == "muscle") return mass;  // "muscle" now maps to "mass"
    if (type == "mass") return mass;
    if (type == "mass_lower") return mass_lower;
    throw std::runtime_error("Invalid actuator type: " + type);
}

Controller::Controller(const ControllerConfig& config)
    : mSkeleton(config.skeleton),
      mNumDofs(config.skeleton->getNumDofs()),
      mRootDof(config.skeleton->getRootJoint()->getNumDofs()),
      mNumMuscleDof(config.numMuscleDof),
      mKp(config.kp),
      mKv(config.kv),
      mActuatorType(config.actuatorType),
      mInferencePerSim(config.inferencePerSim),
      mScaleTauOnWeight(config.scaleTauOnWeight),
      mTorqueMassRatio(config.torqueMassRatio),
      mNumSubSteps(config.numSubSteps)
{
    mCachedSPDTorque = Eigen::VectorXd::Zero(mNumDofs);
}

Controller::~Controller() {}

void Controller::setVirtualRootForceKp(double kp) {
    mVfKp = kp;
    double vf_kv = 2.0 * std::sqrt(kp);  // Critical damping
    // Directly set root translation gains (DOF 3-5)
    mKp.segment<3>(3).setConstant(kp);
    mKv.segment<3>(3).setConstant(vf_kv);
}

void Controller::setVirtualRootRefPosition(const Eigen::Vector3d& ref_pos) {
    mVfRefPosition = ref_pos;
}

void Controller::setVirtualRootRefOrientation(const Eigen::Matrix3d& ref_rot) {
    mVfRefOrientation = ref_rot;
}

void Controller::setVirtualRootRefOverride(bool enabled, const Eigen::Vector3d& ref_pos) {
    mVfRefOverrideEnabled = enabled;
    mVfRefOverridePos = ref_pos;
}

Eigen::Vector3d Controller::computeRootVirtualSPD() {
    if (mVfKp <= 0.0) {
        mCachedVirtualRootForce.setZero();
        return mCachedVirtualRootForce;
    }

    double dt = mSkeleton->getTimeStep() * mInferencePerSim;
    double kp = mVfKp;
    double kv = 2.0 * std::sqrt(kp);  // Critical damping
    double mass = mSkeleton->getMass();
    Eigen::Vector3d gravity(0.0, -9.81 * mass, 0.0);  // World frame gravity force

    // === Translation SPD (world frame) ===
    Eigen::Vector3d p = mSkeleton->getPositions().segment<3>(3);
    Eigen::Vector3d v = mSkeleton->getRootBodyNode()->getLinearVelocity();
    Eigen::Vector3d ref_pos = getVirtualRootRefPosition();

    Eigen::Vector3d p_next = p + v * dt;
    Eigen::Vector3d p_err = ref_pos - p_next;

    // Solve for linear acceleration
    Eigen::Vector3d rhs_lin = kp * p_err - kv * v - gravity;
    double m_eff = mass + dt * kv;
    Eigen::Vector3d a_lin = rhs_lin / m_eff;

    // SPD force (position only)
    mCachedVirtualRootForce = kp * p_err - kv * v - dt * kv * a_lin;

    return mCachedVirtualRootForce;
}

ControllerOutput Controller::step(const ControllerInput& input)
{
    ControllerOutput output;
    output.torque = Eigen::VectorXd::Zero(mNumDofs);
    output.activations = Eigen::VectorXf::Zero(0);

    switch (mActuatorType)
    {
    case tor:
        // Pass through pre-set torque
        output.torque = input.torque;
        mCachedSPDTorque = input.torque;
        break;

    case pd:
        // Compute SPD torque
        output.torque = computeSPDForces(input.pdTarget);
        mCachedSPDTorque = output.torque;
        break;

    case mass:
    case mass_lower:
    {
        if (!input.muscleTuple) {
            LOG_ERROR("[Controller] mass mode requires muscleTuple");
            break;
        }

        // Build external force from JtP
        // When isMirror: MuscleTuple.JtP is mirrored, but SPD needs actual frame
        Eigen::VectorXd fullJtp = Eigen::VectorXd::Zero(mNumDofs);
        if (input.includeJtPinSPD) {
            fullJtp.tail(mNumDofs - mRootDof) = input.muscleTuple->JtP;
            // Un-mirror JtP for SPD physics (skeleton state is in actual frame)
            if (input.isMirror && mMirrorCharacter) {
                fullJtp = mMirrorCharacter->getMirrorPosition(fullJtp);
            }
        }

        // SPD computes in actual frame
        mCachedSPDTorque = computeSPDForces(input.pdTarget, fullJtp);

        // Mirror tau for muscle NN (NN uses mirrored MuscleTuple when isMirror)
        Eigen::VectorXd tauForNN = Eigen::VectorXd::Zero(mNumDofs);
        tauForNN.segment(mRootDof, mNumMuscleDof) = mCachedSPDTorque.segment(mRootDof, mNumMuscleDof);
        if (input.isMirror && mMirrorCharacter) {
            tauForNN = mMirrorCharacter->getMirrorPosition(tauForNN);
        }
        output.torque = mCachedSPDTorque;
        output.torque.segment(mRootDof, mNumMuscleDof).setZero();

        // Compute activations
        output.activations = computeActivationsFromTorque(
            *input.muscleTuple, tauForNN, input.includeJtPinSPD);

        // Un-mirror activations for application to actual skeleton
        if (input.isMirror && mMirrorCharacter) {
            output.activations = mMirrorCharacter->getMirrorActivation(
                output.activations.cast<double>()).cast<float>();
        }
        break;
    }
    }

    return output;
}

Eigen::VectorXd Controller::computeSPDForces(const Eigen::VectorXd& pdTarget,
                                              const Eigen::VectorXd& ext)
{
    Eigen::VectorXd q = mSkeleton->getPositions();
    Eigen::VectorXd dq = mSkeleton->getVelocities();
    double dt = mSkeleton->getTimeStep() * mInferencePerSim;

    Eigen::VectorXd qdqdt = q + dq * dt;

    // Note: Virtual root force is now computed separately in computeWorldFrameSPD()
    Eigen::VectorXd p_target = pdTarget;

    Eigen::VectorXd p_diff = -mKp.cwiseProduct(mSkeleton->getPositionDifferences(qdqdt, p_target));
    Eigen::VectorXd v_diff = -mKv.cwiseProduct(dq);

    int numDofs = mSkeleton->getNumDofs();
    Eigen::VectorXd extForces = ext.size() > 0 ? ext : Eigen::VectorXd::Zero(numDofs);

    Eigen::VectorXd cg = mSkeleton->getCoriolisAndGravityForces();
    Eigen::VectorXd constraint = mSkeleton->getConstraintForces();
    Eigen::MatrixXd M_inv = (mSkeleton->getMassMatrix() + Eigen::MatrixXd(dt * mKv.asDiagonal())).inverse();
    Eigen::VectorXd ddq = M_inv * (-cg + p_diff + v_diff + constraint + extForces);
    Eigen::VectorXd tau = p_diff + v_diff - dt * mKv.cwiseProduct(ddq);

    tau.head<3>().setZero();
    tau.segment<3>(3) = computeRootVirtualSPD();
    return tau;
}

Eigen::VectorXf Controller::computeActivationsFromTorque(const MuscleTuple& mt,
                                                          const Eigen::VectorXd& desiredTorque,
                                                          bool includeJtPinSPD)
{
    // Extract dt for muscle NN (excluding root DOFs)
    Eigen::VectorXd dt = desiredTorque.tail(mt.JtP.rows());

    // If not including JtP in SPD, subtract it here
    // (This matches the original calcActivation logic)
    if (!includeJtPinSPD) {
        dt -= mt.JtP;
    }

    int numMuscles = mt.JtA.cols();
    std::vector<Eigen::VectorXf> prev_activations;

    for (int j = 0; j < (int)mPrevNetworks.size() + 1; j++) {
        prev_activations.push_back(Eigen::VectorXf::Zero(numMuscles));
    }

    // For base network
    if (mPrevNetworks.size() > 0 && mPrevNetworks[0]) {
        prev_activations[0] = mPrevNetworks[0]->unnormalized_no_grad_forward(mt.JtA_reduced, dt, nullptr, 1.0);
    }

    // Hierarchical networks
    for (size_t j = 1; j < mPrevNetworks.size(); j++) {
        if (!mPrevNetworks[j]) continue;

        Eigen::VectorXf prev_activation = Eigen::VectorXf::Zero(numMuscles);
        if (j < mChildNetworks.size()) {
            for (int k : mChildNetworks[j]) {
                prev_activation += prev_activations[k];
            }
        }

        double weight = (j < mWeights.size()) ? mWeights[j] : 1.0;
        bool useWeight = (j * 2 + 1 < mUseWeights.size()) ? mUseWeights[j * 2 + 1] : true;

        prev_activations[j] = (useWeight ? 1.0f : 0.0f) * static_cast<float>(weight) *
            mPrevNetworks[j]->unnormalized_no_grad_forward(mt.JtA_reduced, dt, &prev_activation, weight);
    }

    // Current Network
    if (mLoadedMuscleNN && mMuscleNN) {
        Eigen::VectorXf prev_activation = Eigen::VectorXf::Zero(numMuscles);

        if (!mChildNetworks.empty()) {
            for (int k : mChildNetworks.back()) {
                prev_activation += prev_activations[k];
            }
        }

        double weight = mWeights.empty() ? 1.0 : mWeights.back();
        bool useWeight = mUseWeights.empty() ? true : mUseWeights.back();

        if (mPrevNetworks.size() > 0) {
            prev_activations.back() = (useWeight ? 1.0f : 0.0f) * static_cast<float>(weight) *
                mMuscleNN->unnormalized_no_grad_forward(mt.JtA_reduced, dt, &prev_activation, weight);
        } else {
            prev_activations.back() = mMuscleNN->unnormalized_no_grad_forward(mt.JtA_reduced, dt, nullptr, 1.0);
        }
    }

    // Sum all activations
    Eigen::VectorXf activations = Eigen::VectorXf::Zero(numMuscles);
    for (const Eigen::VectorXf& a : prev_activations) {
        activations += a;
    }

    // Apply activation filter
    if (mMuscleNN) {
        activations = mMuscleNN->forward_filter(activations);
    }

    // Training data sampling (1/numSubSteps chance per call)
    if (thread_safe_uniform(0.0, 1.0) < 1.0 / static_cast<double>(mNumSubSteps) || !mTupleFilled) {
        mRandomDesiredTorque = dt;
        if (mUseCascading && !mChildNetworks.empty()) {
            Eigen::VectorXf prev_activation = Eigen::VectorXf::Zero(numMuscles);
            for (int k : mChildNetworks.back()) {
                prev_activation += prev_activations[k];
            }
            mRandomPrevOut = prev_activation.cast<double>();
            mRandomWeight = mWeights.empty() ? 1.0 : mWeights.back();
        }
        mTupleFilled = true;
    }

    return activations;
}
