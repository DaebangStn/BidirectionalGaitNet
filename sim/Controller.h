#ifndef __MS_CONTROLLER_H__
#define __MS_CONTROLLER_H__

#include <Eigen/Dense>
#include "dart/dart.hpp"
#include "MuscleNN.h"

// Forward declaration
class Character;
struct MuscleTuple;

/**
 * Actuator types for the controller.
 * Note: 'mus' has been removed - muscle simulation uses 'mass' or 'mass_lower'.
 */
enum ActuatorType
{
    tor,        // Direct torque control
    pd,         // PD (SPD) control
    mass,       // Muscle simulation with full-body SPD reference
    mass_lower  // Muscle simulation for lower body, PD for upper body
};

/**
 * Parse actuator type from string.
 * @param type String representation ("torque", "pd", "muscle", "mass", "mass_lower")
 * @return Corresponding ActuatorType enum value
 * @throws std::runtime_error if type is invalid
 */
ActuatorType getActuatorType(std::string type);

/**
 * Input struct for Controller::step()
 * Contains all information needed for one control step.
 * Controller handles all mirroring internally when isMirror=true.
 */
struct ControllerInput
{
    Eigen::VectorXd pdTarget;       // Target pose for PD control (in actual frame)
    MuscleTuple* muscleTuple;       // Muscle Jacobians (mirrored if isMirror=true)
    bool isMirror;                  // Whether mirroring is active
    Eigen::VectorXd torque;         // Pre-set torque (for tor mode)
    bool includeJtPinSPD;           // Whether to include JtP in SPD computation

    ControllerInput() : muscleTuple(nullptr), isMirror(false), includeJtPinSPD(false) {}
};

/**
 * Output struct from Controller::step()
 * Contains computed forces/activations to apply to the character.
 */
struct ControllerOutput
{
    Eigen::VectorXd torque;         // Torque to apply (includes upper body for mass_lower)
    Eigen::VectorXf activations;    // Muscle activations (mass/mass_lower)

    ControllerOutput() {}
};

/**
 * Configuration struct for Controller initialization.
 * Consolidates all setup parameters into a single struct.
 */
struct ControllerConfig
{
    dart::dynamics::SkeletonPtr skeleton;
    Eigen::VectorXd kp;
    Eigen::VectorXd kv;
    ActuatorType actuatorType = mass;
    Eigen::VectorXd maxTorque;
    int inferencePerSim = 1;
    int upperBodyStart = 24;        // rootDof(6) + lowerBodyDof(18)
    bool scaleTauOnWeight = false;
    double torqueMassRatio = 1.0;
    int numSubSteps = 1;
};

/**
 * Controller class - owns control logic (SPD, MuscleNN).
 *
 * Clean separation of concerns:
 * - Environment (RL interface) - handles state/action/reward/reset
 * - Controller (control computation) - computes forces from policy output
 * - Character (physics entity) - applies forces, tracks metrics
 *
 * The Controller holds a skeleton reference for efficient state access,
 * but does NOT own the Character. Only Environment applies outputs to Character.
 */
class Controller
{
public:
    /**
     * Construct Controller with configuration.
     * @param config ControllerConfig containing all setup parameters
     */
    explicit Controller(const ControllerConfig& config);
    ~Controller();

    /**
     * Main control step - computes outputs based on inputs.
     * For mass/mass_lower modes, computes SPD only. Environment must:
     * 1. Mirror the cached SPD torque
     * 2. Call computeActivationsFromTorque() with mirrored torque
     * 3. Mirror the resulting activations
     * @param input ControllerInput containing all necessary data
     * @return ControllerOutput with torque and/or activations (activations empty for mass modes)
     */
    ControllerOutput step(const ControllerInput& input);

    /**
     * Compute SPD (Stable PD) forces.
     * @param pdTarget Target pose
     * @param ext External forces (e.g., JtP from muscles)
     * @return Computed torque vector
     */
    Eigen::VectorXd computeSPDForces(const Eigen::VectorXd& pdTarget,
                                      const Eigen::VectorXd& ext = Eigen::VectorXd());

    /**
     * Compute muscle activations from desired torque and muscle tuple.
     * Called by Environment after mirroring the SPD torque.
     * @param mt Muscle tuple with Jacobians (already mirrored if needed)
     * @param desiredTorque Desired torque (mirrored by Environment if needed)
     * @param includeJtPinSPD Whether JtP was included in SPD calculation
     * @return Muscle activations (in mirrored frame, Environment must mirror back)
     */
    Eigen::VectorXf computeActivationsFromTorque(const MuscleTuple& mt,
                                                  const Eigen::VectorXd& desiredTorque,
                                                  bool includeJtPinSPD);

    // === Accessors ===

    /** Get the last computed SPD torque (cached for visualization) */
    const Eigen::VectorXd& getCachedSPDTorque() const { return mCachedSPDTorque; }

    /** Get current actuator type */
    ActuatorType getActuatorType() const { return mActuatorType; }

    /** Set actuator type */
    void setActuatorType(ActuatorType type) { mActuatorType = type; }

    // === MuscleNN management ===

    /** Set the muscle neural network */
    void setMuscleNN(MuscleNN nn) { mMuscleNN = nn; mLoadedMuscleNN = true; }

    /** Get reference to the muscle neural network */
    MuscleNN& getMuscleNN() { return mMuscleNN; }

    /** Check if MuscleNN has been loaded */
    bool hasLoadedMuscleNN() const { return mLoadedMuscleNN; }

    // === Cascading network management ===

    /** Add a previous network for hierarchical control */
    void addPrevNetwork(MuscleNN nn) { mPrevNetworks.push_back(nn); }

    /** Get previous networks */
    std::vector<MuscleNN>& getPrevNetworks() { return mPrevNetworks; }

    /** Set child network indices for hierarchical control */
    void setChildNetworks(const std::vector<std::vector<int>>& children) { mChildNetworks = children; }

    /** Get child networks */
    std::vector<std::vector<int>>& getChildNetworks() { return mChildNetworks; }

    /** Set use weights flags */
    void setUseWeights(const std::vector<bool>& flags) { mUseWeights = flags; }

    /** Get use weights flags */
    std::vector<bool>& getUseWeights() { return mUseWeights; }

    /** Set weights for hierarchical control */
    void setWeights(const std::vector<double>& weights) { mWeights = weights; }

    /** Get weights */
    std::vector<double>& getWeights() { return mWeights; }

    // === Mirror Character ===

    /**
     * Set Character pointer for mirroring operations.
     * Required for mass/mass_lower modes when isMirror=true.
     */
    void setMirrorCharacter(Character* character) {
        mMirrorCharacter = character;
    }

    /** Check if mirror character is set */
    bool hasMirrorCharacter() const { return mMirrorCharacter != nullptr; }

    // === Configuration ===

    /** Enable/disable upper body torque scaling based on mass */
    void setScaleTauOnWeight(bool scale, double ratio = 1.0) {
        mScaleTauOnWeight = scale;
        mTorqueMassRatio = ratio;
    }

    /** Get torque mass ratio */
    double getTorqueMassRatio() const { return mTorqueMassRatio; }

    /** Set upper body start DOF index */
    void setUpperBodyStart(int start) { mUpperBodyStart = start; }

    /** Get upper body start DOF index */
    int getUpperBodyStart() const { return mUpperBodyStart; }

    /** Set max torque for SPD clipping */
    void setMaxTorque(const Eigen::VectorXd& maxTorque) { mMaxTorque = maxTorque; }

    /** Set inference per sim ratio (for time step scaling) */
    void setInferencePerSim(int inferencePerSim) { mInferencePerSim = inferencePerSim; }

    // === Training data sampling ===

    /** Get randomly sampled desired torque (for muscle training) */
    const Eigen::VectorXd& getRandomDesiredTorque() const { return mRandomDesiredTorque; }

    /** Get randomly sampled previous output (for cascading training) */
    const Eigen::VectorXd& getRandomPrevOut() const { return mRandomPrevOut; }

    /** Get random weight for cascading */
    double getRandomWeight() const { return mRandomWeight; }

    /** Check if training tuple has been filled this step */
    bool isTupleFilled() const { return mTupleFilled; }

    /** Reset tuple filled flag */
    void resetTupleFilled() { mTupleFilled = false; }

    /** Set number of substeps (for training data sampling) */
    void setNumSubSteps(int numSubSteps) { mNumSubSteps = numSubSteps; }

    /** Set using cascading mode */
    void setUseCascading(bool useCascading) { mUseCascading = useCascading; }

    /** Check if using cascading */
    bool getUseCascading() const { return mUseCascading; }

private:
    // === Skeleton reference (for efficient state access) ===
    dart::dynamics::SkeletonPtr mSkeleton;

    // === Control configuration ===
    ActuatorType mActuatorType = mass;

    // PD gains
    Eigen::VectorXd mKp;
    Eigen::VectorXd mKv;

    // SPD torque clipping
    Eigen::VectorXd mMaxTorque;

    // Inference ratio
    int mInferencePerSim = 1;

    // === Cached SPD result ===
    Eigen::VectorXd mCachedSPDTorque;

    // === MuscleNN ===
    MuscleNN mMuscleNN;
    bool mLoadedMuscleNN = false;

    // === Cascading networks ===
    std::vector<MuscleNN> mPrevNetworks;
    std::vector<std::vector<int>> mChildNetworks;
    std::vector<bool> mUseWeights;
    std::vector<double> mWeights;
    bool mUseCascading = false;

    // === Upper body config (for mass_lower) ===
    int mUpperBodyStart = 24;  // rootDof(6) + lowerBodyDof(18)
    bool mScaleTauOnWeight = false;
    double mTorqueMassRatio = 1.0;

    // === Training data sampling ===
    int mNumSubSteps = 1;
    bool mTupleFilled = false;
    Eigen::VectorXd mRandomDesiredTorque;
    Eigen::VectorXd mRandomPrevOut;
    double mRandomWeight = 1.0;

    // === Mirror character ===
    Character* mMirrorCharacter = nullptr;
};

#endif
