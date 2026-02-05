#ifndef __MS_CHARACTER_H__
#define __MS_CHARACTER_H__
#include "dart/dart.hpp"
#include "DARTHelper.h"
#include "Muscle.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

struct ModifyInfo
{
    ModifyInfo() : ModifyInfo(1.0, 1.0, 1.0, 1.0, 0.0) {}
    ModifyInfo(double lx, double ly, double lz, double sc, double to)
    {
        value[0] = lx; // length x (ratio)
        value[1] = ly; // length y (ratio)
        value[2] = lz; // length z (ratio)
        value[3] = sc; // scale (ration)
        value[4] = to; // torsion (angle)
    };
    double value[5];
    double &operator[](int idx) { return value[idx]; }
    double operator[](int idx) const { return value[idx]; }
};
using BoneInfo = std::tuple<std::string, ModifyInfo>;

// Note: ActuatorType enum has been moved to Controller.h

enum MetabolicType
{
    LEGACY,  // No metabolic computation (backward compatible)
    A,       // abs(activation)
    A2,      // activation^2
    MA,      // mass * abs(activation)
    MA2      // mass * activation^2
};

struct MuscleTuple
{
    // Eigen::VectorXd dt;
    Eigen::VectorXd JtA_reduced;
    Eigen::VectorXd JtP;
    Eigen::MatrixXd JtA;
};
class Character
{
public:
    Character(std::string path, int skelFlags = SKEL_DEFAULT);
    ~Character();

    dart::dynamics::SkeletonPtr getSkeleton() { return mSkeleton; }
    std::map<std::string, std::vector<std::string>> getBVHMap() { return mBVHMap; }

    // Skeleton metadata accessors (for export)
    const std::map<std::string, std::string>& getContactFlags() const { return mContactFlags; }
    const std::map<std::string, std::string>& getObjFileLabels() const { return mObjFileLabels; }

    // For Mirror Function

    std::vector<std::pair<Joint *, Joint *>> getPairs() { return mPairs; }
    Eigen::VectorXd getMirrorPosition(Eigen::VectorXd pos);

    void setPDTarget(Eigen::VectorXd _pdtarget) { mPDTarget = _pdtarget; }
    Eigen::VectorXd getPDTarget() { return mPDTarget; }

    Eigen::VectorXd getUpperBodyTorque() const { return mUpperBodyTorque; }
    int getUpperBodyDim() const { return mUpperBodyDim; }
    void setZeroForces();
    void setActivations(Eigen::VectorXd _activation);
    Eigen::VectorXd getActivations() { return mActivations; }

    // Force application methods (actuator-agnostic)
    void applyTorque(const Eigen::VectorXd& torque);
    void addTorque(const Eigen::VectorXd& torque);
    void applyMuscleForces();

    // Simulation step (metabolic tracking only)
    void step();

    std::vector<dart::dynamics::BodyNode *> getEndEffectors() { return mEndEffectors; }
    const Eigen::VectorXd& getKpVector() const { return mKp; }
    const Eigen::VectorXd& getKvVector() const { return mKv; }

    Eigen::VectorXd heightCalibration(dart::simulation::WorldPtr _world);
    std::vector<Eigen::Matrix3d> getBodyNodeTransform() { return mBodyNodeTransform; }
    void setMuscles(std::string path, bool meshLbsWeight = false);
    void setMusclesXML(std::string path, bool meshLbsWeight);
    void setMusclesYAML(std::string path);
    void setSortMuscleLogging(bool enable) { mSortMuscleLogs = enable; }
    void clearMuscles();
    const std::vector<Muscle *> &getMuscles() { return mMuscles; }
    Muscle* getMuscleByName(const std::string& name) const;

    void clearLogs();

    const std::vector<Eigen::Vector3d> &getCOMLogs() { return mCOMLogs; }
    const std::vector<Eigen::Vector3d> &getHeadVelLogs() { return mHeadVelLogs; }
    const std::vector<Eigen::VectorXd> &getMuscleTorqueLogs() { return mMuscleTorqueLogs; }

    Eigen::VectorXd getMirrorActivation(Eigen::VectorXd _activation);

    // Metabolic Energy Methods
    void cacheMuscleMass();
    void setMetabolicType(MetabolicType type) { mMetabolicType = type; }
    MetabolicType getMetabolicType() const { return mMetabolicType; }
    double getMetabolicEnergy() const { return mMetabolicEnergy; }
    double getMetabolicStepEnergy() const { return mMetabolicStepEnergy; }

    // Torque Energy Methods (separate from metabolic)
    void setTorqueEnergyCoeff(double coeff) { mTorqueEnergyCoeff = coeff; }
    double getTorqueEnergyCoeff() const { return mTorqueEnergyCoeff; }
    double getTorqueEnergy() const { return mTorqueEnergy; }
    double getTorqueStepEnergy() const { return mTorqueStepEnergy; }

    // Knee Loading Methods
    double getKneeLoading() const { return mKneeLoading; }
    double getKneeLoadingStep() const { return mKneeLoadingStep; }
    double getKneeLoadingMax() const { return mKneeLoadingMax; }

    // Combined Step-Based Metrics Methods (evaluates metabolic, torque, and knee loading)
    void evalStep();        // Evaluates all step-based metrics
    void resetStep();       // Resets all step-based metrics
    void resetKneeLoadingMax() { mKneeLoadingMax = 0.0; mStepComplete = true; }  // Reset max for new gait cycle
    double getEnergy() const { return mMetabolicEnergy + mTorqueEnergy; }

    // Muscle Parameter Modification
    void setMuscleParam(const std::string& muscleName, const std::string& paramType, double value);

    // MuscleTuple getMuscleTuple(Eigen::VectorXd dt, bool isMirror = false);
    const MuscleTuple& getMuscleTuple(bool isMirror = false);

    // Cache invalidation - call when state changes that affects muscle Jacobians
    void invalidateMuscleTuple();

    int getNumMuscles() { return mMuscles.size(); }
    int getNumMuscleRelatedDof() { return mNumMuscleRelatedDof; }

    Eigen::VectorXd addPositions(Eigen::VectorXd pos1, Eigen::VectorXd pos2, bool includeRoot = true);

    // Motion Interpolation
    Eigen::VectorXd interpolatePose(const Eigen::VectorXd& pose1,
                                    const Eigen::VectorXd& pose2,
                                    double t,
                                    bool extrapolate_root = false);

    // Body Parameter
    double getSkelParamValue(std::string name);
    double getTorsionValue(std::string name);
    void setSkelParam(std::vector<std::pair<std::string, double>> _skel_info);
    void applySkeletonLength(const std::vector<BoneInfo> &info);
    void applySkeletonBodyNode(const std::vector<BoneInfo> &info, dart::dynamics::SkeletonPtr skel);

    // For Drawing
    void updateRefSkelParam(dart::dynamics::SkeletonPtr skel) { applySkeletonBodyNode(mSkelInfos, skel); }
    double getGlobalRatio() { return mGlobalRatio; }
    void setBodyMass(double targetMass);
    void setTorqueClipping(bool _torqueClipping) { mTorqueClipping = _torqueClipping; }
    bool getTorqueClipping() { return mTorqueClipping; }

    void setIncludeJtPinSPD(bool _includeJtPinSPD) { mIncludeJtPinSPD = _includeJtPinSPD; }
    bool getIncludeJtPinSPD() { return mIncludeJtPinSPD; }

    // Upper body torque scaling based on body mass (stabilizes light bodies)
    void setScaleTauOnWeight(bool enable) { mScaleTauOnWeight = enable; }
    bool getScaleTauOnWeight() const { return mScaleTauOnWeight; }
    void updateTorqueMassRatio();
    double getTorqueMassRatio() const { return mTorqueMassRatio; }

    // Max torque for SPD clipping (used by Controller)
    const Eigen::VectorXd& getMaxTorque() const { return mMaxTorque; }

    // Muscle force scaling based on body mass (f0 scales with mass^(2/3))
    void setScaleF0OnWeight(bool enable) { mScaleF0OnWeight = enable; }
    bool getScaleF0OnWeight() const { return mScaleF0OnWeight; }
    void updateMuscleForceRatio();

    // Clip lm_norm for passive force calculation
    void setClipLmNorm(double clip);

    // SPD torque clipping: tau <= min(M_diag * max_acc, max_torque)
    void setMaxAcc(double maxAcc) { mMaxAcc = maxAcc; updateMaxTorque(); }
    double getMaxAcc() const { return mMaxAcc; }
    void setMaxTorqueLimit(double maxTorque) { mMaxTorqueLimit = maxTorque; updateMaxTorque(); }
    double getMaxTorqueLimit() const { return mMaxTorqueLimit; }
    void updateMaxTorque();

    // Critical damping: Kv = 2 * sqrt(Kp * M_diag) for numerical stability
    void setUseCriticalDamping(bool enable) { mUseCriticalDamping = enable; }
    bool getUseCriticalDamping() const { return mUseCriticalDamping; }
    void updateCriticalDamping();

    Eigen::VectorXd posToSixDof(Eigen::VectorXd pos);
    Eigen::VectorXd sixDofToPos(Eigen::VectorXd raw_pos);

private:
    double calculateKneeLoadingStep(bool isLeft = false);
    bool mTorqueClipping;

    void parseSkeletonMetadataFromYAML(const std::string& resolvedPath);
    void parseSkeletonMetadataFromXML(const std::string& resolvedPath);

    dart::dynamics::SkeletonPtr mSkeleton;
    dart::dynamics::SkeletonPtr mRefSkeleton;

    Eigen::VectorXd mKp;
    Eigen::VectorXd mKv;
    Eigen::VectorXd mTorqueWeight;

    std::vector<dart::dynamics::BodyNode *> mEndEffectors;
    std::map<std::string, std::vector<std::string>> mBVHMap;

    // Skeleton metadata (parsed once during construction, immutable)
    std::map<std::string, std::string> mContactFlags;   // body_node → "On"/"Off"
    std::map<std::string, std::string> mObjFileLabels;  // body_node → "mesh.obj"

    std::vector<std::pair<Joint *, Joint *>> mPairs;
    std::vector<Eigen::Matrix3d> mBodyNodeTransform;

    Eigen::VectorXd mUpperBodyTorque;  // Cached upper body torque (for visualization)
    int mUpperBodyDim = 0;             // Cached upper body DOF dimension
    Eigen::VectorXd mPDTarget;

    // Muscle
    std::vector<Muscle *> mMuscles;
    std::map<std::string, Muscle*> mMuscleNameCache;  // Fast lookup by name

    // MuscleTuple caching (lazy computation)
    MuscleTuple mCachedMuscleTuple;           // Raw (non-mirrored)
    MuscleTuple mCachedMuscleTupleMirrored;   // Mirrored version
    bool mMuscleTupleRawValid = false;        // Raw cache validity
    bool mMuscleTupleMirroredValid = false;   // Mirrored cache validity
    void computeMuscleTupleRaw();             // Compute raw tuple
    void computeMuscleTupleMirrored();        // Compute mirrored from raw

    int mNumMuscleRelatedDof;
    Eigen::VectorXd mActivations;
    bool mSortMuscleLogs;

    // Step-Based Metrics Tracking System
    double mStepDivisor;            // Unified divisor for all step-based metrics

    // Metabolic Energy Tracking
    MetabolicType mMetabolicType;
    Eigen::VectorXd mMuscleMassCache;
    double mMetabolicEnergyAccum;
    double mMetabolicEnergy;
    double mMetabolicStepEnergy;

    // Torque Energy Tracking
    double mTorqueEnergyCoeff;      // Coefficient for torque-based energy calculation
    double mTorqueEnergyAccum;      // Accumulated torque energy
    double mTorqueEnergy;           // Final torque energy value
    double mTorqueStepEnergy;       // Torque energy per step

    // Knee Loading Tracking
    double mKneeLoadingAccum;       // Accumulated knee loading
    double mKneeLoading;            // Final averaged knee loading value
    double mKneeLoadingStep;        // Knee loading per step
    double mKneeLoadingMax;         // Maximum knee loading within current step
    bool mStepComplete;             // Flag indicating if step evaluation is complete

    // Log
    std::vector<Eigen::Vector3d> mCOMLogs;
    std::vector<Eigen::Vector3d> mHeadVelLogs;
    std::vector<Eigen::VectorXd> mMuscleTorqueLogs;

    // Skeleton Parameters
    std::vector<BoneInfo> mSkelInfos;
    double mGlobalRatio;
    std::map<dart::dynamics::BodyNode *, ModifyInfo> modifyLog;
    bool mIncludeJtPinSPD;

    // Upper body torque scaling for light bodies
    bool mScaleTauOnWeight = false;
    double mTorqueMassRatio = 1.0;

    // Muscle force scaling for body mass
    bool mScaleF0OnWeight = false;

    // Target mass for preservation across skeleton parameter changes
    double mTargetMass = -1.0;  // -1 means no target mass set

    // SPD torque clipping parameters
    double mMaxAcc = -1.0;  // -1 means no clipping (mass-dependent)
    double mMaxTorqueLimit = -1.0;  // -1 means no clipping (absolute limit)
    Eigen::VectorXd mMaxTorque;  // Cached per-DOF: min(M_diag * mMaxAcc, mMaxTorqueLimit)

    // Critical damping: Kv = 2 * sqrt(Kp * M_diag)
    bool mUseCriticalDamping = false;

    static constexpr double kRefMass = 70.0;  // Reference mass for stable simulation
};
#endif
