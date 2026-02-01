#include "RenderEnvironment.h"
#include "Log.h"
#include <Eigen/Geometry>
#include <cmath>

RenderEnvironment::RenderEnvironment(const std::string& filepath, CBufferData<double>* graph_data)
    : Environment(filepath), mGraphData(graph_data),
      mVelocityX_Estimator(1000), mVelocityZ_Estimator(1000),
      mXZ_Regression(1000), mVelocityMethod(0)
{
    reset();
}

RenderEnvironment::~RenderEnvironment()
{
}

void RenderEnvironment::step() {
    preStep();

    for (int i = 0; i < getNumSubSteps(); i++) {
        muscleStep();
        RecordGraphData();

        // Call registered callback if set (for per-substep data collection)
        if (mOnSubStepCallback) {
            mOnSubStepCallback();
        }
    }
    postStep();

    // Record info data after step completion
    RecordInfoData();
}

void RenderEnvironment::RecordInfoData() {
    // Get info map from environment and log to graph data
    const std::map<std::string, double>& infoMap = getInfoMap();
    for (const auto& pair : infoMap) {
        if (mGraphData->key_exists(pair.first)) {
            mGraphData->push(pair.first, pair.second);
        }
    }
}

void RenderEnvironment::RecordGraphData() {
    // Extract contact and GRF data
    Eigen::Vector2i contact = getIsContact();
    Eigen::Vector2d grf = getFootGRF();

    // Log contact data
    if (mGraphData->key_exists("contact_left"))
        mGraphData->push("contact_left", static_cast<double>(contact[0]));
    if (mGraphData->key_exists("contact_right"))
        mGraphData->push("contact_right", static_cast<double>(contact[1]));
    // NOTE: contact_phaseR stores raw contact state, not GaitPhase state transitions
    if (mGraphData->key_exists("contact_phaseR"))
        mGraphData->push("contact_phaseR", static_cast<double>(contact[1]));

    // Log GRF data
    if (mGraphData->key_exists("grf_left"))
        mGraphData->push("grf_left", grf[0]);
    if (mGraphData->key_exists("grf_right"))
        mGraphData->push("grf_right", grf[1]);

    // Log kinematic data
    auto skel = getCharacter()->getSkeleton();
    
    if (mGraphData->key_exists("sway_Foot_Rx")) {
        const double root_x = skel->getRootBodyNode()->getCOM()[0];
        const double footRx = skel->getBodyNode("TalusR")->getCOM()[0] - root_x;
        mGraphData->push("sway_Foot_Rx", footRx);
    }

    if (mGraphData->key_exists("sway_Foot_Lx")) {
        const double root_x = skel->getRootBodyNode()->getCOM()[0];
        const double footLx = skel->getBodyNode("TalusL")->getCOM()[0] - root_x;
        mGraphData->push("sway_Foot_Lx", footLx);
    }

    if (mGraphData->key_exists("sway_Toe_Ry")) {
        const double toeRy = skel->getBodyNode("FootThumbR")->getCOM()[1];
        mGraphData->push("sway_Toe_Ry", toeRy);
    }

    if (mGraphData->key_exists("sway_Toe_Ly")) {
        const double toeLy = skel->getBodyNode("FootThumbL")->getCOM()[1];
        mGraphData->push("sway_Toe_Ly", toeLy);
    }

    if (mGraphData->key_exists("sway_FPAr")) {
        const Eigen::Isometry3d& footTransform = skel->getBodyNode("TalusR")->getWorldTransform();
        Eigen::Vector3d footForward = footTransform.linear() * Eigen::Vector3d::UnitZ();
        Eigen::Vector2d footForwardXZ(footForward[0], footForward[2]);

        double fpAr = 0.0;
        const double normXZ = footForwardXZ.norm();
        if (normXZ > 1e-8) {
            footForwardXZ /= normXZ;
            fpAr = std::atan2(footForwardXZ[0], footForwardXZ[1]) * 180.0 / M_PI;
        }

        // gaitnet_narrow_model.xml has reversed direction in right foot
        if (fpAr > 90.0) fpAr -= 180.0;
        else if (fpAr < -90.0) fpAr += 180.0;

        mGraphData->push("sway_FPAr", fpAr);

        if (mGraphData->key_exists("sway_AnteversionR")) {
            const double jointIRR = (skel->getJoint("FemurR")->getPosition(1) + skel->getJoint("TalusR")->getPosition(1)) * 180.0 / M_PI;
            const double anteversionR = jointIRR - fpAr;
            mGraphData->push("sway_AnteversionR", anteversionR);
        }
    }

    if (mGraphData->key_exists("sway_FPAl")) {
        const Eigen::Isometry3d& footTransform = skel->getBodyNode("TalusL")->getWorldTransform();
        Eigen::Vector3d footForward = footTransform.linear() * Eigen::Vector3d::UnitZ();
        Eigen::Vector2d footForwardXZ(footForward[0], footForward[2]);

        double fpAl = 0.0;
        const double normXZ = footForwardXZ.norm();
        if (normXZ > 1e-8) {
            footForwardXZ /= normXZ;
            fpAl = std::atan2(footForwardXZ[0], footForwardXZ[1]) * 180.0 / M_PI;
        }
        mGraphData->push("sway_FPAl", fpAl);
        if (mGraphData->key_exists("sway_AnteversionL")) {
            const double jointIRL = (skel->getJoint("FemurL")->getPosition(1) + skel->getJoint("TalusL")->getPosition(1)) * 180.0 / M_PI;
            const double anteversionL = jointIRL - fpAl;
            mGraphData->push("sway_AnteversionL", anteversionL);
        }
    }

        

    if (mGraphData->key_exists("sway_Torso_X")) {
        const double root_x = skel->getRootBodyNode()->getCOM()[0];
        const double torsoX = skel->getBodyNode("Torso")->getCOM()[0] - root_x;
        mGraphData->push("sway_Torso_X", torsoX);
    }
    
    if (mGraphData->key_exists("angle_HipR")) {
        const double angleHipR = skel->getJoint("FemurR")->getPosition(0) * 180.0 / M_PI;
        mGraphData->push("angle_HipR", -angleHipR);
    }
    
    if (mGraphData->key_exists("angle_HipIRR")) {
        const double angleHipIRR = skel->getJoint("FemurR")->getPosition(1) * 180.0 / M_PI;
        mGraphData->push("angle_HipIRR", angleHipIRR);
    }
    
    if (mGraphData->key_exists("angle_HipAbR")) {
        const double angleHipAbR = skel->getJoint("FemurR")->getPosition(2) * 180.0 / M_PI;
        mGraphData->push("angle_HipAbR", angleHipAbR);
    }
    
    if (mGraphData->key_exists("angle_KneeR")) {
        const double angleKneeR = skel->getJoint("TibiaR")->getPosition(0) * 180.0 / M_PI;
        mGraphData->push("angle_KneeR", angleKneeR);
    }
    
    if (mGraphData->key_exists("angle_AnkleR")) {
        const double angleAnkleR = skel->getJoint("TalusR")->getPosition(0) * 180.0 / M_PI;
        mGraphData->push("angle_AnkleR", -angleAnkleR);
    }
    
    if (mGraphData->key_exists("angle_Rotation")) {
        const double angleRotation = skel->getJoint("Pelvis")->getPosition(1) * 180.0 / M_PI;
        mGraphData->push("angle_Rotation", angleRotation);
    }
    
    if (mGraphData->key_exists("angle_Obliquity")) {
        const double angleObliquity = skel->getJoint("Pelvis")->getPosition(2) * 180.0 / M_PI;
        mGraphData->push("angle_Obliquity", angleObliquity);
    }
    
    if (mGraphData->key_exists("angle_Tilt")) {
        const double angleTilt = skel->getJoint("Pelvis")->getPosition(0) * 180.0 / M_PI;
        mGraphData->push("angle_Tilt", angleTilt);
    }

    const auto character = getCharacter();
    // Log metabolic energy
    if (mGraphData->key_exists("energy_metabolic_step")) {
        const double metabolicStepEnergy = character->getMetabolicStepEnergy();
        mGraphData->push("energy_metabolic_step", metabolicStepEnergy);
    }
    if (mGraphData->key_exists("energy_metabolic")) {
        const double metabolicEnergy = character->getMetabolicEnergy();
        mGraphData->push("energy_metabolic", metabolicEnergy);
    }
    // Log torque energy
    if (mGraphData->key_exists("energy_torque_step")) {
        const double torqueStepEnergy = character->getTorqueStepEnergy();
        mGraphData->push("energy_torque_step", torqueStepEnergy);
    }
    if (mGraphData->key_exists("energy_torque")) {
        const double torqueEnergy = character->getTorqueEnergy();
        mGraphData->push("energy_torque", torqueEnergy);
    }
    // Log combined energy
    if (mGraphData->key_exists("energy_combined")) {
        const double combinedEnergy = character->getEnergy();
        mGraphData->push("energy_combined", combinedEnergy);
    }

    // Log knee loading (max value)
    if (mGraphData->key_exists("knee_loading_max")) {
        const double kneeLoading = character->getKneeLoadingMax();
        mGraphData->push("knee_loading_max", kneeLoading);
    }

    // Log local phase
    if (mGraphData->key_exists("local_phase")) mGraphData->push("local_phase", getGaitPhase()->getAdaptivePhase());

    // Log COM position
    Eigen::Vector3d com = skel->getCOM();
    // Eigen::Vector3d com = skel->getBodyNode("Head")->getCOM();
    if (mGraphData->key_exists("com_x")) mGraphData->push("com_x", com[0]);
    if (mGraphData->key_exists("com_z")) mGraphData->push("com_z", com[2]);
    // Calculate COM velocity based on selected method
    double vel_x = 0.0, vel_z = 0.0;

    if (mVelocityMethod == 0) {
        // Method 0: Least Squares Regression
        double globalTime = getSimTime();
        mVelocityX_Estimator.update(globalTime, com[0]);
        mVelocityZ_Estimator.update(globalTime, com[2]);
        vel_x = mVelocityX_Estimator.getSlope();
        vel_z = mVelocityZ_Estimator.getSlope();
    } else {
        // Method 1: Average over Horizon Steps
        Eigen::Vector3d avgVel = getAvgVelocity();
        vel_x = avgVel[0];
        vel_z = avgVel[2];
    }

    // Log COM velocity
    if (mGraphData->key_exists("com_vel_x")) mGraphData->push("com_vel_x", vel_x);
    if (mGraphData->key_exists("com_vel_z")) mGraphData->push("com_vel_z", vel_z);

    // Calculate X-Z regression (lateral deviation): X = f(Z)
    mXZ_Regression.update(com[2], com[0]);  // Z as independent, X as dependent

    // Calculate mean regression error over all buffered data
    double mean_error = 0.0;
    const std::vector<Eigen::Vector3d>& comLogs = getCharacter()->getCOMLogs();

    if (mXZ_Regression.size() >= 2 && !comLogs.empty()) {
        double slope = mXZ_Regression.getSlope();
        double intercept = mXZ_Regression.getIntercept();

        // Calculate error for each point in the buffer
        size_t buffer_size = std::min(static_cast<size_t>(mXZ_Regression.size()), comLogs.size());
        for (size_t i = comLogs.size() - buffer_size; i < comLogs.size(); i++) {
            double z = comLogs[i][2];
            double x = comLogs[i][0];
            double predicted_x = slope * z + intercept;
            mean_error += std::abs(x - predicted_x);
        }
        mean_error /= buffer_size;
    }

    // Log mean regression error
    if (mGraphData->key_exists("com_deviation")) mGraphData->push("com_deviation", mean_error * 1000.0);

    // Log joint loading (joint constraint forces) for hip, knee, and ankle
    std::vector<std::pair<std::string, std::string>> joints = {
        {"hip", "FemurR"},
        {"knee", "TibiaR"},
        {"ankle", "TalusR"}
    };

    for (const auto& joint_pair : joints) {
        const std::string& prefix = joint_pair.first;
        const std::string& joint_name = joint_pair.second;

        auto joint = skel->getJoint(joint_name);
        if (joint) {
            Eigen::Vector6d wrench = joint->getWrenchToChildBodyNode();

            // Extract components: [torque_x, torque_y, torque_z, force_x, force_y, force_z]
            double tx = wrench[0], ty = wrench[1], tz = wrench[2];
            double fx = wrench[3] / 1000.0, fy = wrench[4] / 1000.0, fz = wrench[5] / 1000.0;

            // Calculate magnitudes
            double force_mag = std::sqrt(fx*fx + fy*fy + fz*fz);
            double torque_mag = std::sqrt(tx*tx + ty*ty + tz*tz);

            // Log individual components
            if (mGraphData->key_exists(prefix + "_force_x")) mGraphData->push(prefix + "_force_x", fx);
            if (mGraphData->key_exists(prefix + "_force_y")) mGraphData->push(prefix + "_force_y", fy);
            if (mGraphData->key_exists(prefix + "_force_z")) mGraphData->push(prefix + "_force_z", fz);
            if (mGraphData->key_exists(prefix + "_torque_x")) mGraphData->push(prefix + "_torque_x", tx);
            if (mGraphData->key_exists(prefix + "_torque_y")) mGraphData->push(prefix + "_torque_y", ty);
            if (mGraphData->key_exists(prefix + "_torque_z")) mGraphData->push(prefix + "_torque_z", tz);
            if (mGraphData->key_exists(prefix + "_force_mag")) mGraphData->push(prefix + "_force_mag", force_mag);
            if (mGraphData->key_exists(prefix + "_torque_mag")) mGraphData->push(prefix + "_torque_mag", torque_mag);
        }
    }

    // Record muscle metrics
    for (const auto& muscle : character->getMuscles()) {
        const auto& muscle_name = muscle->GetName();
        // Activation
        if (mGraphData->key_exists("act_" + muscle_name)) {
            mGraphData->push("act_" + muscle_name, muscle->GetActivation());
        }
        // Passive force
        if (mGraphData->key_exists("fp_" + muscle_name)) {
            mGraphData->push("fp_" + muscle_name, muscle->Getf_p());
        }
        // Active force
        if (mGraphData->key_exists("fa_" + muscle_name)) {
            mGraphData->push("fa_" + muscle_name, muscle->GetActiveForce());
        }
        // Total force
        if (mGraphData->key_exists("ft_" + muscle_name)) {
            mGraphData->push("ft_" + muscle_name, muscle->GetForce());
        }
        // Normalized muscle length
        if (mGraphData->key_exists("lm_" + muscle_name)) {
            mGraphData->push("lm_" + muscle_name, muscle->GetLmNorm());
        }
    }

    // Log torque per DOF (excluding root joint)
    const Eigen::VectorXd& torques = getLastDesiredTorque();
    std::vector<std::string> axisSuffixes = {"_x", "_y", "_z"};
    for (size_t i = 1; i < skel->getNumJoints(); ++i) {  // Skip root (i=0)
        auto joint = skel->getJoint(i);
        std::string jointName = joint->getName();
        int numDofs = joint->getNumDofs();

        for (int d = 0; d < numDofs; ++d) {
            std::string suffix = (d < 3) ? axisSuffixes[d] : "_" + std::to_string(d);
            std::string key = "torque_" + jointName + suffix;
            if (mGraphData->key_exists(key)) {
                int dofIdx = joint->getIndexInSkeleton(d);
                mGraphData->push(key, torques[dofIdx]);
            }
        }
    }

    // Record activation noise from NoiseInjector (or 0 if disabled)
    auto muscles = character->getMuscles();
    bool noiseActive = getNoiseInjector() &&
                       getNoiseInjector()->isEnabled() &&
                       getNoiseInjector()->isActivationEnabled();

    for (size_t i = 0; i < muscles.size(); i++) {
        const auto& muscle_name = muscles[i]->GetName();
        if(muscle_name.find("R_") != std::string::npos) {
            std::string key = "noise_" + muscle_name;
            if (mGraphData->key_exists(key)) {
                double noiseValue = 0.0;
                if (noiseActive) {
                    const auto& viz = getNoiseInjector()->getVisualization();
                    const auto& activationNoises = viz.activationNoises;
                    if (i < activationNoises.size()) {
                        noiseValue = activationNoises[i];
                    }
                }
                mGraphData->push(key, noiseValue);
            }
        }
    }

    // Record gait phase metrics only on foot strikes
    if (getGaitPhase() && getGaitPhase()->isStepComplete()) {
        // Determine which foot just struck
        bool isLeftStance = getGaitPhase()->isLeftLegStance();

        // Push stride length for whichever foot just struck
        double strideLength = isLeftStance ? getGaitPhase()->getStrideLengthL() : getGaitPhase()->getStrideLengthR();
        if (mGraphData->key_exists("stride_length"))
            mGraphData->push("stride_length", strideLength);

        // Push phase total for whichever foot just struck
        double phaseTotal = isLeftStance ? getGaitPhase()->getPhaseTotalL() : getGaitPhase()->getPhaseTotalR();
        if (mGraphData->key_exists("phase_total"))
            mGraphData->push("phase_total", phaseTotal);
    }
}
