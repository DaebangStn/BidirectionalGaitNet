#include "RenderEnvironment.h"
#include <Eigen/Geometry>
#include <cmath>

RenderEnvironment::RenderEnvironment(std::string metadata, CBufferData<double>* graph_data)
    : mGraphData(graph_data) 
{
    mEnv = new Environment();
    mEnv->initialize(metadata);
    mEnv->reset();
}

RenderEnvironment::~RenderEnvironment() 
{
    delete mEnv;
}

void RenderEnvironment::step() {

    for (int i = 0; i < mEnv->getNumSubSteps(); i++) {
        mEnv->muscleStep();
        RecordGraphData();
    }
    mEnv->postStep();

    // Record info data after step completion
    RecordInfoData();
}

void RenderEnvironment::RecordInfoData() {
    // Get info map from environment and log to graph data
    const std::map<std::string, double>& infoMap = mEnv->getInfoMap();
    for (const auto& pair : infoMap) {
        if (mGraphData->key_exists(pair.first)) {
            mGraphData->push(pair.first, pair.second);
        }
    }
}

void RenderEnvironment::RecordGraphData() {
    // Extract contact and GRF data
    Eigen::Vector2i contact = mEnv->getIsContact();
    Eigen::Vector2d grf = mEnv->getFootGRF();
    
    // Log contact data
    if (mGraphData->key_exists("contact_left"))
        mGraphData->push("contact_left", static_cast<double>(contact[0]));
    if (mGraphData->key_exists("contact_right"))
        mGraphData->push("contact_right", static_cast<double>(contact[1]));
    if (mGraphData->key_exists("contact_phaseR"))
        mGraphData->push("contact_phaseR", static_cast<double>(contact[1]));
    
    // Log GRF data
    if (mGraphData->key_exists("grf_left"))
        mGraphData->push("grf_left", grf[0]);
    if (mGraphData->key_exists("grf_right"))
        mGraphData->push("grf_right", grf[1]);
    
    // Log kinematic data
    auto skel = mEnv->getCharacter()->getSkeleton();
    
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

    const auto character = mEnv->getCharacter();
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

    // Record muscle activations
    for (const auto& muscle : character->getMuscles()) {
        const auto& muscle_name = muscle->GetName();
        if(muscle_name.find("R_") != std::string::npos) {
            std::string key = "act_" + muscle_name;
            if (mGraphData->key_exists(key)) {
                mGraphData->push(key, muscle->GetActivation());
            }
        }
    }

    // Record activation noise from NoiseInjector (or 0 if disabled)
    auto muscles = character->getMuscles();
    bool noiseActive = mEnv->getNoiseInjector() &&
                       mEnv->getNoiseInjector()->isEnabled() &&
                       mEnv->getNoiseInjector()->isActivationEnabled();

    for (size_t i = 0; i < muscles.size(); i++) {
        const auto& muscle_name = muscles[i]->GetName();
        if(muscle_name.find("R_") != std::string::npos) {
            std::string key = "noise_" + muscle_name;
            if (mGraphData->key_exists(key)) {
                double noiseValue = 0.0;
                if (noiseActive) {
                    const auto& viz = mEnv->getNoiseInjector()->getVisualization();
                    const auto& activationNoises = viz.activationNoises;
                    if (i < activationNoises.size()) {
                        noiseValue = activationNoises[i];
                    }
                }
                mGraphData->push(key, noiseValue);
            }
        }
    }
}
