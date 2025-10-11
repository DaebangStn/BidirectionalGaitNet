#include "RenderEnvironment.h"
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

    // Record reward data after step completion
    RecordRewardData();
}

void RenderEnvironment::RecordRewardData() {
    // Get reward map from environment and log to graph data
    const std::map<std::string, double>& rewardMap = mEnv->getRewardMap();
    for (const auto& pair : rewardMap) {
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
    auto skel = mEnv->getCharacter(0)->getSkeleton();
    
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
        mGraphData->push("angle_HipIRR", -angleHipIRR);
    }
    
    if (mGraphData->key_exists("angle_HipAbR")) {
        const double angleHipAbR = skel->getJoint("FemurR")->getPosition(2) * 180.0 / M_PI;
        mGraphData->push("angle_HipAbR", -angleHipAbR);
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
}

