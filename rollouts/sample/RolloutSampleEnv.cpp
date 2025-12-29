#include "RolloutSampleEnv.h"
#include "PyRolloutRecord.h"
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <iostream>

RolloutSampleEnv::RolloutSampleEnv(const std::string& metadata) {
    // Use initialize() which auto-detects XML vs YAML format
    mEnv.initialize(metadata);
    mEnv.reset();
    mUseMuscle = mEnv.isTwoLevelController();
}

RolloutSampleEnv::~RolloutSampleEnv() = default;

void RolloutSampleEnv::LoadRecordConfig(const std::string& yaml_path) {
    mRecordConfig = RecordConfig::LoadFromYAML(yaml_path);

    // Apply MetabolicType override if enabled
    if (mRecordConfig.metabolic.enabled) {
        const std::string& type = mRecordConfig.metabolic.type;
        if (type == "LEGACY") {
            mEnv.getCharacter()->setMetabolicType(LEGACY);
        } else if (type == "A") {
            mEnv.getCharacter()->setMetabolicType(A);
        } else if (type == "A2") {
            mEnv.getCharacter()->setMetabolicType(A2);
        } else if (type == "MA") {
            mEnv.getCharacter()->setMetabolicType(MA);
        } else if (type == "MA2") {
            mEnv.getCharacter()->setMetabolicType(MA2);
        } else {
            std::cerr << "Warning: Unknown MetabolicType '" << type
                      << "', using LEGACY" << std::endl;
            mEnv.getCharacter()->setMetabolicType(LEGACY);
        }
        std::cout << "MetabolicType set to: " << type << std::endl;
    }

    // Also load target cycles from YAML if present
    try {
        YAML::Node root = YAML::LoadFile(yaml_path);
        if (root["sample"] && root["sample"]["cycle"]) {
            mTargetCycles = root["sample"]["cycle"].as<int>();
        }
    } catch (const YAML::Exception& e) {
        std::cerr << "Note: Using default target cycles: " << mTargetCycles << std::endl;
    }
}

void RolloutSampleEnv::LoadPolicyWeights(py::dict state_dict) {
    // Create policy network if not exists
    if (!mPolicy) {
        int num_states = GetStateDim();
        int num_actions = GetActionDim();
        mPolicy = std::make_unique<PolicyNetImpl>(num_states, num_actions, torch::kCPU);
    }

    // Load weights from Python state_dict
    mPolicy->load_state_dict(state_dict);
    mPolicyLoaded = true;
    std::cout << "Policy weights loaded successfully" << std::endl;
}

void RolloutSampleEnv::LoadMuscleWeights(py::object weights) {
    if (!mUseMuscle) {
        std::cerr << "Warning: LoadMuscleWeights called but not using muscle network" << std::endl;
        return;
    }
    mEnv.setMuscleNetworkWeight(weights);
    std::cout << "Muscle network weights loaded successfully" << std::endl;
}

void RolloutSampleEnv::SetParameters(const std::map<std::string, double>& params) {
    if (params.empty()) {
        // Empty parameters - sample from default distribution
        mEnv.updateParamState();
        return;
    }

    // Get parameter names from environment
    const std::vector<std::string>& param_names = mEnv.getParamName();

    // Get default parameter values
    Eigen::VectorXd param_state = mEnv.getParamDefault();

    // Fill in provided parameters (overriding defaults)
    for (size_t i = 0; i < param_names.size(); ++i) {
        auto it = params.find(param_names[i]);
        if (it != params.end()) {
            param_state[i] = it->second;
        }
    }

    // Set the parameter state
    mEnv.setParamState(param_state);
}

Eigen::VectorXd RolloutSampleEnv::InterpolatePose(const Eigen::VectorXd& pose1,
                                                   const Eigen::VectorXd& pose2,
                                                   double t,
                                                   bool extrapolate_root) {
    return mEnv.getCharacter()->interpolatePose(pose1, pose2, t, extrapolate_root);
}

void RolloutSampleEnv::RecordStep(PyRolloutRecord* record) {
    std::unordered_map<std::string, float> data;

    // Basic fields (always recorded)
    data["step"] = mEnv.getSimulationCount();
    data["time"] = mEnv.getSimTime();
    data["cycle"] = mEnv.getGaitPhase()->getAdaptiveCycleCount();

    auto skel = mEnv.getCharacter()->getSkeleton();

    // Contact and GRF fields
    if (mRecordConfig.foot.enabled) {
        Eigen::Vector2i contact = mEnv.getIsContact();
        Eigen::Vector2d grf = mEnv.getFootGRF();

        if (mRecordConfig.foot.contact_left) {
            data["contact/left"] = static_cast<float>(contact[0]);
        }
        if (mRecordConfig.foot.contact_right) {
            data["contact/right"] = static_cast<float>(contact[1]);
        }
        if (mRecordConfig.foot.grf_left) {
            data["grf/left"] = grf[0];
        }
        if (mRecordConfig.foot.grf_right) {
            data["grf/right"] = grf[1];
        }
    }

    // Kinematics fields
    if (mRecordConfig.kinematics.enabled) {
        // All joint positions as a single vector
        if (mRecordConfig.kinematics.all) {
            record->addVector("motions", mEnv.getSimulationCount() - 1, skel->getPositions().cast<float>());
        }

        // Root position
        if (mRecordConfig.kinematics.root) {
            auto root_body = skel->getRootBodyNode();
            Eigen::Vector3d root_pos = root_body->getCOM();
            data["root/x"] = root_pos[0];
            data["root/y"] = root_pos[1];
            data["root/z"] = root_pos[2];
        }

        // Angle fields
        if (mRecordConfig.kinematics.angle.enabled) {
            if (mRecordConfig.kinematics.angle.hip) {
                float angle = skel->getJoint("FemurR")->getPosition(0) * 180.0 / M_PI;
                data["angle/HipR"] = -angle;
            }
            if (mRecordConfig.kinematics.angle.hip_ir) {
                float angle = skel->getJoint("FemurR")->getPosition(1) * 180.0 / M_PI;
                data["angle/HipIRR"] = -angle;
            }
            if (mRecordConfig.kinematics.angle.hip_ab) {
                float angle = skel->getJoint("FemurR")->getPosition(2) * 180.0 / M_PI;
                data["angle/HipAbR"] = -angle;
            }
            if (mRecordConfig.kinematics.angle.knee) {
                float angle = skel->getJoint("TibiaR")->getPosition(0) * 180.0 / M_PI;
                data["angle/KneeR"] = angle;
            }
            if (mRecordConfig.kinematics.angle.ankle) {
                float angle = skel->getJoint("TalusR")->getPosition(0) * 180.0 / M_PI;
                data["angle/AnkleR"] = -angle;
            }
            if (mRecordConfig.kinematics.angle.pelvic_tilt) {
                float angle = skel->getJoint("Pelvis")->getPosition(0) * 180.0 / M_PI;
                data["angle/Tilt"] = angle;
            }
            if (mRecordConfig.kinematics.angle.pelvic_rotation) {
                float angle = skel->getJoint("Pelvis")->getPosition(1) * 180.0 / M_PI;
                data["angle/Rotation"] = angle;
            }
            if (mRecordConfig.kinematics.angle.pelvic_obliquity) {
                float angle = skel->getJoint("Pelvis")->getPosition(2) * 180.0 / M_PI;
                data["angle/Obliquity"] = angle;
            }
        }

        // Angular velocity fields
        if (mRecordConfig.kinematics.anvel.enabled) {
            if (mRecordConfig.kinematics.anvel.hip) {
                data["anvel/HipR"] = skel->getJoint("FemurR")->getVelocity(0);
            }
            if (mRecordConfig.kinematics.anvel.knee) {
                data["anvel/KneeR"] = skel->getJoint("TibiaR")->getVelocity(0);
            }
            if (mRecordConfig.kinematics.anvel.ankle) {
                data["anvel/AnkleR"] = skel->getJoint("TalusR")->getVelocity(0);
            }
        }

        // Sway recording
        if (mRecordConfig.kinematics.sway.enabled) {
            const double root_x = skel->getRootBodyNode()->getCOM()[0];

            if (mRecordConfig.kinematics.sway.foot) {
                data["sway/FootR"] = skel->getBodyNode("TalusR")->getCOM()[0] - root_x;
                data["sway/FootL"] = skel->getBodyNode("TalusL")->getCOM()[0] - root_x;
            }

            if (mRecordConfig.kinematics.sway.toe) {
                data["sway/ToeR"] = skel->getBodyNode("FootThumbR")->getCOM()[1];
                data["sway/ToeL"] = skel->getBodyNode("FootThumbL")->getCOM()[1];
            }

            if (mRecordConfig.kinematics.sway.fpa || mRecordConfig.kinematics.sway.anteversion) {
                // Right FPA
                const Eigen::Isometry3d& footTransformR = skel->getBodyNode("TalusR")->getWorldTransform();
                Eigen::Vector3d footForwardR = footTransformR.linear() * Eigen::Vector3d::UnitZ();
                Eigen::Vector2d footForwardXZR(footForwardR[0], footForwardR[2]);
                double fpaR = 0.0;
                if (footForwardXZR.norm() > 1e-8) {
                    footForwardXZR.normalize();
                    fpaR = std::atan2(footForwardXZR[0], footForwardXZR[1]) * 180.0 / M_PI;
                    if (fpaR > 90.0) fpaR -= 180.0;
                    else if (fpaR < -90.0) fpaR += 180.0;
                }

                // Left FPA
                const Eigen::Isometry3d& footTransformL = skel->getBodyNode("TalusL")->getWorldTransform();
                Eigen::Vector3d footForwardL = footTransformL.linear() * Eigen::Vector3d::UnitZ();
                Eigen::Vector2d footForwardXZL(footForwardL[0], footForwardL[2]);
                double fpaL = 0.0;
                if (footForwardXZL.norm() > 1e-8) {
                    footForwardXZL.normalize();
                    fpaL = std::atan2(footForwardXZL[0], footForwardXZL[1]) * 180.0 / M_PI;
                }

                if (mRecordConfig.kinematics.sway.fpa) {
                    data["sway/FPA_R"] = fpaR;
                    data["sway/FPA_L"] = -fpaL;
                }

                if (mRecordConfig.kinematics.sway.anteversion) {
                    const double jointIRR = (skel->getJoint("FemurR")->getPosition(1) +
                                             skel->getJoint("TalusR")->getPosition(1)) * 180.0 / M_PI;
                    const double jointIRL = (skel->getJoint("FemurL")->getPosition(1) +
                                             skel->getJoint("TalusL")->getPosition(1)) * 180.0 / M_PI;
                    data["sway/AnteversionR"] = jointIRR - fpaR;
                    data["sway/AnteversionL"] = jointIRL - fpaL;
                }
            }

            if (mRecordConfig.kinematics.sway.torso) {
                data["sway/Torso"] = skel->getBodyNode("Torso")->getCOM()[0] - root_x;
            }
        }
    }

    // Metabolic energy recording
    if (mRecordConfig.metabolic.enabled) {
        float stepEnergy = mEnv.getCharacter()->getMetabolicStepEnergy();
        data["metabolic/step_energy"] = stepEnergy;
    }

    // Muscle recording (vector data -> matrix_data)
    if (mRecordConfig.muscle.enabled) {
        auto character = mEnv.getCharacter();
        const auto& muscles = character->getMuscles();
        int numMuscles = muscles.size();

        if (mRecordConfig.muscle.activation) {
            Eigen::VectorXd activations = character->getActivations();
            record->addVector("muscle/activation", mEnv.getSimulationCount() - 1,
                             activations.cast<float>());
        }

        if (mRecordConfig.muscle.passive) {
            Eigen::VectorXf passive(numMuscles);
            for (int i = 0; i < numMuscles; i++) {
                passive[i] = static_cast<float>(muscles[i]->Getf_p());
            }
            record->addVector("muscle/passive", mEnv.getSimulationCount() - 1, passive);
        }

        if (mRecordConfig.muscle.force) {
            Eigen::VectorXf force(numMuscles);
            for (int i = 0; i < numMuscles; i++) {
                force[i] = static_cast<float>(muscles[i]->GetForce());
            }
            record->addVector("muscle/force", mEnv.getSimulationCount() - 1, force);
        }

        if (mRecordConfig.muscle.lm_norm) {
            Eigen::VectorXf lm_norm(numMuscles);
            for (int i = 0; i < numMuscles; i++) {
                lm_norm[i] = static_cast<float>(muscles[i]->GetLmNorm());
            }
            record->addVector("muscle/lm_norm", mEnv.getSimulationCount() - 1, lm_norm);
        }
    }

    record->add(mEnv.getSimulationCount() - 1, data);
}

py::dict RolloutSampleEnv::CollectRollout(py::object param_dict) {
    if (!mPolicyLoaded) {
        throw std::runtime_error("Policy weights not loaded. Call load_policy_weights() first.");
    }

    // 1. Set parameters (or sample random)
    if (param_dict.is_none()) {
        mEnv.updateParamState();  // Random sampling
    } else {
        auto params = param_dict.cast<std::map<std::string, double>>();
        SetParameters(params);
    }

    // 2. Reset environment
    mEnv.reset();

    // 3. Create record
    PyRolloutRecord record(mRecordConfig);

    // 4. Run until target cycles or termination
    int current_cycle = 0;
    int step_count = 0;
    float cumulative_energy = 0.0f;
    int last_cycle = -1;

    while (current_cycle < mTargetCycles && !mEnv.isTerminated()) {
        // Get state
        Eigen::VectorXd state_d = mEnv.getState();
        Eigen::VectorXf state = state_d.cast<float>();

        // Policy inference (single forward pass)
        auto [action, value, logprob] = mPolicy->sample_action(state);

        // Step environment
        mEnv.setAction(action.cast<double>().eval());
        mEnv.preStep();

        int numSubSteps = mEnv.getNumSubSteps();
        for (int i = 0; i < numSubSteps; i++) {
            mEnv.muscleStep();
            // Clear gait cycle flag immediately after each muscleStep
            // to prevent mWorldPhaseCount from incrementing multiple times
            mEnv.clearGaitCycleComplete();
            RecordStep(&record);

            // Track cumulative energy per cycle
            if (mRecordConfig.metabolic.enabled && mRecordConfig.metabolic.cumulative) {
                cumulative_energy += mEnv.getCharacter()->getMetabolicStepEnergy();
            }
        }
        mEnv.postStep();

        // Check cycle completion
        current_cycle = mEnv.getGaitPhase()->getAdaptiveCycleCount();

        // Record cycle-level attributes when cycle changes
        if (current_cycle != last_cycle && last_cycle >= 0) {
            if (mRecordConfig.metabolic.enabled && mRecordConfig.metabolic.cumulative) {
                record.addCycleAttribute(last_cycle, "metabolic/cumulative", cumulative_energy);
                cumulative_energy = 0.0f;  // Reset for next cycle
            }
            // Add step count as cycle attribute
            record.addCycleAttribute(last_cycle, "cycle_steps", static_cast<float>(step_count));
        }
        last_cycle = current_cycle;

        step_count++;
    }

    // Record final cycle attributes
    if (last_cycle >= 0 && mRecordConfig.metabolic.enabled && mRecordConfig.metabolic.cumulative) {
        record.addCycleAttribute(last_cycle, "metabolic/cumulative", cumulative_energy);
    }

    // 5. Build return dict
    py::dict result;
    result["data"] = record.get_data_array();
    result["matrix_data"] = record.get_matrix_data();
    result["fields"] = record.get_fields();

    // Parameter state
    Eigen::VectorXd param_state = mEnv.getParamState(false);
    py::array_t<float> param_arr(param_state.size());
    auto param_buf = param_arr.mutable_unchecked<1>();
    for (int i = 0; i < param_state.size(); ++i) {
        param_buf(i) = static_cast<float>(param_state[i]);
    }
    result["param_state"] = param_arr;

    result["cycle_attributes"] = record.get_cycle_attributes();
    result["success"] = (current_cycle >= mTargetCycles);

    // Metrics
    py::dict metrics;
    metrics["steps"] = static_cast<int>(record.get_nrow());
    metrics["cycles"] = current_cycle;
    metrics["terminated"] = mEnv.isTerminated();
    result["metrics"] = metrics;

    return result;
}

int RolloutSampleEnv::GetStateDim() const {
    // Get state dimension from environment
    return const_cast<Environment&>(mEnv).getState().size();
}

int RolloutSampleEnv::GetActionDim() const {
    return const_cast<Environment&>(mEnv).getNumAction();
}

int RolloutSampleEnv::GetSkeletonDOF() const {
    return const_cast<Environment&>(mEnv).getCharacter()->getSkeleton()->getNumDofs();
}

double RolloutSampleEnv::GetMass() const {
    return const_cast<Environment&>(mEnv).getCharacter()->getSkeleton()->getMass();
}

std::vector<std::string> RolloutSampleEnv::GetParameterNames() {
    const std::vector<std::string>& param_names = mEnv.getParamName();
    return std::vector<std::string>(param_names.begin(), param_names.end());
}

std::vector<std::string> RolloutSampleEnv::GetMuscleNames() {
    const auto& muscles = mEnv.getCharacter()->getMuscles();
    std::vector<std::string> names;
    names.reserve(muscles.size());
    for (const auto& muscle : muscles) {
        names.push_back(muscle->GetName());
    }
    return names;
}

std::vector<std::string> RolloutSampleEnv::GetRecordFields() const {
    int skeleton_dof = const_cast<Environment&>(mEnv).getCharacter()->getSkeleton()->getNumDofs();
    return RolloutRecord::FieldsFromConfig(mRecordConfig, skeleton_dof);
}
