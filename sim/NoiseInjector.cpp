#include "NoiseInjector.h"
#include "Character.h"
#include "UriResolver.h"
#include "Log.h"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cmath>
#include <random>

NoiseInjector::NoiseInjector(const std::string& config_path, double time_step)
    : mTimeCounter(0.0),
      mTimeStep(time_step),
      mEnabled(false),
      mPositionEnabled(false),
      mForceEnabled(false),
      mActivationEnabled(false),
      mPositionAmplitude(0.005),
      mForceAmplitude(40.0),
      mActivationAmplitude(0.05),
      mPositionFrequency(0.25),
      mForceFrequency(0.25),
      mActivationFrequency(0.25)
{
    // Initialize Perlin noise generator
    mNoiseGenerator.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
    mNoiseGenerator.SetFrequency(1.0);  // Base frequency (will be multiplied per noise type)

    // Load configuration if path provided
    if (!config_path.empty()) {
        loadConfig(config_path);
    }

    LOG_VERBOSE("[NoiseInjector] Initialized with config: " << config_path);
}

void NoiseInjector::reset()
{
    // Reset time counter
    mTimeCounter = 0.0;

    // Reseed with random value for episode variety
    std::random_device rd;
    int seed = rd();
    mNoiseGenerator.SetSeed(seed);

    // Clear visualization data
    mViz.forceArrows.clear();
    mViz.positionNoises.clear();
    mViz.activationNoises.clear();

    LOG_VERBOSE("[NoiseInjector] Reset with seed: " << seed);
}

void NoiseInjector::step(Character* character)
{
    if (!mEnabled || !character) return;

    // Increment time counter
    mTimeCounter += mTimeStep;

    // Apply enabled noise types
    if (mPositionEnabled) {
        applyPositionNoise(character, mTimeCounter);
    }

    if (mForceEnabled) {
        applyForceNoise(character, mTimeCounter);
    }

    if (mActivationEnabled) {
        applyActivationNoise(character, mTimeCounter);
    }
}

void NoiseInjector::applyPositionNoise(Character* character, double time)
{
    mViz.positionNoises.clear();

    auto skeleton = character->getSkeleton();
    Eigen::VectorXd current_pos = skeleton->getPositions();

    // Apply noise only to specified body nodes
    int node_idx = 0;
    for (const std::string& node_name : mPositionTargetNodes) {
        auto* bn = skeleton->getBodyNode(node_name);
        if (!bn) continue;

        auto* joint = bn->getParentJoint();
        if (!joint) continue;

        // Create unique noise offset for this body node using hash
        std::hash<std::string> hasher;
        float node_offset = static_cast<float>(hasher(node_name) % 10000);

        // Apply noise to all DOFs of this joint
        int start_dof = joint->getIndexInSkeleton(0);
        int num_dofs = joint->getNumDofs();

        Eigen::Vector3d total_offset = Eigen::Vector3d::Zero();
        for (int i = 0; i < num_dofs; i++) {
            int dof_idx = start_dof + i;
            // Each body node gets independent noise using node_offset
            // Each DOF within the node gets different phase using i
            float noise = mNoiseGenerator.GetNoise(
                static_cast<float>(time * mPositionFrequency),
                node_offset + static_cast<float>(i * 100)
            );
            double delta = mPositionAmplitude * noise;
            current_pos[dof_idx] += delta;

            // Accumulate offset for visualization
            if (i < 3) {
                total_offset[i] = delta;
            }
        }

        // Store for visualization with actual offset vector
        mViz.positionNoises.push_back({node_name, total_offset});
        node_idx++;
    }

    skeleton->setPositions(current_pos);
}

void NoiseInjector::applyForceNoise(Character* character, double time)
{
    mViz.forceArrows.clear();

    auto skeleton = character->getSkeleton();

    for (const std::string& node_name : mForceTargetNodes) {
        auto* bn = skeleton->getBodyNode(node_name);
        if (bn) {
            // Create unique noise offset for this body node using hash
            std::hash<std::string> hasher;
            float node_offset = static_cast<float>(hasher(node_name) % 10000);

            // Generate 3D force vector using Perlin noise
            // Each body node gets independent noise in each axis
            float time_f = static_cast<float>(time * mForceFrequency);
            Eigen::Vector3d noise_force(
                mForceAmplitude * mNoiseGenerator.GetNoise(time_f, node_offset + 0.0f),
                mForceAmplitude * mNoiseGenerator.GetNoise(time_f, node_offset + 100.0f),
                mForceAmplitude * mNoiseGenerator.GetNoise(time_f, node_offset + 200.0f)
            );

            // Apply external force to body node
            bn->setExtForce(noise_force, Eigen::Vector3d::Zero(), false, true);

            // Store for visualization
            Eigen::Vector3d pos = bn->getWorldTransform().translation();
            mViz.forceArrows.push_back({pos, noise_force});
        }
    }
}

void NoiseInjector::applyActivationNoise(Character* character, double time)
{
    Eigen::VectorXd activations = character->getActivations();
    auto muscles = character->getMuscles();

    mViz.activationNoises.clear();
    mViz.activationNoises.resize(activations.size());

    for (int i = 0; i < activations.size(); i++) {
        // Create unique noise offset for each muscle using name hash
        std::string muscle_name = muscles[i]->GetName();
        std::hash<std::string> hasher;
        float muscle_offset = static_cast<float>(hasher(muscle_name) % 10000);

        // Each muscle gets independent noise
        float noise = mNoiseGenerator.GetNoise(
            static_cast<float>(time * mActivationFrequency),
            muscle_offset
        );

        // Use absolute value for positive-only noise (maintains zero mean)
        double noise_delta = mActivationAmplitude * std::abs(noise);

        // Store the actual noise value for visualization (used for green intensity)
        mViz.activationNoises[i] = noise_delta;

        // Apply clamped noise to activation
        activations[i] = std::clamp(activations[i] + noise_delta, 0.0, 1.0);
    }

    character->setActivations(activations);
}

void NoiseInjector::loadConfig(const std::string& config_path)
{
    try {
        // Resolve URI if needed
        std::string resolved_path = PMuscle::URIResolver::getInstance().resolve(config_path);

        YAML::Node config = YAML::LoadFile(resolved_path);
        YAML::Node ni_config = config["noise_injection"];

        if (!ni_config) {
            LOG_ERROR("[NoiseInjector] No 'noise_injection' key found in config: " << resolved_path);
            return;
        }

        // Load master enable
        if (ni_config["enabled"]) {
            mEnabled = ni_config["enabled"].as<bool>();
        }

        // Load position noise config
        if (ni_config["position"]) {
            auto pos = ni_config["position"];
            if (pos["enabled"]) mPositionEnabled = pos["enabled"].as<bool>();
            if (pos["amplitude"]) mPositionAmplitude = pos["amplitude"].as<double>();
            if (pos["frequency"]) mPositionFrequency = pos["frequency"].as<double>();

            // Load target nodes for position
            if (pos["target_nodes"]) {
                mPositionTargetNodes.clear();
                for (const auto& node : pos["target_nodes"]) {
                    mPositionTargetNodes.push_back(node.as<std::string>());
                }
            }
        }

        // Load force noise config
        if (ni_config["force"]) {
            auto force = ni_config["force"];
            if (force["enabled"]) mForceEnabled = force["enabled"].as<bool>();
            if (force["amplitude"]) mForceAmplitude = force["amplitude"].as<double>();
            if (force["frequency"]) mForceFrequency = force["frequency"].as<double>();

            // Load target nodes for force
            if (force["target_nodes"]) {
                mForceTargetNodes.clear();
                for (const auto& node : force["target_nodes"]) {
                    mForceTargetNodes.push_back(node.as<std::string>());
                }
            }
        }

        // Load activation noise config
        if (ni_config["activation"]) {
            auto act = ni_config["activation"];
            if (act["enabled"]) mActivationEnabled = act["enabled"].as<bool>();
            if (act["amplitude"]) mActivationAmplitude = act["amplitude"].as<double>();
            if (act["frequency"]) mActivationFrequency = act["frequency"].as<double>();
        }

        LOG_INFO("[NoiseInjector] Loaded config from: " << resolved_path);
        LOG_INFO("[NoiseInjector]   Enabled: " << mEnabled);
        LOG_INFO("[NoiseInjector]   Position: " << mPositionEnabled << " (amp=" << mPositionAmplitude << ", freq=" << mPositionFrequency << ", nodes=" << mPositionTargetNodes.size() << ")");
        LOG_INFO("[NoiseInjector]   Force: " << mForceEnabled << " (amp=" << mForceAmplitude << ", freq=" << mForceFrequency << ", nodes=" << mForceTargetNodes.size() << ")");
        LOG_INFO("[NoiseInjector]   Activation: " << mActivationEnabled << " (amp=" << mActivationAmplitude << ", freq=" << mActivationFrequency << ")");

    } catch (const YAML::Exception& e) {
        LOG_ERROR("[NoiseInjector] YAML parsing error: " << e.what());
    } catch (const std::exception& e) {
        LOG_ERROR("[NoiseInjector] Error loading config: " << e.what());
    }
}
