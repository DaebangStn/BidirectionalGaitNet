#ifndef __NOISE_INJECTOR_H__
#define __NOISE_INJECTOR_H__

#include <string>
#include <vector>
#include <Eigen/Core>
#include "../libs/FastNoiseLite.h"

// Forward declarations
class Character;

class NoiseInjector
{
public:
    // Constructor: optionally load from YAML config file
    NoiseInjector(const std::string& config_path = "", double time_step = 0.001);
    ~NoiseInjector() = default;

    // Core functionality
    void reset();  // Reset for new episode (reseed Perlin noise)
    void step(Character* character);

    // Configuration setters (for ImGui runtime updates)
    void setEnabled(bool enabled) { mEnabled = enabled; }
    void setPositionNoiseEnabled(bool enabled) { mPositionEnabled = enabled; }
    void setForceNoiseEnabled(bool enabled) { mForceEnabled = enabled; }
    void setActivationNoiseEnabled(bool enabled) { mActivationEnabled = enabled; }

    void setPositionAmplitude(double amp) { mPositionAmplitude = amp; }
    void setForceAmplitude(double amp) { mForceAmplitude = amp; }
    void setActivationAmplitude(double amp) { mActivationAmplitude = amp; }

    void setPositionFrequency(double freq) { mPositionFrequency = freq; }
    void setForceFrequency(double freq) { mForceFrequency = freq; }
    void setActivationFrequency(double freq) { mActivationFrequency = freq; }

    void setForceTargetNodes(const std::vector<std::string>& nodes) { mForceTargetNodes = nodes; }
    void setPositionTargetNodes(const std::vector<std::string>& nodes) { mPositionTargetNodes = nodes; }

    // Configuration getters (for ImGui display)
    bool isEnabled() const { return mEnabled; }
    bool isPositionEnabled() const { return mPositionEnabled; }
    bool isForceEnabled() const { return mForceEnabled; }
    bool isActivationEnabled() const { return mActivationEnabled; }

    double getPositionAmplitude() const { return mPositionAmplitude; }
    double getForceAmplitude() const { return mForceAmplitude; }
    double getActivationAmplitude() const { return mActivationAmplitude; }

    double getPositionFrequency() const { return mPositionFrequency; }
    double getForceFrequency() const { return mForceFrequency; }
    double getActivationFrequency() const { return mActivationFrequency; }

    const std::vector<std::string>& getForceTargetNodes() const { return mForceTargetNodes; }
    const std::vector<std::string>& getPositionTargetNodes() const { return mPositionTargetNodes; }

    // Visualization support
    struct NoiseVisualization {
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> forceArrows;  // <position, force vector>
        std::vector<std::pair<std::string, Eigen::Vector3d>> positionNoises;  // <joint name, position offset>
        std::vector<double> activationNoises;  // Per-muscle noise magnitudes for color intensity
    };

    const NoiseVisualization& getVisualization() const { return mViz; }

private:
    // Noise application methods
    void applyPositionNoise(Character* character, double time);
    void applyForceNoise(Character* character, double time);
    void applyActivationNoise(Character* character, double time);

    // Load configuration from YAML file
    void loadConfig(const std::string& config_path);

    // Perlin noise generator
    FastNoiseLite mNoiseGenerator;

    // Visualization data
    NoiseVisualization mViz;

    // Time tracking
    double mTimeCounter;
    double mTimeStep;

    // Configuration state
    bool mEnabled;
    bool mPositionEnabled;
    bool mForceEnabled;
    bool mActivationEnabled;

    double mPositionAmplitude;
    double mForceAmplitude;
    double mActivationAmplitude;

    double mPositionFrequency;
    double mForceFrequency;
    double mActivationFrequency;

    std::vector<std::string> mForceTargetNodes;
    std::vector<std::string> mPositionTargetNodes;
};

#endif  // __NOISE_INJECTOR_H__
