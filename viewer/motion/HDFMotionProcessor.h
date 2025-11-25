#pragma once

#include "IMotionProcessor.h"
#include "PlaybackContext.h"
#include <memory>

// Forward declarations
class HDF;

/**
 * @brief Motion processor for HDF5 format
 *
 * Handles loading, playback, and rendering of HDF motion files.
 * HDF motions contain skeleton pose data with optional gait parameters.
 *
 * File formats supported:
 * - Single-cycle HDF5 (.h5, .hdf5) with flat structure
 *
 * Features:
 * - Skeleton-aware pose interpolation
 * - Gait parameter support
 * - Cycle accumulation for walking progression
 */
class HDFMotionProcessor : public IMotionProcessor {
public:
    HDFMotionProcessor();
    ~HDFMotionProcessor() override;

    // ==================== IMotionProcessor Implementation ====================

    Motion* load(const std::string& path,
                 Character* character,
                 WorldPtr world) override;

    MotionProcessorContext computePlayback(Motion* motion,
                                           double viewerTime,
                                           double viewerPhase,
                                           PlaybackViewerState& state,
                                           Character* character) override;

    void render(const MotionProcessorContext& context,
                Character* character,
                ShapeRenderer* renderer,
                const DrawFlags& flags) override;

    std::string getSourceType() const override { return "hdf"; }

    std::vector<std::string> getSupportedExtensions() const override {
        return {".h5", ".hdf5"};
    }

    bool supportsParameters() const override { return true; }

    // ==================== Parameter Operations ====================

    bool hasParameters(Motion* motion) const override;
    std::vector<std::string> getParameterNames(Motion* motion) const override;
    std::vector<float> getParameterValues(Motion* motion) const override;
    bool applyParametersToEnvironment(Motion* motion, Environment* env) const override;

    // ==================== Cycle Distance ====================

    Eigen::Vector3d computeCycleDistance(Motion* motion) const override;

private:
    /**
     * @brief Evaluate skeleton pose at given frame
     * @param motion HDF motion
     * @param frameFloat Frame position (with fractional part)
     * @param character Character for interpolation
     * @param state Playback state with offsets
     * @return Evaluated pose with offsets applied
     */
    Eigen::VectorXd evaluatePose(Motion* motion,
                                 double frameFloat,
                                 Character* character,
                                 const PlaybackViewerState& state);
};
