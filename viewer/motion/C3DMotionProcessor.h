#pragma once

#include "IMotionProcessor.h"
#include "PlaybackContext.h"
#include <memory>

// Forward declarations
class C3DMotion;
class C3D_Reader;
class Environment;

/**
 * @brief Conversion parameters for C3D to skeleton
 */
struct C3DProcessorParams {
    double femurTorsionL = 0.0;
    double femurTorsionR = 0.0;
};

/**
 * @brief Motion processor for C3D marker format
 *
 * Handles loading, playback, and rendering of C3D motion capture files.
 * C3D motions contain both marker positions and IK-converted skeleton poses.
 *
 * Features:
 * - Dual data: skeleton poses + raw markers
 * - Marker interpolation for smooth visualization
 * - Integrated with C3D_Reader for IK conversion
 */
class C3DMotionProcessor : public IMotionProcessor {
public:
    /**
     * @brief Construct C3D processor with required dependencies
     * @param reader C3D_Reader for IK conversion (must remain valid)
     */
    explicit C3DMotionProcessor(C3D_Reader* reader);
    ~C3DMotionProcessor() override;

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

    std::string getSourceType() const override { return "c3d"; }

    std::vector<std::string> getSupportedExtensions() const override {
        return {".c3d"};
    }

    bool supportsMarkers() const override { return true; }
    bool supportsParameters() const override { return false; }

    // ==================== Cycle Distance ====================

    Eigen::Vector3d computeCycleDistance(Motion* motion) const override;

    // ==================== C3D-Specific Methods ====================

    /**
     * @brief Set IK conversion parameters
     * @param params Conversion parameters (femur torsion, etc.)
     */
    void setConversionParams(const C3DProcessorParams& params);

    /**
     * @brief Get current conversion parameters
     */
    C3DProcessorParams getConversionParams() const { return mParams; }

    /**
     * @brief Get markers at specific frame
     * @param motion C3DMotion to query
     * @param frameIdx Frame index
     * @return Vector of marker positions
     */
    std::vector<Eigen::Vector3d> getMarkersAtFrame(Motion* motion, int frameIdx) const;

    /**
     * @brief Get interpolated markers at frame float
     * @param motion C3DMotion to query
     * @param frameFloat Frame position with fractional part
     * @return Vector of interpolated marker positions
     */
    std::vector<Eigen::Vector3d> getInterpolatedMarkers(Motion* motion, double frameFloat) const;

private:
    C3D_Reader* mReader;  // Non-owning pointer to C3D_Reader
    C3DProcessorParams mParams;

    /**
     * @brief Evaluate skeleton pose at given frame
     * @param motion C3D motion
     * @param frameFloat Frame position (with fractional part)
     * @param character Character for interpolation
     * @param state Playback state with offsets
     * @return Evaluated pose with offsets applied
     */
    Eigen::VectorXd evaluatePose(Motion* motion,
                                 double frameFloat,
                                 Character* character,
                                 const PlaybackViewerState& state);

    /**
     * @brief Update marker data in context
     * @param context Context to update
     * @param motion C3DMotion source
     * @param frameFloat Current frame position
     * @param state Playback state with offsets
     */
    void updateMarkers(MotionProcessorContext& context,
                       Motion* motion,
                       double frameFloat,
                       const PlaybackViewerState& state);
};
