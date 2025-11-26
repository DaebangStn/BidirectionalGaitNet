#pragma once

#include <memory>
#include <string>
#include <vector>
#include <Eigen/Core>

// Forward declarations
class Motion;
class RenderCharacter;
class Environment;
class C3D_Reader;
struct PlaybackViewerState;
struct DrawFlags;
class ShapeRenderer;
struct MotionProcessorContext;

namespace dart::simulation {
    class World;
}
using WorldPtr = std::shared_ptr<dart::simulation::World>;

// C3D-specific conversion parameters
struct C3DProcessorParams {
    double femurTorsionL = 0.0;
    double femurTorsionR = 0.0;
};

/**
 * @brief Unified motion processor for HDF and C3D motion files
 *
 * Handles loading, playback computation, and rendering for all motion types.
 * Uses the Motion base class interface for type-agnostic operations, with
 * conditional handling for type-specific features (markers for C3D, parameters for HDF).
 *
 * Design Rationale:
 * - Single class replaces previous HDFMotionProcessor/C3DMotionProcessor hierarchy
 * - Eliminates ~200 lines of duplicated code between processors
 * - Uses file extension dispatch for loading, runtime type checking for features
 */
class MotionProcessor {
public:
    MotionProcessor();
    ~MotionProcessor();

    // ==================== Dependency Injection ====================

    /**
     * @brief Set C3D reader for loading C3D files
     * @param reader Pointer to C3D_Reader (owned externally)
     */
    void setC3DReader(C3D_Reader* reader) { mReader = reader; }

    /**
     * @brief Set conversion parameters for C3D IK conversion
     * @param params Femur torsion parameters
     */
    void setC3DConversionParams(const C3DProcessorParams& params) { mParams = params; }

    // ==================== Core Operations ====================

    /**
     * @brief Load motion from file (HDF or C3D)
     * @param path Path to motion file
     * @param character Character for motion preprocessing
     * @param world Simulation world for calibration
     * @return Loaded Motion object (caller takes ownership), nullptr on failure
     */
    Motion* load(const std::string& path, RenderCharacter* character, WorldPtr world);

    /**
     * @brief Compute playback context for current time
     * @param motion Motion to evaluate
     * @param viewerTime Current viewer time in seconds
     * @param viewerPhase Current viewer phase [0, 1)
     * @param state Playback state (modified with cycle accumulation)
     * @param character Character for interpolation
     * @return Computed playback context
     */
    MotionProcessorContext computePlayback(Motion* motion,
                                           double viewerTime,
                                           double viewerPhase,
                                           PlaybackViewerState& state,
                                           RenderCharacter* character);

    /**
     * @brief Render motion at computed context
     * @param context Playback context from computePlayback()
     * @param character Character skeleton for rendering
     * @param renderer Shape renderer
     * @param flags Draw flags for visualization options
     */
    void render(const MotionProcessorContext& context,
                RenderCharacter* character,
                ShapeRenderer* renderer,
                const DrawFlags& flags);

    // ==================== Cycle Distance ====================

    /**
     * @brief Compute cycle distance for motion
     * @param motion Motion to analyze
     * @return Cycle distance vector (root translation per cycle)
     */
    Eigen::Vector3d computeCycleDistance(Motion* motion) const;

    // ==================== Parameter Methods ====================

    /**
     * @brief Check if loaded motion has parameters
     * @param motion Motion to check
     * @return true if motion file contains parameter data
     */
    bool hasParameters(Motion* motion) const;

    /**
     * @brief Get parameter names from motion
     * @param motion Motion to query
     * @return Vector of parameter names (empty if no parameters)
     */
    std::vector<std::string> getParameterNames(Motion* motion) const;

    /**
     * @brief Get parameter values from motion
     * @param motion Motion to query
     * @return Vector of parameter values (empty if no parameters)
     */
    std::vector<float> getParameterValues(Motion* motion) const;

    /**
     * @brief Apply motion parameters to environment
     * @param motion Motion with parameters
     * @param env Environment to apply parameters to
     * @return true if parameters were applied successfully
     */
    bool applyParametersToEnvironment(Motion* motion, Environment* env) const;

    // ==================== Marker Methods (C3D-specific) ====================

    /**
     * @brief Get markers at specific frame
     * @param motion Motion to query (must be C3D type)
     * @param frameIdx Frame index
     * @return Vector of marker positions (empty if not C3D or no markers)
     */
    std::vector<Eigen::Vector3d> getMarkersAtFrame(Motion* motion, int frameIdx) const;

    /**
     * @brief Get interpolated markers at fractional frame
     * @param motion Motion to query (must be C3D type)
     * @param frameFloat Fractional frame index
     * @return Vector of interpolated marker positions (empty if not C3D or no markers)
     */
    std::vector<Eigen::Vector3d> getInterpolatedMarkers(Motion* motion, double frameFloat) const;

    // ==================== Direct Pose Evaluation ====================

    /**
     * @brief Evaluate pose at specific frame (for integration with existing code)
     *
     * This is a simpler interface than computePlayback() for cases where
     * frame computation is already done externally.
     *
     * @param motion Motion to evaluate
     * @param frameFloat Fractional frame index (already computed)
     * @param character Character for skeleton-aware interpolation
     * @param state Playback state (used for cycle accumulation offsets)
     * @return Interpolated pose with position offsets applied
     */
    Eigen::VectorXd evaluatePoseAtFrame(Motion* motion, double frameFloat,
                                        RenderCharacter* character, const PlaybackViewerState& state);

    /**
     * @brief Update markers at specific frame (for C3D motions)
     * @param motion Motion to get markers from
     * @param frameIdx Frame index
     * @param state Playback state (used for position offsets)
     * @return Vector of marker positions with offsets applied
     */
    std::vector<Eigen::Vector3d> getMarkersAtFrameWithOffsets(Motion* motion, int frameIdx,
                                                               const PlaybackViewerState& state);

    // ==================== Type Information ====================

    /**
     * @brief Check if extension is HDF type
     * @param ext File extension (lowercase, with dot)
     * @return true if extension is .h5 or .hdf5
     */
    static bool isHDFExtension(const std::string& ext);

    /**
     * @brief Check if extension is C3D type
     * @param ext File extension (lowercase, with dot)
     * @return true if extension is .c3d
     */
    static bool isC3DExtension(const std::string& ext);

    /**
     * @brief Extract extension from file path
     * @param path File path
     * @return Extension (lowercase, including dot)
     */
    static std::string extractExtension(const std::string& path);

private:
    // ==================== Private Methods ====================

    /**
     * @brief Evaluate pose at fractional frame
     * @param motion Motion to evaluate
     * @param frameFloat Fractional frame index
     * @param character Character for skeleton-aware interpolation
     * @param state Playback state for cycle accumulation
     * @return Interpolated pose with position offsets applied
     */
    Eigen::VectorXd evaluatePose(Motion* motion,
                                 double frameFloat,
                                 RenderCharacter* character,
                                 const PlaybackViewerState& state);

    /**
     * @brief Update markers in context (C3D-specific)
     * @param context Context to update
     * @param motion Motion to get markers from
     * @param frameFloat Fractional frame index
     * @param state Playback state for position offsets
     */
    void updateMarkers(MotionProcessorContext& context,
                       Motion* motion,
                       double frameFloat,
                       const PlaybackViewerState& state);

    // ==================== Members ====================

    C3D_Reader* mReader = nullptr;
    C3DProcessorParams mParams;
};
