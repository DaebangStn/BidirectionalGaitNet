#pragma once

#include <memory>
#include <string>
#include <vector>
#include <Eigen/Core>

// Forward declarations
class Motion;
class Character;
class Environment;
struct PlaybackViewerState;
struct DrawFlags;
class ShapeRenderer;

namespace dart::simulation {
    class World;
}
using WorldPtr = std::shared_ptr<dart::simulation::World>;

// Forward declaration of PlaybackContext (defined in PlaybackContext.h)
struct MotionProcessorContext;

/**
 * @brief Abstract interface for motion processing operations
 *
 * Encapsulates loading, playback computation, and rendering for a motion type.
 * Each processor handles one family of motion formats (HDF or C3D).
 *
 * Design Rationale:
 * - Replaces scattered getSourceType() string comparisons with virtual dispatch
 * - Separates loading, playback, and rendering concerns
 * - Enables type-specific optimizations without polluting shared code
 */
class IMotionProcessor {
public:
    virtual ~IMotionProcessor() = default;

    // ==================== Core Operations ====================

    /**
     * @brief Load motion from file
     * @param path Path to motion file
     * @param character Character for motion preprocessing
     * @param world Simulation world for calibration
     * @return Loaded Motion object (caller takes ownership)
     */
    virtual Motion* load(const std::string& path,
                         Character* character,
                         WorldPtr world) = 0;

    /**
     * @brief Compute playback context for current time
     * @param motion Motion to evaluate
     * @param viewerTime Current viewer time in seconds
     * @param viewerPhase Current viewer phase [0, 1)
     * @param state Playback state (modified with cycle accumulation)
     * @param character Character for interpolation
     * @return Computed playback context
     */
    virtual MotionProcessorContext computePlayback(Motion* motion,
                                                   double viewerTime,
                                                   double viewerPhase,
                                                   PlaybackViewerState& state,
                                                   Character* character) = 0;

    /**
     * @brief Render motion at computed context
     * @param context Playback context from computePlayback()
     * @param character Character skeleton for rendering
     * @param renderer Shape renderer
     * @param flags Draw flags for visualization options
     */
    virtual void render(const MotionProcessorContext& context,
                        Character* character,
                        ShapeRenderer* renderer,
                        const DrawFlags& flags) = 0;

    // ==================== Type Information ====================

    /**
     * @brief Get source type identifier
     * @return Type string ("hdf" or "c3d")
     */
    virtual std::string getSourceType() const = 0;

    /**
     * @brief Get supported file extensions
     * @return Vector of extensions (e.g., {".h5", ".hdf5"})
     */
    virtual std::vector<std::string> getSupportedExtensions() const = 0;

    /**
     * @brief Check if processor supports marker data
     * @return true for C3D, false for HDF
     */
    virtual bool supportsMarkers() const { return false; }

    /**
     * @brief Check if processor supports gait parameters
     * @return true for HDF, false for C3D
     */
    virtual bool supportsParameters() const { return false; }

    // ==================== Parameter Operations (HDF only) ====================

    /**
     * @brief Check if loaded motion has parameters
     * @param motion Motion to check
     * @return true if motion file contains parameter data
     */
    virtual bool hasParameters(Motion* motion) const { return false; }

    /**
     * @brief Get parameter names from motion
     * @param motion Motion to query
     * @return Vector of parameter names (empty if no parameters)
     */
    virtual std::vector<std::string> getParameterNames(Motion* motion) const { return {}; }

    /**
     * @brief Get parameter values from motion
     * @param motion Motion to query
     * @return Vector of parameter values (empty if no parameters)
     */
    virtual std::vector<float> getParameterValues(Motion* motion) const { return {}; }

    /**
     * @brief Apply motion parameters to environment
     * @param motion Motion with parameters
     * @param env Environment to apply parameters to
     * @return true if parameters were applied successfully
     */
    virtual bool applyParametersToEnvironment(Motion* motion, Environment* env) const { return false; }

    // ==================== Cycle Distance ====================

    /**
     * @brief Compute cycle distance for motion
     * @param motion Motion to analyze
     * @return Cycle distance vector (root translation per cycle)
     */
    virtual Eigen::Vector3d computeCycleDistance(Motion* motion) const = 0;
};
