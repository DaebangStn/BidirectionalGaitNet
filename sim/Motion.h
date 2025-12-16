#ifndef __MOTION_H__
#define __MOTION_H__

#include <Eigen/Core>
#include <string>
#include "dart/dart.hpp"

// Forward declarations
class Character;
class Environment;

/**
 * @brief Abstract base class for motion data (BVH, NPZ, etc.)
 *
 * Provides a unified interface for accessing reference motion data
 * with both time-based (phase) and frame-based access patterns.
 */
class Motion {
public:
    virtual ~Motion() = default;

    // ==================== Pure Virtual Methods ====================

    /**
     * @brief Get target pose at given phase with any preprocessing (e.g., symmetry)
     * @param phase Phase value in [0, 1] representing motion cycle position
     * @return Pose vector (full DOF including root)
     */
    virtual Eigen::VectorXd getTargetPose(double phase) = 0;

    /**
     * @brief Get pose at specific frame index
     * @param frameIdx Frame index (0 to numFrames-1)
     * @return Pose vector at given frame
     */
    virtual Eigen::VectorXd getPose(int frameIdx) = 0;

    /**
     * @brief Get interpolated pose at given phase
     * @param phase Phase value in [0, 1]
     * @return Interpolated pose vector
     */
    virtual Eigen::VectorXd getPose(double phase) = 0;

    /**
     * @brief Get total duration of motion in seconds
     * @return Motion duration
     */
    virtual double getMaxTime() const = 0;

    /**
     * @brief Get total number of frames in motion
     * @return Number of frames
     */
    virtual int getNumFrames() const = 0;

    /**
     * @brief Get time duration of each frame in seconds
     * @return Frame time (e.g., 1/60 for 60Hz)
     */
    virtual double getFrameTime() const = 0;

    /**
     * @brief Get name/identifier of motion
     * @return Motion name (typically filename)
     */
    virtual std::string getName() const = 0;

    /**
     * @brief Initialize motion with character and world for preprocessing
     * @param character Character skeleton to apply motion to
     * @param world Simulation world for collision/calibration
     */
    virtual void setRefMotion(Character* character, dart::simulation::WorldPtr world) = 0;

    // ==================== Virtual Methods with Default Implementation ====================

    /**
     * @brief Get source type identifier
     * @return Source format string ("npz", "hdfSingle", "bvh", "hdfRollout")
     */
    virtual std::string getSourceType() const = 0;

    /**
     * @brief Get log header for debug output
     * @return Log header string (e.g., "[NPZ]", "[BVH]", "[HDF Single]")
     */
    virtual std::string getLogHeader() const = 0;

    /**
     * @brief Check if motion has associated parameters
     * @return True if motion file contains parameter data
     */
    virtual bool hasParameters() const { return false; }

    /**
     * @brief Get parameter names from motion file
     * @return Vector of parameter name strings (empty if no parameters)
     */
    virtual std::vector<std::string> getParameterNames() const { return {}; }

    /**
     * @brief Get parameter values from motion file
     * @return Vector of parameter values (empty if no parameters)
     */
    virtual std::vector<float> getParameterValues() const { return {}; }

    /**
     * @brief Get flattened raw motion data array
     * @return Flattened motion data (all frames concatenated)
     */
    virtual Eigen::VectorXd getRawMotionData() const { return Eigen::VectorXd(); }

    /**
     * @brief Get cycle distance for forward progression
     * @return Cycle distance vector (root translation per cycle)
     */
    virtual Eigen::Vector3d getCycleDistance() const { return Eigen::Vector3d::Zero(); }

    /**
     * @brief Get number of values per frame (DOF count)
     * @return Values per frame (e.g., 56 for HDF/BVH, 101 for NPZ)
     */
    virtual int getValuesPerFrame() const { return 56; }

    /**
     * @brief Get timestamps for each frame
     * @return Vector of timestamps in seconds (empty if not available)
     */
    virtual std::vector<double> getTimestamps() const { return {}; }

    /**
     * @brief Get total timesteps (for HDF rollout files)
     * @return Total timesteps across all cycles
     */
    virtual int getTotalTimesteps() const { return getNumFrames(); }

    /**
     * @brief Get timesteps per cycle (for HDF files)
     * @return Average timesteps per gait cycle
     */
    virtual int getTimestepsPerCycle() const { return getNumFrames(); }

    /**
     * @brief Apply motion parameters to environment
     * @param env Environment to apply parameters to
     * @return True if parameters were applied successfully, false if count mismatch (caller should use defaults)
     */
    virtual bool applyParametersToEnvironment(Environment* env) const { return false; }

protected:
    Character* mCharacter = nullptr;
};

#endif // __MOTION_H__
