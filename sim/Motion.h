#ifndef __MOTION_H__
#define __MOTION_H__

#include <Eigen/Core>
#include <string>
#include "dart/dart.hpp"

// Forward declaration
class Character;

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
     * @brief Enable/disable height calibration
     * @param enable True to enable height calibration
     */
    virtual void setHeightCalibration(bool enable) { mHeightCalibration = enable; }

    /**
     * @brief Check if height calibration is enabled
     * @return True if height calibration is enabled
     */
    virtual bool getHeightCalibration() const { return mHeightCalibration; }

protected:
    bool mHeightCalibration = false;
    Character* mCharacter = nullptr;
};

#endif // __MOTION_H__
