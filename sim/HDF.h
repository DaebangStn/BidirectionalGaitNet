#ifndef __HDF_H__
#define __HDF_H__

#include "Motion.h"
#include <H5Cpp.h>
#include <Eigen/Core>
#include <string>

// Forward declaration
class Character;

/**
 * @brief HDF5 motion format implementation
 *
 * Loads motion data from HDF5 files containing:
 * - Structure: param_N/cycle_M/{motions, phase, time, ...}
 * - "motions": Frame sequence (num_frames × 56 DOF in angle format)
 * - "phase": Phase values for each frame
 * - "time": Time values for each frame
 */
class HDF : public Motion {
public:
    /**
     * @brief Construct HDF motion from file and indices
     * @param filepath Path to .h5/.hdf5 file
     * @param param_idx Parameter index (e.g., 0 for param_0)
     * @param cycle_idx Cycle index (e.g., 0 for cycle_0)
     */
    explicit HDF(const std::string& filepath, int param_idx, int cycle_idx);
    ~HDF() override = default;

    // ==================== Motion Interface Implementation ====================

    Eigen::VectorXd getTargetPose(double phase) override;
    Eigen::VectorXd getPose(int frameIdx) override;
    Eigen::VectorXd getPose(double phase) override;

    double getMaxTime() const override;
    int getNumFrames() const override { return mNumFrames; }
    double getFrameTime() const override { return mFrameTime; }
    std::string getName() const override { return mFilename; }

    void setRefMotion(Character* character, dart::simulation::WorldPtr world) override;

    // ==================== HDF-Specific Methods ====================

    /**
     * @brief Get phase data for all frames
     * @return Phase vector (numFrames,)
     */
    Eigen::VectorXd getPhaseData() const { return mPhaseData; }

    /**
     * @brief Get time data for all frames
     * @return Time vector (numFrames,)
     */
    Eigen::VectorXd getTimeData() const { return mTimeData; }

    /**
     * @brief Get parameter index
     * @return Parameter index used in construction
     */
    int getParamIdx() const { return mParamIdx; }

    /**
     * @brief Get cycle index
     * @return Cycle index used in construction
     */
    int getCycleIdx() const { return mCycleIdx; }

private:
    std::string mFilename;
    int mParamIdx;
    int mCycleIdx;

    Eigen::MatrixXd mMotionData;  ///< Motion data in angle format: (numFrames, 56)
    Eigen::VectorXd mPhaseData;   ///< Phase values: (numFrames,)
    Eigen::VectorXd mTimeData;    ///< Time values: (numFrames,)

    int mNumFrames;               ///< Number of frames in this cycle
    int mDofPerFrame;             ///< Values per frame in angle format (56)
    double mFrameTime;            ///< Time per frame in seconds (default: 1/60)

    // Height calibration state
    Eigen::Isometry3d mRootTransform;
    double mHeightOffset = 0.0;
    double mXOffset = 0.0;

    // ==================== Private Helper Methods ====================

    /**
     * @brief Load HDF5 file and extract cycle data
     * @param filepath Path to .h5/.hdf5 file
     * @param param_idx Parameter index
     * @param cycle_idx Cycle index
     */
    void loadFromFile(const std::string& filepath, int param_idx, int cycle_idx);

    /**
     * @brief Interpolate between two frames
     * @param frame1 First frame index
     * @param frame2 Second frame index
     * @param t Interpolation parameter [0, 1]
     * @return Interpolated pose
     */
    Eigen::VectorXd interpolatePose(int frame1, int frame2, double t) const;
};

#endif // __HDF_H__
