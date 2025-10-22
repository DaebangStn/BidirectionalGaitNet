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
     * @brief Construct HDF motion from single-cycle extracted file
     * @param filepath Path to .h5/.hdf5 file with flat structure (/motions, /phase, /time)
     */
    explicit HDF(const std::string& filepath);
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

    std::string getSourceType() const override { return "hdfSingle"; }
    std::string getLogHeader() const override { return "[HDF Single]"; }

    bool hasParameters() const override { return !mParameterNames.empty(); }
    std::vector<std::string> getParameterNames() const override { return mParameterNames; }
    std::vector<float> getParameterValues() const override { return mParameterValues; }
    bool applyParametersToEnvironment(RenderEnvironment* env) const override;

    // Extended interface for legacy ViewerMotion compatibility
    Eigen::VectorXd getRawMotionData() const override;
    int getValuesPerFrame() const override { return 56; }  // HDF uses skeleton DOF
    std::vector<double> getTimestamps() const override;
    int getTotalTimesteps() const override { return mMotionData.rows(); }
    int getTimestepsPerCycle() const override { return mMotionData.rows(); }

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

private:
    std::string mFilename;

    Eigen::MatrixXd mMotionData;  ///< Motion data in angle format: (numFrames, 56)
    Eigen::VectorXd mPhaseData;   ///< Phase values: (numFrames,)
    Eigen::VectorXd mTimeData;    ///< Time values: (numFrames,)

    int mNumFrames;               ///< Number of frames in this cycle
    int mDofPerFrame;             ///< Values per frame in angle format (56)
    double mFrameTime;            ///< Time per frame in seconds (default: 1/60)

    // Parameter data (from HDF5 single files)
    std::vector<std::string> mParameterNames;  ///< Parameter names from /parameter_names dataset
    std::vector<float> mParameterValues;       ///< Parameter values from /param_state dataset

    // Height calibration state
    Eigen::Isometry3d mRootTransform;
    double mHeightOffset = 0.0;
    double mXOffset = 0.0;

    // ==================== Private Helper Methods ====================

    /**
     * @brief Load HDF5 file with flat structure
     * @param filepath Path to .h5/.hdf5 file
     */
    void loadFromFile(const std::string& filepath);

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
