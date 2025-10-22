#ifndef __HDFROLLOUT_H__
#define __HDFROLLOUT_H__

#include "Motion.h"
#include <H5Cpp.h>
#include <Eigen/Core>
#include <string>
#include <vector>

// Forward declaration
class Character;

/**
 * @brief HDF5 rollout motion format implementation
 *
 * Loads motion data from HDF5 rollout files with hierarchical structure:
 * - Root level: /parameter_names (dataset)
 * - Param groups: /param_0, /param_1, ... (each with param_state dataset)
 * - Cycle groups: /param_N/cycle_0, /param_N/cycle_1, ...
 * - Motion data: /param_N/cycle_M/{motions, phase, time}
 *
 * This class handles navigation and loading of specific param/cycle combinations.
 */
class HDFRollout : public Motion {
public:
    /**
     * @brief Construct HDFRollout motion from rollout file
     * @param filepath Path to .h5/.hdf5 rollout file
     */
    explicit HDFRollout(const std::string& filepath);
    ~HDFRollout() override = default;

    // ==================== Motion Interface Implementation ====================

    Eigen::VectorXd getTargetPose(double phase) override;
    Eigen::VectorXd getPose(int frameIdx) override;
    Eigen::VectorXd getPose(double phase) override;

    double getMaxTime() const override;
    int getNumFrames() const override { return mNumFrames; }
    double getFrameTime() const override { return mFrameTime; }
    std::string getName() const override;

    void setRefMotion(Character* character, dart::simulation::WorldPtr world) override;

    std::string getSourceType() const override { return "hdfRollout"; }
    std::string getLogHeader() const override { return "[HDF Rollout]"; }

    bool hasParameters() const override { return !mParameterNames.empty(); }
    std::vector<std::string> getParameterNames() const override { return mParameterNames; }
    std::vector<float> getParameterValues() const override { return mParameterValues; }
    bool applyParametersToEnvironment(Environment* env) const override;

    // Extended interface for legacy ViewerMotion compatibility
    Eigen::VectorXd getRawMotionData() const override;
    int getValuesPerFrame() const override { return 56; }
    std::vector<double> getTimestamps() const override;
    int getTotalTimesteps() const override { return mMotionData.rows(); }
    int getTimestepsPerCycle() const override { return mMotionData.rows(); }

    // ==================== HDFRollout-Specific Methods ====================

    /**
     * @brief Scan file structure to populate param/cycle lists
     */
    void scanStructure();

    /**
     * @brief Load specific param/cycle combination
     * @param paramIdx Parameter group index (0 to getNumParams()-1)
     * @param cycleIdx Cycle index (0 to getNumCycles()-1)
     */
    void loadParamCycle(int paramIdx, int cycleIdx);

    /**
     * @brief Get number of parameter groups in file
     * @return Number of param groups (param_0, param_1, ...)
     */
    int getNumParams() const { return mParamGroups.size(); }

    /**
     * @brief Get number of cycles in current param group
     * @return Number of cycle groups (cycle_0, cycle_1, ...)
     */
    int getNumCycles() const { return mCycleGroups.size(); }

    /**
     * @brief Get currently loaded parameter index
     * @return Current param index (-1 if none loaded)
     */
    int getCurrentParamIdx() const { return mSelectedParamIdx; }

    /**
     * @brief Get currently loaded cycle index
     * @return Current cycle index (-1 if none loaded)
     */
    int getCurrentCycleIdx() const { return mSelectedCycleIdx; }

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
    std::string mFilePath;

    // File structure (discovered by scanStructure)
    std::vector<std::string> mParamGroups;     ///< Available param groups ["param_0", "param_1", ...]
    std::vector<std::string> mCycleGroups;     ///< Available cycles in current param ["cycle_0", "cycle_1", ...]

    // Currently loaded param/cycle
    int mSelectedParamIdx;                     ///< Currently loaded param index (-1 if none)
    int mSelectedCycleIdx;                     ///< Currently loaded cycle index (-1 if none)

    // Parameter metadata (from root level)
    std::vector<std::string> mParameterNames;  ///< Parameter names from /parameter_names
    std::vector<float> mParameterValues;       ///< Current param values from param_N/param_state

    // Current motion data (from param_N/cycle_M)
    Eigen::MatrixXd mMotionData;               ///< Motion data in angle format: (numFrames, 56)
    Eigen::VectorXd mPhaseData;                ///< Phase values: (numFrames,)
    Eigen::VectorXd mTimeData;                 ///< Time values: (numFrames,)

    int mNumFrames;                            ///< Number of frames in loaded cycle
    int mDofPerFrame;                          ///< Values per frame in angle format (56)
    double mFrameTime;                         ///< Time per frame in seconds

    // Height calibration state
    Eigen::Isometry3d mRootTransform;
    double mHeightOffset = 0.0;
    double mXOffset = 0.0;

    // ==================== Private Helper Methods ====================

    /**
     * @brief Load root-level parameter names
     */
    void loadParameterNames();

    /**
     * @brief Scan param group for available cycles
     * @param paramIdx Parameter index to scan
     */
    void scanCycles(int paramIdx);

    /**
     * @brief Interpolate between two frames
     * @param frame1 First frame index
     * @param frame2 Second frame index
     * @param t Interpolation parameter [0, 1]
     * @return Interpolated pose
     */
    Eigen::VectorXd interpolatePose(int frame1, int frame2, double t) const;
};

#endif // __HDFROLLOUT_H__
