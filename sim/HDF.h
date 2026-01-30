#ifndef __HDF_H__
#define __HDF_H__

#include "Motion.h"
#include <H5Cpp.h>
#include <Eigen/Core>
#include <string>
#include <map>
#include <functional>

// Forward declaration
class Character;

/**
 * @brief Kinematics summary data for HDF5 export
 */
struct KinematicsExportData {
    std::vector<std::string> jointKeys;
    std::map<std::string, std::vector<double>> mean;   // 100-element vectors
    std::map<std::string, std::vector<double>> std;    // 100-element vectors
    int numCycles = 0;
};

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

    std::string getSourceType() const override { return "hdf"; }
    std::string getLogHeader() const override { return "[HDF]"; }

    bool hasParameters() const override { return !mParameterNames.empty(); }
    std::vector<std::string> getParameterNames() const override { return mParameterNames; }
    std::vector<float> getParameterValues() const override { return mParameterValues; }
    bool applyParametersToEnvironment(Environment* env) const override;

    // Extended interface for legacy ViewerMotion compatibility
    Eigen::VectorXd getRawMotionData() const override;
    int getValuesPerFrame() const override { return mDofPerFrame; }  // Actual DOF from file
    std::vector<double> getTimestamps() const override;
    int getTotalTimesteps() const override { return mMotionData.rows(); }
    int getTimestepsPerCycle() const override { return mMotionData.rows(); }
    Eigen::Vector3d getCycleDistance() const override { return mCycleDistance; }
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
     * @brief Apply Y-axis rotation to all frames
     * @param angleDegrees Rotation angle in degrees
     */
    void applyYRotation(double angleDegrees);

    /**
     * @brief Apply Y-axis rotation to a range of frames
     * @param startFrame Start frame (0-based, clamped to valid range)
     * @param endFrame End frame (inclusive, clamped to valid range)
     * @param angleDegrees Rotation angle in degrees
     */
    void applyYRotationToRange(int startFrame, int endFrame, double angleDegrees);

    /**
     * @brief Apply height offset to all frames
     * @param offset Height offset to add to Y translation (index 4)
     */
    void applyHeightOffset(double offset);

    /**
     * @brief Apply height offset to a range of frames
     * @param startFrame Start frame (0-based, clamped to valid range)
     * @param endFrame End frame (inclusive, clamped to valid range)
     * @param offset Height offset to add to Y translation (index 4)
     */
    void applyHeightOffsetToRange(int startFrame, int endFrame, double offset);

    /**
     * @brief Trim motion data in-place
     * @param startFrame Start frame (0-based)
     * @param endFrame End frame (inclusive)
     */
    void trim(int startFrame, int endFrame);

    /**
     * @brief Keep only specified frame ranges, concatenating them in order
     * @param ranges Vector of (startFrame, endFrame) pairs to keep
     *
     * Used for multi-interval trimming (e.g., after straightening backward intervals).
     * Each range is inclusive. Ranges are concatenated in the order provided.
     * Phase and time data are regenerated for the new frame count.
     */
    void keepFrameRanges(const std::vector<std::pair<int, int>>& ranges);

    /**
     * @brief Keep frame ranges with interpolation between them
     * @param ranges Vector of (startFrame, endFrame) pairs to keep
     * @param interpFrames Number of interpolation frames between consecutive ranges
     * @param interpolateFunc Callback for pose interpolation (pose1, pose2, t) -> interpolated pose
     */
    using InterpolateFn = std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&, double)>;
    void keepFrameRangesWithInterpolation(
        const std::vector<std::pair<int, int>>& ranges,
        int interpFrames,
        InterpolateFn interpolateFunc);

    /**
     * @brief Export current motion data to HDF5 file
     * @param outputPath Output file path
     * @param metadata Optional metadata map for attributes
     *
     * Writes:
     * - /motions: (numFrames x DOF) float32
     * - /phase: (numFrames,) normalized 0-1
     * - /time: (numFrames,)
     * - Attributes: frame_rate, num_frames, dof_per_frame, + metadata
     */
    void exportToFile(
        const std::string& outputPath,
        const std::map<std::string, std::string>& metadata = {},
        const KinematicsExportData* kinematics = nullptr) const;

    /**
     * @brief Get stride attribute from HDF5 file
     * @param defaultValue Value to return if attribute not found
     * @return Stride value or defaultValue
     */
    double getStrideAttribute(double defaultValue = -1.0) const;

    /**
     * @brief Check if reference kinematics data is available
     * @return true if /kinematics group was loaded from file
     */
    bool hasReferenceKinematics() const { return mHasReferenceKinematics; }

    /**
     * @brief Get reference kinematics data (mean/std over gait cycle)
     * @return Kinematics data with 100-element vectors per joint key
     */
    const KinematicsExportData& getReferenceKinematics() const { return mReferenceKinematics; }

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

    // Cycle distance for forward progression (computed in loadFromFile)
    Eigen::Vector3d mCycleDistance = Eigen::Vector3d::Zero();

    // Stride from file attribute (-1 = not set)
    double mStrideAttribute = -1.0;

    // Reference kinematics (loaded from /kinematics group if present)
    KinematicsExportData mReferenceKinematics;  ///< Mean/std over 100 gait cycle bins
    bool mHasReferenceKinematics = false;       ///< True if /kinematics group was loaded

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
