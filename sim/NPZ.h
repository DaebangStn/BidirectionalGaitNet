#ifndef __NPZ_H__
#define __NPZ_H__

#include "Motion.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

/**
 * @brief NPZ motion format implementation
 *
 * Loads motion data from NumPy .npz files containing:
 * - "motions": Flattened pose sequence (e.g., 6060 = 101 frames × 60 DOF)
 * - "params": Parameter vector (e.g., 279 elements)
 */
class NPZ : public Motion {
public:
    /**
     * @brief Construct NPZ motion from file
     * @param filepath Path to .npz file
     */
    explicit NPZ(const std::string& filepath);
    ~NPZ() override = default;

    // ==================== Motion Interface Implementation ====================

    Eigen::VectorXd getTargetPose(double phase) override;
    Eigen::VectorXd getPose(int frameIdx) override;
    Eigen::VectorXd getPose(double phase) override;

    double getMaxTime() const override;
    int getNumFrames() const override { return mNumFrames; }
    double getFrameTime() const override { return mFrameTime; }
    std::string getName() const override { return mFilename; }

    void setRefMotion(Character* character, dart::simulation::WorldPtr world) override;

    // ==================== NPZ-Specific Methods ====================

    /**
     * @brief Get parameter vector from NPZ file
     * @return Parameter vector
     */
    Eigen::VectorXd getParams() const { return mParams; }

    /**
     * @brief Set degrees of freedom per frame (default: 60)
     * @param dof DOF per frame
     */
    void setDofPerFrame(int dof) { mDofPerFrame = dof; }

private:
    std::string mFilename;
    Eigen::MatrixXd mMotionData;  ///< Motion data in 6D rotation format: (numFrames, 101)
    Eigen::VectorXd mParams;      ///< Parameter vector

    int mNumFrames;               ///< Number of frames in motion
    int mDofPerFrame;             ///< Values per frame in 6D rotation format (101)
    double mFrameTime;            ///< Time per frame in seconds (default: 1/60)

    bool mNeedsConversion;        ///< Whether data needs 6D-to-angle conversion

    // Height calibration state
    Eigen::Isometry3d mRootTransform;
    double mHeightOffset = 0.0;
    double mXOffset = 0.0;

    // ==================== Private Helper Methods ====================

    /**
     * @brief Load NPZ file using pybind11 and numpy
     * @param filepath Path to .npz file
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

#endif // __NPZ_H__
