#include "NPZ.h"
#include "Character.h"
#include "dart/dynamics/FreeJoint.hpp"
#include "../viewer/Log.h"
#include <iostream>
#include <algorithm>

using namespace dart::dynamics;

NPZ::NPZ(const std::string& filepath)
    : mFilename(filepath)
    , mFrameTime(1.0 / 60.0)
    , mNumFrames(0)
    , mDofPerFrame(101)  // NPZ format: 101 values per frame (6D rotation representation)
    , mNeedsConversion(true)
{
    loadFromFile(filepath);
}

void NPZ::loadFromFile(const std::string& filepath)
{
    py::gil_scoped_acquire gil;

    try {
        // Load NPZ file using numpy
        py::module np = py::module::import("numpy");
        py::object npz_file = np.attr("load")(filepath);

        // Extract motions array
        py::array_t<double> motions_array = npz_file.attr("__getitem__")("motions");
        auto motions_info = motions_array.request();

        if (motions_info.ndim != 2) {
            throw std::runtime_error("Expected 2D motions array, got " + std::to_string(motions_info.ndim) + "D");
        }

        // Get dimensions
        int num_sequences = motions_info.shape[0];
        int total_elements = motions_info.shape[1];

        if (num_sequences == 0) {
            throw std::runtime_error("No motion sequences found in NPZ file");
        }

        // Calculate number of frames from total elements
        // Example: 6060 elements ÷ 60 DOF/frame = 101 frames
        mNumFrames = total_elements / mDofPerFrame;

        if (total_elements % mDofPerFrame != 0) {
            LOG_WARN("[NPZ] Warning: Total elements (" << total_elements
                      << ") not evenly divisible by DOF per frame (" << mDofPerFrame << ")");
        }

        // Reshape flattened data to (numFrames, dofPerFrame)
        mMotionData.resize(mNumFrames, mDofPerFrame);

        double* data_ptr = static_cast<double*>(motions_info.ptr);

        // Copy data: row-major format
        for (int frame = 0; frame < mNumFrames; frame++) {
            for (int dof = 0; dof < mDofPerFrame; dof++) {
                mMotionData(frame, dof) = data_ptr[frame * mDofPerFrame + dof];
            }
        }

        // Extract params array (optional)
        try {
            py::array_t<double> params_array = npz_file.attr("__getitem__")("params");
            auto params_info = params_array.request();

            if (params_info.ndim == 2) {
                int num_params = params_info.shape[1];
                mParams.resize(num_params);
                double* params_ptr = static_cast<double*>(params_info.ptr);
                for (int i = 0; i < num_params; i++) {
                    mParams[i] = params_ptr[i];
                }
            } else if (params_info.ndim == 1) {
                int num_params = params_info.shape[0];
                mParams.resize(num_params);
                double* params_ptr = static_cast<double*>(params_info.ptr);
                for (int i = 0; i < num_params; i++) {
                    mParams[i] = params_ptr[i];
                }
            }
        } catch (const std::exception& e) {
            // Params are optional
            mParams.resize(0);
        }

    } catch (const std::exception& e) {
        std::cerr << "[NPZ] Error loading " << filepath << ": " << e.what() << std::endl;
        throw;
    }
}

Eigen::VectorXd NPZ::getPose(int frameIdx)
{
    // Clamp to valid range
    frameIdx = std::clamp(frameIdx, 0, mNumFrames - 1);

    Eigen::VectorXd raw_pose = mMotionData.row(frameIdx);

    // Convert from 6D rotation format to DART angle format
    Eigen::VectorXd pose;
    if (mNeedsConversion && mCharacter) {
        pose = mCharacter->sixDofToPos(raw_pose);
    } else {
        pose = raw_pose;
    }

    // Apply height calibration offsets if enabled
    if (mHeightCalibration) {
        pose[4] += mHeightOffset;  // Y position (height)
        pose[3] -= mXOffset;        // X position
    }

    return pose;
}

Eigen::VectorXd NPZ::getPose(double phase)
{
    // Normalize phase to [0, 1]
    while (phase < 0.0) phase += 1.0;
    while (phase >= 1.0) phase -= 1.0;

    // Convert phase to frame index with interpolation
    double frame_float = phase * (mNumFrames - 1);
    int frame1 = static_cast<int>(std::floor(frame_float));
    int frame2 = static_cast<int>(std::ceil(frame_float));
    double t = frame_float - frame1;

    // Clamp to valid range
    frame1 = std::clamp(frame1, 0, mNumFrames - 1);
    frame2 = std::clamp(frame2, 0, mNumFrames - 1);

    if (frame1 == frame2) {
        return getPose(frame1);
    }

    return interpolatePose(frame1, frame2, t);
}

Eigen::VectorXd NPZ::interpolatePose(int frame1, int frame2, double t) const
{
    Eigen::VectorXd raw_pose1 = mMotionData.row(frame1);
    Eigen::VectorXd raw_pose2 = mMotionData.row(frame2);

    // Linear interpolation in 6D rotation space
    Eigen::VectorXd interpolated_raw = (1.0 - t) * raw_pose1 + t * raw_pose2;

    // Convert to angle format if needed
    if (mNeedsConversion && mCharacter) {
        return mCharacter->sixDofToPos(interpolated_raw);
    } else {
        return interpolated_raw;
    }
}

Eigen::VectorXd NPZ::getTargetPose(double phase)
{
    return getPose(phase);
}

double NPZ::getMaxTime() const
{
    return mNumFrames * mFrameTime;
}

void NPZ::setRefMotion(Character* character, dart::simulation::WorldPtr world)
{
    mCharacter = character;

    if (mNumFrames == 0) {
        LOG_WARN("[NPZ] Warning: No frames loaded");
        return;
    }

    // Apply height calibration if enabled
    if (mHeightCalibration && world) {
        // Convert first frame from 6D to angle format
        Eigen::VectorXd raw_initial_pose = mMotionData.row(0);
        Eigen::VectorXd initial_pose = character->sixDofToPos(raw_initial_pose);
        character->getSkeleton()->setPositions(initial_pose);

        // Store initial root transform
        Eigen::Isometry3d initial_transform = FreeJoint::convertToTransform(initial_pose.head(6));

        // Perform height calibration
        character->heightCalibration(world, false);

        // Get calibrated pose and calculate offsets
        Eigen::VectorXd calibrated_pose = character->getSkeleton()->getPositions();
        mRootTransform = FreeJoint::convertToTransform(calibrated_pose.head(6));

        mHeightOffset = mRootTransform.translation()[1] - initial_transform.translation()[1];
        mXOffset = mRootTransform.translation()[0] - initial_transform.translation()[0];
    }
}

// Extended interface implementations for legacy ViewerMotion compatibility

Eigen::VectorXd NPZ::getRawMotionData() const
{
    // Flatten mMotionData (numFrames x 101) into 1D vector
    Eigen::VectorXd flattened(mMotionData.rows() * mMotionData.cols());
    for (int i = 0; i < mMotionData.rows(); ++i) {
        flattened.segment(i * mMotionData.cols(), mMotionData.cols()) = mMotionData.row(i);
    }
    return flattened;
}
