#include "HDF.h"
#include "Character.h"
#include "dart/dynamics/FreeJoint.hpp"
#include <iostream>
#include <algorithm>
#include <sstream>

using namespace dart::dynamics;

HDF::HDF(const std::string& filepath)
    : mFilename(filepath)
    , mFrameTime(1.0 / 60.0)
    , mNumFrames(0)
    , mDofPerFrame(56)  // HDF format: 56 values per frame (angle format, no conversion needed)
{
    loadFromFile(filepath);
}

void HDF::loadFromFile(const std::string& filepath)
{
    try {
        // Open HDF5 file
        H5::H5File file(filepath, H5F_ACC_RDONLY);

        // Check for flat structure (extracted single-cycle files only)
        if (!H5Lexists(file.getId(), "/motions", H5P_DEFAULT)) {
            std::cerr << "[HDF] Error: File must have flat structure with /motions dataset at top level." << std::endl;
            std::cerr << "[HDF] Use extract_cycle tool to extract single cycles from rollout files." << std::endl;
            throw std::runtime_error("HDF file does not have required flat structure");
        }

        // Use flat structure paths
        std::string path_motions = "/motions";
        std::string path_phase = "/phase";
        std::string path_time = "/time";

        // Load motions dataset
        H5::DataSet dataset_motions = file.openDataSet(path_motions);
        H5::DataSpace dataspace_motions = dataset_motions.getSpace();

        // Get dimensions
        hsize_t dims_motions[2];
        int ndims = dataspace_motions.getSimpleExtentDims(dims_motions, nullptr);

        if (ndims != 2) {
            throw std::runtime_error("[HDF] Expected 2D motions array, got " + std::to_string(ndims) + "D");
        }

        mNumFrames = static_cast<int>(dims_motions[0]);
        int values_per_frame = static_cast<int>(dims_motions[1]);

        if (values_per_frame != mDofPerFrame) {
            std::cerr << "[HDF] Warning: Expected " << mDofPerFrame << " DOF per frame, got "
                      << values_per_frame << std::endl;
            mDofPerFrame = values_per_frame;
        }

        // Allocate and read motions data
        mMotionData.resize(mNumFrames, mDofPerFrame);
        std::vector<float> buffer_motions(mNumFrames * mDofPerFrame);
        dataset_motions.read(buffer_motions.data(), H5::PredType::NATIVE_FLOAT);

        // Copy to Eigen matrix (convert float to double)
        for (int i = 0; i < mNumFrames; i++) {
            for (int j = 0; j < mDofPerFrame; j++) {
                mMotionData(i, j) = static_cast<double>(buffer_motions[i * mDofPerFrame + j]);
            }
        }

        // Load phase dataset
        H5::DataSet dataset_phase = file.openDataSet(path_phase);
        H5::DataSpace dataspace_phase = dataset_phase.getSpace();
        hsize_t dims_phase[1];
        dataspace_phase.getSimpleExtentDims(dims_phase, nullptr);

        mPhaseData.resize(static_cast<int>(dims_phase[0]));
        std::vector<float> buffer_phase(dims_phase[0]);
        dataset_phase.read(buffer_phase.data(), H5::PredType::NATIVE_FLOAT);

        for (int i = 0; i < mPhaseData.size(); i++) {
            mPhaseData[i] = static_cast<double>(buffer_phase[i]);
        }

        // Load time dataset
        H5::DataSet dataset_time = file.openDataSet(path_time);
        H5::DataSpace dataspace_time = dataset_time.getSpace();
        hsize_t dims_time[1];
        dataspace_time.getSimpleExtentDims(dims_time, nullptr);

        mTimeData.resize(static_cast<int>(dims_time[0]));
        std::vector<float> buffer_time(dims_time[0]);
        dataset_time.read(buffer_time.data(), H5::PredType::NATIVE_FLOAT);

        for (int i = 0; i < mTimeData.size(); i++) {
            mTimeData[i] = static_cast<double>(buffer_time[i]);
        }

        // Calculate frame time from time data if available
        if (mTimeData.size() >= 2) {
            double avg_dt = (mTimeData[mTimeData.size() - 1] - mTimeData[0]) / (mTimeData.size() - 1);
            if (avg_dt > 0) {
                mFrameTime = avg_dt;
            }
        }

        file.close();

        std::cout << "[HDF] Loaded " << filepath
                  << " with " << mNumFrames << " frames (" << mDofPerFrame << " DOF/frame, "
                  << mFrameTime << " s/frame)" << std::endl;

    } catch (const H5::Exception& e) {
        std::cerr << "[HDF] HDF5 error loading " << filepath << ": " << e.getDetailMsg() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "[HDF] Error loading " << filepath << ": " << e.what() << std::endl;
        throw;
    }
}

Eigen::VectorXd HDF::getPose(int frameIdx)
{
    // Clamp to valid range
    frameIdx = std::clamp(frameIdx, 0, mNumFrames - 1);

    Eigen::VectorXd pose = mMotionData.row(frameIdx);

    // Apply height calibration offsets if enabled
    if (mHeightCalibration && pose.size() >= 6) {
        pose[4] += mHeightOffset;  // Y position (height)
        pose[3] -= mXOffset;        // X position
    }

    return pose;
}

Eigen::VectorXd HDF::getPose(double phase)
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

Eigen::VectorXd HDF::interpolatePose(int frame1, int frame2, double t) const
{
    Eigen::VectorXd pose1 = mMotionData.row(frame1);
    Eigen::VectorXd pose2 = mMotionData.row(frame2);

    // Use Character's skeleton-aware interpolation
    return mCharacter->interpolatePose(pose1, pose2, t, false);
}

Eigen::VectorXd HDF::getTargetPose(double phase)
{
    return getPose(phase);
}

double HDF::getMaxTime() const
{
    return mNumFrames * mFrameTime;
}

void HDF::setRefMotion(Character* character, dart::simulation::WorldPtr world)
{
    mCharacter = character;

    if (mNumFrames == 0) {
        std::cerr << "[HDF] Warning: No frames loaded" << std::endl;
        return;
    }

    // Apply height calibration if enabled
    if (mHeightCalibration && world) {
        // Get initial pose (already in angle format, no conversion needed)
        Eigen::VectorXd initial_pose = mMotionData.row(0);
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

        std::cout << "[HDF] Height calibration applied: Y offset = " << mHeightOffset
                  << ", X offset = " << mXOffset << std::endl;
    }
}
