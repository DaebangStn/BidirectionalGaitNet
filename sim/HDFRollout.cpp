#include "HDFRollout.h"
#include "Character.h"
#include "Environment.h"
#include "dart/dynamics/FreeJoint.hpp"
#include "../viewer/Log.h"
#include <iostream>
#include <algorithm>
#include <sstream>

using namespace dart::dynamics;

HDFRollout::HDFRollout(const std::string& filepath)
    : mFilePath(filepath)
    , mSelectedParamIdx(-1)
    , mSelectedCycleIdx(-1)
    , mNumFrames(0)
    , mDofPerFrame(56)
    , mFrameTime(1.0 / 60.0)
{
    // Load parameter names from root level
    loadParameterNames();

    // Scan file structure
    scanStructure();

    LOG_INFO("[HDFRollout] Loaded " << mFilePath);
    LOG_VERBOSE("[HDFRollout] Found " << mParamGroups.size() << " parameter groups");
}

void HDFRollout::loadParameterNames()
{
    try {
        H5::H5File file(mFilePath, H5F_ACC_RDONLY);

        // Read parameter_names from root level (shared across all params)
        if (!H5Lexists(file.getId(), "/parameter_names", H5P_DEFAULT)) {
            LOG_WARN("[HDFRollout] Warning: No parameter_names in file");
            file.close();
            return;
        }

        H5::DataSet param_names_ds = file.openDataSet("/parameter_names");
        H5::DataSpace param_names_space = param_names_ds.getSpace();

        hsize_t num_param_names;
        param_names_space.getSimpleExtentDims(&num_param_names);

        // Read variable-length strings
        H5::DataType dtype = param_names_ds.getDataType();
        std::vector<char*> c_strs(num_param_names);
        param_names_ds.read(c_strs.data(), dtype);

        mParameterNames.resize(num_param_names);
        for (size_t i = 0; i < num_param_names; i++) {
            mParameterNames[i] = std::string(c_strs[i]);
            free(c_strs[i]);  // Free HDF5-allocated strings
        }

        dtype.close();
        param_names_space.close();
        param_names_ds.close();
        file.close();

        LOG_VERBOSE("[HDFRollout] Loaded " << mParameterNames.size() << " parameter names");

    } catch (const H5::Exception& e) {
        std::cerr << "[HDFRollout] Error loading parameter names: " << e.getDetailMsg() << std::endl;
    }
}

void HDFRollout::scanStructure()
{
    try {
        H5::H5File file(mFilePath, H5F_ACC_RDONLY);

        // Scan for param groups (param_0, param_1, ...)
        mParamGroups.clear();
        hsize_t num_objs = file.getNumObjs();

        for (hsize_t i = 0; i < num_objs; i++) {
            std::string obj_name = file.getObjnameByIdx(i);
            if (obj_name.find("param_") == 0) {
                mParamGroups.push_back(obj_name);
            }
        }

        // Sort param groups numerically
        std::sort(mParamGroups.begin(), mParamGroups.end(),
                  [](const std::string& a, const std::string& b) {
                      int num_a = std::stoi(a.substr(6));  // Extract number from "param_N"
                      int num_b = std::stoi(b.substr(6));
                      return num_a < num_b;
                  });

        file.close();

    } catch (const H5::Exception& e) {
        std::cerr << "[HDFRollout] Error scanning structure: " << e.getDetailMsg() << std::endl;
    }
}

void HDFRollout::scanCycles(int paramIdx)
{
    if (paramIdx < 0 || paramIdx >= mParamGroups.size()) {
        LOG_WARN("[HDFRollout] Invalid param index: " << paramIdx);
        return;
    }

    try {
        H5::H5File file(mFilePath, H5F_ACC_RDONLY);
        H5::Group param_group = file.openGroup(mParamGroups[paramIdx]);

        // Scan for cycle groups
        mCycleGroups.clear();
        hsize_t num_cycles = param_group.getNumObjs();

        for (hsize_t i = 0; i < num_cycles; i++) {
            std::string obj_name = param_group.getObjnameByIdx(i);
            if (obj_name.find("cycle_") == 0) {
                mCycleGroups.push_back(obj_name);
            }
        }

        // Sort cycle groups numerically
        std::sort(mCycleGroups.begin(), mCycleGroups.end(),
                  [](const std::string& a, const std::string& b) {
                      int num_a = std::stoi(a.substr(6));  // Extract number from "cycle_N"
                      int num_b = std::stoi(b.substr(6));
                      return num_a < num_b;
                  });

        param_group.close();
        file.close();

    } catch (const H5::Exception& e) {
        std::cerr << "[HDFRollout] Error scanning cycles: " << e.getDetailMsg() << std::endl;
    }
}

void HDFRollout::loadParamCycle(int paramIdx, int cycleIdx)
{
    if (paramIdx < 0 || paramIdx >= mParamGroups.size()) {
        LOG_WARN("[HDFRollout] Invalid param index: " << paramIdx);
        return;
    }

    // Scan cycles if not already done or if param changed
    if (paramIdx != mSelectedParamIdx || mCycleGroups.empty()) {
        scanCycles(paramIdx);
    }

    if (cycleIdx < 0 || cycleIdx >= mCycleGroups.size()) {
        LOG_WARN("[HDFRollout] Invalid cycle index: " << cycleIdx);
        return;
    }

    try {
        H5::H5File file(mFilePath, H5F_ACC_RDONLY);

        // Construct paths
        std::string param_path = mParamGroups[paramIdx];
        std::string cycle_path = param_path + "/" + mCycleGroups[cycleIdx];

        LOG_VERBOSE("[HDFRollout] Loading " << cycle_path);

        // Load parameter values from param_N/param_state
        std::string param_state_path = param_path + "/param_state";
        if (H5Lexists(file.getId(), param_state_path.c_str(), H5P_DEFAULT)) {
            H5::DataSet param_state_ds = file.openDataSet(param_state_path);
            H5::DataSpace param_state_space = param_state_ds.getSpace();

            hsize_t num_param_values;
            param_state_space.getSimpleExtentDims(&num_param_values);

            mParameterValues.resize(num_param_values);
            param_state_ds.read(mParameterValues.data(), H5::PredType::NATIVE_FLOAT);

            param_state_space.close();
            param_state_ds.close();
        }

        // Open cycle group
        H5::Group cycle_group = file.openGroup(cycle_path);

        // Load motions dataset
        std::string motions_path = cycle_path + "/motions";
        H5::DataSet motions_ds = file.openDataSet(motions_path);
        H5::DataSpace motions_space = motions_ds.getSpace();

        hsize_t dims_motions[2];
        motions_space.getSimpleExtentDims(dims_motions, nullptr);

        mNumFrames = static_cast<int>(dims_motions[0]);
        int values_per_frame = static_cast<int>(dims_motions[1]);

        if (values_per_frame != mDofPerFrame) {
            LOG_WARN("[HDFRollout] Warning: Expected " << mDofPerFrame
                      << " DOF per frame, got " << values_per_frame);
            mDofPerFrame = values_per_frame;
        }

        // Read motion data
        mMotionData.resize(mNumFrames, mDofPerFrame);
        std::vector<float> buffer_motions(mNumFrames * mDofPerFrame);
        motions_ds.read(buffer_motions.data(), H5::PredType::NATIVE_FLOAT);

        for (int i = 0; i < mNumFrames; i++) {
            for (int j = 0; j < mDofPerFrame; j++) {
                mMotionData(i, j) = static_cast<double>(buffer_motions[i * mDofPerFrame + j]);
            }
        }

        motions_space.close();
        motions_ds.close();

        // Load phase dataset
        std::string phase_path = cycle_path + "/phase";
        H5::DataSet phase_ds = file.openDataSet(phase_path);
        H5::DataSpace phase_space = phase_ds.getSpace();

        hsize_t dims_phase[1];
        phase_space.getSimpleExtentDims(dims_phase, nullptr);

        mPhaseData.resize(static_cast<int>(dims_phase[0]));
        std::vector<float> buffer_phase(dims_phase[0]);
        phase_ds.read(buffer_phase.data(), H5::PredType::NATIVE_FLOAT);

        for (int i = 0; i < mPhaseData.size(); i++) {
            mPhaseData[i] = static_cast<double>(buffer_phase[i]);
        }

        phase_space.close();
        phase_ds.close();

        // Load time dataset
        std::string time_path = cycle_path + "/time";
        H5::DataSet time_ds = file.openDataSet(time_path);
        H5::DataSpace time_space = time_ds.getSpace();

        hsize_t dims_time[1];
        time_space.getSimpleExtentDims(dims_time, nullptr);

        mTimeData.resize(static_cast<int>(dims_time[0]));
        std::vector<float> buffer_time(dims_time[0]);
        time_ds.read(buffer_time.data(), H5::PredType::NATIVE_FLOAT);

        for (int i = 0; i < mTimeData.size(); i++) {
            mTimeData[i] = static_cast<double>(buffer_time[i]);
        }

        time_space.close();
        time_ds.close();

        // Calculate frame time from time data
        if (mTimeData.size() >= 2) {
            double avg_dt = (mTimeData[mTimeData.size() - 1] - mTimeData[0]) / (mTimeData.size() - 1);
            if (avg_dt > 0) {
                mFrameTime = avg_dt;
            }
        }

        cycle_group.close();
        file.close();

        // Update selection
        mSelectedParamIdx = paramIdx;
        mSelectedCycleIdx = cycleIdx;

        LOG_VERBOSE("[HDFRollout] Loaded " << mNumFrames << " frames from "
                  << param_path << "/" << mCycleGroups[cycleIdx]);

    } catch (const H5::Exception& e) {
        std::cerr << "[HDFRollout] Error loading param/cycle: " << e.getDetailMsg() << std::endl;
        throw;
    }
}

Eigen::VectorXd HDFRollout::getPose(int frameIdx)
{
    if (mSelectedParamIdx < 0 || mSelectedCycleIdx < 0) {
        LOG_WARN("[HDFRollout] No param/cycle loaded");
        return Eigen::VectorXd::Zero(mDofPerFrame);
    }

    // Clamp to valid range
    frameIdx = std::clamp(frameIdx, 0, mNumFrames - 1);

    Eigen::VectorXd pose = mMotionData.row(frameIdx);

    // Apply height calibration if enabled
    if (mHeightCalibration && pose.size() >= 6) {
        pose[4] += mHeightOffset;  // Y position (height)
        pose[3] -= mXOffset;        // X position
    }

    return pose;
}

Eigen::VectorXd HDFRollout::getPose(double phase)
{
    if (mSelectedParamIdx < 0 || mSelectedCycleIdx < 0) {
        LOG_WARN("[HDFRollout] No param/cycle loaded");
        return Eigen::VectorXd::Zero(mDofPerFrame);
    }

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

Eigen::VectorXd HDFRollout::interpolatePose(int frame1, int frame2, double t) const
{
    Eigen::VectorXd pose1 = mMotionData.row(frame1);
    Eigen::VectorXd pose2 = mMotionData.row(frame2);

    // Use Character's skeleton-aware interpolation if available
    if (mCharacter) {
        return mCharacter->interpolatePose(pose1, pose2, t, false);
    } else {
        // Fallback to linear interpolation
        return (1.0 - t) * pose1 + t * pose2;
    }
}

Eigen::VectorXd HDFRollout::getTargetPose(double phase)
{
    return getPose(phase);
}

double HDFRollout::getMaxTime() const
{
    return mNumFrames * mFrameTime;
}

std::string HDFRollout::getName() const
{
    if (mSelectedParamIdx >= 0 && mSelectedCycleIdx >= 0) {
        return mFilePath + "/" + mParamGroups[mSelectedParamIdx] + "/" + mCycleGroups[mSelectedCycleIdx];
    }
    return mFilePath;
}

void HDFRollout::setRefMotion(Character* character, dart::simulation::WorldPtr world)
{
    mCharacter = character;

    if (mNumFrames == 0 || mSelectedParamIdx < 0 || mSelectedCycleIdx < 0) {
        LOG_WARN("[HDFRollout] Warning: No motion loaded for calibration");
        return;
    }

    // Apply height calibration if enabled
    if (mHeightCalibration && world) {
        // Get initial pose (already in angle format)
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

        LOG_VERBOSE("[HDFRollout] Height calibration applied: Y offset = " << mHeightOffset
                  << ", X offset = " << mXOffset);
    }
}

// Extended interface implementations for legacy ViewerMotion compatibility

Eigen::VectorXd HDFRollout::getRawMotionData() const
{
    // Flatten mMotionData (numFrames x 56) into 1D vector
    Eigen::VectorXd flattened(mMotionData.rows() * mMotionData.cols());
    for (int i = 0; i < mMotionData.rows(); ++i) {
        flattened.segment(i * mMotionData.cols(), mMotionData.cols()) = mMotionData.row(i);
    }
    return flattened;
}

std::vector<double> HDFRollout::getTimestamps() const
{
    // Convert Eigen::VectorXd to std::vector<double>
    std::vector<double> timestamps(mTimeData.size());
    for (int i = 0; i < mTimeData.size(); ++i) {
        timestamps[i] = mTimeData[i];
    }
    return timestamps;
}

bool HDFRollout::applyParametersToEnvironment(Environment* env) const
{
    if (!hasParameters() || !env) {
        return false;
    }

    // Get simulation parameter names and current state
    const std::vector<std::string>& sim_param_names = env->getParamName();
    Eigen::VectorXd current_params = env->getParamState();

    // Name-based matching: build new parameter vector with matched values
    Eigen::VectorXd new_params = current_params;  // Start with defaults
    int matched_count = 0;

    // Match HDFRollout parameter names with simulation parameter names
    for (size_t i = 0; i < mParameterNames.size(); i++) {
        for (size_t j = 0; j < sim_param_names.size(); j++) {
            if (mParameterNames[i] == sim_param_names[j]) {
                new_params[j] = static_cast<double>(mParameterValues[i]);
                matched_count++;
                break;
            }
        }
    }

    std::cout << "[HDFRollout] Matched " << matched_count << " / " << mParameterNames.size()
              << " parameters (Environment has " << sim_param_names.size() << " parameters)" << std::endl;

    // Apply matched parameters
    env->setParamState(new_params, false, true);
    return true;
}
