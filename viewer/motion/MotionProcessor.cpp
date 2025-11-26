#include "MotionProcessor.h"
#include "PlaybackController.h"
#include "PlaybackContext.h"
#include "HDF.h"
#include "C3D.h"
#include "Motion.h"
#include "Character.h"
#include "Environment.h"
#include "../C3D_Reader.h"
#include "../GLFWApp.h"  // For PlaybackViewerState, PlaybackNavigationMode, DrawFlags

#include <iostream>
#include <cmath>
#include <algorithm>

MotionProcessor::MotionProcessor() = default;
MotionProcessor::~MotionProcessor() = default;

// ==================== Type Information ====================

bool MotionProcessor::isHDFExtension(const std::string& ext)
{
    return ext == ".h5" || ext == ".hdf5";
}

bool MotionProcessor::isC3DExtension(const std::string& ext)
{
    return ext == ".c3d";
}

std::string MotionProcessor::extractExtension(const std::string& path)
{
    size_t dot = path.rfind('.');
    if (dot == std::string::npos)
        return "";

    std::string ext = path.substr(dot);
    // Convert to lowercase
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

// ==================== Core Operations ====================

Motion* MotionProcessor::load(const std::string& path,
                               Character* character,
                               WorldPtr world)
{
    std::string ext = extractExtension(path);

    Motion* motion = nullptr;

    if (isHDFExtension(ext)) {
        HDF* hdf = new HDF(path);
        motion = hdf;
    }
    else if (isC3DExtension(ext)) {
        if (!mReader) {
            std::cerr << "[MotionProcessor::load] Error: C3D_Reader not set" << std::endl;
            return nullptr;
        }

        // Create conversion params struct for C3D_Reader
        C3DConversionParams params;
        params.femurTorsionL = mParams.femurTorsionL;
        params.femurTorsionR = mParams.femurTorsionR;

        // Load and convert C3D (IK conversion happens here)
        motion = mReader->loadC3D(path, params);
    }
    else {
        std::cerr << "[MotionProcessor::load] Unsupported file extension: " << ext << std::endl;
        return nullptr;
    }

    if (motion) {
        motion->setRefMotion(character, world);
    }

    return motion;
}

MotionProcessorContext MotionProcessor::computePlayback(Motion* motion,
                                                         double viewerTime,
                                                         double viewerPhase,
                                                         PlaybackViewerState& state,
                                                         Character* character)
{
    MotionProcessorContext context;
    context.reset();

    if (!motion) {
        return context;
    }

    context.motion = motion;
    context.state = &state;
    context.character = character;
    context.viewerTime = viewerTime;
    context.viewerPhase = viewerPhase;

    // Get motion info
    context.totalFrames = motion->getTotalTimesteps();
    context.valuesPerFrame = motion->getValuesPerFrame();

    if (context.totalFrames <= 0 || context.valuesPerFrame <= 0) {
        return context;
    }

    // Compute phase (use viewer phase for simplicity)
    context.phase = viewerPhase;

    // Determine frame based on navigation mode
    context.frameFloat = PlaybackController::determineMotionFrame(motion, state, context.phase);

    // Wrap frame to valid range
    context.wrappedFrameFloat = PlaybackController::wrapFrameFloat(
        context.frameFloat, context.totalFrames);

    // Compute integer frame index
    context.frameIndex = PlaybackController::computeFrameIndex(
        context.wrappedFrameFloat, context.totalFrames);

    // Detect cycle wrap
    context.cycleWrapped = PlaybackController::detectCycleWrap(
        context.frameIndex, state.lastFrameIdx, context.totalFrames);

    // Update cycle accumulation
    PlaybackController::updateCycleAccumulationStandard(state, context.frameIndex);

    // Copy accumulated values to context
    context.cycleAccumulation = state.cycleAccumulation;
    context.displayOffset = state.displayOffset;

    // Evaluate pose
    context.currentPose = evaluatePose(motion, context.wrappedFrameFloat, character, state);

    // Conditionally update markers (C3D-specific)
    if (motion->getSourceType() == "c3d") {
        updateMarkers(context, motion, context.wrappedFrameFloat, state);
    } else {
        context.hasMarkers = false;
    }

    // Update state
    state.currentPose = context.currentPose;
    if (context.hasMarkers) {
        state.currentMarkers = context.currentMarkers;
    }

    context.valid = (context.currentPose.size() > 0);

    return context;
}

void MotionProcessor::render(const MotionProcessorContext& context,
                              Character* character,
                              ShapeRenderer* renderer,
                              const DrawFlags& flags)
{
    // Render validation only - actual rendering is done by GLFWApp::drawSkeleton
    // This method will be called from GLFWApp which has access to drawSkeleton
    // The context provides currentPose which GLFWApp uses for rendering

    if (!context.valid || !context.state || !context.state->render)
        return;

    if (context.currentPose.size() == 0)
        return;

    // Note: Actual skeleton rendering is handled by GLFWApp::drawSkeleton
    // which is called after computePlayback() populates context.currentPose
    // Marker rendering is handled by GLFWApp using context.currentMarkers
    // The processor's responsibility is to populate context with pose and markers
}

// ==================== Cycle Distance ====================

Eigen::Vector3d MotionProcessor::computeCycleDistance(Motion* motion) const
{
    return PlaybackController::computeCycleDistance(motion);
}

// ==================== Parameter Methods ====================

bool MotionProcessor::hasParameters(Motion* motion) const
{
    if (!motion)
        return false;
    return motion->hasParameters();
}

std::vector<std::string> MotionProcessor::getParameterNames(Motion* motion) const
{
    if (!motion)
        return {};
    return motion->getParameterNames();
}

std::vector<float> MotionProcessor::getParameterValues(Motion* motion) const
{
    if (!motion)
        return {};
    return motion->getParameterValues();
}

bool MotionProcessor::applyParametersToEnvironment(Motion* motion, Environment* env) const
{
    if (!motion || !env)
        return false;
    return motion->applyParametersToEnvironment(env);
}

// ==================== Marker Methods (C3D-specific) ====================

std::vector<Eigen::Vector3d> MotionProcessor::getMarkersAtFrame(Motion* motion, int frameIdx) const
{
    if (!motion)
        return {};

    // Check if motion is C3D
    C3D* c3dMotion = dynamic_cast<C3D*>(motion);
    if (!c3dMotion)
        return {};

    return std::vector<Eigen::Vector3d>(c3dMotion->getMarkers(frameIdx));
}

std::vector<Eigen::Vector3d> MotionProcessor::getInterpolatedMarkers(Motion* motion, double frameFloat) const
{
    if (!motion)
        return {};

    // Check if motion is C3D
    C3D* c3dMotion = dynamic_cast<C3D*>(motion);
    if (!c3dMotion)
        return {};

    return c3dMotion->getInterpolatedMarkers(frameFloat);
}

// ==================== Direct Pose Evaluation (Public) ====================

Eigen::VectorXd MotionProcessor::evaluatePoseAtFrame(Motion* motion, double frameFloat,
                                                      Character* character, const PlaybackViewerState& state)
{
    // Delegate to private evaluatePose
    return evaluatePose(motion, frameFloat, character, state);
}

std::vector<Eigen::Vector3d> MotionProcessor::getMarkersAtFrameWithOffsets(Motion* motion, int frameIdx,
                                                                            const PlaybackViewerState& state)
{
    if (!motion)
        return {};

    // Check if motion is C3D
    C3D* c3dMotion = dynamic_cast<C3D*>(motion);
    if (!c3dMotion)
        return {};

    std::vector<Eigen::Vector3d> markers = c3dMotion->getMarkers(frameIdx);
    if (markers.empty())
        return {};

    // Apply display offset and cycle accumulation to markers
    for (auto& marker : markers) {
        marker += state.displayOffset + state.cycleAccumulation;
    }

    return markers;
}

// ==================== Private Methods ====================

Eigen::VectorXd MotionProcessor::evaluatePose(Motion* motion,
                                               double frameFloat,
                                               Character* character,
                                               const PlaybackViewerState& state)
{
    if (!motion || !character) {
        return Eigen::VectorXd();
    }

    int valuesPerFrame = motion->getValuesPerFrame();
    int totalFrames = motion->getTotalTimesteps();

    if (totalFrames <= 0 || valuesPerFrame <= 0)
        return Eigen::VectorXd();

    // Clamp frame_float to valid range
    if (frameFloat < 0) frameFloat = 0;
    if (frameFloat >= totalFrames) frameFloat = std::fmod(frameFloat, totalFrames);

    int currentFrameIdx = static_cast<int>(frameFloat);
    currentFrameIdx = std::max(0, std::min(currentFrameIdx, totalFrames - 1));

    // Extract and interpolate frame data
    Eigen::VectorXd interpolatedFrame;
    double weight = frameFloat - std::floor(frameFloat);

    Eigen::VectorXd rawMotion = motion->getRawMotionData();

    // Safety check: ensure motion data is large enough
    int requiredSize = totalFrames * valuesPerFrame;
    if (rawMotion.size() < requiredSize) {
        std::cerr << "[MotionProcessor::evaluatePose] Warning: Motion data too small! Expected "
                  << requiredSize << " but got " << rawMotion.size() << std::endl;
        return Eigen::VectorXd::Zero(valuesPerFrame);
    }

    if (weight > 1e-6) {
        int nextFrameIdx = (currentFrameIdx + 1) % totalFrames;
        Eigen::VectorXd p1 = rawMotion.segment(currentFrameIdx * valuesPerFrame, valuesPerFrame);
        Eigen::VectorXd p2 = rawMotion.segment(nextFrameIdx * valuesPerFrame, valuesPerFrame);

        // Use Character's skeleton-aware interpolation
        bool phaseOverflow = (nextFrameIdx < currentFrameIdx);  // Detect cycle wraparound
        interpolatedFrame = PlaybackController::interpolatePose(p1, p2, weight, character, phaseOverflow);
    } else {
        // No interpolation needed, use exact frame
        interpolatedFrame = rawMotion.segment(currentFrameIdx * valuesPerFrame, valuesPerFrame);
    }

    // Apply position offset (additive)
    Eigen::VectorXd motionPos = interpolatedFrame;
    motionPos[3] += state.cycleAccumulation[0] + state.displayOffset[0];
    motionPos[4] += state.displayOffset[1];
    motionPos[5] += state.cycleAccumulation[2] + state.displayOffset[2];

    return motionPos;
}

void MotionProcessor::updateMarkers(MotionProcessorContext& context,
                                     Motion* motion,
                                     double frameFloat,
                                     const PlaybackViewerState& state)
{
    // Check if motion is C3D
    C3D* c3dMotion = dynamic_cast<C3D*>(motion);
    if (!c3dMotion) {
        context.hasMarkers = false;
        return;
    }

    // Get interpolated markers
    std::vector<Eigen::Vector3d> markers = c3dMotion->getInterpolatedMarkers(frameFloat);
    if (markers.empty()) {
        context.hasMarkers = false;
        return;
    }

    // Apply display offset and cycle accumulation to markers
    for (auto& marker : markers) {
        marker += state.displayOffset + state.cycleAccumulation;
    }

    context.currentMarkers = std::move(markers);
    context.hasMarkers = true;
}
