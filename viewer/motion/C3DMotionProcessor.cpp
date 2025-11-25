#include "C3DMotionProcessor.h"
#include "PlaybackController.h"
#include "C3DMotion.h"
#include "Motion.h"
#include "Character.h"
#include "../C3D_Reader.h"
#include "../GLFWApp.h"  // For PlaybackViewerState, PlaybackNavigationMode, DrawFlags

#include <iostream>
#include <cmath>

C3DMotionProcessor::C3DMotionProcessor(C3D_Reader* reader)
    : mReader(reader)
{
}

C3DMotionProcessor::~C3DMotionProcessor() = default;

// ==================== Core Operations ====================

Motion* C3DMotionProcessor::load(const std::string& path,
                                  Character* character,
                                  WorldPtr world)
{
    if (!mReader) {
        std::cerr << "[C3DMotionProcessor::load] Error: C3D_Reader not set" << std::endl;
        return nullptr;
    }

    // Create conversion params struct for C3D_Reader
    C3DConversionParams params;
    params.femurTorsionL = mParams.femurTorsionL;
    params.femurTorsionR = mParams.femurTorsionR;

    // Load and convert C3D to C3DMotion (IK conversion happens here)
    C3DMotion* motion = mReader->loadC3D(path, params);

    if (motion) {
        motion->setRefMotion(character, world);
    }

    return motion;
}

MotionProcessorContext C3DMotionProcessor::computePlayback(Motion* motion,
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

    // Update markers (C3D-specific)
    updateMarkers(context, motion, context.wrappedFrameFloat, state);

    // Update state
    state.currentPose = context.currentPose;
    state.currentMarkers = context.currentMarkers;

    context.valid = (context.currentPose.size() > 0);

    return context;
}

void C3DMotionProcessor::render(const MotionProcessorContext& context,
                                Character* character,
                                ShapeRenderer* renderer,
                                const DrawFlags& flags)
{
    // Render validation only - actual rendering is done by GLFWApp
    // GLFWApp::drawSkeleton for skeleton, GLFWApp::drawPlayableMarkers for markers

    if (!context.valid || !context.state || !context.state->render)
        return;

    if (context.currentPose.size() == 0)
        return;

    // Note: Actual skeleton rendering is handled by GLFWApp::drawSkeleton
    // Marker rendering is handled by GLFWApp using context.currentMarkers
    // The processor's responsibility is to populate context with pose and markers
}

// ==================== Cycle Distance ====================

Eigen::Vector3d C3DMotionProcessor::computeCycleDistance(Motion* motion) const
{
    return PlaybackController::computeCycleDistance(motion);
}

// ==================== C3D-Specific Methods ====================

void C3DMotionProcessor::setConversionParams(const C3DProcessorParams& params)
{
    mParams = params;
}

std::vector<Eigen::Vector3d> C3DMotionProcessor::getMarkersAtFrame(Motion* motion, int frameIdx) const
{
    if (!motion)
        return {};

    // Check if motion is C3DMotion
    C3DMotion* c3dMotion = dynamic_cast<C3DMotion*>(motion);
    if (!c3dMotion)
        return {};

    return std::vector<Eigen::Vector3d>(c3dMotion->getMarkers(frameIdx));
}

std::vector<Eigen::Vector3d> C3DMotionProcessor::getInterpolatedMarkers(Motion* motion, double frameFloat) const
{
    if (!motion)
        return {};

    // Check if motion is C3DMotion
    C3DMotion* c3dMotion = dynamic_cast<C3DMotion*>(motion);
    if (!c3dMotion)
        return {};

    return c3dMotion->getInterpolatedMarkers(frameFloat);
}

// ==================== Private Methods ====================

Eigen::VectorXd C3DMotionProcessor::evaluatePose(Motion* motion,
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
        std::cerr << "[C3DMotionProcessor::evaluatePose] Warning: Motion data too small! Expected "
                  << requiredSize << " but got " << rawMotion.size() << std::endl;
        return Eigen::VectorXd::Zero(valuesPerFrame);
    }

    if (weight > 1e-6) {
        int nextFrameIdx = (currentFrameIdx + 1) % totalFrames;
        Eigen::VectorXd p1 = rawMotion.segment(currentFrameIdx * valuesPerFrame, valuesPerFrame);
        Eigen::VectorXd p2 = rawMotion.segment(nextFrameIdx * valuesPerFrame, valuesPerFrame);

        // C3D motion: use Character's skeleton-aware interpolation
        bool phaseOverflow = (nextFrameIdx < currentFrameIdx);  // Detect cycle wraparound
        interpolatedFrame = PlaybackController::interpolatePose(p1, p2, weight, character, phaseOverflow);
    } else {
        // No interpolation needed, use exact frame
        interpolatedFrame = rawMotion.segment(currentFrameIdx * valuesPerFrame, valuesPerFrame);
    }

    // Apply position offset (same as HDF approach: additive)
    Eigen::VectorXd motionPos = interpolatedFrame;
    motionPos[3] += state.cycleAccumulation[0] + state.displayOffset[0];
    motionPos[4] += state.displayOffset[1];
    motionPos[5] += state.cycleAccumulation[2] + state.displayOffset[2];

    return motionPos;
}

void C3DMotionProcessor::updateMarkers(MotionProcessorContext& context,
                                       Motion* motion,
                                       double frameFloat,
                                       const PlaybackViewerState& state)
{
    // Check if motion is C3DMotion
    C3DMotion* c3dMotion = dynamic_cast<C3DMotion*>(motion);
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
