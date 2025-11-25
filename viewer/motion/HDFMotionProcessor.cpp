#include "HDFMotionProcessor.h"
#include "PlaybackController.h"
#include "HDF.h"
#include "Motion.h"
#include "Character.h"
#include "Environment.h"
#include "../GLFWApp.h"  // For PlaybackViewerState, PlaybackNavigationMode, DrawFlags
#include "../ShapeRenderer.h"

#include <iostream>
#include <cmath>

HDFMotionProcessor::HDFMotionProcessor() = default;
HDFMotionProcessor::~HDFMotionProcessor() = default;

// ==================== Core Operations ====================

Motion* HDFMotionProcessor::load(const std::string& path,
                                  Character* character,
                                  WorldPtr world)
{
    HDF* hdf = new HDF(path);
    hdf->setRefMotion(character, world);
    return hdf;
}

MotionProcessorContext HDFMotionProcessor::computePlayback(Motion* motion,
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

    // HDF doesn't have markers
    context.hasMarkers = false;

    // Update state
    state.currentPose = context.currentPose;

    context.valid = (context.currentPose.size() > 0);

    return context;
}

void HDFMotionProcessor::render(const MotionProcessorContext& context,
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
    // This allows GLFWApp to maintain consistent rendering style across all motion types

    // The processor's primary responsibility is to:
    // 1. Compute the pose (done in computePlayback)
    // 2. Store it in context.currentPose
    // 3. Validate render conditions (done above)

    // GLFWApp will call drawSkeleton(context.currentPose, color, isLineSkeleton)
}

// ==================== Parameter Operations ====================

bool HDFMotionProcessor::hasParameters(Motion* motion) const
{
    if (!motion)
        return false;
    return motion->hasParameters();
}

std::vector<std::string> HDFMotionProcessor::getParameterNames(Motion* motion) const
{
    if (!motion)
        return {};
    return motion->getParameterNames();
}

std::vector<float> HDFMotionProcessor::getParameterValues(Motion* motion) const
{
    if (!motion)
        return {};
    return motion->getParameterValues();
}

bool HDFMotionProcessor::applyParametersToEnvironment(Motion* motion, Environment* env) const
{
    if (!motion || !env)
        return false;
    return motion->applyParametersToEnvironment(env);
}

// ==================== Cycle Distance ====================

Eigen::Vector3d HDFMotionProcessor::computeCycleDistance(Motion* motion) const
{
    return PlaybackController::computeCycleDistance(motion);
}

// ==================== Private Methods ====================

Eigen::VectorXd HDFMotionProcessor::evaluatePose(Motion* motion,
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
        std::cerr << "[HDFMotionProcessor::evaluatePose] Warning: Motion data too small! Expected "
                  << requiredSize << " but got " << rawMotion.size() << std::endl;
        return Eigen::VectorXd::Zero(valuesPerFrame);
    }

    if (weight > 1e-6) {
        int nextFrameIdx = (currentFrameIdx + 1) % totalFrames;
        Eigen::VectorXd p1 = rawMotion.segment(currentFrameIdx * valuesPerFrame, valuesPerFrame);
        Eigen::VectorXd p2 = rawMotion.segment(nextFrameIdx * valuesPerFrame, valuesPerFrame);

        // HDF motion: use Character's skeleton-aware interpolation
        bool phaseOverflow = (nextFrameIdx < currentFrameIdx);  // Detect cycle wraparound
        interpolatedFrame = PlaybackController::interpolatePose(p1, p2, weight, character, phaseOverflow);
    } else {
        // No interpolation needed, use exact frame
        interpolatedFrame = rawMotion.segment(currentFrameIdx * valuesPerFrame, valuesPerFrame);
    }

    // Apply position offset (HDF approach: additive)
    Eigen::VectorXd motionPos = interpolatedFrame;
    motionPos[3] += state.cycleAccumulation[0] + state.displayOffset[0];
    motionPos[4] += state.displayOffset[1];
    motionPos[5] += state.cycleAccumulation[2] + state.displayOffset[2];

    return motionPos;
}
