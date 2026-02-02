#include "GaitPhase.h"
#include "Log.h"
#include <iomanip>
#include <sstream>

// ============================================================================
// Debug Flags
// ============================================================================
#define LOG_STATE_CHANGES 0  // Set to 1 to enable state change logging, 0 to disable
// #define LOG_STATE_CHANGES 1

GaitPhase::GaitPhase(Character* character,
                     dart::simulation::WorldPtr world,
                     double motionCycleTime,
                     double refStride,
                     UpdateMode mode,
                     double controlHz,
                     double simulationHz)
    : mCharacter(character),
      mWorld(world),
      mMode(mode),
      mMotionCycleTime(motionCycleTime),
      mRefStride(refStride),
      mCadence(1.0),
      mStride(1.0),
      mSimTime(0.0),
      mAdaptiveTime(0.0),
      mControlHz(controlHz),
      mSimulationHz(simulationHz),
      mPhaseAction(0.0),
      mCachedContact(Eigen::Vector2i::Zero()),
      mCachedGRF(Eigen::Vector2d::Zero()),
      mState(RIGHT_STANCE),
      mIsLeftLegStance(false),
      mCycleCount(0),
      mIsGaitCycleComplete(false),
      mGaitCycleCompletePD(false),
      mIsStepComplete(false),
      mStepCompletePD(false),
      mStepMinRatio(0.3),
      mGRFThreshold(0.2),
      mContactDebounceAlpha(0.25),
      mContactDebounceThreshold(0.5),
      mDebouncedContactL(0.0),
      mDebouncedContactR(0.0),
      mIsFirstContactUpdate(true)
{
    mPrevContact = Eigen::Vector2i::Zero();
    mCurrentFoot = Eigen::Vector3d::Zero();
    mCurrentTargetFoot = Eigen::Vector3d::Zero();
    mNextTargetFoot = Eigen::Vector3d::Zero();
    mCurrentStanceFoot = Eigen::Vector3d::Zero();
    mCurCycleCOM = Eigen::Vector3d::Zero();
    mPrevCycleCOM = Eigen::Vector3d::Zero();
}

void GaitPhase::reset(double time)
{
    // Reset counters and flags
    mCycleCount = 0;
    mIsGaitCycleComplete = false;
    mGaitCycleCompletePD = false;
    mIsStepComplete = false;
    mStepCompletePD = false;

    // Reset timing
    mSimTime = time;
    mAdaptiveTime = time;
    mSwingTimeR = 0.0;
    mSwingTimeL = 0.0;
    mStanceTimeR = 0.0;
    mStanceTimeL = 0.0;
    mRightPhaseUpTime = 0.0;
    mLeftPhaseUpTime = 0.0;
    mPhaseTotalR = 0.0;
    mPhaseTotalL = 0.0;
    mStrideLengthR = 0.0;
    mStrideLengthL = 0.0;

    // Reset cycle tracking
    mCycleTime = 0.0;
    mCurCycleTime = 0.0;
    mPrevCycleTime = 0.0;
    mCycleDist = 0.0;
    mCurrentCadence = 0.0;
    mStanceRatioR = 0.0;
    mStanceRatioL = 0.0;

    // Get initial state from skeleton and motion
    auto skel = mCharacter->getSkeleton();
    Eigen::Vector3d footRPos = getLowestPointFromBodyNodes({"TalusR", "FootPinkyR", "FootThumbR"});
    Eigen::Vector3d footLPos = getLowestPointFromBodyNodes({"TalusL", "FootPinkyL", "FootThumbL"});

    // Initialize contact state (will be updated in first step())
    mCachedContact = Eigen::Vector2i::Zero();
    mCachedGRF = Eigen::Vector2d::Zero();
    mPrevContact = Eigen::Vector2i::Zero();

    // Reset debouncer state
    mDebouncedContactL = 0.0;
    mDebouncedContactR = 0.0;
    mIsFirstContactUpdate = true;

    // Determine state from foot heights
    // Lowest foot is in contact; if height differs < 0.005m, both are in contact
    double heightDiff = std::abs(footRPos[1] - footLPos[1]);
    bool bothContact = heightDiff < 0.005;
    bool rightLower = footRPos[1] < footLPos[1];

    if (bothContact) {
        // Both feet in contact - double support phase
        // Use phase to determine which leg is transitioning
        if (getAdaptivePhase() < 0.4) {
            mState = RIGHT_STANCE;
            mIsLeftLegStance = false;
            mCurrentFoot = footRPos;
            mCurrentStanceFoot = footRPos;
        } else {
            mState = LEFT_STANCE;
            mIsLeftLegStance = true;
            mCurrentFoot = footLPos;
            mCurrentStanceFoot = footLPos;
        }
    } else if (rightLower) {
        // Right foot lower (in contact), left foot in swing
        mState = LEFT_TAKEOFF;
        mIsLeftLegStance = false;
        mCurrentFoot = footRPos;
        mCurrentStanceFoot = footRPos;
    } else {
        // Left foot lower (in contact), right foot in swing
        mState = RIGHT_TAKEOFF;
        mIsLeftLegStance = true;
        mCurrentFoot = footLPos;
        mCurrentStanceFoot = footLPos;
    }

    resetFootPos();
    // Initialize foot targets
    mCurrentTargetFoot = mCurrentStanceFoot;
    mNextTargetFoot = mCurrentStanceFoot;

    double targetStride = mStride * mRefStride * mCharacter->getGlobalRatio();
    mNextTargetFoot[2] += targetStride / 2.0;
    mNextTargetFoot[0] *= -1.0;

    // Initialize COM tracking
    mCurCycleCOM = skel->getBodyNode("Pelvis")->getCOM();
    mPrevCycleCOM = mCurCycleCOM;
}

void GaitPhase::resetFootPos()
{
    auto skel = mCharacter->getSkeleton();
    mFootPrevRz = skel->getBodyNode("TalusR")->getCOM()[2];
    mFootPrevLz = skel->getBodyNode("TalusL")->getCOM()[2];
}

void GaitPhase::step()
{
    // Update time
    mSimTime += 1.0 / mSimulationHz;
    mAdaptiveTime += (1.0 + mPhaseAction) / mSimulationHz;

    // Hard phase clipping (always enabled)
    double cycleTime = mMotionCycleTime / mCadence;
    int currentGlobalCount = mSimTime / cycleTime;
    int currentLocalCount = mAdaptiveTime / cycleTime;

    if (currentGlobalCount > currentLocalCount) mAdaptiveTime = mSimTime;
    else if (currentGlobalCount < currentLocalCount) mAdaptiveTime = currentLocalCount * cycleTime;

    mIsGaitCycleComplete = false;  // Reset flag
    mIsStepComplete = false;       // Reset flag

    // Detect raw contact state
    Eigen::Vector2i rawContact = detectContact(mWorld);

    // Apply debouncer with exponential moving average filter
    if (mIsFirstContactUpdate) {
        // First update: use raw values directly without filtering
        mDebouncedContactL = rawContact[0];
        mDebouncedContactR = rawContact[1];
        mIsFirstContactUpdate = false;
    } else {
        // Apply exponential moving average filter
        mDebouncedContactL = mContactDebounceAlpha * rawContact[0] + (1.0 - mContactDebounceAlpha) * mDebouncedContactL;
        mDebouncedContactR = mContactDebounceAlpha * rawContact[1] + (1.0 - mContactDebounceAlpha) * mDebouncedContactR;
    }

    // Convert debounced continuous values to binary contact state using threshold
    mCachedContact[0] = (mDebouncedContactL > mContactDebounceThreshold) ? 1 : 0;
    mCachedContact[1] = (mDebouncedContactR > mContactDebounceThreshold) ? 1 : 0;

    // Get raw GRF and normalize by body weight
    Eigen::Vector2d rawGRF = getFootGRF(mWorld);
    double mass = mCharacter->getSkeleton()->getMass();
    mCachedGRF = rawGRF / (9.81 * mass);  // Normalize by body weight

    if (mMode == CONTACT) updateFromContact();
    else updateFromPhase();
}

bool GaitPhase::isGaitCycleComplete()
{
    return mGaitCycleCompletePD;
}

void GaitPhase::clearGaitCycleComplete()
{
    mIsGaitCycleComplete = false;
    mGaitCycleCompletePD = false;
}

bool GaitPhase::isStepComplete()
{
    return mStepCompletePD;
}

void GaitPhase::clearStepComplete()
{
    mIsStepComplete = false;
    mStepCompletePD = false;
}

// ========== Contact Detection ==========

Eigen::Vector3d GaitPhase::getLowestPointFromBodyNodes(const std::vector<std::string>& bodyNodeNames)
{
    auto skel = mCharacter->getSkeleton();
    Eigen::Vector3d lowestPoint = Eigen::Vector3d::Zero();
    double lowestY = std::numeric_limits<double>::max();
    bool foundAny = false;

    for (const auto& nodeName : bodyNodeNames) {
        auto bodyNode = skel->getBodyNode(nodeName);
        if (!bodyNode) continue;

        // Iterate through all shape nodes in this body node
        for (std::size_t i = 0; i < bodyNode->getNumShapeNodes(); ++i) {
            auto shapeNode = bodyNode->getShapeNode(i);
            auto shape = shapeNode->getShape();

            // Get bounding box in local shape coordinates
            const auto& bbox = shape->getBoundingBox();

            // Get the minimum Y point in local coordinates
            Eigen::Vector3d localMinPoint = bbox.getMin();

            // Transform to world coordinates
            Eigen::Isometry3d worldTransform = shapeNode->getWorldTransform();
            Eigen::Vector3d worldMinPoint = worldTransform * localMinPoint;

            // Track the lowest point
            if (!foundAny || worldMinPoint[1] < lowestY) {
                lowestY = worldMinPoint[1];
                lowestPoint = worldMinPoint;
                foundAny = true;
            }
        }
    }

    // LOG_INFO("Lowest point Y: " << lowestY << " at position: ["
                // << lowestPoint[0] << ", " << lowestPoint[1] << ", " << lowestPoint[2] << "]");

    return lowestPoint;
}

Eigen::Vector2i GaitPhase::detectContact(dart::simulation::WorldPtr world)
{
    Eigen::Vector2i result(0, 0);
    const auto results = world->getConstraintSolver()->getLastCollisionResult();

    for (auto bn : results.getCollidingBodyNodes()) {
        std::string name = bn->getName();
        if (name == "TalusL" || name == "FootPinkyL" || name == "FootThumbL")
            result[0] = 1;
        if (name == "TalusR" || name == "FootPinkyR" || name == "FootThumbR")
            result[1] = 1;
    }

    return result;
}

Eigen::Vector2d GaitPhase::getFootGRF(dart::simulation::WorldPtr world)
{
    Eigen::Vector3d grfVectorL = Eigen::Vector3d::Zero();
    Eigen::Vector3d grfVectorR = Eigen::Vector3d::Zero();
    const auto results = world->getConstraintSolver()->getLastCollisionResult();

    for (std::size_t i = 0; i < results.getNumContacts(); ++i) {
        const auto& contact = results.getContact(i);
        const auto* sf1 = contact.collisionObject1->getShapeFrame();
        const auto* sf2 = contact.collisionObject2->getShapeFrame();

        // Convert to ShapeNode to access parent BodyNode
        const auto* sn1 = dynamic_cast<const dart::dynamics::ShapeNode*>(sf1);
        const auto* sn2 = dynamic_cast<const dart::dynamics::ShapeNode*>(sf2);
        if (!sn1 || !sn2) continue;

        // Check if either shape is ground (has no parent BodyNode or is named "ground")
        bool contacted = false;
        const dart::dynamics::ShapeNode* footShapeNode = nullptr;

        auto bn1 = sn1->getBodyNodePtr();
        auto bn2 = sn2->getBodyNodePtr();

        // Determine which is ground and which is foot
        if (!bn1 || bn1->getName() == "ground" || bn1->getName() == "Ground") {
            contacted = true;
            footShapeNode = sn2;
        } else if (!bn2 || bn2->getName() == "ground" || bn2->getName() == "Ground") {
            contacted = true;
            footShapeNode = sn1;
        }

        if (!contacted || !footShapeNode) continue;

        // Get the foot body node
        auto footBodyNode = footShapeNode->getBodyNodePtr();
        if (!footBodyNode) continue;

        std::string footName = footBodyNode->getName();

        // Check left foot - sum force vectors component-wise (only ground contacts)
        if (footName == "TalusL" || footName == "FootPinkyL" || footName == "FootThumbL") {
            grfVectorL += contact.force;
        }

        // Check right foot - sum force vectors component-wise (only ground contacts)
        if (footName == "TalusR" || footName == "FootPinkyR" || footName == "FootThumbR") {
            grfVectorR += contact.force;
        }
    }

    // Take norm of the summed vectors at the end
    Eigen::Vector2d grf;
    grf[0] = grfVectorL.norm();
    grf[1] = grfVectorR.norm();
    return grf;
}

// ========== Contact-Based Update ==========

void GaitPhase::updateFromContact()
{
    // Use cached contact and GRF (already computed and normalized in step())
    Eigen::Vector2i contact = mCachedContact;
    Eigen::Vector2d grfNorm = mCachedGRF;  // Already normalized by body weight
    bool contactR = contact[1] > 0;
    bool contactL = contact[0] > 0;

    double time = mWorld->getTime();
    auto skel = mCharacter->getSkeleton();
    Eigen::Vector3d footRPos = skel->getBodyNode("TalusR")->getCOM();
    Eigen::Vector3d footLPos = skel->getBodyNode("TalusL")->getCOM();

    // Calculate target stride and minimum step using stored parameters
    double targetStride = mStride * mRefStride * mCharacter->getGlobalRatio();
    double stepMin = mStepMinRatio * targetStride;

    // 4-State Gait Machine
    switch (mState) {
        case RIGHT_STANCE: {
            // Right foot in stance, waiting for left foot liftoff
            // Transition conditions to LEFT_TAKEOFF:
            // - contactR: Right foot maintains collision contact (stance foot on ground)
            // - !contactL: Left foot loses contact (begins swing phase)
            // Note: No GRF check for liftoff - just contact loss is sufficient
            if (contactR && !contactL) {
                mState = LEFT_TAKEOFF;
                mStanceTimeL = time - mLeftPhaseUpTime;
                mLeftPhaseUpTime = time;
                mPhaseTotalL = mStanceTimeL + mSwingTimeL;
#if LOG_STATE_CHANGES
                LOG_INFO("✓ STATE CHANGE: RIGHT_STANCE → LEFT_TAKEOFF (swing left)");
#endif
            } else {
#if LOG_STATE_CHANGES
                std::ostringstream oss;
                oss << std::setprecision(3);
                oss << "✗ RIGHT_STANCE held | Reasons: ";
                if (!contactR) oss << "[No right contact] ";
                if (contactL) oss << "[Left still in contact] ";
                LOG_INFO(oss.str());
#endif
            }
            break;
        }

        case LEFT_TAKEOFF: {
            // Left foot in swing phase, waiting for left heel strike
            // Transition conditions to LEFT_STANCE:
            // - contactL: Left foot establishes collision contact (foot touches ground)
            // - grfNorm[0] > mGRFThreshold: Left foot has sufficient GRF (loading response complete)
            // - progression > stepMin: Left foot has moved forward enough (valid step, not in-place contact)
            // All three conditions ensure valid heel strike with forward progression
            double progression = footLPos[2] - mFootPrevLz;
            if (contactL && grfNorm[0] > mGRFThreshold && progression > stepMin) {
#if LOG_STATE_CHANGES
                std::ostringstream oss;
                oss << std::setprecision(3)
                    << "✓ STATE CHANGE: LEFT_TAKEOFF → LEFT_STANCE | Step: " << progression
                    << " (min: " << stepMin << ")";
                LOG_INFO(oss.str());
#endif

                mState = LEFT_STANCE;
                mSwingTimeL = time - mLeftPhaseUpTime;
                mLeftPhaseUpTime = time;

                // Update foot targets for next stride
                mCurrentTargetFoot = mNextTargetFoot;
                mNextTargetFoot = mCurrentStanceFoot + Eigen::Vector3d::UnitZ() * targetStride;
                mCurrentStanceFoot = footLPos;

                // Update stride tracking
                mStrideLengthL = progression;
                mFootPrevLz = footLPos[2];
                mCurrentFoot = footLPos;
                mIsLeftLegStance = true;

                // Step complete (left heel strike)
                mIsStepComplete = true;
                mStepCompletePD = true;  // Set PD-level persistent flag
            } else {
#if LOG_STATE_CHANGES
                std::ostringstream oss;
                oss << std::setprecision(3);
                oss << "✗ LEFT_TAKEOFF held | Reasons: ";
                if (!contactL) oss << "[No left contact] ";
                if (grfNorm[0] <= mGRFThreshold) oss << "[Low left GRF: " << grfNorm[0] << " ≤ " << mGRFThreshold << "] ";
                if (progression <= stepMin) oss << "[Short step: " << progression << " ≤ " << stepMin << "] ";
                LOG_INFO(oss.str());
#endif
            }
            break;
        }

        case LEFT_STANCE: {
            // Left foot in stance, waiting for right foot liftoff
            // Transition conditions to RIGHT_TAKEOFF:
            // - contactL: Left foot maintains collision contact (stance foot on ground)
            // - !contactR: Right foot loses contact (begins swing phase)
            // Note: No GRF check for liftoff - just contact loss is sufficient
            if (contactL && !contactR) {
                mState = RIGHT_TAKEOFF;
                mStanceTimeR = time - mRightPhaseUpTime;
                mRightPhaseUpTime = time;
                mPhaseTotalR = mStanceTimeR + mSwingTimeR;
#if LOG_STATE_CHANGES
                LOG_INFO("✓ STATE CHANGE: LEFT_STANCE → RIGHT_TAKEOFF (swing right)");
#endif
            } else {
#if LOG_STATE_CHANGES
                std::ostringstream oss;
                oss << std::setprecision(3);
                oss << "✗ LEFT_STANCE held | Reasons: ";
                if (!contactL) oss << "[No left contact] ";
                if (contactR) oss << "[Right still in contact] ";
                LOG_INFO(oss.str());
#endif
            }
            break;
        }

        case RIGHT_TAKEOFF: {
            // Right foot in swing phase, waiting for right heel strike
            // Transition conditions to RIGHT_STANCE:
            // - contactR: Right foot establishes collision contact (foot touches ground)
            // - grfNorm[1] > mGRFThreshold: Right foot has sufficient GRF (loading response complete)
            // - progression > stepMin: Right foot has moved forward enough (valid step, not in-place contact)
            // All three conditions ensure valid heel strike with forward progression
            // Right heel strike marks the completion of a full gait cycle
            double progression = footRPos[2] - mFootPrevRz;
            if (contactR && grfNorm[1] > mGRFThreshold && progression > stepMin) {
#if LOG_STATE_CHANGES
                std::ostringstream oss;
                oss << std::setprecision(3)
                    << "✓ STATE CHANGE: RIGHT_TAKEOFF → RIGHT_STANCE | Step: " << progression
                    << " (min: " << stepMin << ") [CYCLE COMPLETE]";
                LOG_INFO(oss.str());
#endif

                mState = RIGHT_STANCE;
                mSwingTimeR = time - mRightPhaseUpTime;
                mRightPhaseUpTime = time;

                // Update foot targets for next stride
                mCurrentTargetFoot = mNextTargetFoot;
                mNextTargetFoot = mCurrentStanceFoot + Eigen::Vector3d::UnitZ() * targetStride;
                mCurrentStanceFoot = footRPos;

                // Update stride tracking
                mStrideLengthR = progression;
                mFootPrevRz = footRPos[2];
                mCurrentFoot = footRPos;
                mIsLeftLegStance = false;

                // Gait cycle complete (right heel strike defines cycle)
                auto skel = mCharacter->getSkeleton();
                mCurCycleCOM = skel->getBodyNode("Pelvis")->getCOM();
                mCycleDist = (mCurCycleCOM - mPrevCycleCOM).norm();
                mPrevCycleCOM = mCurCycleCOM;

                mCurCycleTime = mSimTime;
                mCycleTime = mCurCycleTime - mPrevCycleTime;
                mPrevCycleTime = mCurCycleTime;

                if (mCycleTime > 0.0) {
                    mCurrentCadence = 2.0 / mCycleTime;
                    mStanceRatioR = mStanceTimeR / mCycleTime;
                    mStanceRatioL = mStanceTimeL / mCycleTime;
                }

                mIsGaitCycleComplete = true;
                mGaitCycleCompletePD = true;  // Set PD-level persistent flag
                mIsStepComplete = true;
                mStepCompletePD = true;  // Set PD-level persistent flag (right heel strike)
                mCycleCount++;
            } else {
#if LOG_STATE_CHANGES
                std::ostringstream oss;
                oss << std::setprecision(3);
                oss << "✗ RIGHT_TAKEOFF held | Reasons: ";
                if (!contactR) oss << "[No right contact] ";
                if (grfNorm[1] <= mGRFThreshold) oss << "[Low right GRF: " << grfNorm[1] << " ≤ " << mGRFThreshold << "] ";
                if (progression <= stepMin) oss << "[Short step: " << progression << " ≤ " << stepMin << "] ";
                LOG_INFO(oss.str());
#endif
            }
            break;
        }
    }

    mPrevContact = contact;
}

// ========== Phase-Based Update ==========

void GaitPhase::updateFromPhase()
{
    double phase = getAdaptivePhase();
    double targetStride = mStride * mRefStride * mCharacter->getGlobalRatio();
    auto skel = mCharacter->getSkeleton();

    // Phase-based stance detection (using 0.33-0.83 range for right stance)
    if (0.33 < phase && phase <= 0.83) {
        // Right leg stance phase
        if (mIsLeftLegStance) {
            // Transition from left to right
            mCurrentTargetFoot = mNextTargetFoot;
            mNextTargetFoot = mCurrentFoot + Eigen::Vector3d::UnitZ() * targetStride;
            mIsGaitCycleComplete = true;
            mGaitCycleCompletePD = true;  // Set PD-level persistent flag
        }

        mIsLeftLegStance = false;
        mCurrentFoot = skel->getBodyNode("TalusR")->getCOM();
        mState = RIGHT_STANCE;
        mCycleCount++;
    } else {
        // Left leg stance phase
        if (!mIsLeftLegStance) {
            // Transition from right to left
            mCurrentTargetFoot = mNextTargetFoot;
            mNextTargetFoot = mCurrentFoot + Eigen::Vector3d::UnitZ() * targetStride;
        }

        mIsLeftLegStance = true;
        mCurrentFoot = skel->getBodyNode("TalusL")->getCOM();
        mState = LEFT_STANCE;
    }

    // Zero out Y component (height)
    mCurrentTargetFoot[1] = 0.0;
    mNextTargetFoot[1] = 0.0;
}

void GaitPhase::setPhaseAction(double action)
{
    if (action < (-1.0 / mControlHz)) action = -1.0 / mControlHz;
    mPhaseAction = action;
}