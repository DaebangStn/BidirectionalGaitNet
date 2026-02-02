#ifndef __MS_GAIT_PHASE_H__
#define __MS_GAIT_PHASE_H__

#include "dart/dart.hpp"
#include "Character.h"
#include <Eigen/Dense>
#include <string>

namespace dart {
namespace simulation {
class World;
}
}

class Character;

/**
 * GaitPhase - Encapsulates all gait tracking logic
 *
 * Manages foot placement, contact detection, and gait cycle tracking.
 * Supports two modes:
 *   - CONTACT: Physics-based contact detection with 4-state machine
 *   - PHASE: Time-based phase tracking (backward compatible)
 */
class GaitPhase {
public:
    /**
     * Explicit 4-state gait cycle machine
     * RIGHT_STANCE -> LEFT_TAKEOFF -> LEFT_STANCE -> RIGHT_TAKEOFF -> (cycle)
     */
    enum GaitState {
        RIGHT_STANCE,    // Right foot in stance, left in swing
        LEFT_TAKEOFF,    // Right in stance, left about to contact
        LEFT_STANCE,     // Left foot in stance, right ingetAdaptivePhase swing
        RIGHT_TAKEOFF    // Left in stance, right about to contact
    };

    /**
     * Update mode selection
     */
    enum UpdateMode {
        CONTACT,  // Physics-based contact detection
        PHASE     // Time-based phase tracking
    };

    /**
     * Constructor
     * @param character Pointer to Character for accessing skeleton and state
     * @param world World pointer for contact detection
     * @param motionCycleTime Motion cycle duration (NOT Motion pointer)
     * @param refStride Reference stride length (typically 1.34m)
     * @param mode Update mode (CONTACT or PHASE)
     * @param controlHz Control frequency
     * @param simulationHz Simulation frequency
     */
    GaitPhase(Character* character,
              dart::simulation::WorldPtr world,
              double motionCycleTime,
              double refStride,
              UpdateMode mode,
              double controlHz,
              double simulationHz);

    /**
     * Main update method - call every simulation step (600Hz)
     * Uses stored parameters including mPhaseDisplacement
     */
    void step();

    /**
     * Reset gait state
     */
    void reset(double time);

    /**
     * Reset foot previous Z positions from current skeleton state
     * Call after skeleton positions are updated (e.g., after zeroing ankle DOF)
     */
    void resetFootPos();

    // ========== State Accessors ==========

    /**
     * Get current gait state (4-state machine)
     */
    GaitState getCurrentState() const { return mState; }

    /**
     * Check if left leg is currently in stance
     */
    bool isLeftLegStance() const { return mIsLeftLegStance; }

    /**
     * Get current stance foot position
     */
    Eigen::Vector3d getCurrentFoot() const { return mCurrentFoot; }

    /**
     * Get current target foot position
     */
    Eigen::Vector3d getCurrentTargetFoot() const { return mCurrentTargetFoot; }

    /**
     * Get next target foot position
     */
    Eigen::Vector3d getNextTargetFoot() const { return mNextTargetFoot; }

    /**
     * Check if gait cycle completed (PD-step level flag)
     * This flag persists across multiple muscle steps until cleared
     */
    bool isGaitCycleComplete();

    /**
     * Clear the PD-level gait cycle completion flag
     * Call this after consuming the completion event
     */
    void clearGaitCycleComplete();

    /**
     * Check if a step completed (PD-step level flag)
     * This flag is set on both right and left foot strikes
     * and persists across multiple muscle steps until cleared
     */
    bool isStepComplete();

    /**
     * Clear the PD-level step completion flag
     * Call this after consuming the step completion event
     */
    void clearStepComplete();

    /**
     * Get total gait cycle count
     */
    int getCycleCount() const { return mCycleCount; }

    // ========== Detailed Metrics ==========

    double getStanceTimeR() const { return mStanceTimeR; }
    double getStanceTimeL() const { return mStanceTimeL; }
    double getSwingTimeR() const { return mSwingTimeR; }
    double getSwingTimeL() const { return mSwingTimeL; }
    double getPhaseTotalR() const { return mPhaseTotalR; }
    double getPhaseTotalL() const { return mPhaseTotalL; }
    double getCadence() const { return mCurrentCadence; }
    double getStanceRatioR() const { return mStanceRatioR; }
    double getStanceRatioL() const { return mStanceRatioL; }
    double getStrideLengthR() const { return mStrideLengthR; }
    double getStrideLengthL() const { return mStrideLengthL; }
    Eigen::Vector3d getCycleCOM() const { return mCurCycleCOM; }
    double getCycleDist() const { return mCycleDist; }
    double getCycleTime() const { return mCycleTime; }

    // ========== Configuration ==========

    /**
     * Set minimum step progression ratio (default: 0.5)
     * Step must be at least (stepMinRatio * stride) to trigger heel strike
     */
    void setStepMinRatio(double ratio) { mStepMinRatio = ratio; }

    /**
     * Set GRF threshold for phase change (default: 0.2)
     * Normalized GRF must exceed this to confirm contact
     */
    void setGRFThreshold(double threshold) { mGRFThreshold = threshold; }

    /**
     * Set contact debouncer alpha (default: 0.25)
     * Controls smoothing: lower = more smoothing, higher = faster response
     */
    void setContactDebounceAlpha(double alpha) { mContactDebounceAlpha = alpha; }

    /**
     * Set contact debouncer threshold (default: 0.5)
     * Threshold for converting debounced value to binary contact
     */
    void setContactDebounceThreshold(double threshold) { mContactDebounceThreshold = threshold; }

    /**
     * Set update mode (can be changed at runtime)
     */
    void setUpdateMode(UpdateMode mode) { mMode = mode; }

    /**
     * Get current update mode
     */
    UpdateMode getUpdateMode() const { return mMode; }

    // ========== Parameter Setters ==========

    /**
     * Set stride ratio (dynamic parameter)
     */
    void setStride(double stride) { mStride = stride; }

    /**
     * Set cadence ratio (dynamic parameter)
     */
    void setCadence(double cadence) { mCadence = cadence; }

    /**
     * Get motion cycle time
     */
    double getMotionCycleTime() const { return mMotionCycleTime; }

    // ========== Time Management ==========

    /**
     * Get current local time
     */
    double getAdaptiveTime() const { return mAdaptiveTime; }

    /**
     * Get global simulation time (linear, no phase adjustment)
     */
    double getSimTime() const { return mSimTime; }

    double getAdaptivePhase() const { 
        double phase = mAdaptiveTime / (mMotionCycleTime / mCadence);
        phase = phase - floor(phase);  // Normalize to [0, 1)
        return phase;
    }

    int getAdaptiveCycleCount() const { return floor(mAdaptiveTime / (mMotionCycleTime / mCadence)); }
    int getAdaptiveCycleCount(double adaptiveTime) const { return floor(adaptiveTime / (mMotionCycleTime / mCadence)); }

    /**
     * Set phase action (updated each control step)
     */
    void setPhaseAction(double action);

    // ========== Contact Detection (Cached) ==========

    /**
     * Get cached foot contact state (updated in step())
     * @return [left, right] contact as 0 (no contact) or 1 (contact)
     */
    Eigen::Vector2i getContactState() const { return mCachedContact; }

    /**
     * Get cached normalized ground reaction forces (updated in step())
     * @return [left, right] GRF normalized by body weight
     */
    Eigen::Vector2d getNormalizedGRF() const { return mCachedGRF; }

private:
    // ========== Internal Contact Detection ==========

    /**
     * Detect foot contact from collision results
     * @return [left_contact, right_contact] as 0/1
     */
    Eigen::Vector2i detectContact(dart::simulation::WorldPtr world);

    /**
     * Calculate ground reaction forces for both feet
     * @return [left_GRF, right_GRF] in Newtons
     */
    Eigen::Vector2d getFootGRF(dart::simulation::WorldPtr world);

    /**
     * Get the lowest point (in world coordinates) from a set of body nodes
     * @param bodyNodeNames Vector of body node names to check
     * @return World position of the lowest point across all shapes
     */
    Eigen::Vector3d getLowestPointFromBodyNodes(const std::vector<std::string>& bodyNodeNames);

    // ========== Update Implementations ==========

    /**
     * Contact-based update with 4-state machine (uses stored members)
     */
    void updateFromContact();

    /**
     * Phase-based update (uses stored members)
     */
    void updateFromPhase();


    // ========== Dependencies ==========

    Character* mCharacter;                  // Character reference for skeleton access
    dart::simulation::WorldPtr mWorld;      // World for contact detection
    UpdateMode mMode;                       // Current update mode

    // ========== Parameters (Stored) ==========

    double mMotionCycleTime;   // Motion cycle duration
    double mRefStride;         // Reference stride (constant, typically 1.34m)
    double mCadence;           // Current cadence ratio
    double mStride;            // Current stride ratio

    // ========== Time Management ==========

    double mSimTime;                // Global simulation time (linear, no phase adjustment)
    double mAdaptiveTime;           // Local simulation time (adjusted by phase action)
    double mControlHz;           // Control frequency
    double mSimulationHz;        // Simulation frequency
    double mPhaseAction;   // Phase displacement (passed to step())

    // ========== Cached Contact State ==========

    Eigen::Vector2i mCachedContact;    // Cached contact state [left, right]
    Eigen::Vector2d mCachedGRF;        // Cached normalized GRF [left, right]

    // ========== Core State ==========

    GaitState mState;                      // Current 4-state gait state
    bool mIsLeftLegStance;                 // True if left leg in stance
    Eigen::Vector3d mCurrentFoot;          // Current stance foot position
    Eigen::Vector3d mCurrentTargetFoot;    // Current foot target position
    Eigen::Vector3d mNextTargetFoot;       // Next foot target position
    Eigen::Vector3d mCurrentStanceFoot;    // Last confirmed stance foot position

    // ========== Contact Tracking ==========

    Eigen::Vector2i mPrevContact;    // Previous contact state [left, right]
    double mFootPrevRz;              // Previous right foot Z position
    double mFootPrevLz;              // Previous left foot Z position
    double mDebouncedContactL;       // Left foot debounced state (0.0-1.0)
    double mDebouncedContactR;       // Right foot debounced state (0.0-1.0)
    bool mIsFirstContactUpdate;      // Flag to skip filtering on first update

    // ========== Per-Foot Timing ==========

    double mSwingTimeR;           // Right foot swing duration
    double mSwingTimeL;           // Left foot swing duration
    double mStanceTimeR;          // Right foot stance duration
    double mStanceTimeL;          // Left foot stance duration
    double mRightPhaseUpTime;     // Time of last right phase change
    double mLeftPhaseUpTime;      // Time of last left phase change
    double mPhaseTotalR;          // Right foot total cycle time
    double mPhaseTotalL;          // Left foot total cycle time
    double mStrideLengthR;        // Right foot stride length
    double mStrideLengthL;        // Left foot stride length

    // ========== Cycle Tracking ==========

    int mCycleCount;                      // Total gait cycles completed
    bool mIsGaitCycleComplete;            // Flag: cycle completed (muscle-step level, resets in step())
    bool mGaitCycleCompletePD;            // Flag: cycle completed (PD-step level, persists until cleared)
    bool mIsStepComplete;                 // Flag: step completed (muscle-step level, resets in step())
    bool mStepCompletePD;                 // Flag: step completed (PD-step level, persists until cleared)
    double mCycleTime;                    // Last cycle duration
    double mCurCycleTime;                 // Current cycle start time
    double mPrevCycleTime;                // Previous cycle start time
    Eigen::Vector3d mCurCycleCOM;         // Current cycle COM position
    Eigen::Vector3d mPrevCycleCOM;        // Previous cycle COM position
    double mCycleDist;                    // Distance traveled in last cycle
    double mCurrentCadence;               // Current cadence (cycles/sec)
    double mStanceRatioR;                 // Right stance time / cycle time
    double mStanceRatioL;                 // Left stance time / cycle time

    // ========== Configuration ==========

    double mStepMinRatio;             // Minimum step progression ratio (default: 0.5)
    double mGRFThreshold;             // GRF threshold for contact (default: 0.2)
    double mContactDebounceAlpha;     // Debouncer filter alpha (default: 0.25)
    double mContactDebounceThreshold; // Debouncer threshold for binary conversion (default: 0.5)
};

#endif // __MS_GAIT_PHASE_H__
