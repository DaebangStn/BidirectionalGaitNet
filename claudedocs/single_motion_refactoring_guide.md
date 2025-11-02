# Single Motion Refactoring Guide

Complete step-by-step guide to convert GLFWApp from multi-motion vector to single Motion* architecture.

## Overview
- **From**: `vector<Motion*> mMotions` + `vector<PlaybackViewerState> mMotionStates` + `int mMotionIdx`
- **To**: `Motion* mMotion` + `PlaybackViewerState mMotionState`
- **Pattern**: Delete old motion when loading new one (new/delete lifecycle)

---

## ✅ COMPLETED (Header File Changes)

### viewer/GLFWApp.h
Already done:
- Line 343: Changed to `Motion* mMotion;`
- Line 344: Changed to `PlaybackViewerState mMotionState;`
- Removed `int mMotionIdx;` and fallback variables
- Line 542: Added `void setMotion(Motion* motion);` helper

---

## STEP-BY-STEP CPP FILE CHANGES

### 1. Constructor (Line ~85)

**FIND:**
```cpp
    // Initialize motion navigation control
    mFallbackMotionNavigationMode = PLAYBACK_SYNC;
    mFallbackManualFrameIndex = 0;
```

**REPLACE WITH:**
```cpp
    // Initialize single motion architecture
    mMotion = nullptr;
```

### 2. Destructor (Line ~508-513)

**FIND:**
```cpp
    // Clean up Motion* pointers in new architecture
    for (Motion* motion : mMotions) {
        delete motion;
    }
    mMotions.clear();
```

**REPLACE WITH:**
```cpp
    // Clean up single motion
    delete mMotion;
    mMotion = nullptr;
```

### 3. Add setMotion Helper (Insert before alignMotionToSimulation, ~line 4617)

**INSERT BEFORE `void GLFWApp::alignMotionToSimulation()`:**
```cpp
void GLFWApp::setMotion(Motion* motion)
{
    // Delete old motion and assign new one
    delete mMotion;
    mMotion = motion;

    // Initialize viewer state
    if (motion) {
        mMotionState.cycleDistance = computeMotionCycleDistance(motion);
        mMotionState.maxFrameIndex = std::max(0, motion->getNumFrames() - 1);
        mMotionState.currentPose.resize(0);
        mMotionState.displayOffset.setZero();
        mMotionState.displayOffset[0] = 1.0;
        mMotionState.navigationMode = PLAYBACK_SYNC;
        mMotionState.manualFrameIndex = 0;
        mMotionState.render = true;
    }
}

```

### 4. Fix C3D Auto-Load (Line ~1210-1220)

**FIND:**
```cpp
            if (c3dMotion) {
                // Add to motion list
                mMotions.push_back(c3dMotion);

                // Create viewer state
                PlaybackViewerState state;
                state.cycleDistance = computeMotionCycleDistance(c3dMotion);
                state.maxFrameIndex = std::max(0, c3dMotion->getNumFrames() - 1);
                mMotionStates.push_back(state);

                // Set as active motion
                mMotionIdx = static_cast<int>(mMotions.size()) - 1;

                // Align to simulation
                alignMotionToSimulation();
```

**REPLACE WITH:**
```cpp
            if (c3dMotion) {
                setMotion(c3dMotion);
                alignMotionToSimulation();
```

### 5. Fix Motion Display Text (Line ~1408)

**FIND:**
```cpp
ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Motion Loaded (%zu)", mMotions.size());
```

**REPLACE WITH:**
```cpp
if (mMotion) {
    ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Motion: %s", mMotion->getName().c_str());
}
```

### 6. REMOVE Motion Selection UI (Line ~1455-1505)

**ENTIRE SECTION TO DELETE:**
```cpp
            if (mMotion != nullptr) {
                ImGui::Text("Available Motions");
                if (ImGui::BeginListBox("##motions", ImVec2(-FLT_MIN, 5 * ImGui::GetTextLineHeightWithSpacing())))
                {
                    for (int i = 0; i < mMotions.size(); i++)
                    {
                        // ... entire listbox code ...
                    }
                    ImGui::EndListBox();
                }
            }
```

**REPLACE WITH:**
```cpp
            // Single motion architecture - no selection UI needed
```

### 7. Fix C3D Manual Load (Line ~1560-1573)

**FIND:**
```cpp
                                if (c3dMotion) {
                                    // Add to motion list
                                    mMotions.push_back(c3dMotion);

                                    // Create viewer state
                                    PlaybackViewerState state;
                                    state.cycleDistance = computeMotionCycleDistance(c3dMotion);
                                    state.maxFrameIndex = std::max(0, c3dMotion->getNumFrames() - 1);
                                    mMotionStates.push_back(state);

                                    // Set as active motion
                                    mMotionIdx = static_cast<int>(mMotions.size()) - 1;

                                    // Align to simulation
                                    alignMotionToSimulation();
```

**REPLACE WITH:**
```cpp
                                if (c3dMotion) {
                                    setMotion(c3dMotion);
                                    alignMotionToSimulation();
```

### 8. Fix Motion State Access (Line ~1804)

**FIND:**
```cpp
        if (!mMotionStates.empty() && mMotionIdx >= 0 && static_cast<size_t>(mMotionIdx) < mMotionStates.size()) {
            motionStatePtr = &mMotionStates[mMotionIdx];
        }
```

**REPLACE WITH:**
```cpp
        if (mMotion != nullptr) {
            motionStatePtr = &mMotionState;
        }
```

### 9. Fix NPZ Loading (Line ~5553-5565)

**FIND:**
```cpp
			mMotions.push_back(npz);

			PlaybackViewerState state;
			state.cycleDistance = computeMotionCycleDistance(npz);
			state.maxFrameIndex = std::max(0, npz->getNumFrames() - 1);
			mMotionStates.push_back(state);

			if (npz->hasParameters()) {
```

**REPLACE WITH:**
```cpp
			setMotion(npz);

			if (npz->hasParameters()) {
```

### 10. Fix Rollout Loading (Line ~5618-5631)

**FIND:**
```cpp
					mMotions.push_back(rollout);

					PlaybackViewerState state;
					state.cycleDistance = computeMotionCycleDistance(rollout);
					state.maxFrameIndex = std::max(0, rollout->getNumFrames() - 1);
					mMotionStates.push_back(state);

					if (rollout->hasParameters()) {
```

**REPLACE WITH:**
```cpp
					setMotion(rollout);

					if (rollout->hasParameters()) {
```

### 11. Fix BVH Loading (Line ~5678-5691)

**FIND:**
```cpp
				mMotions.push_back(bvh);

				PlaybackViewerState state;
				state.cycleDistance = computeMotionCycleDistance(bvh);
				state.maxFrameIndex = std::max(0, bvh->getNumFrames() - 1);
				mMotionStates.push_back(state);

				LOG_VERBOSE(bvh->getLogHeader() << " Loaded " << bvh->getName() << " with " << bvh->getNumFrames() << " frames");
```

**REPLACE WITH:**
```cpp
				setMotion(bvh);

				LOG_VERBOSE(bvh->getLogHeader() << " Loaded " << bvh->getName() << " with " << bvh->getNumFrames() << " frames");
```

### 12. Fix HDF Loading (Line ~5726-5739)

**FIND:**
```cpp
				mMotions.push_back(hdf);

				PlaybackViewerState state;
				state.cycleDistance = computeMotionCycleDistance(hdf);
				state.maxFrameIndex = std::max(0, hdf->getNumFrames() - 1);
				mMotionStates.push_back(state);

				if (hdf->hasParameters()) {
```

**REPLACE WITH:**
```cpp
				setMotion(hdf);

				if (hdf->hasParameters()) {
```

### 13. Fix Playback Loops (Line ~1631, ~1746)

**FIND (both locations):**
```cpp
for (auto* motion : mMotions) {
    // ... render code ...
}
```

**REPLACE WITH:**
```cpp
if (mMotion) {
    Motion* motion = mMotion;
    // ... render code ...
}
```

### 14. Fix unloadMotion (Line ~6116-6122)

**FIND:**
```cpp
    for (Motion* motion : mMotions) {
        delete motion;
    }
    mMotions.clear();
    mMotionStates.clear();

    // Reset motion playback indices (-1 indicates no motion selected)
    mMotionIdx = -1;
```

**REPLACE WITH:**
```cpp
    delete mMotion;
    mMotion = nullptr;
    mMotionState = PlaybackViewerState();  // Reset to default
```

### 15. Fix alignMotionToSimulation Safety Checks (Line ~4620)

**FIND:**
```cpp
    if (mMotions.empty() || mMotionIdx < 0 || mMotionIdx >= mMotions.size()) {
        LOG_ERROR("[alignMotionToSimulation] No motions loaded or invalid index");
        return;
    }

    if (mMotionStates.size() <= static_cast<size_t>(mMotionIdx)) {
        LOG_ERROR("[alignMotionToSimulation] Motion state out of range for index " << mMotionIdx);
        return;
    }

    PlaybackViewerState& state = mMotionStates[mMotionIdx];
```

**REPLACE WITH:**
```cpp
    if (mMotion == nullptr) {
        LOG_ERROR("[alignMotionToSimulation] No motion loaded");
        return;
    }

    PlaybackViewerState& state = mMotionState;
```

### 16. Fix motionPoseEval calls

**FIND (all occurrences):**
```cpp
motionPoseEval(mMotions[mMotionIdx], mMotionIdx, frame_float);
```

**REPLACE WITH:**
```cpp
motionPoseEval(mMotion, 0, frame_float);  // motionIdx parameter kept for signature compatibility
```

---

## GLOBAL SEARCH & REPLACE (Use IDE)

After manual changes above, do these global replacements:

1. **`mMotions[mMotionIdx]`** → **`mMotion`**
2. **`mMotionStates[mMotionIdx]`** → **`mMotionState`**
3. **`!mMotions.empty() && mMotion`** → **`mMotion`** (redundant check)
4. **`mMotions.empty()`** → **`mMotion == nullptr`**

---

## VERIFICATION

After changes:
```bash
ninja -C build/release
```

Expected: Clean build with no errors.

---

## TESTING

1. Load a C3D file - should work
2. Load different motion types (BVH, HDF5, NPZ) - each replaces previous
3. Verify no crashes when switching motions
4. Check that state resets properly between motion loads
