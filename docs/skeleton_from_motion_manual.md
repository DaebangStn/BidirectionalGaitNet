# How to predict and verify "skeleton from motion"

## Overview
When the "Load BGN" button is clicked at `viewer/GLFWApp.cpp:638`, the system loads a Backward GaitNet model that can predict skeletal configurations from motion data. Here's the complete flow:

## 1. BGN Model Loading (`viewer/GLFWApp.cpp:638-648`)

```cpp
if (ImGui::Button("Load BGN"))
{
    mGVAELoaded = true;
    py::object load_gaitvae = py::module::import("advanced_vae").attr("load_gaitvae");
    int rows = mEnv->getCharacter(0)->posToSixDof(mEnv->getCharacter(0)->getSkeleton()->getPositions()).rows();
    mGVAE = load_gaitvae(mBGNList[selected_fgn], rows, 60, mEnv->getNumKnownParam(), mEnv->getNumParamState());

    mPredictedMotion.motion = mMotions[mMotionIdx].motion;
    mPredictedMotion.param = mMotions[mMotionIdx].param;
    mPredictedMotion.name = "Unpredicted";
}
```

**What happens:**
- Loads the BGN (Backward GaitNet) model from the selected file
- Initializes `mPredictedMotion` with current motion data
- Sets up the neural network for motion-to-skeleton prediction

## 2. Skeleton Prediction from Motion (`viewer/GLFWApp.cpp:786-796`)

```cpp
if (ImGui::Button("predict new motion"))
{
    Eigen::VectorXd input = Eigen::VectorXd::Zero(mMotions[mMotionIdx].motion.rows() + mEnv->getNumKnownParam());
    input << mMotions[mMotionIdx].motion, mEnv->getNormalizedParamStateFromParam(mMotions[mMotionIdx].param.head(mEnv->getNumKnownParam()));
    py::tuple res = mGVAE.attr("render_forward")(input.cast<float>());
    Eigen::VectorXd motion = res[0].cast<Eigen::VectorXd>();
    Eigen::VectorXd param = res[1].cast<Eigen::VectorXd>();

    mPredictedMotion.motion = motion;
    mPredictedMotion.param = mEnv->getParamStateFromNormalized(param);
}
```

**What happens:**
1. **Input preparation**: Combines motion data with known parameters
2. **Neural network inference**: `mGVAE.render_forward()` predicts skeletal parameters
3. **Result storage**: Updates `mPredictedMotion` with predicted motion and parameters

## 3. Applying Predicted Skeletal Configuration (`viewer/GLFWApp.cpp:806`)

```cpp
if (ImGui::Button("Set to predicted param"))
    mEnv->setParamState(mPredictedMotion.param, false, true);
```

**What happens:**
- Applies the predicted parameters to the actual skeleton in the simulation
- This modifies the character's physical properties

## 4. Combined Prediction and Application (`viewer/GLFWApp.cpp:808-819`)

```cpp
if (ImGui::Button("Predict and set param"))
{
    Eigen::VectorXd input = Eigen::VectorXd::Zero(mMotions[mMotionIdx].motion.rows() + mEnv->getNumKnownParam());
    input << mMotions[mMotionIdx].motion, mEnv->getNormalizedParamStateFromParam(mMotions[mMotionIdx].param.head(mEnv->getNumKnownParam()));
    py::tuple res = mGVAE.attr("render_forward")(input.cast<float>());
    Eigen::VectorXd motion = res[0].cast<Eigen::VectorXd>();
    Eigen::VectorXd param = res[1].cast<Eigen::VectorXd>();

    mPredictedMotion.motion = motion;
    mPredictedMotion.param = mEnv->getParamStateFromNormalized(param);
    mEnv->setParamState(mPredictedMotion.param, false, true);
}
```

**What happens:**
- Performs prediction AND immediately applies results to the skeleton
- One-click solution for motion-to-skeleton prediction and verification

## 5. Parameter Application Implementation (`sim/Environment.cpp:1477-1500`)

```cpp
void Environment::setParamState(Eigen::VectorXd _param_state, bool onlyMuscle, bool doOptimization)
{
    int idx = 0;
    // skeleton parameter
    if (!onlyMuscle)
    {
        std::vector<std::pair<std::string, double>> skel_info;
        for (auto name : mParamName)
        {
            // gait parameter
            if (name.find("stride") != std::string::npos)
                mStride = _param_state[idx];

            if (name.find("cadence") != std::string::npos)
                mCadence = _param_state[idx];

            if (name.find("skeleton") != std::string::npos)
                skel_info.push_back(std::make_pair((name.substr(9)), _param_state[idx]));
            
            idx++;
        }
        mCharacters[0]->setSkelParam(skel_info, doOptimization);
    }
    // ... muscle parameter handling continues
}
```

**What this does:**
- **Stride/Cadence**: Updates gait timing parameters
- **Skeleton parameters**: Modifies bone lengths, joint limits, and scaling factors
- **Muscle parameters**: Adjusts muscle force and length properties
- **Optimization**: Calls `setSkelParam()` to apply changes with optional geometry optimization

## 6. Skeleton Parameter Application (`sim/Character.cpp:692-740`)

```cpp
void Character::setSkelParam(std::vector<std::pair<std::string, double>> _skel_info, bool doOptimization)
{
    // Global Setting
    for (auto s_i : _skel_info)
    {
        if (std::get<0>(s_i) == "global")
        {
            mGlobalRatio = std::get<1>(s_i);
            for (auto jn : mSkeleton->getJoints())
            {
                // Apply global scaling to all joints
                if (jn->getNumDofs() > 0)
                {
                    Eigen::VectorXd upper = jn->getPositionUpperLimits();
                    Eigen::VectorXd lower = jn->getPositionLowerLimits();
                    jn->setPositionUpperLimits(upper * mGlobalRatio);
                    jn->setPositionLowerLimits(lower * mGlobalRatio);
                }
            }
        }
    }
    
    if (doOptimization)
        applySkeletonLength(_skel_info, doOptimization);
}
```

**What this does:**
- Applies global scaling to joint limits
- Calls `applySkeletonLength()` for detailed bone modifications

## 7. Verification Through Visualization (`viewer/GLFWApp.cpp:1652`)

```cpp
// Currently commented out - uncomment to enable visualization
// drawMotions(mPredictedMotion.motion, mPredictedMotion.param, 
//            Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector4d(0.8, 0.8, 0.2, 0.7));
```

**For verification:**
- The `drawMotions()` function would render the predicted skeletal configuration
- Compare predicted vs. original motion visually
- Check if the skeleton matches the motion characteristics

## 8. Sampling Alternative Parameters (`viewer/GLFWApp.cpp:798-803`)

```cpp
if (ImGui::Button("Sampling 1000 params"))
{
    Eigen::VectorXd input = Eigen::VectorXd::Zero(mMotions[mMotionIdx].motion.rows() + mEnv->getNumKnownParam());
    input << mMotions[mMotionIdx].motion, mEnv->getNormalizedParamStateFromParam(mMotions[mMotionIdx].param.head(mEnv->getNumKnownParam()));
    mGVAE.attr("sampling")(input.cast<float>(), mMotions[mMotionIdx].param);
}
```

**What this does:**
- Generates multiple parameter variations for the same motion
- Useful for exploring parameter space and uncertainty quantification

## Workflow Summary

1. **Load BGN Model** (`GLFWApp.cpp:638`) → Neural network ready for inference
2. **Select Motion** → Choose input motion data from `mMotions[mMotionIdx]`
3. **Predict Skeleton** (`GLFWApp.cpp:786`) → Neural network predicts skeletal parameters from motion
4. **Apply Parameters** (`GLFWApp.cpp:806`) → Update simulation character with predicted skeleton
5. **Verify Results** (`GLFWApp.cpp:1652`) → Visual comparison of predicted vs. actual motion

This pipeline enables **motion-to-skeleton inference**, allowing you to determine what skeletal configuration would produce a given motion pattern.

## Key Files and Components

- **`viewer/GLFWApp.cpp`**: Main UI and workflow control
- **`sim/Environment.cpp`**: Parameter management and application
- **`sim/Character.cpp`**: Skeleton modification and optimization
- **`python/advanced_vae.py`**: BGN neural network implementation
- **`bgn/`**: Directory containing trained BGN models
- **`motions/`**: Directory containing input motion files

## Usage Tips

1. Ensure BGN model files are present in the `bgn/` directory
2. Load motion data before attempting prediction
3. Use "Predict and set param" for immediate feedback
4. Enable visualization (uncomment line 1652) to see prediction results
5. Experiment with "Sampling 1000 params" to explore parameter variations