#include <cmath>
#include <utility>
#include <algorithm>
#include <tinyxml2.h>
#include <ezc3d/ezc3d_all.h>
#include "C3D_Reader.h"
#include "C3D.h"
#include "Log.h"
#include "ascii.h"

Eigen::MatrixXd getRotationMatrixFromPoints(Eigen::Vector3d p0, Eigen::Vector3d p1, Eigen::Vector3d p2)
{
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

    Eigen::Vector3d _axis = p1 - p0;
    Eigen::Vector3d axis1 = p2 - p0;
    Eigen::Vector3d axis2 = _axis.cross(axis1);
    Eigen::Vector3d axis3 = axis1.cross(axis2);

    axis1.normalize();
    axis2.normalize();
    axis3.normalize();

    R.col(0) = axis1;
    R.col(1) = axis2;
    R.col(2) = axis3;

    return R;
}

// ============================================================================
// SECTION 1: Infrastructure & Configuration
// ============================================================================

C3D_Reader::C3D_Reader(std::string marker_path, RenderCharacter *character)
{
    mCharacter = character;
    mVirtSkeleton = character->getSkeleton();  // Get skeleton from Character

    mFrameRate = 60;

    tinyxml2::XMLDocument doc;
    doc.LoadFile(marker_path.c_str());
    if (doc.Error())
    {
        std::cout << "Error loading marker set file: " << marker_path << std::endl;
        std::cout << doc.ErrorName() << std::endl;
        std::cout << doc.ErrorStr() << std::endl;
        return;
    }

    mSkelInfos.clear();
    mMarkerSet.clear();

    for (auto bn : mVirtSkeleton->getBodyNodes())
    {
        ModifyInfo SkelInfo;
        mSkelInfos.push_back(std::make_pair(bn->getName(), SkelInfo));
    }

    auto marker = doc.FirstChildElement("Markers");
    for (TiXmlElement *s = marker->FirstChildElement(); s != NULL; s = s->NextSiblingElement())
    {
        std::string name = std::string(s->Attribute("name"));
        std::string bn = std::string(s->Attribute("bn"));
        Eigen::Vector3d offset = string_to_vector3d(s->Attribute("offset"));

        MocapMarker m;
        m.name = name;
        m.bn = mVirtSkeleton->getBodyNode(bn);
        m.offset = offset;

        mMarkerSet.push_back(m);
    }

    femurL_torsion = 0.0;
    femurR_torsion = 0.0;

    // Load skeleton fitting config
    mFittingConfig = loadSkeletonFittingConfig("data/config/skeleton_fitting.yaml");
}

C3D_Reader::~C3D_Reader()
{
}

// Load default bone mappings
void SkeletonFittingConfig::loadDefaults() {
    frameStart = 0;
    frameEnd = 0;
    maxIterations = 50;
    convergenceThreshold = 1e-6;
    plotConvergence = true;

    boneMappings.clear();
    // Lower body
    boneMappings.push_back({"Pelvis",  {10, 11, 12}, false});
    boneMappings.push_back({"FemurR",  {25, 13, 14}, true});
    boneMappings.push_back({"TibiaR",  {14, 15, 16}, false});
    boneMappings.push_back({"TalusR",  {16, 17, 18}, false});
    boneMappings.push_back({"FemurL",  {26, 19, 20}, true});
    boneMappings.push_back({"TibiaL",  {20, 21, 22}, false});
    boneMappings.push_back({"TalusL",  {22, 23, 24}, false});
    // Upper body
    boneMappings.push_back({"Head",    {0, 1, 2}, false});
    boneMappings.push_back({"Torso",   {3, 4, 7}, false});
    boneMappings.push_back({"ArmR",    {3, 5, 6}, false});
    boneMappings.push_back({"ArmL",    {7, 8, 9}, false});
}

// Load skeleton fitting config from YAML
SkeletonFittingConfig C3D_Reader::loadSkeletonFittingConfig(const std::string& configPath) {
    SkeletonFittingConfig config;

    try {
        YAML::Node yaml = YAML::LoadFile(configPath);
        auto sf = yaml["skeleton_fitting"];

        if (!sf) {
            LOG_WARN("[C3D_Reader] No skeleton_fitting section in config, using defaults");
            config.loadDefaults();
            return config;
        }

        if (sf["frame_range"]) {
            config.frameStart = sf["frame_range"]["start"].as<int>(0);
            config.frameEnd = sf["frame_range"]["end"].as<int>(0);
        }

        if (sf["optimization"]) {
            config.maxIterations = sf["optimization"]["max_iterations"].as<int>(50);
            config.convergenceThreshold = sf["optimization"]["convergence_threshold"].as<double>(1e-6);
            config.plotConvergence = sf["optimization"]["plot_convergence"].as<bool>(true);
        }

        if (sf["bone_mappings"]) {
            for (const auto& bm : sf["bone_mappings"]) {
                SkeletonFittingConfig::BoneMapping mapping;
                mapping.boneName = bm["bone"].as<std::string>();
                mapping.markerIndices = bm["markers"].as<std::vector<int>>();
                mapping.hasVirtualMarker = bm["virtual_first"].as<bool>(false);
                config.boneMappings.push_back(mapping);
            }
        }

        LOG_INFO("[C3D_Reader] Loaded skeleton fitting config:");
        LOG_INFO("  - frameRange: " << config.frameStart << " to " << config.frameEnd);
        LOG_INFO("  - maxIterations: " << config.maxIterations);
        LOG_INFO("  - convergenceThreshold: " << config.convergenceThreshold);
        LOG_INFO("  - plotConvergence: " << (config.plotConvergence ? "true" : "false"));
        LOG_INFO("  - boneMappings: " << config.boneMappings.size() << " bones");

    } catch (const std::exception& e) {
        LOG_WARN("[C3D_Reader] Failed to load config from " << configPath << ": " << e.what());
        LOG_WARN("[C3D_Reader] Using default configuration");
        config.loadDefaults();
    }

    return config;
}

void C3D_Reader::reloadFittingConfig() {
    mFittingConfig = loadSkeletonFittingConfig("data/config/skeleton_fitting.yaml");
}

void C3D_Reader::resetSkeletonToDefault() {
    // Reset all bone parameters to default (scale = 1.0)
    for (auto& info : mSkelInfos) {
        std::get<1>(info) = ModifyInfo();
    }
    if (mCharacter) {
        mCharacter->applySkeletonBodyNode(mSkelInfos, mVirtSkeleton);
    }
    // Set skeleton to zero pose
    if (mVirtSkeleton) {
        mVirtSkeleton->setPositions(Eigen::VectorXd::Zero(mVirtSkeleton->getNumDofs()));
    }
    LOG_INFO("[C3D_Reader] Skeleton reset to default scales and zero pose");
}

// ============================================================================
// SECTION 2: Main Entry Point
// ============================================================================

C3D* C3D_Reader::loadC3D(const std::string& path, const C3DConversionParams& params)
{
    LOG_VERBOSE("[C3D_Reader] loadC3D started for: " << path);

    // Step 1: Load marker data from C3D file
    C3D* c3dData = loadMarkerData(path);
    if (!c3dData) {
        return nullptr;
    }

    // Step 2: Load ezc3d data and setup
    ezc3d::c3d c3d(path);
    mFrameRate = static_cast<int>(std::lround(c3d.header().frameRate()));

    const size_t numFrames = c3d.data().nbFrames();
    if (numFrames == 0) {
        LOG_ERROR("[C3D_Reader] No frames found in C3D file");
        delete c3dData;
        return nullptr;
    }

    // Step 2.5: Extract all markers and apply backward walking correction before IK
    std::vector<std::vector<Eigen::Vector3d>> allMarkers;
    allMarkers.reserve(numFrames);
    for (size_t frameIdx = 0; frameIdx < numFrames; ++frameIdx) {
        allMarkers.push_back(extractMarkersFromFrame(c3d, frameIdx));
    }

    // Detect and correct backward walking (matches sim/C3D.cpp behavior)
    C3D::detectAndCorrectBackwardWalking(allMarkers);

    // Step 3: Initialize skeleton pose for IK
    // First, set skeleton to zero pose
    Eigen::VectorXd pos = mVirtSkeleton->getPositions();
    pos.setZero();
    pos[mVirtSkeleton->getJoint("ForeArmR")->getIndexInSkeleton(0)] = M_PI * 0.5;
    pos[mVirtSkeleton->getJoint("ForeArmL")->getIndexInSkeleton(0)] = M_PI * 0.5;
    mVirtSkeleton->setPositions(pos);

    // Use multi-frame fitting for anisotropic bone scales
    // This handles: lower body, head, torso, arms with alternating optimization
    // Plus: ForeArm and Spine with naive uniform scale approach
    fitSkeletonMultiFrame(allMarkers, params, true);  // true = show convergence plots in terminal

    // Update c3dData markers with augmented hip joint data from mOriginalMarkers
    // fitSkeletonMultiFrame() computed hip joints (indices 25, 26) via Harrington method
    for (size_t frameIdx = 0; frameIdx < mOriginalMarkers.size(); ++frameIdx) {
        c3dData->setMarkers(static_cast<int>(frameIdx), mOriginalMarkers[frameIdx]);
    }

    // Store reference markers and transformations
    mRefMarkers.clear();
    for (auto m : mMarkerSet)
        mRefMarkers.push_back(m.getGlobalPos());

    mRefBnTransformation.clear();
    for (auto bn : mVirtSkeleton->getBodyNodes())
        mRefBnTransformation.push_back(bn->getTransform());

    // Step 4: Convert all frames to skeleton poses via IK using corrected markers
    // NOTE: mOriginalMarkers is already populated by fitSkeletonMultiFrame() with augmented hip joints
    // Use mOriginalMarkers (which has hip joint data at indices 25, 26) instead of allMarkers
    // Only generate poses for fitted frame range (mFitFrameStart to mFitFrameEnd)
    mCurrentMotion.clear();
    std::vector<Eigen::VectorXd> motion;

    int numFitFrames = mFitFrameEnd - mFitFrameStart + 1;
    motion.reserve(numFitFrames);

    for (int i = 0; i < numFitFrames; ++i)
    {
        int globalFrame = mFitFrameStart + i;
        // Use new function with optimizer's R, t for pelvis
        Eigen::VectorXd pose = getPoseFromC3D_Optimized(i, mOriginalMarkers[globalFrame]);

        // Log first 3 frames for debugging marker-skeleton alignment
        if (i < 3) {
            Eigen::Vector3d rootPos(pose[3], pose[4], pose[5]);
            Eigen::Vector3d pelvisCenter = (mOriginalMarkers[globalFrame][10] + mOriginalMarkers[globalFrame][11] + mOriginalMarkers[globalFrame][12]) / 3.0;

            LOG_VERBOSE("[C3D_Reader] Fit frame " << i << " (global " << globalFrame << "):");
            LOG_VERBOSE("  - Root position (skeleton): [" << rootPos[0] << ", " << rootPos[1] << ", " << rootPos[2] << "]");
            LOG_VERBOSE("  - Pelvis center (markers): [" << pelvisCenter[0] << ", " << pelvisCenter[1] << ", " << pelvisCenter[2] << "]");
            LOG_VERBOSE("  - Difference (root - pelvis): [" << (rootPos[0] - pelvisCenter[0]) << ", "
                        << (rootPos[1] - pelvisCenter[1]) << ", " << (rootPos[2] - pelvisCenter[2]) << "]");
        }

        motion.push_back(pose);
    }
    // Step 5: Apply post-processing (reordering, zeroing, marker alignment)
    // - deprecated: it is required only for give offset to 3/8 of total frames
    // applyMotionPostProcessing(motion, c3dData);

    // Step 6: Set skeleton poses on C3D object and return

    c3dData->setSkeletonPoses(motion);
    c3dData->setSourceFile(path);

    LOG_VERBOSE("[C3D_Reader] C3D created successfully");
    LOG_VERBOSE("[C3D_Reader] - NumFrames: " << c3dData->getNumFrames());

    return c3dData;
}

// ============================================================================
// SECTION 3: loadC3D Step 1 - Load Marker Data
// ============================================================================

C3D* C3D_Reader::loadMarkerData(const std::string& path)
{
    LOG_VERBOSE("[C3D_Reader] Loading marker data from: " << path);

    C3D* markerData = new C3D(path);
    if (!markerData || markerData->getNumFrames() == 0) {
        LOG_ERROR("[C3D_Reader] Failed to load marker data or no frames found");
        delete markerData;
        return nullptr;
    }

    LOG_VERBOSE("[C3D_Reader] Marker data loaded: " << markerData->getNumFrames() << " frames");
    return markerData;
}

// ============================================================================
// SECTION 4: loadC3D Step 2 - Extract Markers from Frame
// ============================================================================

std::vector<Eigen::Vector3d> C3D_Reader::extractMarkersFromFrame(const ezc3d::c3d& c3d, size_t frameIdx)
{
    const auto& frame = c3d.data().frame(frameIdx);
    const auto& points = frame.points();
    const size_t numPoints = points.nbPoints();

    std::vector<Eigen::Vector3d> markers;
    markers.reserve(numPoints);

    for (size_t i = 0; i < numPoints; ++i)
    {
        const auto& point = points.point(i);
        Eigen::Vector3d marker;
        // Convert from C3D coordinate system (mm) to meters with axis reordering
        marker[0] = 0.001 * point.y();
        marker[1] = 0.001 * point.z();
        marker[2] = 0.001 * point.x();
        markers.emplace_back(marker);
    }

    return markers;
}

// ============================================================================
// SECTION 5: loadC3D Step 3 - Multi-Frame Skeleton Fitting (Orchestrator)
// ============================================================================

// Multi-frame skeleton fitting with anisotropic scales
// Uses multi-stage pipeline:
//   Stage 0: Fit pelvis (root body) - extracts R, t, S in world coordinates
//   Stage 1: Compute hip joint centers using Harrington method
//   Stage 2: Fit remaining bones using bone-local coordinates (S only)
//   Stage 3: Apply all scales to skeleton
void C3D_Reader::fitSkeletonMultiFrame(
    const std::vector<std::vector<Eigen::Vector3d>>& allMarkersConst,
    const C3DConversionParams& params,
    bool plotConvergence)
{
    // Reset all bone parameters to default
    for (auto& info : mSkelInfos) {
        std::get<1>(info) = ModifyInfo();
    }
    if (mCharacter) {
        mCharacter->applySkeletonBodyNode(mSkelInfos, mVirtSkeleton);
    }

    // Make a mutable copy for augmentation (hip joint centers)
    std::vector<std::vector<Eigen::Vector3d>> allMarkers = allMarkersConst;

    LOG_INFO("[C3D_Reader] Starting multi-stage skeleton fitting...");

    // Populate mRefMarkers before fitting (skeleton is at reference pose after reset above)
    // These are the expected marker world positions used as reference 'q' in the optimizer
    mRefMarkers.clear();
    for (auto& m : mMarkerSet) {
        mRefMarkers.push_back(m.getGlobalPos());
    }

    // Clear previous fitting results
    mBoneR_frames.clear();
    mBoneT_frames.clear();

    // ============ STAGE 1: Compute Hip Joint Centers ============
    // Uses Harrington anthropometric regression based on ASIS/SACR markers
    // This augments allMarkers with computed hip joint positions at indices 25, 26
    if (plotConvergence) {
        std::cout << "\n=== Stage 1: Computing Hip Joint Centers (Harrington) ===\n" << std::endl;
    }
    augmentMarkersWithHipJoints(allMarkers);

    // Store augmented markers back to mOriginalMarkers so viewer can display them
    mOriginalMarkers = allMarkers;

    // ============ STAGE 2: Fit ALL Bones from Config ============
    // All bones (including Pelvis) are now in the config and fitted uniformly
    // Each bone extracts S (scale) and stores R, t (global transforms)

    for (const auto& mapping : mFittingConfig.boneMappings) {
        // FemurR/FemurL now have real hip joint data (indices 25, 26)
        // No more virtual marker workaround needed!
        bool hasVirtual = (mapping.boneName != "FemurR" && mapping.boneName != "FemurL")
                           && mapping.hasVirtualMarker;

        fitBoneLocal(mapping.boneName, mapping.markerIndices, hasVirtual, allMarkers, plotConvergence);

        // Apply scale immediately so bone transforms are correct for subsequent bones
        if (mCharacter) {
            mCharacter->applySkeletonBodyNode(mSkelInfos, mVirtSkeleton);
        }
    }

    // ============ STAGE 3: Apply All Scales ============
    // LOG_INFO("[Stage 3] Applying scales to skeleton...");

    // Apply femur torsions from params
    BodyNode* femurRBn = mVirtSkeleton->getBodyNode("FemurR");
    BodyNode* femurLBn = mVirtSkeleton->getBodyNode("FemurL");
    if (femurRBn && femurLBn) {
        int femurR_idx = femurRBn->getIndexInSkeleton();
        int femurL_idx = femurLBn->getIndexInSkeleton();
        std::get<1>(mSkelInfos[femurR_idx]).value[4] = params.femurTorsionR;
        std::get<1>(mSkelInfos[femurL_idx]).value[4] = params.femurTorsionL;
        femurR_torsion = params.femurTorsionR;
        femurL_torsion = params.femurTorsionL;
    }

    // Apply to skeleton
    if (mCharacter) {
        mCharacter->applySkeletonBodyNode(mSkelInfos, mVirtSkeleton);
    }

    // Update reference markers after fitting
    mRefMarkers.clear();
    for (auto& m : mMarkerSet) {
        mRefMarkers.push_back(m.getGlobalPos());
    }

    // Print summary
    LOG_INFO("[C3D_Reader] Multi-stage skeleton fitting complete. Final scales:");
    LOG_INFO("  Pelvis: [" << std::get<1>(mSkelInfos[mVirtSkeleton->getBodyNode("Pelvis")->getIndexInSkeleton()]).value[0]
             << ", " << std::get<1>(mSkelInfos[mVirtSkeleton->getBodyNode("Pelvis")->getIndexInSkeleton()]).value[1]
             << ", " << std::get<1>(mSkelInfos[mVirtSkeleton->getBodyNode("Pelvis")->getIndexInSkeleton()]).value[2] << "]");
    for (const auto& mapping : mFittingConfig.boneMappings) {
        BodyNode* bn = mVirtSkeleton->getBodyNode(mapping.boneName);
        if (bn) {
            int idx = bn->getIndexInSkeleton();
            auto& modInfo = std::get<1>(mSkelInfos[idx]);
            LOG_INFO("  " << mapping.boneName << ": [" << modInfo.value[0] << ", "
                     << modInfo.value[1] << ", " << modInfo.value[2] << "]");
        }
    }
}

// ============================================================================
// SECTION 6: fitSkeletonMultiFrame Sub-Methods (in call order)
// ============================================================================

// Augment marker data with computed hip joint centers
void C3D_Reader::augmentMarkersWithHipJoints(
    std::vector<std::vector<Eigen::Vector3d>>& allMarkers)
{
    // Marker indices from marker_set.xml
    const int IDX_RASI = 10;
    const int IDX_LASI = 11;
    const int IDX_VSAC = 12;  // SACR equivalent
    const int IDX_RHJC = 25;  // R_Hip virtual marker
    const int IDX_LHJC = 26;  // L_Hip virtual marker

    for (size_t frame = 0; frame < allMarkers.size(); ++frame) {
        auto& markers = allMarkers[frame];

        // Ensure marker array is large enough
        if (markers.size() <= (size_t)IDX_LHJC) {
            markers.resize(IDX_LHJC + 1, Eigen::Vector3d::Zero());
        }

        const Eigen::Vector3d& RASI = markers[IDX_RASI];
        const Eigen::Vector3d& LASI = markers[IDX_LASI];
        const Eigen::Vector3d& SACR = markers[IDX_VSAC];

        // Compute hip joint centers using Harrington method
        markers[IDX_RHJC] = computeHipJointCenter(LASI, RASI, SACR, false);  // Right
        markers[IDX_LHJC] = computeHipJointCenter(LASI, RASI, SACR, true);   // Left
    }

    // LOG_INFO("[Stage 1] Augmented markers with Harrington hip joint centers");
    // if (!allMarkers.empty()) {
    //     // Print computed hip joint positions for debugging
    //     LOG_INFO("  RHJC[0] = (" << allMarkers[0][IDX_RHJC].transpose() << ")");
    //     LOG_INFO("  LHJC[0] = (" << allMarkers[0][IDX_LHJC].transpose() << ")");

    //     // Print ASIS distances for validation
    //     double d_ASIS = (allMarkers[0][IDX_LASI] - allMarkers[0][IDX_RASI]).norm();
    //     double d_depth = ((allMarkers[0][IDX_LASI] + allMarkers[0][IDX_RASI]) / 2.0 - allMarkers[0][IDX_VSAC]).norm();
    //     LOG_INFO("  d_ASIS = " << d_ASIS * 1000.0 << " mm, d_depth = " << d_depth * 1000.0 << " mm");
    // }
}

// Stage 1: Harrington Hip Joint Center Estimation
// Reference: Harrington et al. (2007) "Prediction of the hip joint centre"
// Journal of Biomechanics 40(3): 595-602
Eigen::Vector3d C3D_Reader::computeHipJointCenter(
    const Eigen::Vector3d& LASI,
    const Eigen::Vector3d& RASI,
    const Eigen::Vector3d& SACR,
    bool isLeft)
{
    // 1) Pelvis origin (midpoint of ASIS)
    Eigen::Vector3d O = (LASI + RASI) / 2.0;

    // 2) Posterior point (SACR serves as P since we don't have LPSI/RPSI)
    Eigen::Vector3d P = SACR;

    // 3) Pelvis coordinate system (unit axes)
    // Eigen::Vector3d e_ML = (LASI - RASI).normalized();  // Medial-Lateral (left-positive)
    Eigen::Vector3d e_ML = (RASI - LASI).normalized();  // Medial-Lateral (left-positive)
    // Eigen::Vector3d v_post = P - O;
    Eigen::Vector3d v_post = O - P;
    Eigen::Vector3d e_SI = (e_ML.cross(v_post)).normalized();  // Superior-Inferior
    Eigen::Vector3d e_AP = e_SI.cross(e_ML);  // Anterior-Posterior

    // 4) Marker-derived lengths (in meters)
    double d_ASIS = (LASI - RASI).norm();
    double d_depth = (O - P).norm();

    // 5) Harrington regression offsets (meters)
    // These coefficients are from the original paper for adult subjects
    double ML, AP, SI;
    if (isLeft) {
        ML = -(0.33 * d_ASIS + 0.0073);  // Negative = towards left
        AP = -0.24 * d_depth - 0.0099;    // Posterior offset
        SI = -0.30 * d_ASIS - 0.0109;     // Inferior offset
    } else {  // Right
        ML = 0.33 * d_ASIS + 0.0073;      // Positive = towards right
        AP = -0.24 * d_depth - 0.0099;
        SI = -0.30 * d_ASIS - 0.0109;
    }

    // 6) Hip joint center in world coordinates
    return O + ML * e_ML + AP * e_AP + SI * e_SI;
}

// Unified bone fitting - extracts S (scale) and stores R, t (global transforms)
// Works for all bones including Pelvis
void C3D_Reader::fitBoneLocal(
    const std::string& boneName,
    const std::vector<int>& markerIndices,
    bool /* hasVirtualMarker - no longer used, hip joints computed by Harrington */,
    const std::vector<std::vector<Eigen::Vector3d>>& allMarkers,
    bool plotConvergence)
{
    BodyNode* bn = mVirtSkeleton->getBodyNode(boneName);
    if (!bn) {
        LOG_WARN("[Fitting] Bone not found: " << boneName);
        return;
    }

    // Frame range from config
    int totalFrames = (int)allMarkers.size();
    int startFrame = std::max(0, mFittingConfig.frameStart);
    int endFrame = (mFittingConfig.frameEnd < 0) ? totalFrames : std::min(mFittingConfig.frameEnd + 1, totalFrames);
    int K = endFrame - startFrame;

    if (K <= 0) {
        LOG_WARN("[Fitting] Invalid frame range, using frame 0 only");
        startFrame = 0;
        K = 1;
    }

    // Store frame range (needed for getPoseFromC3D_Optimized)
    mFitFrameStart = startFrame;
    mFitFrameEnd = endFrame - 1;  // Convert to inclusive end

    // Extract frames for fitting (global/world coordinates)
    std::vector<std::vector<Eigen::Vector3d>> globalP(K);
    for (int k = 0; k < K; ++k) {
        int frameIdx = startFrame + k;
        globalP[k] = allMarkers[frameIdx];  // Pass all markers for the frame
    }

    // Run optimization - estimateScaleAlternating handles coordinate transforms
    bool shouldPlot = mFittingConfig.plotConvergence && plotConvergence;
    if (shouldPlot) {
        std::cout << "\n=== Fitting " << boneName << " ===" << std::endl;
    }

    auto result = estimateScaleAlternating(bn, markerIndices, globalP,
        mFittingConfig.maxIterations,
        mFittingConfig.convergenceThreshold,
        shouldPlot);

    if (result.valid) {
        // Store scale
        int idx = bn->getIndexInSkeleton();
        auto& modInfo = std::get<1>(mSkelInfos[idx]);
        modInfo.value[0] = result.scale(0);
        modInfo.value[1] = result.scale(1);
        modInfo.value[2] = result.scale(2);
        modInfo.value[3] = 1.0;

        // Store per-frame global transforms (NEW)
        mBoneR_frames[boneName] = result.R_frames;
        mBoneT_frames[boneName] = result.t_frames;

        // Legacy compatibility for Pelvis
        if (boneName == "Pelvis") {
            mPelvisR_frames = result.R_frames;
            mPelvisT_frames = result.t_frames;
            mPelvisRotation = result.R_frames[0];
            mPelvisTranslation = result.t_frames[0];
        }

        LOG_INFO("[Fitting] " << boneName << " scale: ["
                 << result.scale.transpose() << "] RMS=" << result.finalRMS * 1000.0 << "mm"
                 << " (" << result.iterations << " iters, " << K << " frames)");
    }
}

// ============================================================================
// SECTION 7: Core Algorithm - Alternating Scale Estimation
// ============================================================================

// Alternating optimization for anisotropic scale estimation
// Accepts BodyNode, marker indices, and GLOBAL marker positions
// Internally handles world-to-local transformation and returns GLOBAL transforms
BoneFitResult C3D_Reader::estimateScaleAlternating(
    BodyNode* bn,
    const std::vector<int>& markerIndices,
    const std::vector<std::vector<Eigen::Vector3d>>& globalP,
    int maxIterations,
    double convergenceThreshold,
    bool plotConvergence)
{
    BoneFitResult out;
    out.valid = false;
    out.scale = Eigen::Vector3d::Ones();
    out.iterations = 0;
    out.finalRMS = 0.0;

    if (!bn) return out;

    // =====================================================
    // Step 0: Setup coordinate transforms
    // =====================================================
    Eigen::Isometry3d bnTransform = bn->getTransform();
    Eigen::Isometry3d invTransform = bnTransform.inverse();

    // Get bone size for computing q from marker offsets
    auto* shapeNode = bn->getShapeNodeWith<VisualAspect>(0);
    if (!shapeNode) return out;
    const auto* boxShape = dynamic_cast<const BoxShape*>(shapeNode->getShape().get());
    if (!boxShape) return out;
    Eigen::Vector3d size = boxShape->getSize();

    // =====================================================
    // Step 1: Compute reference markers q (bone-local)
    // =====================================================
    std::vector<Eigen::Vector3d> q;
    for (int idx : markerIndices) {
        const Eigen::Vector3d& offset = mMarkerSet[idx].offset;
        Eigen::Vector3d localPos(
            std::abs(size[0]) * 0.5 * offset[0],
            std::abs(size[1]) * 0.5 * offset[1],
            std::abs(size[2]) * 0.5 * offset[2]
        );
        q.push_back(localPos);
    }

    // =====================================================
    // Step 2: Transform global p to bone-local coordinates
    // =====================================================
    const int K = (int)globalP.size();
    const int N = (int)q.size();
    if (K == 0 || N < 2) return out;

    std::vector<std::vector<Eigen::Vector3d>> p(K);
    for (int k = 0; k < K; ++k) {
        for (size_t i = 0; i < markerIndices.size(); ++i) {
            int idx = markerIndices[i];
            if (idx < (int)globalP[k].size()) {
                // Transform from world to bone-local coordinates
                Eigen::Vector3d localPos = invTransform * globalP[k][idx];
                p[k].push_back(localPos);
            } else {
                p[k].push_back(Eigen::Vector3d::Zero());
            }
        }
    }

    // Debug: Print input markers
    if (plotConvergence) {
        std::cout << "\n--- INPUT MARKERS (N=" << N << ", K=" << K << ") ---" << std::endl;
        std::cout << "Reference markers q (bone-local):" << std::endl;
        for (int i = 0; i < N; ++i) {
            std::cout << "  q[" << i << "] = (" << q[i].x() << ", " << q[i].y() << ", " << q[i].z() << ")" << std::endl;
        }
        std::cout << "Measured markers p[0] (bone-local, frame 0):" << std::endl;
        for (int i = 0; i < N; ++i) {
            std::cout << "  p[0][" << i << "] = (" << p[0][i].x() << ", " << p[0][i].y() << ", " << p[0][i].z() << ")" << std::endl;
        }
        // // Print distance between q and p for each marker
        // std::cout << "Initial marker distances |q[i] - p[0][i]|:" << std::endl;
        // for (int i = 0; i < N; ++i) {
        //     double dist = (q[i] - p[0][i]).norm();
        //     std::cout << "  marker " << i << ": " << dist * 1000.0 << " mm" << std::endl;
        // }
    }

    // Initialize scale to identity
    Eigen::Vector3d S(1.0, 1.0, 1.0);
    Eigen::Vector3d S_prev = S;

    // Storage for per-frame poses
    std::vector<Eigen::Matrix3d> R(K);
    std::vector<Eigen::Vector3d> t(K);

    // Convergence tracking for plotting
    std::vector<double> rmsHistory;
    std::vector<double> scaleChangeHistory;

    for (int iter = 0; iter < maxIterations; ++iter) {
        // =====================================================
        // Step A: Per-frame pose estimation (fix S)
        // =====================================================

        // Compute scaled reference markers
        std::vector<Eigen::Vector3d> x(N);
        Eigen::Vector3d xbar = Eigen::Vector3d::Zero();
        for (int i = 0; i < N; ++i) {
            x[i] = S.asDiagonal() * q[i];
            xbar += x[i];
        }
        xbar /= (double)N;

        // Centered scaled references
        std::vector<Eigen::Vector3d> xc(N);
        for (int i = 0; i < N; ++i) {
            xc[i] = x[i] - xbar;
        }

        // For each frame, compute R_k and t_k via Kabsch
        for (int k = 0; k < K; ++k) {
            // Centroid of measured markers
            Eigen::Vector3d pbar_k = Eigen::Vector3d::Zero();
            for (int i = 0; i < N; ++i) {
                pbar_k += p[k][i];
            }
            pbar_k /= (double)N;

            // Centered measured markers
            std::vector<Eigen::Vector3d> pc(N);
            for (int i = 0; i < N; ++i) {
                pc[i] = p[k][i] - pbar_k;
            }

            // Cross-covariance matrix H_k
            Eigen::Matrix3d H_k = Eigen::Matrix3d::Zero();
            for (int i = 0; i < N; ++i) {
                H_k += pc[i] * xc[i].transpose();
            }

            // Kabsch SVD
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(H_k, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();
            Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
            if ((U * V.transpose()).determinant() < 0.0) D(2, 2) = -1.0;

            R[k] = U * D * V.transpose();
            t[k] = pbar_k - R[k] * xbar;
        }

        // =====================================================
        // Compute Reprojection RMS (metric 1)
        // =====================================================
        double reproj_sum = 0.0;
        for (int k = 0; k < K; ++k) {
            for (int i = 0; i < N; ++i) {
                Eigen::Vector3d p_hat = R[k] * (S.asDiagonal() * q[i]) + t[k];
                reproj_sum += (p_hat - p[k][i]).squaredNorm();
            }
        }
        double rms = std::sqrt(reproj_sum / (K * N));
        rmsHistory.push_back(rms);

        // =====================================================
        // Step B: Shared scale update (fix {R_k, t_k})
        // =====================================================
        Eigen::Vector3d num = Eigen::Vector3d::Zero();
        Eigen::Vector3d den = Eigen::Vector3d::Zero();

        for (int k = 0; k < K; ++k) {
            for (int i = 0; i < N; ++i) {
                // Transform measured marker to body frame
                Eigen::Vector3d y = R[k].transpose() * (p[k][i] - t[k]);

                // Accumulate per-axis
                num += y.cwiseProduct(q[i]);
                den += q[i].cwiseProduct(q[i]);
            }
        }

        // Update scale (elementwise division)
        const double eps = 1e-12;
        for (int a = 0; a < 3; ++a) {
            S(a) = (den(a) < eps) ? 1.0 : num(a) / den(a);
        }

        // =====================================================
        // Compute Scale-change norm (metric 2)
        // =====================================================
        double scaleChange = (S - S_prev).norm();
        scaleChangeHistory.push_back(scaleChange);

        // Debug: Print per-iteration values
        if (plotConvergence) {
            std::cout << "Iter " << iter << ": S=(" << S.x() << ", " << S.y() << ", " << S.z()
                      << ") RMS=" << rms * 1000.0 << "mm dS=" << scaleChange << std::endl;

            // Print translation and marker coordinate differences for frame 0
            // if (K > 0) {
            //     std::cout << "  t[0] = (" << t[0].x() * 1000.0 << ", " << t[0].y() * 1000.0 << ", " << t[0].z() * 1000.0 << ") mm" << std::endl;
            //     std::cout << "  Marker diff (R*S*q + t - p) (frame 0):" << std::endl;
            //     for (int i = 0; i < N; ++i) {
            //         Eigen::Vector3d p_hat = R[0] * (S.asDiagonal() * q[i]) + t[0];
            //         Eigen::Vector3d diff = p_hat - p[0][i];
            //         int actualIdx = (i < (int)markerIndices.size()) ? markerIndices[i] : i;
            //         std::cout << "    marker " << actualIdx << ": (" << diff.x() * 1000.0 << ", " << diff.y() * 1000.0 << ", " << diff.z() * 1000.0 << ") mm" << std::endl;
            //     }
            // }
        }

        S_prev = S;
        out.iterations = iter + 1;

        if (scaleChange < convergenceThreshold) {
            if (plotConvergence) {
                std::cout << "Converged at iteration " << iter << std::endl;
            }
            break;
        }
    }

    // =====================================================
    // Plot convergence using ASCII chart
    // =====================================================
    if (plotConvergence && !rmsHistory.empty()) {
        // Normalize for better visualization (scale change is much smaller)
        double maxRms = *std::max_element(rmsHistory.begin(), rmsHistory.end());
        double maxScaleChange = *std::max_element(scaleChangeHistory.begin(), scaleChangeHistory.end());

        std::vector<double> rmsNorm, scaleNorm;
        for (size_t i = 0; i < rmsHistory.size(); ++i) {
            rmsNorm.push_back(rmsHistory[i] * 1000.0);  // Convert to mm
            // Scale up scale-change for visibility
            if (maxScaleChange > 1e-12) {
                scaleNorm.push_back(scaleChangeHistory[i] * (maxRms / maxScaleChange) * 0.5 * 1000.0);
            } else {
                scaleNorm.push_back(0.0);
            }
        }

        ascii::Asciichart chart({
            {"RMS (mm)", rmsNorm},
            {"Scale chg", scaleNorm}
        });

        std::cout << chart.show_legend(true).height(6).Plot()
                  << "Final RMS: " << rmsHistory.back() * 1000.0 << " mm"
                  << ", Scale: [" << S(0) << ", " << S(1) << ", " << S(2) << "]"
                  << " (" << out.iterations << " iters)" << std::endl;
    }

    out.scale = S;
    out.finalRMS = rmsHistory.empty() ? 0.0 : rmsHistory.back();
    out.valid = true;

    // =====================================================
    // Step 5: Output R, t as GLOBAL transforms
    // =====================================================
    // The optimization finds R, t in bone-local coordinates
    // We convert to global/world coordinates by multiplying with bone's default transform
    // bnTransform was already computed in Step 0
    out.R_frames.resize(K);
    out.t_frames.resize(K);
    for (int k = 0; k < K; ++k) {
        // Convert bone-local R, t to global
        // Global_T = bnTransform * Local_T
        Eigen::Isometry3d localT = Eigen::Isometry3d::Identity();
        localT.linear() = R[k];
        localT.translation() = t[k];

        Eigen::Isometry3d globalT = bnTransform * localT;
        out.R_frames[k] = globalT.linear();
        out.t_frames[k] = globalT.translation();
    }

    if (plotConvergence && K > 0) {
        std::cout << "Output GLOBAL transform (frame 0):" << std::endl;
        std::cout << "  R[0]:\n" << out.R_frames[0] << std::endl;
        std::cout << "  t[0] = (" << out.t_frames[0].x() * 1000.0 << ", "
                  << out.t_frames[0].y() * 1000.0 << ", " << out.t_frames[0].z() * 1000.0 << ") mm" << std::endl;
    }

    return out;
}

// ============================================================================
// SECTION 8: loadC3D Step 4 - Pose Extraction
// ============================================================================

// Use optimizer's global R, t for all fitted bones
// With SKEL_FREE_JOINTS, each bone is independent (6 DOF)
Eigen::VectorXd C3D_Reader::getPoseFromC3D_Optimized(int fitFrameIdx, std::vector<Eigen::Vector3d>& _pos)
{
    // Initialize with zero pose
    Eigen::VectorXd pos = mVirtSkeleton->getPositions();
    pos.setZero();
    mVirtSkeleton->setPositions(pos);

    // Process bones in config order (pelvis first, then children)
    // This ensures parent transforms are set before computing child joint positions
    for (const auto& mapping : mFittingConfig.boneMappings) {
        const std::string& boneName = mapping.boneName;

        // Check if we have transforms for this bone
        auto it = mBoneR_frames.find(boneName);
        if (it == mBoneR_frames.end()) continue;

        const auto& R_frames = it->second;
        if (fitFrameIdx >= (int)R_frames.size()) continue;

        auto* bn = mVirtSkeleton->getBodyNode(boneName);
        if (!bn) continue;

        auto* joint = bn->getParentJoint();
        if (!joint) continue;

        // Get stored global transform (bodynode world position/orientation from fitting)
        Eigen::Isometry3d bodynodeGlobalT = Eigen::Isometry3d::Identity();
        bodynodeGlobalT.linear() = R_frames[fitFrameIdx];
        bodynodeGlobalT.translation() = mBoneT_frames.at(boneName)[fitFrameIdx];

        Eigen::Isometry3d jointT;

        // Formula:
        // joint_tx = parent_bn_global * parent_to_joint * joint_angle
        // child_global = joint_tx * child_to_joint
        //
        // Therefore:
        // child_global = parent_bn_global * parent_to_joint * joint_angle * child_to_joint
        // joint_angle = parent_to_joint.inverse() * parent_bn_global.inverse() * child_global * child_to_joint.inverse()

        Eigen::Isometry3d parentToJoint = joint->getTransformFromParentBodyNode();
        Eigen::Isometry3d childToJoint = joint->getTransformFromChildBodyNode();

        // Get parent bodynode global transform
        Eigen::Isometry3d parentBnGlobal = Eigen::Isometry3d::Identity();
        if (boneName == "Pelvis") {
            // Pelvis has no parent bodynode, parent is world (identity)
            parentBnGlobal = Eigen::Isometry3d::Identity();
        } else {
            // Get parent bodynode's current global transform from skeleton
            auto* parentBn = bn->getParentBodyNode();
            if (parentBn) {
                parentBnGlobal = parentBn->getTransform();
            }
        }

        // Compute joint angle (local rotation/translation)
        jointT = parentToJoint.inverse() * parentBnGlobal.inverse() * bodynodeGlobalT * childToJoint.inverse();

        // Convert joint transform to FreeJoint positions and update skeleton
        int jn_idx = joint->getIndexInSkeleton(0);
        int jn_dof = joint->getNumDofs();
        if (jn_idx >= 0 && jn_idx + jn_dof <= pos.size()) {
            Eigen::VectorXd jointPos = FreeJoint::convertToPositions(jointT);
            pos.segment(jn_idx, jn_dof) = jointPos;

            // Update skeleton so next bone can get correct joint global position
            mVirtSkeleton->setPositions(pos);
        }
    }

    return pos;
}

// ============================================================================
// SECTION 9: Legacy Methods (kept for reference/compatibility)
// ============================================================================

Eigen::VectorXd C3D_Reader::getPoseFromC3D(std::vector<Eigen::Vector3d>& _pos)
{
    int jn_idx = 0;
    int jn_dof = 0;
    Eigen::Matrix3d T = Eigen::Matrix3d::Identity();

    Eigen::VectorXd pos = mVirtSkeleton->getPositions();
    pos.setZero();
    mVirtSkeleton->setPositions(pos);

    // Pelvis

    jn_idx = mVirtSkeleton->getJoint("Pelvis")->getIndexInSkeleton(0);
    jn_dof = mVirtSkeleton->getJoint("Pelvis")->getNumDofs();

    Eigen::Matrix3d origin_pelvis = getRotationMatrixFromPoints(mRefMarkers[10], mRefMarkers[11], mRefMarkers[12]);
    Eigen::Matrix3d current_pelvis = getRotationMatrixFromPoints(_pos[10], _pos[11], _pos[12]);
    Eigen::Isometry3d current_pelvis_T = Eigen::Isometry3d::Identity();
    current_pelvis_T.linear() = current_pelvis * origin_pelvis.transpose();
    current_pelvis_T.translation() = (_pos[10] + _pos[11] + _pos[12]) / 3.0 - (mRefMarkers[10] + mRefMarkers[11] + mRefMarkers[12]) / 3.0;

    pos.segment(jn_idx, jn_dof) = FreeJoint::convertToPositions(current_pelvis_T);

    mVirtSkeleton->getJoint("Pelvis")->setPositions(FreeJoint::convertToPositions(current_pelvis_T));
    // Right Leg

    // FemurR
    jn_idx = mVirtSkeleton->getJoint("FemurR")->getIndexInSkeleton(0);
    jn_dof = mVirtSkeleton->getJoint("FemurR")->getNumDofs();

    Eigen::Matrix3d origin_femurR = getRotationMatrixFromPoints(mMarkerSet[25].getGlobalPos() , mMarkerSet[13].getGlobalPos() , mMarkerSet[14].getGlobalPos() );
    Eigen::Matrix3d current_femurR = getRotationMatrixFromPoints(mMarkerSet[25].getGlobalPos() , _pos[13] , _pos[14] );
    Eigen::Isometry3d pT = mVirtSkeleton->getJoint("FemurR")->getParentBodyNode()->getTransform() * mVirtSkeleton->getJoint("FemurR")->getTransformFromParentBodyNode();

    T = current_femurR * (origin_femurR.transpose());
    mVirtSkeleton->getJoint("FemurR")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));


    // TibiaR
    jn_idx = mVirtSkeleton->getJoint("TibiaR")->getIndexInSkeleton(0);
    jn_dof = mVirtSkeleton->getJoint("TibiaR")->getNumDofs();

    Eigen::Matrix3d origin_kneeR = getRotationMatrixFromPoints(mMarkerSet[14].getGlobalPos(), mMarkerSet[15].getGlobalPos(), mMarkerSet[16].getGlobalPos());
    Eigen::Matrix3d current_kneeR = getRotationMatrixFromPoints(_pos[14], _pos[15], _pos[16]);
    T = (current_kneeR * origin_kneeR.transpose());

    pT = mVirtSkeleton->getJoint("TibiaR")->getParentBodyNode()->getTransform() * mVirtSkeleton->getJoint("TibiaR")->getTransformFromParentBodyNode();
    // Extract only the first component for 1-DOF knee joint
    Eigen::VectorXd kneeR_angles = pT.linear().transpose() * BallJoint::convertToPositions(T);
    mVirtSkeleton->getJoint("TibiaR")->setPosition(0, kneeR_angles[0]);

    // TalusR
    jn_idx = mVirtSkeleton->getJoint("TalusR")->getIndexInSkeleton(0);
    jn_dof = mVirtSkeleton->getJoint("TalusR")->getNumDofs();

    Eigen::Matrix3d origin_talusR = getRotationMatrixFromPoints(mMarkerSet[16].getGlobalPos(), mMarkerSet[17].getGlobalPos(), mMarkerSet[18].getGlobalPos());
    Eigen::Matrix3d current_talusR = getRotationMatrixFromPoints(_pos[16], _pos[17], _pos[18]);
    T = (current_talusR * origin_talusR.transpose());
    pT = mVirtSkeleton->getJoint("TalusR")->getParentBodyNode()->getTransform() * mVirtSkeleton->getJoint("TalusR")->getTransformFromParentBodyNode();
    mVirtSkeleton->getJoint("TalusR")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));
    // FemurL
    jn_idx = mVirtSkeleton->getJoint("FemurL")->getIndexInSkeleton(0);
    jn_dof = mVirtSkeleton->getJoint("FemurL")->getNumDofs();

    Eigen::Matrix3d origin_femurL = getRotationMatrixFromPoints(mMarkerSet[26].getGlobalPos(), mMarkerSet[19].getGlobalPos(), mMarkerSet[20].getGlobalPos());
    Eigen::Matrix3d current_femurL = getRotationMatrixFromPoints(mMarkerSet[26].getGlobalPos(), _pos[19], _pos[20]);
    T = current_femurL * origin_femurL.transpose();
    pT = mVirtSkeleton->getJoint("FemurL")->getParentBodyNode()->getTransform() * mVirtSkeleton->getJoint("FemurL")->getTransformFromParentBodyNode();

    mVirtSkeleton->getJoint("FemurL")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    // TibiaL
    jn_idx = mVirtSkeleton->getJoint("TibiaL")->getIndexInSkeleton(0);
    jn_dof = mVirtSkeleton->getJoint("TibiaL")->getNumDofs();

    Eigen::Matrix3d origin_kneeL = getRotationMatrixFromPoints(mMarkerSet[20].getGlobalPos(), mMarkerSet[21].getGlobalPos(), mMarkerSet[22].getGlobalPos());
    Eigen::Matrix3d current_kneeL = getRotationMatrixFromPoints(_pos[20], _pos[21], _pos[22]);
    T = current_kneeL * origin_kneeL.transpose();
    pT = mVirtSkeleton->getJoint("TibiaL")->getParentBodyNode()->getTransform() * mVirtSkeleton->getJoint("TibiaL")->getTransformFromParentBodyNode();

    // Extract only the first component for 1-DOF knee joint
    Eigen::VectorXd kneeL_angles = pT.linear().transpose() * BallJoint::convertToPositions(T);
    mVirtSkeleton->getJoint("TibiaL")->setPosition(0, kneeL_angles[0]);

    // TalusL
    jn_idx = mVirtSkeleton->getJoint("TalusL")->getIndexInSkeleton(0);
    jn_dof = mVirtSkeleton->getJoint("TalusL")->getNumDofs();

    Eigen::Matrix3d origin_talusL = getRotationMatrixFromPoints(mMarkerSet[22].getGlobalPos(), mMarkerSet[23].getGlobalPos(), mMarkerSet[24].getGlobalPos());
    Eigen::Matrix3d current_talusL = getRotationMatrixFromPoints(_pos[22], _pos[23], _pos[24]);
    T = current_talusL * origin_talusL.transpose();
    pT = mVirtSkeleton->getJoint("TalusL")->getParentBodyNode()->getTransform() * mVirtSkeleton->getJoint("TalusL")->getTransformFromParentBodyNode();

    mVirtSkeleton->getJoint("TalusL")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));


    // Spine and Torso
    Eigen::Matrix3d origin_torso = getRotationMatrixFromPoints(mMarkerSet[3].getGlobalPos(), mMarkerSet[4].getGlobalPos(), mMarkerSet[7].getGlobalPos());
    Eigen::Matrix3d current_torso = getRotationMatrixFromPoints(_pos[3], _pos[4], _pos[7]);
    T = current_torso * origin_torso.transpose();
    pT = mVirtSkeleton->getJoint("Torso")->getParentBodyNode()->getTransform() * mVirtSkeleton->getJoint("Torso")->getTransformFromParentBodyNode();
    Eigen::Quaterniond tmp_T = Eigen::Quaterniond(T).slerp(0.5, Eigen::Quaterniond::Identity());
    T = tmp_T.toRotationMatrix();

    // Spine
    jn_idx = mVirtSkeleton->getJoint("Spine")->getIndexInSkeleton(0);
    jn_dof = mVirtSkeleton->getJoint("Spine")->getNumDofs();
    mVirtSkeleton->getJoint("Spine")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));
    // Torso
    jn_idx = mVirtSkeleton->getJoint("Torso")->getIndexInSkeleton(0);
    jn_dof = mVirtSkeleton->getJoint("Torso")->getNumDofs();
    mVirtSkeleton->getJoint("Torso")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    // Neck and Head
    Eigen::Matrix3d origin_head = getRotationMatrixFromPoints(mMarkerSet[0].getGlobalPos(), mMarkerSet[1].getGlobalPos(), mMarkerSet[2].getGlobalPos());
    Eigen::Matrix3d current_head = getRotationMatrixFromPoints(_pos[0], _pos[1], _pos[2]);
    T = current_head * origin_head.transpose();
    pT = mVirtSkeleton->getJoint("Head")->getParentBodyNode()->getTransform() * mVirtSkeleton->getJoint("Head")->getTransformFromParentBodyNode();
    tmp_T = Eigen::Quaterniond(T).slerp(0.5, Eigen::Quaterniond::Identity());
    T = tmp_T.toRotationMatrix();

    // // Neck
    // jn_idx = mBVHSkeleton->getJoint("Neck")->getIndexInSkeleton(0);
    // jn_dof = mBVHSkeleton->getJoint("Neck")->getNumDofs();
    // mBVHSkeleton->getJoint("Neck")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));
    // // Head
    // jn_idx = mBVHSkeleton->getJoint("Head")->getIndexInSkeleton(0);
    // jn_dof = mBVHSkeleton->getJoint("Head")->getNumDofs();
    // mBVHSkeleton->getJoint("Head")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    // Arm

    // ArmR

    // Elbow Angle
    Eigen::Vector3d v1 = _pos[3] - _pos[5];
    Eigen::Vector3d v2 = _pos[6] - _pos[5];

    double angle = abs(atan2(v1.cross(v2).norm(), v1.dot(v2)));

    if(angle > M_PI * 0.5)
        angle = M_PI - angle;

    jn_idx = mVirtSkeleton->getJoint("ForeArmR")->getIndexInSkeleton(0);
    pos[jn_idx] = angle;
    mVirtSkeleton->getJoint("ForeArmR")->setPosition(0, angle);

    Eigen::Matrix3d origin_armR = getRotationMatrixFromPoints(mMarkerSet[3].getGlobalPos(), mMarkerSet[5].getGlobalPos(), mMarkerSet[6].getGlobalPos());
    Eigen::Matrix3d current_armR = getRotationMatrixFromPoints(mMarkerSet[3].getGlobalPos(), _pos[5], _pos[6]);
    T = current_armR * origin_armR.transpose();
    pT = mVirtSkeleton->getJoint("ArmR")->getParentBodyNode()->getTransform() * mVirtSkeleton->getJoint("ArmR")->getTransformFromParentBodyNode();

    mVirtSkeleton->getJoint("ArmR")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    // ArmL

    // Elbow Angle
    v1 = _pos[8] - _pos[7];
    v2 = _pos[8] - _pos[9];

    angle = abs(atan2(v1.cross(v2).norm(), v1.dot(v2)));

    if(angle > M_PI * 0.5)
        angle = M_PI - angle;

    jn_idx = mVirtSkeleton->getJoint("ForeArmL")->getIndexInSkeleton(0);
    pos[jn_idx] = angle;
    mVirtSkeleton->getJoint("ForeArmL")->setPosition(0, angle);

    Eigen::Matrix3d origin_armL = getRotationMatrixFromPoints(mMarkerSet[7].getGlobalPos(), mMarkerSet[8].getGlobalPos(), mMarkerSet[9].getGlobalPos());
    Eigen::Matrix3d current_armL = getRotationMatrixFromPoints(mMarkerSet[7].getGlobalPos(), _pos[8], _pos[9]);
    T = current_armL * origin_armL.transpose();
    pT = mVirtSkeleton->getJoint("ArmL")->getParentBodyNode()->getTransform() * mVirtSkeleton->getJoint("ArmL")->getTransformFromParentBodyNode();

    mVirtSkeleton->getJoint("ArmL")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    return mVirtSkeleton->getPositions();
}

void C3D_Reader::fitSkeletonToMarker(std::vector<Eigen::Vector3d> init_marker, double torsionL, double torsionR)
{
    // Reset skeleton parameters to default before fitting
    // This ensures each C3D file starts with a clean skeleton state
    for (auto& info : mSkelInfos) {
        std::get<1>(info) = ModifyInfo();  // Reset to default (1,1,1,1,0)
    }

    // Apply reset to skeleton before computing new values
    if (mCharacter) {
        mCharacter->applySkeletonBodyNode(mSkelInfos, mVirtSkeleton);
    }

    std::vector<Eigen::Vector3d> ref_markers;
    for(auto m: mMarkerSet)
        ref_markers.push_back(m.getGlobalPos());

    // Pelvis size
    int idx = mVirtSkeleton->getBodyNode("Pelvis")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[11] - init_marker[10]).norm() / (ref_markers[11] - ref_markers[10]).norm();

    // FemurR size
    idx = mVirtSkeleton->getBodyNode("FemurR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[11] - init_marker[20]).norm() / (ref_markers[11] - ref_markers[20]).norm();

    // TibiaR size
    idx = mVirtSkeleton->getBodyNode("TibiaR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[16] - init_marker[14]).norm() / (ref_markers[16] - ref_markers[14]).norm();

    // TalusR size
    idx = mVirtSkeleton->getBodyNode("TalusR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[18] - init_marker[19]).norm() / (ref_markers[18] - ref_markers[19]).norm();

    // FemurL size
    idx = mVirtSkeleton->getBodyNode("FemurL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[11] - init_marker[20]).norm() / (ref_markers[11] - ref_markers[20]).norm();

    // TibiaL size
    idx = mVirtSkeleton->getBodyNode("TibiaL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[20] - init_marker[22]).norm() / (ref_markers[20] - ref_markers[22]).norm();

    // TalusL size
    idx = mVirtSkeleton->getBodyNode("TalusL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[18] - init_marker[19]).norm() / (ref_markers[18] - ref_markers[19]).norm();

    // Upper Body
    idx = mVirtSkeleton->getBodyNode("Spine")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mVirtSkeleton->getBodyNode("Torso")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mVirtSkeleton->getBodyNode("Neck")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mVirtSkeleton->getBodyNode("Head")->getIndexInSkeleton();
    // std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mVirtSkeleton->getBodyNode("ShoulderR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mVirtSkeleton->getBodyNode("ShoulderL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mVirtSkeleton->getBodyNode("ArmR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mVirtSkeleton->getBodyNode("ArmL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mVirtSkeleton->getBodyNode("ForeArmR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mVirtSkeleton->getBodyNode("ForeArmL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mVirtSkeleton->getBodyNode("HandR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[4] - init_marker[3]).norm() / (ref_markers[4] - ref_markers[3]).norm();

    idx = mVirtSkeleton->getBodyNode("HandL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[4] - init_marker[3]).norm() / (ref_markers[4] - ref_markers[3]).norm();

    // Torsion
    idx = mVirtSkeleton->getBodyNode("FemurR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[4] = torsionR;

    idx = mVirtSkeleton->getBodyNode("FemurL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[4] = torsionL;

    femurR_torsion = torsionR;
    femurL_torsion = torsionL;

    // Apply final skeleton modifications
    if (mCharacter) {
        mCharacter->applySkeletonBodyNode(mSkelInfos, mVirtSkeleton);
    }
}

// ============================================================================
// SECTION 10: Helper & Public API Methods
// ============================================================================

// Get marker position in BONE-LOCAL frame (for Stage 1 alternating optimization)
// Returns q_i^(b) in segment-local coordinates, WITHOUT world transform
Eigen::Vector3d C3D_Reader::getMarkerLocalPos(int markerIdx)
{
    if (markerIdx < 0 || markerIdx >= (int)mMarkerSet.size()) {
        LOG_WARN("[C3D_Reader] Invalid marker index: " << markerIdx);
        return Eigen::Vector3d::Zero();
    }

    const MocapMarker& marker = mMarkerSet[markerIdx];
    BodyNode* bn = marker.bn;

    // Get bone size
    Eigen::Vector3d size = (dynamic_cast<const BoxShape*>(
        bn->getShapeNodeWith<VisualAspect>(0)->getShape().get()))->getSize();

    // Return LOCAL position = offset * (size/2)
    // WITHOUT applying bn->getTransform() - this is the key fix!
    // Stage 1 model: p_k = R_k * S * q_local + t_k
    return Eigen::Vector3d(
        std::abs(size[0]) * 0.5 * marker.offset[0],
        std::abs(size[1]) * 0.5 * marker.offset[1],
        std::abs(size[2]) * 0.5 * marker.offset[2]
    );
}

// TODO: Re-enable when Environment dependency is resolved
// This function requires Environment methods (getParamState, getRefStride, getRefCadence)
MotionData
C3D_Reader::convertToMotion()
{
    LOG_WARN("[C3D_Reader] convertToMotion() temporarily disabled - requires Environment");
    MotionData motion;
    motion.name = "C3D";
    motion.motion = Eigen::VectorXd::Zero(6060);
    motion.param = Eigen::VectorXd::Zero(13);
    return motion;
}

// ============================================================================
// SECTION 11: Unused/Deprecated Methods (kept for reference)
// ============================================================================

void C3D_Reader::initializeSkeletonForIK(const std::vector<Eigen::Vector3d>& firstFrameMarkers,
                                          const C3DConversionParams& params)
{
    LOG_VERBOSE("[C3D_Reader] Initializing skeleton pose for IK (single-frame fallback)");

    // Initialize skeleton to zero pose
    Eigen::VectorXd pos = mVirtSkeleton->getPositions();
    pos.setZero();

    // Set initial arm positions (forearms at 90 degrees)
    pos[mVirtSkeleton->getJoint("ForeArmR")->getIndexInSkeleton(0)] = M_PI * 0.5;
    pos[mVirtSkeleton->getJoint("ForeArmL")->getIndexInSkeleton(0)] = M_PI * 0.5;

    // Initialize knee joints (1-DOF revolute joints)
    pos[mVirtSkeleton->getJoint("TibiaR")->getIndexInSkeleton(0)] = 0.0;
    pos[mVirtSkeleton->getJoint("TibiaL")->getIndexInSkeleton(0)] = 0.0;

    mVirtSkeleton->setPositions(pos);

    // Fit skeleton to first frame markers (naive approach for ForeArm/Spine)
    fitSkeletonToMarker(firstFrameMarkers, params.femurTorsionL, params.femurTorsionR);

    // Store reference markers and transformations
    mRefMarkers.clear();
    for (auto m : mMarkerSet)
        mRefMarkers.push_back(m.getGlobalPos());

    mRefBnTransformation.clear();
    for (auto bn : mVirtSkeleton->getBodyNodes())
        mRefBnTransformation.push_back(bn->getTransform());
}

std::vector<Eigen::VectorXd> C3D_Reader::convertFramesToSkeletonPoses(const ezc3d::c3d& c3d, size_t numFrames)
{
    LOG_VERBOSE("[C3D_Reader] Converting " << numFrames << " frames to skeleton poses");

    std::vector<Eigen::VectorXd> motion;
    motion.reserve(numFrames);

    mOriginalMarkers.clear();
    mOriginalMarkers.reserve(numFrames);

    for (size_t frameIdx = 0; frameIdx < numFrames; ++frameIdx)
    {
        std::vector<Eigen::Vector3d> markers = extractMarkersFromFrame(c3d, frameIdx);
        Eigen::VectorXd pose = getPoseFromC3D(markers);

        // Log first 3 frames for debugging marker-skeleton alignment (before moving markers)
        if (frameIdx < 3) {
            Eigen::Vector3d rootPos(pose[3], pose[4], pose[5]);
            Eigen::Vector3d pelvisCenter = (markers[10] + markers[11] + markers[12]) / 3.0;

            LOG_VERBOSE("[C3D_Reader] Frame " << frameIdx << ":");
            LOG_VERBOSE("  - Root position (skeleton): [" << rootPos[0] << ", " << rootPos[1] << ", " << rootPos[2] << "]");
            LOG_VERBOSE("  - Pelvis center (markers): [" << pelvisCenter[0] << ", " << pelvisCenter[1] << ", " << pelvisCenter[2] << "]");
            LOG_VERBOSE("  - Difference (root - pelvis): [" << (rootPos[0] - pelvisCenter[0]) << ", "
                        << (rootPos[1] - pelvisCenter[1]) << ", " << (rootPos[2] - pelvisCenter[2]) << "]");
        }

        motion.push_back(pose);
        mOriginalMarkers.push_back(std::move(markers));
    }

    LOG_VERBOSE("[C3D_Reader] Conversion complete: " << motion.size() << " poses");
    return motion;
}

// deprecated: it is required only for give offset to 3/8 of total frames
void C3D_Reader::applyMotionPostProcessing(std::vector<Eigen::VectorXd>& motion, C3D* markerData)
{
    LOG_VERBOSE("[C3D_Reader] Applying post-processing to motion and markers");

    if (motion.empty()) {
        LOG_WARN("[C3D_Reader] Empty motion data, skipping post-processing");
        return;
    }

    // Calculate frame offset for walking cycle removal (3/8 of total frames)
    int offset = motion.size() * 3 / 8;

    // Step 1: Zero skeleton positions relative to frame 0
    Eigen::Vector3d initial_skeleton_offset(motion[0][3], 0.0, motion[0][5]);

    for (size_t i = 1; i < motion.size(); i++)
    {
        motion[i][3] -= motion[0][3];
        motion[i][5] -= motion[0][5];
    }

    motion[0][3] = 0;
    motion[0][5] = 0;

    // Step 2: Reorder frames (walking offset removal) for both skeleton and markers
    std::vector<Eigen::VectorXd> new_motion;
    new_motion.reserve(motion.size());

    std::vector<std::vector<Eigen::Vector3d>> new_markers;
    new_markers.reserve(motion.size());

    mCurrentMotion.clear();
    mCurrentMotion.reserve(motion.size());

    // Copy frames from offset to end
    for (size_t i = offset; i < motion.size(); i++)
    {
        new_motion.push_back(motion[i]);
        new_markers.push_back(mOriginalMarkers[i]);
        mCurrentMotion.push_back(motion[i]);
    }

    // Calculate position offset for continuity
    Eigen::Vector3d offset_pos = new_motion.back().segment(3,3);
    offset_pos[1] = 0.0;

    // Copy frames from start to offset with position adjustment
    for (int i = 0; i < offset; i++)
    {
        motion[i].segment(3,3) += offset_pos;
        new_motion.push_back(motion[i]);
        new_markers.push_back(mOriginalMarkers[i]);
        mCurrentMotion.push_back(motion[i]);
    }

    // Step 3: Zero positions relative to new frame 0
    Eigen::Vector3d final_skeleton_offset(new_motion[0][3], 0.0, new_motion[0][5]);

    for (size_t i = 1; i < new_motion.size(); i++)
    {
        new_motion[i][3] -= new_motion[0][3];
        new_motion[i][5] -= new_motion[0][5];

        mCurrentMotion[i][3] -= mCurrentMotion[0][3];
        mCurrentMotion[i][5] -= mCurrentMotion[0][5];
    }

    new_motion[0][3] = 0;
    new_motion[0][5] = 0;
    mCurrentMotion[0][3] = 0;
    mCurrentMotion[0][5] = 0;

    // Step 4: Apply same offset to markers for alignment AND reorder them
    Eigen::Vector3d total_marker_offset = initial_skeleton_offset + final_skeleton_offset;

    LOG_VERBOSE("[C3D_Reader] initial_skeleton_offset: [" << initial_skeleton_offset[0] << ", " << initial_skeleton_offset[1] << ", " << initial_skeleton_offset[2] << "]");
    LOG_VERBOSE("[C3D_Reader] final_skeleton_offset: [" << final_skeleton_offset[0] << ", " << final_skeleton_offset[1] << ", " << final_skeleton_offset[2] << "]");
    LOG_VERBOSE("[C3D_Reader] total_marker_offset: [" << total_marker_offset[0] << ", " << total_marker_offset[1] << ", " << total_marker_offset[2] << "]");

    // Apply offset to reordered markers and update markerData
    for (size_t frameIdx = 0; frameIdx < new_markers.size(); ++frameIdx) {
        const auto& original = new_markers[frameIdx];
        std::vector<Eigen::Vector3d> aligned_markers;
        aligned_markers.reserve(original.size());

        for (const auto& marker : original) {
            Eigen::Vector3d aligned = marker;
            aligned[0] -= total_marker_offset[0];
            aligned[2] -= total_marker_offset[2];
            aligned_markers.push_back(aligned);
        }

        // Update markerData with aligned and reordered markers using proper setter
        markerData->setMarkers(frameIdx, aligned_markers);

        // Log first frame to verify update
        if (frameIdx == 0) {
            Eigen::Vector3d pelvisAfterAlign = (aligned_markers[10] + aligned_markers[11] + aligned_markers[12]) / 3.0;
            LOG_VERBOSE("[C3D_Reader] Frame 0 pelvis center AFTER alignment: [" << pelvisAfterAlign[0] << ", " << pelvisAfterAlign[1] << ", " << pelvisAfterAlign[2] << "]");
        }
    }

    // Replace motion with post-processed version
    motion = std::move(new_motion);
}
