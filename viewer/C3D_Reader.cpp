#include <cmath>
#include <utility>
#include <algorithm>
#include <tinyxml2.h>
#include <ezc3d/ezc3d_all.h>
#include "C3D_Reader.h"
#ifdef USE_CERES
#include "CeresOptimizer.h"
#endif
#include "C3D.h"
#include "UriResolver.h"
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

C3D_Reader::C3D_Reader(std::string fitting_config_path, std::string marker_path, RenderCharacter *free_character, RenderCharacter *motion_character)
{
    mFittingConfigPath = fitting_config_path;
    mFreeCharacter = free_character;
    mMotionCharacter = motion_character;
    mFrameRate = 60;

    // Resolve URI scheme (e.g., @data/ -> absolute path)
    std::string resolvedPath = PMuscle::URIResolver::getInstance().resolve(marker_path);

    tinyxml2::XMLDocument doc;
    doc.LoadFile(resolvedPath.c_str());
    if (doc.Error())
    {
        std::cout << "Error loading marker set file: " << marker_path << std::endl;
        std::cout << doc.ErrorName() << std::endl;
        std::cout << doc.ErrorStr() << std::endl;
        return;
    }

    mSkelInfos.clear();
    mMarkerSet.clear();

    for (auto bn : mFreeCharacter->getSkeleton()->getBodyNodes())
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
        m.bn = mFreeCharacter->getSkeleton()->getBodyNode(bn);
        m.offset = offset;

        mMarkerSet.push_back(m);
    }

    femurL_torsion = 0.0;
    femurR_torsion = 0.0;
}

C3D_Reader::~C3D_Reader()
{
}

// ============================================================================
// MarkerResolver Implementation
// ============================================================================

void MarkerResolver::setC3DLabels(const std::vector<std::string>& labels) {
    mC3DLabels = labels;
    mLabelToIndex.clear();
    for (size_t i = 0; i < labels.size(); ++i) {
        mLabelToIndex[labels[i]] = static_cast<int>(i);
    }
}

int MarkerResolver::resolve(const MarkerReference& ref) const {
    if (ref.type == MarkerReference::Type::DataIndex) {
        return ref.dataIndex;
    }
    // Type::DataLabel - lookup in C3D labels
    auto it = mLabelToIndex.find(ref.dataLabel);
    if (it != mLabelToIndex.end()) {
        return it->second;
    }
    return -1;  // Not found
}

bool MarkerResolver::resolveAll(std::vector<MarkerReference>& refs, std::vector<int>& out) const {
    out.clear();
    out.reserve(refs.size());
    bool allResolved = true;
    for (auto& ref : refs) {
        int idx = resolve(ref);
        if (idx < 0) {
            allResolved = false;
            LOG_WARN("[MarkerResolver] Failed to resolve marker: name='" << ref.name
                     << "' label='" << ref.dataLabel << "'");
        } else {
            ref.dataIndex = idx;  // Cache resolved index
        }
        out.push_back(idx);
    }
    return allResolved;
}

void MarkerResolver::setMarkerSet(const std::vector<MocapMarker>& markerSet) {
    mNameToInfo.clear();
    for (const auto& m : markerSet) {
        if (m.bn) {
            mNameToInfo[m.name] = { m.offset, m.bn->getName() };
        }
    }
}

bool MarkerResolver::resolveOffset(MarkerReference& ref) const {
    auto it = mNameToInfo.find(ref.name);
    if (it != mNameToInfo.end()) {
        ref.offset = it->second.offset;
        ref.boneName = it->second.boneName;
        return true;
    }
    return false;
}

// ============================================================================
// C3D_Reader: Marker Resolution
// ============================================================================

void C3D_Reader::resolveMarkerReferences(const std::vector<std::string>& c3dLabels) {
    mResolver.setC3DLabels(c3dLabels);
    mResolver.setMarkerSet(mMarkerSet);  // Setup skeleton marker lookup

    // Resolve all references in markerMappings
    for (auto& ref : mFittingConfig.markerMappings) {
        // Resolve C3D data index from label
        if (ref.needsResolution()) {
            int resolvedIdx = mResolver.resolve(ref);
            if (resolvedIdx >= 0) {
                ref.dataIndex = resolvedIdx;
            } else {
                LOG_WARN("[C3D_Reader] Failed to resolve marker data label: " << ref.name << " (label: " << ref.dataLabel << ")");
            }
        }

        // Resolve skeleton offset from mMarkerSet
        if (!mResolver.resolveOffset(ref)) {
            LOG_WARN("[C3D_Reader] Failed to resolve marker offset: " << ref.name << " (not found in skeleton)");
        }
    }

    LOG_INFO("[C3D_Reader] Marker reference resolution complete");
}

// Load skeleton fitting config from YAML
SkeletonFittingConfig C3D_Reader::loadSkeletonFittingConfig(const std::string& configPath) {
    SkeletonFittingConfig config;

    try {
        YAML::Node yaml = YAML::LoadFile(configPath);
        auto sf = yaml["skeleton_fitting"];

        if (!sf) {
            LOG_ERROR("[C3D_Reader] No skeleton_fitting section in config file");
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
            // SVD-based optimizer targets (3+ markers)
            if (sf["optimization"]["target_svd"]) {
                for (const auto& t : sf["optimization"]["target_svd"]) {
                    config.targetSvd.push_back(t.as<std::string>());
                }
            }

            config.lambdaRot = sf["optimization"]["lambda_rot"].as<double>(10.0);
            config.interpolateRatio = sf["optimization"]["interpolate_ratio"].as<double>(0.5);
            config.skelRatioBound = sf["optimization"]["skel_ratio_bound"].as<double>(1.5);

            // Ceres-based optimizer targets (2+ markers with regularization)
            if (sf["optimization"]["target_ceres"]) {
                for (const auto& t : sf["optimization"]["target_ceres"]) {
                    config.targetCeres.push_back(t.as<std::string>());
                }
            }

            // Motion skeleton conversion targets (joint names)
            if (sf["optimization"]["target_motion_joint"]) {
                for (const auto& joint : sf["optimization"]["target_motion_joint"]) {
                    config.targetMotionJoint.push_back(joint.as<std::string>());
                }
                LOG_INFO("[Config] target_motion_joint: " << config.targetMotionJoint.size() << " joints");
            }

            // Revolute axis selection mode and thresholds
            if (sf["optimization"]["revolute_axis_mode"]) {
                std::string modeStr = sf["optimization"]["revolute_axis_mode"].as<std::string>();
                if (modeStr == "PCA") {
                    config.revoluteAxisMode = SkeletonFittingConfig::RevoluteAxisMode::PCA;
                } else if (modeStr == "FIX") {
                    config.revoluteAxisMode = SkeletonFittingConfig::RevoluteAxisMode::FIX;
                } else {
                    config.revoluteAxisMode = SkeletonFittingConfig::RevoluteAxisMode::BLEND;
                }
                LOG_INFO("[Config] revolute_axis_mode: " << modeStr);
            }
            if (sf["optimization"]["revolute_axis_threshold_low"]) {
                config.revoluteAxisThresholdLow = sf["optimization"]["revolute_axis_threshold_low"].as<double>();
            }
            if (sf["optimization"]["revolute_axis_threshold_high"]) {
                config.revoluteAxisThresholdHigh = sf["optimization"]["revolute_axis_threshold_high"].as<double>();
            }
        }

        // New format: marker_mappings (flat list)
        if (sf["marker_mappings"]) {
            for (const auto& m : sf["marker_mappings"]) {
                if (m.IsMap()) {
                    std::string name = m["name"].as<std::string>("");
                    if (m["data_idx"]) {
                        int idx = m["data_idx"].as<int>();
                        config.markerMappings.push_back(MarkerReference::fromNameAndIndex(name, idx));
                    } else if (m["data_label"]) {
                        std::string label = m["data_label"].as<std::string>();
                        config.markerMappings.push_back(MarkerReference::fromNameAndLabel(name, label));
                    } else {
                        LOG_WARN("[C3D_Reader] marker_mappings entry missing data_idx or data_label for: " << name);
                    }
                }
            }
        }

        LOG_INFO("[C3D_Reader] Loaded skeleton fitting config:");
        LOG_INFO("  - frameRange: " << config.frameStart << " to " << config.frameEnd);
        LOG_INFO("  - maxIterations: " << config.maxIterations);
        LOG_INFO("  - convergenceThreshold: " << config.convergenceThreshold);
        LOG_INFO("  - plotConvergence: " << (config.plotConvergence ? "true" : "false"));
        LOG_INFO("  - markerMappings: " << config.markerMappings.size() << " markers");
        LOG_INFO("  - targetSvd: " << config.targetSvd.size() << " bones");
        LOG_INFO("  - lambdaRot: " << config.lambdaRot);
        LOG_INFO("  - interpolateRatio: " << config.interpolateRatio);
        LOG_INFO("  - skelRatioBound: " << config.skelRatioBound);
        LOG_INFO("  - targetCeres: " << config.targetCeres.size() << " bones");

    } catch (const std::exception& e) {
        LOG_ERROR("[C3D_Reader] Failed to load config from " << configPath << ": " << e.what());
    }

    return config;
}

// ============================================================================
// SECTION 2: Main Entry Point
// ============================================================================

C3D* C3D_Reader::loadC3D(const std::string& path, const C3DConversionParams& params)
{
    LOG_VERBOSE("[C3D_Reader] loadC3D started for: " << path);

    // ========== 0. Load skeleton fitting config ==========
    // Resolve URI scheme if present
    std::string resolvedConfigPath = PMuscle::URIResolver::getInstance().resolve(mFittingConfigPath);
    mFittingConfig = loadSkeletonFittingConfig(resolvedConfigPath);

    // ========== 1. Load C3D file ==========
    // C3D::load() parses the file and stores markers directly in c3dData->mMarkers
    C3D* c3dData = parseC3DFile(path);
    if (!c3dData) {
        return nullptr;
    }

    mCurrentC3D = c3dData;  // Store for calibration access
    mFrameRate = static_cast<int>(std::lround(c3dData->getFrameRate()));

    const size_t numFrames = c3dData->getNumFrames();
    if (numFrames == 0) {
        LOG_ERROR("[C3D_Reader] No frames found in C3D file");
        mCurrentC3D = nullptr;
        delete c3dData;
        return nullptr;
    }

    resolveMarkerReferences(c3dData->getLabels());

    // ========== 2. Correct backward walking (modifies c3dData markers in place) ==========
    // getAllMarkers() returns mutable reference to c3dData's internal marker storage
    C3D::detectAndCorrectBackwardWalking(c3dData->getAllMarkers());

    // ========== 2b. Compute frame range from config ==========
    int totalFrames = static_cast<int>(numFrames);
    mFitFrameStart = std::max(0, mFittingConfig.frameStart);
    int endFrame = (mFittingConfig.frameEnd < 0) ? totalFrames : std::min(mFittingConfig.frameEnd + 1, totalFrames);
    mFitFrameEnd = endFrame - 1;  // Convert to inclusive end

    if (mFitFrameEnd < mFitFrameStart) {
        LOG_WARN("[C3D_Reader] Invalid frame range from config, using all frames");
        mFitFrameStart = 0;
        mFitFrameEnd = totalFrames - 1;
    }

    // ========== 3. Initialize skeleton to T-pose ==========
    Eigen::VectorXd pos = mFreeCharacter->getSkeleton()->getPositions();
    pos.setZero();
    mFreeCharacter->getSkeleton()->setPositions(pos);

    // Skeleton calibration (anisotropic bone scale fitting)
    if (params.doCalibration) {
        calibrateSkeleton(params);
    }

    // ========== 4. Convert to skeleton poses (IK) ==========
    std::vector<Eigen::VectorXd> motion;
    int numFitFrames = mFitFrameEnd - mFitFrameStart + 1;
    motion.reserve(numFitFrames);

    for (int i = 0; i < numFitFrames; ++i) {
        Eigen::VectorXd pose = buildFramePose(i);
        motion.push_back(pose);
    }

    // ========== 5. Convert to motion skeleton (if available) ==========
    if (mMotionCharacter) {
        mMotionResult = convertToMotionSkeleton();
    }

    // ========== 6. Finalize ==========
    c3dData->setSkeletonPoses(motion);
    c3dData->setSourceFile(path);
    mCurrentC3D = nullptr;  // Clear after processing

    LOG_VERBOSE("[C3D_Reader] Loaded " << c3dData->getNumFrames() << " frames");
    return c3dData;
}

// ============================================================================
// SECTION 3: parseC3DFile - Parse C3D file structure
// ============================================================================

C3D* C3D_Reader::parseC3DFile(const std::string& path)
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
// SECTION 4: calibrateSkeleton - Orchestrate skeleton calibration
//   Pipeline: computeHipJointCenters → calibrateBone (per bone)
// ============================================================================

// Multi-frame skeleton fitting with anisotropic scales
// Uses multi-stage pipeline:
//   Stage 0: Fit pelvis (root body) - extracts R, t, S in world coordinates
//   Stage 1: Compute hip joint centers using Harrington method
//   Stage 2: Fit remaining bones using bone-local coordinates (S only)
//   Stage 3: Apply all scales to skeleton
void C3D_Reader::calibrateSkeleton(const C3DConversionParams& params)
{
    if (!mCurrentC3D) {
        LOG_ERROR("[C3D_Reader] No C3D data loaded, cannot calibrate skeleton");
        return;
    }

    // Reset all bone parameters to default
    for (auto& info : mSkelInfos) {
        std::get<1>(info) = ModifyInfo();
    }
    if (mFreeCharacter) mFreeCharacter->applySkeletonBodyNode(mSkelInfos, mFreeCharacter->getSkeleton());
    if (mMotionCharacter) mMotionCharacter->applySkeletonBodyNode(mSkelInfos, mMotionCharacter->getSkeleton());

    LOG_INFO("[C3D_Reader] Starting multi-stage skeleton fitting...");

    // Clear previous fitting results
    mBoneR_frames.clear();
    mBoneT_frames.clear();
    mBoneOrder.clear();

    // ============ STAGE 1: Compute Hip Joint Centers ============
    // Uses Harrington anthropometric regression based on ASIS/SACR markers
    // This augments markers with computed hip joint positions at indices 25, 26
    std::cout << "\n=== Stage 1: Computing Hip Joint Centers (Harrington) ===\n" << std::endl;
    computeHipJointCenters(mCurrentC3D->getAllMarkers());

    // ============ STAGE 2: Fit ALL Bones from Config ============
    // All bones (including Pelvis) are now in the config and fitted uniformly
    // Each bone extracts S (scale) and stores R, t (global transforms)

    // Build bone -> MarkerReference* map using resolved boneName
    // MarkerReference now contains both offset (skeleton) and dataIndex (C3D)
    std::map<std::string, std::vector<const MarkerReference*>> boneToMarkers;
    for (const auto& ref : mFittingConfig.markerMappings) {
        if (ref.hasOffset() && ref.dataIndex >= 0) {
            boneToMarkers[ref.boneName].push_back(&ref);
        }
    }

    // ============ STAGE 2a: SVD-based bones (3+ markers) ============
    std::cout << "\n=== Stage 2a: SVD-based bone fitting (3+ markers) ===\n" << std::endl;
    for (const auto& boneName : mFittingConfig.targetSvd) {
        auto it = boneToMarkers.find(boneName);
        if (it == boneToMarkers.end() || it->second.empty()) {
            LOG_WARN("[C3D_Reader] No marker mappings found for bone: " << boneName);
            continue;
        }

        calibrateBone(boneName, it->second, mCurrentC3D->getAllMarkers());
        mBoneOrder.push_back(boneName);
    }

#ifdef USE_CERES
    // ============ STAGE 2b: Ceres-based bones (2+ markers with regularization) ============
    runCeresBoneFitting(mFittingConfig, boneToMarkers, mCurrentC3D->getAllMarkers(),
                        mCharacter, mSkelInfos, mBoneR_frames, mBoneT_frames);
    for (const auto& boneName : mFittingConfig.targetCeres) {
        mBoneOrder.push_back(boneName);
    }
#else
    // ============ STAGE 2b Fallback: Marker-distance scaling for arms ============
    scaleArmsFallback();
#endif

    // ============ STAGE 2b-2: Copy dependent scales ============
    // Hand = ForeArm, Neck = Head (if Head in target_svd)
    copyDependentScales();

    // ============ STAGE 2b-3: Interpolate dependent bone transforms ============
    interpolateDependent();

    // ============ STAGE 3: Apply All Scales ============
    // LOG_INFO("[Stage 3] Applying scales to skeleton...");

    // Apply femur torsions from params
    BodyNode* femurRBn = mFreeCharacter->getSkeleton()->getBodyNode("FemurR");
    BodyNode* femurLBn = mFreeCharacter->getSkeleton()->getBodyNode("FemurL");
    if (femurRBn && femurLBn) {
        int femurR_idx = femurRBn->getIndexInSkeleton();
        int femurL_idx = femurLBn->getIndexInSkeleton();
        std::get<1>(mSkelInfos[femurR_idx]).value[4] = params.femurTorsionR;
        std::get<1>(mSkelInfos[femurL_idx]).value[4] = params.femurTorsionL;
        femurR_torsion = params.femurTorsionR;
        femurL_torsion = params.femurTorsionL;
    }

    // Apply to skeleton (both free and motion characters)
    if (mFreeCharacter) mFreeCharacter->applySkeletonBodyNode(mSkelInfos, mFreeCharacter->getSkeleton());
    if (mMotionCharacter) mMotionCharacter->applySkeletonBodyNode(mSkelInfos, mMotionCharacter->getSkeleton());

    // Print summary
    LOG_INFO("[C3D_Reader] Multi-stage skeleton fitting complete. Final scales:");
    // Combine SVD and Ceres targets for summary
    std::vector<std::string> allTargets;
    allTargets.insert(allTargets.end(), mFittingConfig.targetSvd.begin(), mFittingConfig.targetSvd.end());
    allTargets.insert(allTargets.end(), mFittingConfig.targetCeres.begin(), mFittingConfig.targetCeres.end());
    for (const auto& boneName : allTargets) {
        BodyNode* bn = mFreeCharacter->getSkeleton()->getBodyNode(boneName);
        if (bn) {
            int idx = bn->getIndexInSkeleton();
            auto& modInfo = std::get<1>(mSkelInfos[idx]);
            LOG_INFO("  " << boneName << ": [" << modInfo.value[0] << ", "
                     << modInfo.value[1] << ", " << modInfo.value[2] << "]");
        }
    }
}

// ============================================================================
// SECTION 6: Calibration Sub-Methods
//   - computeHipJointCenters: Harrington hip joint estimation
//   - computeHipJointCenter: Single hip joint calculation
//   - calibrateBone: Calibrate single bone scale
// ============================================================================

// Augment marker data with computed hip joint centers
void C3D_Reader::computeHipJointCenters(
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

    // Add labels for computed hip joint centers
    if (mCurrentC3D) {
        auto& labels = mCurrentC3D->getLabels();
        // Ensure labels vector is large enough and add HJC labels
        while (labels.size() <= (size_t)IDX_LHJC) {
            labels.push_back("");
        }
        labels[IDX_RHJC] = "R.HJC";
        labels[IDX_LHJC] = "L.HJC";
    }
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
void C3D_Reader::calibrateBone(
    const std::string& boneName,
    const std::vector<const MarkerReference*>& markers,
    const std::vector<std::vector<Eigen::Vector3d>>& allMarkers)
{
    BodyNode* bn = mFreeCharacter->getSkeleton()->getBodyNode(boneName);
    if (!bn) {
        LOG_WARN("[Fitting] Bone not found: " << boneName);
        return;
    }

    // Use pre-computed frame range from loadC3D
    int K = mFitFrameEnd - mFitFrameStart + 1;

    // Extract frames for fitting (global/world coordinates)
    std::vector<std::vector<Eigen::Vector3d>> globalP(K);
    for (int k = 0; k < K; ++k) {
        int frameIdx = mFitFrameStart + k;
        globalP[k] = allMarkers[frameIdx];  // Pass all markers for the frame
    }

    // Run optimization - optimizeBoneScale handles coordinate transforms
    if (mFittingConfig.plotConvergence) {
        std::cout << "\n=== Fitting " << boneName << " ===" << std::endl;
    }

    auto result = optimizeBoneScale(bn, markers, globalP);

    if (result.valid) {
        // Store scale
        int idx = bn->getIndexInSkeleton();
        auto& modInfo = std::get<1>(mSkelInfos[idx]);
        modInfo.value[0] = result.scale(0);
        modInfo.value[1] = result.scale(1);
        modInfo.value[2] = result.scale(2);
        modInfo.value[3] = 1.0;

        // Store per-frame global transforms
        mBoneR_frames[boneName] = result.R_frames;
        mBoneT_frames[boneName] = result.t_frames;

        LOG_INFO("[Fitting] " << boneName << " scale: ["
                 << result.scale.transpose() << "] RMS=" << result.finalRMS * 1000.0 << "mm"
                 << " (" << result.iterations << " iters, " << K << " frames)");
    }
}

// ============================================================================
// SECTION 7: optimizeBoneScale - Core optimization algorithm
//   Alternating optimization: fix S → solve R,t | fix R,t → solve S
// ============================================================================

// Apply scale ratio bound constraint using harmonic mean projection
// Ensures max(S) / min(S) <= ratioBound
// Uses simple (unweighted) harmonic mean with equal weights
static Eigen::Vector3d applyScaleRatioBound(
    const Eigen::Vector3d& u,      // Unconstrained scale from LS
    double ratioBound)             // Max allowed ratio c
{
    const double eps = 1e-12;

    // Ensure positive scales
    Eigen::Vector3d uPos = u.cwiseMax(eps);

    // Check if constraint already satisfied
    double uMax = uPos.maxCoeff();
    double uMin = uPos.minCoeff();
    if (uMax / uMin <= ratioBound) {
        return uPos;  // Already within bounds
    }

    // Compute simple harmonic mean: m = 3 / (1/u_x + 1/u_y + 1/u_z)
    // Equal weights (w=1 for all axes)
    double invSum = 1.0 / uPos(0) + 1.0 / uPos(1) + 1.0 / uPos(2);
    double m = 3.0 / std::max(invSum, eps);

    // Symmetric interval around m in ratio-space: [m/sqrt(c), m*sqrt(c)]
    double sqrtC = std::sqrt(ratioBound);
    double lo = m / sqrtC;
    double hi = m * sqrtC;

    // Clamp each axis to the interval
    Eigen::Vector3d S;
    for (int a = 0; a < 3; ++a) {
        S(a) = std::clamp(uPos(a), lo, hi);
    }

    return S;
}

// Alternating optimization for anisotropic scale estimation
// Accepts BodyNode, marker indices, and GLOBAL marker positions
// Internally handles world-to-local transformation and returns GLOBAL transforms
BoneFitResult C3D_Reader::optimizeBoneScale(
    BodyNode* bn,
    const std::vector<const MarkerReference*>& markers,
    const std::vector<std::vector<Eigen::Vector3d>>& globalP)
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
    if (!shapeNode) {
        LOG_WARN("[Fitting] Bone shape node not found: " << bn->getName());
        return out;
    }
    const auto* boxShape = dynamic_cast<const BoxShape*>(shapeNode->getShape().get());
    if (!boxShape) {
        LOG_WARN("[Fitting] Bone shape not found: " << bn->getName());
        return out;
    }
    Eigen::Vector3d size = boxShape->getSize();

    // =====================================================
    // Step 1: Compute reference markers q (bone-local)
    // Uses MarkerReference.offset directly (skeleton reference)
    // =====================================================
    std::vector<Eigen::Vector3d> q;
    for (const auto* ref : markers) {
        const Eigen::Vector3d& offset = ref->offset;
        Eigen::Vector3d localPos(
            std::abs(size[0]) * 0.5 * offset[0],
            std::abs(size[1]) * 0.5 * offset[1],
            std::abs(size[2]) * 0.5 * offset[2]
        );
        q.push_back(localPos);
    }

    // =====================================================
    // Step 2: Transform global p to bone-local coordinates
    // Uses MarkerReference.dataIndex for C3D data lookup
    // =====================================================
    const int K = (int)globalP.size();
    const int N = (int)q.size();
    if (K == 0 || N < 2) {
        LOG_WARN("[Fitting] No frames or markers to fit: K=" << K << ", N=" << N);
        return out;
    }

    std::vector<std::vector<Eigen::Vector3d>> p(K);
    for (int k = 0; k < K; ++k) {
        for (const auto* ref : markers) {
            if (ref->dataIndex >= 0 && ref->dataIndex < (int)globalP[k].size()) {
                // Transform from world to bone-local coordinates
                Eigen::Vector3d localPos = invTransform * globalP[k][ref->dataIndex];
                p[k].push_back(localPos);
            } else {
                LOG_WARN("[Fitting] Failed to transform marker: " << ref->name << " (dataIndex: " << ref->dataIndex << ")");
                return out;
            }
        }
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

    for (int iter = 0; iter < mFittingConfig.maxIterations; ++iter) {
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

        // Apply scale ratio bound constraint (unweighted harmonic mean)
        S = applyScaleRatioBound(S, mFittingConfig.skelRatioBound);

        // =====================================================
        // Compute Scale-change norm (metric 2)
        // =====================================================
        double scaleChange = (S - S_prev).norm();
        scaleChangeHistory.push_back(scaleChange);

        S_prev = S;
        out.iterations = iter + 1;

        if (scaleChange < mFittingConfig.convergenceThreshold) {
            if (mFittingConfig.plotConvergence) {
                std::cout << "Converged at iteration " << iter << std::endl;
            }
            break;
        }
    }

    // =====================================================
    // Plot convergence using ASCII chart
    // =====================================================
    if (mFittingConfig.plotConvergence && !rmsHistory.empty()) {
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
    return out;
}

// Note: optimizeBoneScaleCeres has been moved to CeresOptimizer.cpp
// It is only available when USE_CERES is defined

// ============================================================================
// SECTION 8: buildFramePose - Build skeleton pose for one frame
// ============================================================================

// Helper: Compute FreeJoint positions from global bodynode transform
// Formula: joint_angle = parent_to_joint.inverse() * parent_bn_global.inverse() * child_global * child_to_joint
// Returns joint positions (6 DOF) if successful, empty vector otherwise
static Eigen::VectorXd computeJointPositions(
    BodyNode* bn,
    const Eigen::Isometry3d& bodynodeGlobalT)
{
    if (!bn) return Eigen::VectorXd();

    auto* joint = bn->getParentJoint();
    if (!joint || joint->getNumDofs() != 6) return Eigen::VectorXd();

    Eigen::Isometry3d parentToJoint = joint->getTransformFromParentBodyNode();
    Eigen::Isometry3d childToJoint = joint->getTransformFromChildBodyNode();

    Eigen::Isometry3d parentBnGlobal = Eigen::Isometry3d::Identity();
    auto* parentBn = bn->getParentBodyNode();
    if (parentBn) {
        parentBnGlobal = parentBn->getTransform();
    }

    Eigen::Isometry3d jointT = parentToJoint.inverse() * parentBnGlobal.inverse() *
                               bodynodeGlobalT * childToJoint;

    return FreeJoint::convertToPositions(jointT);
}

// Use optimizer's global R, t for all fitted bones
// With SKEL_FREE_JOINTS, each bone is independent (6 DOF)
Eigen::VectorXd C3D_Reader::buildFramePose(int fitFrameIdx)
{
    // Initialize with zero pose
    Eigen::VectorXd pos = mFreeCharacter->getSkeleton()->getPositions();
    pos.setZero();
    mFreeCharacter->getSkeleton()->setPositions(pos);

    // Print iteration order (only on first frame)
    if (fitFrameIdx == 0) {
        std::ostringstream oss;
        oss << "[buildFramePose] mBoneOrder: ";
        for (const auto& name : mBoneOrder) {
            oss << name << " -> ";
        }
        LOG_INFO(oss.str());
    }

    // Process all fitted bones in mBoneOrder (preserves config order)
    for (const auto& boneName : mBoneOrder) {

        // Check if we have transforms for this bone
        auto it = mBoneR_frames.find(boneName);
        if (it == mBoneR_frames.end() || it->second.empty()) continue;
        if (fitFrameIdx >= (int)it->second.size()) continue;

        auto* bn = mFreeCharacter->getSkeleton()->getBodyNode(boneName);
        if (!bn) {
            LOG_WARN("[Fitting] Bone not found: " << boneName);
            continue;
        }

        // Get stored global transform (bodynode world position/orientation from fitting)
        Eigen::Isometry3d bodynodeGlobalT = Eigen::Isometry3d::Identity();
        bodynodeGlobalT.linear() = it->second[fitFrameIdx];
        bodynodeGlobalT.translation() = mBoneT_frames.at(boneName)[fitFrameIdx];

        // Compute joint positions from global transform
        Eigen::VectorXd jointPos = computeJointPositions(bn, bodynodeGlobalT);
        if (jointPos.size() > 0) {
            int jn_idx = bn->getParentJoint()->getIndexInSkeleton(0);
            pos.segment(jn_idx, jointPos.size()) = jointPos;
            mFreeCharacter->getSkeleton()->setPositions(pos);
        }
    }

    // Compute arm rotations from marker heuristics
    computeArmRotations(fitFrameIdx, pos);

    return pos;
}

// ============================================================================
// SECTION 8a: Arm Rotation from Marker Heuristics
// ============================================================================

void C3D_Reader::computeArmRotations(int fitFrameIdx, Eigen::VectorXd& pos)
{
    if (!mCurrentC3D) return;

    int frameIdx = mFitFrameStart + fitFrameIdx;
    const auto& allMarkers = mCurrentC3D->getAllMarkers();
    if (frameIdx >= (int)allMarkers.size()) return;

    const auto& markers = allMarkers[frameIdx];

    // Marker indices - separate elbow markers for upper arm and forearm
    int RSHO_idx = mFittingConfig.getDataIndexForMarker("RSHO_arm");
    int RELB_arm_idx = mFittingConfig.getDataIndexForMarker("RELB_arm");
    int RELB_farm_idx = mFittingConfig.getDataIndexForMarker("RELB_farm");
    int RWRI_idx = mFittingConfig.getDataIndexForMarker("RWRI");
    int LSHO_idx = mFittingConfig.getDataIndexForMarker("LSHO_arm");
    int LELB_arm_idx = mFittingConfig.getDataIndexForMarker("LELB_arm");
    int LELB_farm_idx = mFittingConfig.getDataIndexForMarker("LELB_farm");
    int LWRI_idx = mFittingConfig.getDataIndexForMarker("LWRI");

    auto skel = mFreeCharacter->getSkeleton();

    // Helper: get marker local position on bone from mMarkerSet
    auto getMarkerLocalPos = [&](const std::string& markerName) -> Eigen::Vector3d {
        for (const auto& m : mMarkerSet) {
            if (m.name == markerName && m.bn) {
                auto* shape = m.bn->getShapeNodeWith<VisualAspect>(0);
                if (shape) {
                    auto* box = dynamic_cast<const BoxShape*>(shape->getShape().get());
                    if (box) {
                        Eigen::Vector3d size = box->getSize();
                        return Eigen::Vector3d(
                            std::abs(size[0]) * 0.5 * m.offset[0],
                            std::abs(size[1]) * 0.5 * m.offset[1],
                            std::abs(size[2]) * 0.5 * m.offset[2]);
                    }
                }
            }
        }
        return Eigen::Vector3d::Zero();
    };

    // Process both arms
    for (int side = 0; side < 2; ++side) {
        bool isRight = (side == 0);

        // Get marker indices for this arm
        int sho_idx = isRight ? RSHO_idx : LSHO_idx;
        int elb_arm_idx = isRight ? RELB_arm_idx : LELB_arm_idx;
        int elb_farm_idx = isRight ? RELB_farm_idx : LELB_farm_idx;
        int wri_idx = isRight ? RWRI_idx : LWRI_idx;

        // Check valid markers
        if (sho_idx < 0 || elb_arm_idx < 0 || elb_farm_idx < 0 || wri_idx < 0) {
            if (fitFrameIdx == 0) {
                std::cerr << "[ArmDebug] " << (isRight ? "R" : "L")
                          << " SKIPPED: invalid marker idx (sho=" << sho_idx
                          << " elb_arm=" << elb_arm_idx << " elb_farm=" << elb_farm_idx
                          << " wri=" << wri_idx << ")" << std::endl;
            }
            continue;
        }
        if (sho_idx >= (int)markers.size() || elb_arm_idx >= (int)markers.size() ||
            elb_farm_idx >= (int)markers.size() || wri_idx >= (int)markers.size()) continue;

        // 1. Get marker positions
        Eigen::Vector3d S = markers[sho_idx];
        Eigen::Vector3d E_arm = markers[elb_arm_idx];
        Eigen::Vector3d E_farm = markers[elb_farm_idx];
        Eigen::Vector3d W = markers[wri_idx];

        // 2. Compute segment directions
        Eigen::Vector3d u = (E_arm - S).normalized();
        Eigen::Vector3d l = (W - E_farm).normalized();

        // 3. Compute bending plane normal n = normalize(cross(u,l))
        Eigen::Vector3d n_raw = u.cross(l);
        double n_norm = n_raw.norm();

        // Handle degeneracy: if arm is nearly straight, use previous frame's normal
        const double DEGENERACY_THRESHOLD = 0.1;
        Eigen::Vector3d& prevNormal = isRight ? mPrevArmNormalR : mPrevArmNormalL;

        Eigen::Vector3d n;
        if (n_norm < DEGENERACY_THRESHOLD) {
            n = prevNormal;
        } else {
            n = n_raw / n_norm;
            prevNormal = n;
        }

        // 3b. Compute and cache elbow angle using cosine
        // θ = acos(u · l) gives angle in [0, π]
        // Clamp dot product to [-1, 1] for numerical safety
        double cosAngle = std::max(-1.0, std::min(1.0, u.dot(l)));
        double elbowAngle = std::acos(cosAngle);
        // Negate for left arm (opposite rotation direction)
        if (!isRight) {
            elbowAngle = -elbowAngle;
        }
        std::string elbowKey = isRight ? "ForeArmR" : "ForeArmL";
        if ((int)mElbowAngle_frames[elbowKey].size() <= fitFrameIdx) {
            mElbowAngle_frames[elbowKey].resize(fitFrameIdx + 1);
        }
        mElbowAngle_frames[elbowKey][fitFrameIdx] = elbowAngle;

        // 4. Build rotation using Axis-Constrained Basis Alignment
        //
        // Problem: Directly assigning world vectors (u, l, n) to body axes (x, y, z)
        // assumes bone-local marker direction equals body x-axis, which is approximate.
        //
        // Solution: Compute LOCAL reference directions from marker set, then find R
        // such that R * u_local = u_world and R * n_local = n_world.
        //
        // R = B_world * B_local^T  (basis-to-basis rotation)

        std::string upperBoneName = isRight ? "ArmR" : "ArmL";
        std::string lowerBoneName = isRight ? "ForeArmR" : "ForeArmL";

        // Get local marker positions for upper arm
        std::string shoName = isRight ? "RSHO_arm" : "LSHO_arm";
        std::string elbArmName = isRight ? "RELB_arm" : "LELB_arm";
        std::string elbFarmName = isRight ? "RELB_farm" : "LELB_farm";
        std::string wriName = isRight ? "RWRI" : "LWRI";

        Eigen::Vector3d localSho = getMarkerLocalPos(shoName);
        Eigen::Vector3d localElbArm = getMarkerLocalPos(elbArmName);
        Eigen::Vector3d localElbFarm = getMarkerLocalPos(elbFarmName);
        Eigen::Vector3d localWri = getMarkerLocalPos(wriName);

        // Compute LOCAL reference directions for upper arm
        // û_local = local shoulder → elbow direction
        Eigen::Vector3d u_local = (localElbArm - localSho).normalized();

        // n̂_local: Use the joint's revolute axis (Y-axis in joint frame) transformed to parent body frame
        auto* lowerJoint = skel->getJoint(lowerBoneName);
        Eigen::Matrix3d R_p2j = lowerJoint->getTransformFromParentBodyNode().linear();
        // Joint Y-axis in parent body frame = R_p2j * (0, 1, 0)
        Eigen::Vector3d n_local_upper = R_p2j.col(1);  // Joint Y-axis in upper arm body frame

        // Build orthonormal basis in LOCAL frame (Gram-Schmidt on n w.r.t. u)
        // ñ_local = (n_local - u_local * (u_local·n_local)) / ||...||
        double dot_un_local = u_local.dot(n_local_upper);
        Eigen::Vector3d n_tilde_local = (n_local_upper - u_local * dot_un_local).normalized();
        Eigen::Vector3d z_local = u_local.cross(n_tilde_local);

        Eigen::Matrix3d B_local_upper;
        B_local_upper.col(0) = u_local;
        B_local_upper.col(1) = n_tilde_local;
        B_local_upper.col(2) = z_local;

        // Build orthonormal basis in WORLD frame (same Gram-Schmidt)
        double dot_un_world = u.dot(n);
        Eigen::Vector3d n_tilde_world = (n - u * dot_un_world).normalized();
        Eigen::Vector3d z_world = u.cross(n_tilde_world);

        Eigen::Matrix3d B_world_upper;
        B_world_upper.col(0) = u;
        B_world_upper.col(1) = n_tilde_world;
        B_world_upper.col(2) = z_world;

        // Compute upper arm rotation: R = B_world * B_local^T
        Eigen::Matrix3d R_upper_world = B_world_upper * B_local_upper.transpose();

        // Compute LOCAL reference directions for forearm
        // l̂_local = local elbow → wrist direction
        Eigen::Vector3d l_local = (localWri - localElbFarm).normalized();

        // n̂_local for forearm: Use joint's revolute axis in child body frame
        Eigen::Matrix3d R_c2j = lowerJoint->getTransformFromChildBodyNode().linear();
        // Joint Y-axis in child body frame = R_c2j * (0, 1, 0)
        Eigen::Vector3d n_local_lower = R_c2j.col(1);  // Joint Y-axis in forearm body frame

        // Build orthonormal basis in LOCAL frame for forearm
        double dot_ln_local = l_local.dot(n_local_lower);
        Eigen::Vector3d n_tilde_local_lower = (n_local_lower - l_local * dot_ln_local).normalized();
        Eigen::Vector3d z_local_lower = l_local.cross(n_tilde_local_lower);

        Eigen::Matrix3d B_local_lower;
        B_local_lower.col(0) = l_local;
        B_local_lower.col(1) = n_tilde_local_lower;
        B_local_lower.col(2) = z_local_lower;

        // Build orthonormal basis in WORLD frame for forearm
        double dot_ln_world = l.dot(n);
        Eigen::Vector3d n_tilde_world_lower = (n - l * dot_ln_world).normalized();
        Eigen::Vector3d z_world_lower = l.cross(n_tilde_world_lower);

        Eigen::Matrix3d B_world_lower;
        B_world_lower.col(0) = l;
        B_world_lower.col(1) = n_tilde_world_lower;
        B_world_lower.col(2) = z_world_lower;

        // Compute forearm rotation: R = B_world * B_local^T
        Eigen::Matrix3d R_lower_world = B_world_lower * B_local_lower.transpose();

        // DEBUG: Verify alignment
        if (fitFrameIdx == 0) {
            // Check that R * u_local ≈ u_world
            Eigen::Vector3d u_reconstructed = R_upper_world * u_local;
            Eigen::Vector3d l_reconstructed = R_lower_world * l_local;

            // Compute resulting joint transform
            Eigen::Matrix3d R_joint = R_p2j.transpose() * R_upper_world.transpose() * R_lower_world * R_c2j;
            Eigen::AngleAxisd aa_joint(R_joint);

            std::cerr << "[ArmDebug] " << (isRight ? "R" : "L")
                      << " u_local=[" << u_local.transpose() << "]"
                      << " n_local=[" << n_local_upper.transpose() << "]"
                      << std::endl;
            std::cerr << "[ArmDebug] " << (isRight ? "R" : "L")
                      << " dot(R*u_local, u_world)=" << u_reconstructed.dot(u)
                      << " dot(R*l_local, l_world)=" << l_reconstructed.dot(l)
                      << std::endl;
            std::cerr << "[ArmDebug] " << (isRight ? "R" : "L")
                      << " R_joint axis=[" << aa_joint.axis().transpose() << "]"
                      << " angle=" << aa_joint.angle() * 180.0/M_PI << "°"
                      << " θ_cross=" << elbowAngle * 180.0/M_PI << "°"
                      << std::endl;
        }

        // 5. Apply to FreeJoint positions
        // (Reuse localSho, localElbArm, localElbFarm, localWri computed above)

        // Upper arm: average translation from shoulder and elbow markers
        auto* upperBn = skel->getBodyNode(upperBoneName);
        if (upperBn) {
            Eigen::Vector3d t_sho = S - R_upper_world * localSho;
            Eigen::Vector3d t_elb = E_arm - R_upper_world * localElbArm;
            Eigen::Vector3d t_upper = (t_sho + t_elb) * 0.5;

            Eigen::Isometry3d bodynodeGlobalT = Eigen::Isometry3d::Identity();
            bodynodeGlobalT.linear() = R_upper_world;
            bodynodeGlobalT.translation() = t_upper;

            Eigen::VectorXd jointPos = computeJointPositions(upperBn, bodynodeGlobalT);
            if (jointPos.size() > 0) {
                pos.segment(upperBn->getParentJoint()->getIndexInSkeleton(0), jointPos.size()) = jointPos;
                skel->setPositions(pos);
            }

            // DEBUG: After pose applied, check body axis alignment
            if (fitFrameIdx == 0) {
                Eigen::Matrix3d actualR = upperBn->getTransform().linear();
                Eigen::Vector3d actualMarkerDir = actualR * u_local;
                std::cerr << "[ArmDebug] " << (isRight ? "R" : "L")
                          << " upperArm: dot(R*u_local, u_world)=" << actualMarkerDir.dot(u)
                          << std::endl;
            }
        }

        // Forearm: average translation from elbow and wrist markers
        auto* lowerBn = skel->getBodyNode(lowerBoneName);
        if (lowerBn) {
            Eigen::Vector3d t_elb = E_farm - R_lower_world * localElbFarm;
            Eigen::Vector3d t_wri = W - R_lower_world * localWri;
            Eigen::Vector3d t_lower = (t_elb + t_wri) * 0.5;

            Eigen::Isometry3d bodynodeGlobalT = Eigen::Isometry3d::Identity();
            bodynodeGlobalT.linear() = R_lower_world;
            bodynodeGlobalT.translation() = t_lower;

            Eigen::VectorXd jointPos = computeJointPositions(lowerBn, bodynodeGlobalT);
            if (jointPos.size() > 0) {
                pos.segment(lowerBn->getParentJoint()->getIndexInSkeleton(0), jointPos.size()) = jointPos;
                skel->setPositions(pos);
            }

            // DEBUG: After pose applied, check body axis alignment
            if (fitFrameIdx == 0) {
                Eigen::Matrix3d actualR = lowerBn->getTransform().linear();
                Eigen::Vector3d actualMarkerDir = actualR * l_local;
                std::cerr << "[ArmDebug] " << (isRight ? "R" : "L")
                          << " foreArm: dot(R*l_local, l_world)=" << actualMarkerDir.dot(l)
                          << std::endl;
            }
        }

        // ========== Distance-based flip detection (pose-invariant) ==========
        // After applying tentative transform, check if skeleton joints align with markers
        // If skeleton elbow is closer to wrist marker than elbow marker, arm is flipped

        std::string forearmBone = isRight ? "ForeArmR" : "ForeArmL";
        std::string handBone = isRight ? "HandR" : "HandL";

        // Get skeleton joint positions (after tentative transform applied)
        auto* forearmBn = skel->getBodyNode(forearmBone);
        auto* handBn = skel->getBodyNode(handBone);

        if (forearmBn && handBn) {
            Eigen::Vector3d J_E = forearmBn->getTransform().translation();  // Skeleton elbow
            Eigen::Vector3d J_W = handBn->getTransform().translation();     // Skeleton wrist

            // Marker positions (avoid M_E which conflicts with Euler's constant from <cmath>)
            Eigen::Vector3d marker_elbow = E_arm;   // Elbow marker
            Eigen::Vector3d marker_wrist = W;       // Wrist marker

            // Compute distances
            double dEE = (J_E - marker_elbow).norm();  // Skeleton elbow -> elbow marker
            double dEW = (J_E - marker_wrist).norm();  // Skeleton elbow -> wrist marker
            double dWW = (J_W - marker_wrist).norm();  // Skeleton wrist -> wrist marker
            double dWE = (J_W - marker_elbow).norm();  // Skeleton wrist -> elbow marker

            // Symmetric flip detection: skeleton joints closer to wrong markers
            bool armFlipped = (dEE + dWW) > (dEW + dWE);

            if (armFlipped) {
                // Negate both directions
                u = -u;
                l = -l;

                // Recompute bending plane normal
                Eigen::Vector3d n_raw_flip = u.cross(l);
                double n_norm_flip = n_raw_flip.norm();
                Eigen::Vector3d n_flip = (n_norm_flip < 0.1) ? prevNormal : (n_raw_flip / n_norm_flip);
                if (n_norm_flip >= 0.1) prevNormal = n_flip;

                // Rebuild rotation matrices
                Eigen::Vector3d ua_x = u;
                Eigen::Vector3d ua_y = n_flip;
                Eigen::Vector3d ua_z = ua_x.cross(ua_y).normalized();
                ua_y = ua_z.cross(ua_x).normalized();
                R_upper_world.col(0) = ua_x;
                R_upper_world.col(1) = ua_y;
                R_upper_world.col(2) = ua_z;

                Eigen::Vector3d fa_x = l;
                Eigen::Vector3d fa_y = n_flip;
                Eigen::Vector3d fa_z = fa_x.cross(fa_y).normalized();
                fa_y = fa_z.cross(fa_x).normalized();
                R_lower_world.col(0) = fa_x;
                R_lower_world.col(1) = fa_y;
                R_lower_world.col(2) = fa_z;

                // Recompute translations with corrected rotations
                // Upper arm
                if (upperBn) {
                    Eigen::Vector3d localSho = getMarkerLocalPos(isRight ? "RSHO_arm" : "LSHO_arm");
                    Eigen::Vector3d localElb = getMarkerLocalPos(isRight ? "RELB_arm" : "LELB_arm");
                    Eigen::Vector3d t_sho = S - R_upper_world * localSho;
                    Eigen::Vector3d t_elb = E_arm - R_upper_world * localElb;
                    Eigen::Vector3d t_upper = (t_sho + t_elb) * 0.5;

                    Eigen::Isometry3d bodynodeGlobalT = Eigen::Isometry3d::Identity();
                    bodynodeGlobalT.linear() = R_upper_world;
                    bodynodeGlobalT.translation() = t_upper;

                    Eigen::VectorXd jointPos = computeJointPositions(upperBn, bodynodeGlobalT);
                    if (jointPos.size() > 0) {
                        pos.segment(upperBn->getParentJoint()->getIndexInSkeleton(0), jointPos.size()) = jointPos;
                        skel->setPositions(pos);
                    }
                }

                // Forearm
                if (lowerBn) {
                    Eigen::Vector3d localElbFarm = getMarkerLocalPos(isRight ? "RELB_farm" : "LELB_farm");
                    Eigen::Vector3d localWri = getMarkerLocalPos(isRight ? "RWRI" : "LWRI");
                    Eigen::Vector3d t_elb_flip = E_farm - R_lower_world * localElbFarm;
                    Eigen::Vector3d t_wri_flip = W - R_lower_world * localWri;
                    Eigen::Vector3d t_lower = (t_elb_flip + t_wri_flip) * 0.5;

                    Eigen::Isometry3d bodynodeGlobalT = Eigen::Isometry3d::Identity();
                    bodynodeGlobalT.linear() = R_lower_world;
                    bodynodeGlobalT.translation() = t_lower;

                    Eigen::VectorXd jointPos = computeJointPositions(lowerBn, bodynodeGlobalT);
                    if (jointPos.size() > 0) {
                        pos.segment(lowerBn->getParentJoint()->getIndexInSkeleton(0), jointPos.size()) = jointPos;
                        skel->setPositions(pos);
                    }
                }
            }
        }

        // ========== Store final transforms in mBoneR_frames/mBoneT_frames ==========
        // Read from skeleton after all heuristics are applied
        if (upperBn) {
            Eigen::Isometry3d T = upperBn->getTransform();
            mBoneR_frames[upperBoneName][fitFrameIdx] = T.linear();
            mBoneT_frames[upperBoneName][fitFrameIdx] = T.translation();
        }
        if (lowerBn) {
            Eigen::Isometry3d T = lowerBn->getTransform();
            mBoneR_frames[lowerBoneName][fitFrameIdx] = T.linear();
            mBoneT_frames[lowerBoneName][fitFrameIdx] = T.translation();
        }
    }
}

// ============================================================================
// SECTION 8b: Upper Body Helpers
// ============================================================================

// Build rotation matrix from 3 marker positions (forms coordinate frame)
Eigen::Matrix3d C3D_Reader::getRotationMatrixFromPoints(
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2,
    const Eigen::Vector3d& p3)
{
    Eigen::Vector3d x = (p2 - p1).normalized();
    Eigen::Vector3d temp = (p3 - p1).normalized();
    Eigen::Vector3d z = x.cross(temp).normalized();
    Eigen::Vector3d y = z.cross(x).normalized();

    Eigen::Matrix3d R;
    R.col(0) = x;
    R.col(1) = y;
    R.col(2) = z;
    return R;
}

// Update upper body joint positions from markers
// Uses marker positions directly to compute rotations for bones with <3 markers
void C3D_Reader::updateUpperBodyFromMarkers(
    int fitFrameIdx,
    Eigen::VectorXd& pos,
    const std::vector<Eigen::Vector3d>& markers,
    const std::vector<Eigen::Vector3d>& refMarkers)
{
    auto skel = mFreeCharacter->getSkeleton();

    // Marker indices from config
    int RSHO = mFittingConfig.getDataIndexForMarker("RSHO");
    int ROFF = mFittingConfig.getDataIndexForMarker("ROFF");
    int RELB = mFittingConfig.getDataIndexForMarker("RELB_arm");
    int RWRI = mFittingConfig.getDataIndexForMarker("RWRI");
    int LSHO = mFittingConfig.getDataIndexForMarker("LSHO");
    int LELB = mFittingConfig.getDataIndexForMarker("LELB_arm");
    int LWRI = mFittingConfig.getDataIndexForMarker("LWRI");

    if (RSHO < 0 || LSHO < 0) {
        LOG_WARN("[UpperBody] Missing shoulder markers, skipping upper body");
        return;
    }

#ifdef USE_CERES
    // ========== Arm rotation from Ceres optimizer ==========
    applyCeresArmRotations(mFittingConfig, mBoneR_frames, mBoneT_frames, fitFrameIdx, skel, pos);
#endif
}

// ============================================================================
// SECTION 9: buildFramePoseLegacy - Legacy IK method (kept for reference)
// ============================================================================
//
Eigen::VectorXd C3D_Reader::buildFramePoseLegacy(std::vector<Eigen::Vector3d>& _pos)
{
    int jn_idx = 0;
    int jn_dof = 0;
    Eigen::Matrix3d T = Eigen::Matrix3d::Identity();

    // Build reference markers from marker set (T-pose)
    std::vector<Eigen::Vector3d> mRefMarkers;
    for (const auto& m : mMarkerSet) {
        mRefMarkers.push_back(m.getGlobalPos());
    }

    Eigen::VectorXd pos = mFreeCharacter->getSkeleton()->getPositions();
    pos.setZero();
    mFreeCharacter->getSkeleton()->setPositions(pos);

    // Pelvis
    jn_idx = mFreeCharacter->getSkeleton()->getJoint("Pelvis")->getIndexInSkeleton(0);
    jn_dof = mFreeCharacter->getSkeleton()->getJoint("Pelvis")->getNumDofs();

    Eigen::Matrix3d origin_pelvis = getRotationMatrixFromPoints(mRefMarkers[10], mRefMarkers[11], mRefMarkers[12]);
    Eigen::Matrix3d current_pelvis = getRotationMatrixFromPoints(_pos[10], _pos[11], _pos[12]);
    Eigen::Isometry3d current_pelvis_T = Eigen::Isometry3d::Identity();
    current_pelvis_T.linear() = current_pelvis * origin_pelvis.transpose();
    current_pelvis_T.translation() = (_pos[10] + _pos[11] + _pos[12]) / 3.0 - (mRefMarkers[10] + mRefMarkers[11] + mRefMarkers[12]) / 3.0;

    pos.segment(jn_idx, jn_dof) = FreeJoint::convertToPositions(current_pelvis_T);
    mFreeCharacter->getSkeleton()->getJoint("Pelvis")->setPositions(FreeJoint::convertToPositions(current_pelvis_T));

    // Right Leg - FemurR
    jn_idx = mFreeCharacter->getSkeleton()->getJoint("FemurR")->getIndexInSkeleton(0);
    jn_dof = mFreeCharacter->getSkeleton()->getJoint("FemurR")->getNumDofs();
    Eigen::Matrix3d origin_femurR = getRotationMatrixFromPoints(mMarkerSet[25].getGlobalPos(), mMarkerSet[13].getGlobalPos(), mMarkerSet[14].getGlobalPos());
    Eigen::Matrix3d current_femurR = getRotationMatrixFromPoints(mMarkerSet[25].getGlobalPos(), _pos[13], _pos[14]);
    Eigen::Isometry3d pT = mFreeCharacter->getSkeleton()->getJoint("FemurR")->getParentBodyNode()->getTransform() * mFreeCharacter->getSkeleton()->getJoint("FemurR")->getTransformFromParentBodyNode();
    T = current_femurR * (origin_femurR.transpose());
    mFreeCharacter->getSkeleton()->getJoint("FemurR")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    // TibiaR
    jn_idx = mFreeCharacter->getSkeleton()->getJoint("TibiaR")->getIndexInSkeleton(0);
    jn_dof = mFreeCharacter->getSkeleton()->getJoint("TibiaR")->getNumDofs();
    Eigen::Matrix3d origin_kneeR = getRotationMatrixFromPoints(mMarkerSet[14].getGlobalPos(), mMarkerSet[15].getGlobalPos(), mMarkerSet[16].getGlobalPos());
    Eigen::Matrix3d current_kneeR = getRotationMatrixFromPoints(_pos[14], _pos[15], _pos[16]);
    T = (current_kneeR * origin_kneeR.transpose());
    pT = mFreeCharacter->getSkeleton()->getJoint("TibiaR")->getParentBodyNode()->getTransform() * mFreeCharacter->getSkeleton()->getJoint("TibiaR")->getTransformFromParentBodyNode();
    Eigen::VectorXd kneeR_angles = pT.linear().transpose() * BallJoint::convertToPositions(T);
    mFreeCharacter->getSkeleton()->getJoint("TibiaR")->setPosition(0, kneeR_angles[0]);

    // TalusR
    jn_idx = mFreeCharacter->getSkeleton()->getJoint("TalusR")->getIndexInSkeleton(0);
    jn_dof = mFreeCharacter->getSkeleton()->getJoint("TalusR")->getNumDofs();
    Eigen::Matrix3d origin_talusR = getRotationMatrixFromPoints(mMarkerSet[16].getGlobalPos(), mMarkerSet[17].getGlobalPos(), mMarkerSet[18].getGlobalPos());
    Eigen::Matrix3d current_talusR = getRotationMatrixFromPoints(_pos[16], _pos[17], _pos[18]);
    T = (current_talusR * origin_talusR.transpose());
    pT = mFreeCharacter->getSkeleton()->getJoint("TalusR")->getParentBodyNode()->getTransform() * mFreeCharacter->getSkeleton()->getJoint("TalusR")->getTransformFromParentBodyNode();
    mFreeCharacter->getSkeleton()->getJoint("TalusR")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    // Left Leg - FemurL
    jn_idx = mFreeCharacter->getSkeleton()->getJoint("FemurL")->getIndexInSkeleton(0);
    jn_dof = mFreeCharacter->getSkeleton()->getJoint("FemurL")->getNumDofs();
    Eigen::Matrix3d origin_femurL = getRotationMatrixFromPoints(mMarkerSet[26].getGlobalPos(), mMarkerSet[19].getGlobalPos(), mMarkerSet[20].getGlobalPos());
    Eigen::Matrix3d current_femurL = getRotationMatrixFromPoints(mMarkerSet[26].getGlobalPos(), _pos[19], _pos[20]);
    T = current_femurL * origin_femurL.transpose();
    pT = mFreeCharacter->getSkeleton()->getJoint("FemurL")->getParentBodyNode()->getTransform() * mFreeCharacter->getSkeleton()->getJoint("FemurL")->getTransformFromParentBodyNode();
    mFreeCharacter->getSkeleton()->getJoint("FemurL")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    // TibiaL
    jn_idx = mFreeCharacter->getSkeleton()->getJoint("TibiaL")->getIndexInSkeleton(0);
    jn_dof = mFreeCharacter->getSkeleton()->getJoint("TibiaL")->getNumDofs();
    Eigen::Matrix3d origin_kneeL = getRotationMatrixFromPoints(mMarkerSet[20].getGlobalPos(), mMarkerSet[21].getGlobalPos(), mMarkerSet[22].getGlobalPos());
    Eigen::Matrix3d current_kneeL = getRotationMatrixFromPoints(_pos[20], _pos[21], _pos[22]);
    T = current_kneeL * origin_kneeL.transpose();
    pT = mFreeCharacter->getSkeleton()->getJoint("TibiaL")->getParentBodyNode()->getTransform() * mFreeCharacter->getSkeleton()->getJoint("TibiaL")->getTransformFromParentBodyNode();
    Eigen::VectorXd kneeL_angles = pT.linear().transpose() * BallJoint::convertToPositions(T);
    mFreeCharacter->getSkeleton()->getJoint("TibiaL")->setPosition(0, kneeL_angles[0]);

    // TalusL
    jn_idx = mFreeCharacter->getSkeleton()->getJoint("TalusL")->getIndexInSkeleton(0);
    jn_dof = mFreeCharacter->getSkeleton()->getJoint("TalusL")->getNumDofs();
    Eigen::Matrix3d origin_talusL = getRotationMatrixFromPoints(mMarkerSet[22].getGlobalPos(), mMarkerSet[23].getGlobalPos(), mMarkerSet[24].getGlobalPos());
    Eigen::Matrix3d current_talusL = getRotationMatrixFromPoints(_pos[22], _pos[23], _pos[24]);
    T = current_talusL * origin_talusL.transpose();
    pT = mFreeCharacter->getSkeleton()->getJoint("TalusL")->getParentBodyNode()->getTransform() * mFreeCharacter->getSkeleton()->getJoint("TalusL")->getTransformFromParentBodyNode();
    mFreeCharacter->getSkeleton()->getJoint("TalusL")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    // Spine and Torso
    Eigen::Matrix3d origin_torso = getRotationMatrixFromPoints(mMarkerSet[3].getGlobalPos(), mMarkerSet[4].getGlobalPos(), mMarkerSet[7].getGlobalPos());
    Eigen::Matrix3d current_torso = getRotationMatrixFromPoints(_pos[3], _pos[4], _pos[7]);
    T = current_torso * origin_torso.transpose();
    pT = mFreeCharacter->getSkeleton()->getJoint("Torso")->getParentBodyNode()->getTransform() * mFreeCharacter->getSkeleton()->getJoint("Torso")->getTransformFromParentBodyNode();
    Eigen::Quaterniond tmp_T = Eigen::Quaterniond(T).slerp(0.5, Eigen::Quaterniond::Identity());
    T = tmp_T.toRotationMatrix();
    mFreeCharacter->getSkeleton()->getJoint("Spine")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));
    mFreeCharacter->getSkeleton()->getJoint("Torso")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    // Arms - ArmR
    Eigen::Vector3d v1 = _pos[3] - _pos[5];
    Eigen::Vector3d v2 = _pos[6] - _pos[5];
    double angle = abs(atan2(v1.cross(v2).norm(), v1.dot(v2)));
    if (angle > M_PI * 0.5) angle = M_PI - angle;
    jn_idx = mFreeCharacter->getSkeleton()->getJoint("ForeArmR")->getIndexInSkeleton(0);
    pos[jn_idx] = angle;
    mFreeCharacter->getSkeleton()->getJoint("ForeArmR")->setPosition(0, angle);
    Eigen::Matrix3d origin_armR = getRotationMatrixFromPoints(mMarkerSet[3].getGlobalPos(), mMarkerSet[5].getGlobalPos(), mMarkerSet[6].getGlobalPos());
    Eigen::Matrix3d current_armR = getRotationMatrixFromPoints(mMarkerSet[3].getGlobalPos(), _pos[5], _pos[6]);
    T = current_armR * origin_armR.transpose();
    pT = mFreeCharacter->getSkeleton()->getJoint("ArmR")->getParentBodyNode()->getTransform() * mFreeCharacter->getSkeleton()->getJoint("ArmR")->getTransformFromParentBodyNode();
    mFreeCharacter->getSkeleton()->getJoint("ArmR")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    // ArmL
    v1 = _pos[8] - _pos[7];
    v2 = _pos[8] - _pos[9];
    angle = abs(atan2(v1.cross(v2).norm(), v1.dot(v2)));
    if (angle > M_PI * 0.5) angle = M_PI - angle;
    jn_idx = mFreeCharacter->getSkeleton()->getJoint("ForeArmL")->getIndexInSkeleton(0);
    pos[jn_idx] = angle;
    mFreeCharacter->getSkeleton()->getJoint("ForeArmL")->setPosition(0, angle);
    Eigen::Matrix3d origin_armL = getRotationMatrixFromPoints(mMarkerSet[7].getGlobalPos(), mMarkerSet[8].getGlobalPos(), mMarkerSet[9].getGlobalPos());
    Eigen::Matrix3d current_armL = getRotationMatrixFromPoints(mMarkerSet[7].getGlobalPos(), _pos[8], _pos[9]);
    T = current_armL * origin_armL.transpose();
    pT = mFreeCharacter->getSkeleton()->getJoint("ArmL")->getParentBodyNode()->getTransform() * mFreeCharacter->getSkeleton()->getJoint("ArmL")->getTransformFromParentBodyNode();
    mFreeCharacter->getSkeleton()->getJoint("ArmL")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    return mFreeCharacter->getSkeleton()->getPositions();
}

void C3D_Reader::fitSkeletonToMarker(std::vector<Eigen::Vector3d> init_marker, double torsionL, double torsionR)
{
    // Reset skeleton parameters to default before fitting
    // This ensures each C3D file starts with a clean skeleton state
    for (auto& info : mSkelInfos) {
        std::get<1>(info) = ModifyInfo();  // Reset to default (1,1,1,1,0)
    }

    // Apply reset to skeleton before computing new values
    if (mFreeCharacter) {
        mFreeCharacter->applySkeletonBodyNode(mSkelInfos, mFreeCharacter->getSkeleton());
    }

    std::vector<Eigen::Vector3d> ref_markers;
    for(auto m: mMarkerSet)
        ref_markers.push_back(m.getGlobalPos());

    // Pelvis size
    int idx = mFreeCharacter->getSkeleton()->getBodyNode("Pelvis")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[11] - init_marker[10]).norm() / (ref_markers[11] - ref_markers[10]).norm();

    // FemurR size
    idx = mFreeCharacter->getSkeleton()->getBodyNode("FemurR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[11] - init_marker[20]).norm() / (ref_markers[11] - ref_markers[20]).norm();

    // TibiaR size
    idx = mFreeCharacter->getSkeleton()->getBodyNode("TibiaR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[16] - init_marker[14]).norm() / (ref_markers[16] - ref_markers[14]).norm();

    // TalusR size
    idx = mFreeCharacter->getSkeleton()->getBodyNode("TalusR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[18] - init_marker[19]).norm() / (ref_markers[18] - ref_markers[19]).norm();

    // FemurL size
    idx = mFreeCharacter->getSkeleton()->getBodyNode("FemurL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[11] - init_marker[20]).norm() / (ref_markers[11] - ref_markers[20]).norm();

    // TibiaL size
    idx = mFreeCharacter->getSkeleton()->getBodyNode("TibiaL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[20] - init_marker[22]).norm() / (ref_markers[20] - ref_markers[22]).norm();

    // TalusL size
    idx = mFreeCharacter->getSkeleton()->getBodyNode("TalusL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[18] - init_marker[19]).norm() / (ref_markers[18] - ref_markers[19]).norm();

    // Upper Body
    idx = mFreeCharacter->getSkeleton()->getBodyNode("Spine")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mFreeCharacter->getSkeleton()->getBodyNode("Torso")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mFreeCharacter->getSkeleton()->getBodyNode("Neck")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mFreeCharacter->getSkeleton()->getBodyNode("Head")->getIndexInSkeleton();
    // std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mFreeCharacter->getSkeleton()->getBodyNode("ShoulderR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mFreeCharacter->getSkeleton()->getBodyNode("ShoulderL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mFreeCharacter->getSkeleton()->getBodyNode("ArmR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mFreeCharacter->getSkeleton()->getBodyNode("ArmL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mFreeCharacter->getSkeleton()->getBodyNode("ForeArmR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mFreeCharacter->getSkeleton()->getBodyNode("ForeArmL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mFreeCharacter->getSkeleton()->getBodyNode("HandR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[4] - init_marker[3]).norm() / (ref_markers[4] - ref_markers[3]).norm();

    idx = mFreeCharacter->getSkeleton()->getBodyNode("HandL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[4] - init_marker[3]).norm() / (ref_markers[4] - ref_markers[3]).norm();

    // Torsion
    idx = mFreeCharacter->getSkeleton()->getBodyNode("FemurR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[4] = torsionR;

    idx = mFreeCharacter->getSkeleton()->getBodyNode("FemurL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[4] = torsionL;

    femurR_torsion = torsionR;
    femurL_torsion = torsionL;

    // Apply final skeleton modifications
    if (mFreeCharacter) {
        mFreeCharacter->applySkeletonBodyNode(mSkelInfos, mFreeCharacter->getSkeleton());
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
    Eigen::VectorXd pos = mFreeCharacter->getSkeleton()->getPositions();
    pos.setZero();

    // Set initial arm positions (forearms at 90 degrees)
    pos[mFreeCharacter->getSkeleton()->getJoint("ForeArmR")->getIndexInSkeleton(0)] = M_PI * 0.5;
    pos[mFreeCharacter->getSkeleton()->getJoint("ForeArmL")->getIndexInSkeleton(0)] = M_PI * 0.5;

    // Initialize knee joints (1-DOF revolute joints)
    pos[mFreeCharacter->getSkeleton()->getJoint("TibiaR")->getIndexInSkeleton(0)] = 0.0;
    pos[mFreeCharacter->getSkeleton()->getJoint("TibiaL")->getIndexInSkeleton(0)] = 0.0;

    mFreeCharacter->getSkeleton()->setPositions(pos);

    // Fit skeleton to first frame markers (naive approach for ForeArm/Spine)
    fitSkeletonToMarker(firstFrameMarkers, params.femurTorsionL, params.femurTorsionR);
}

// ============================================================================
// SECTION 12: Fallback Scaling for USE_CERES=OFF
// ============================================================================

void C3D_Reader::scaleArmsFallback()
{
    const auto& allMarkers = mCurrentC3D->getAllMarkers();

    // Lookup C3D index by data_label
    auto getIdx = [&](const std::string& label) -> int {
        for (const auto& ref : mFittingConfig.markerMappings) {
            if (ref.dataLabel == label) return ref.dataIndex;
        }
        return -1;
    };

    // Lookup skeleton marker index by name
    auto getSkelIdx = [&](const std::string& name) -> int {
        for (size_t i = 0; i < mMarkerSet.size(); ++i) {
            if (mMarkerSet[i].name == name) return (int)i;
        }
        return -1;
    };

    struct ArmDef {
        const char* bone;
        const char* proxLabel;
        const char* distLabel;
        const char* proxSkel;
        const char* distSkel;
    };

    std::vector<ArmDef> arms = {
        {"ArmR",     "R.Shoulder", "R.Elbow", "RSHO_arm",  "RELB_arm"},
        {"ForeArmR", "R.Elbow",    "R.Wrist", "RELB_farm", "RWRI"},
        {"ArmL",     "L.Shoulder", "L.Elbow", "LSHO_arm",  "LELB_arm"},
        {"ForeArmL", "L.Elbow",    "L.Wrist", "LELB_farm", "LWRI"},
    };

    std::cout << "\n=== Stage 2b Fallback: Marker-distance scaling for arms ===\n" << std::endl;

    for (const auto& arm : arms) {
        int proxIdx = getIdx(arm.proxLabel);
        int distIdx = getIdx(arm.distLabel);
        int proxSkel = getSkelIdx(arm.proxSkel);
        int distSkel = getSkelIdx(arm.distSkel);

        if (proxIdx < 0 || distIdx < 0 || proxSkel < 0 || distSkel < 0) {
            LOG_WARN("[Fallback] Missing markers for " << arm.bone
                     << ": proxIdx=" << proxIdx << " distIdx=" << distIdx
                     << " proxSkel=" << proxSkel << " distSkel=" << distSkel);
            continue;
        }

        double refDist = (mMarkerSet[proxSkel].getGlobalPos() -
                          mMarkerSet[distSkel].getGlobalPos()).norm();
        if (refDist < 1e-6) {
            LOG_WARN("[Fallback] Zero reference distance for " << arm.bone);
            continue;
        }

        // Average across all fitting frames
        double sumScale = 0.0;
        int validFrames = 0;
        for (int f = mFitFrameStart; f <= mFitFrameEnd; ++f) {
            const auto& markers = allMarkers[f];
            double measDist = (markers[proxIdx] - markers[distIdx]).norm();
            if (measDist > 1e-6) {
                sumScale += measDist / refDist;
                validFrames++;
            }
        }

        if (validFrames > 0) {
            double scale = sumScale / validFrames;
            auto* bn = mFreeCharacter->getSkeleton()->getBodyNode(arm.bone);
            if (bn) {
                int idx = bn->getIndexInSkeleton();
                std::get<1>(mSkelInfos[idx]).value[3] = scale;
                LOG_INFO("[Fallback] " << arm.bone << " scale=" << scale);
            }
        }

        mBoneOrder.push_back(arm.bone);
    }

    // Pre-allocate storage for arm transforms (populated in computeArmRotations)
    int numFrames = mFitFrameEnd - mFitFrameStart + 1;
    std::vector<std::string> armBones = {"ArmR", "ForeArmR", "ArmL", "ForeArmL"};
    for (const auto& bone : armBones) {
        mBoneR_frames[bone].resize(numFrames, Eigen::Matrix3d::Identity());
        mBoneT_frames[bone].resize(numFrames, Eigen::Vector3d::Zero());
    }
}

void C3D_Reader::copyDependentScales()
{
    auto copyScale = [&](const char* src, const char* dst) {
        auto* srcBn = mFreeCharacter->getSkeleton()->getBodyNode(src);
        auto* dstBn = mFreeCharacter->getSkeleton()->getBodyNode(dst);
        if (!srcBn || !dstBn) return;

        int srcIdx = srcBn->getIndexInSkeleton();
        int dstIdx = dstBn->getIndexInSkeleton();
        auto& srcInfo = std::get<1>(mSkelInfos[srcIdx]);
        auto& dstInfo = std::get<1>(mSkelInfos[dstIdx]);

        for (int i = 0; i < 4; ++i) {
            dstInfo.value[i] = srcInfo.value[i];
        }
    };

    // Hand = ForeArm
    copyScale("ForeArmR", "HandR");
    copyScale("ForeArmL", "HandL");

    // Neck = Head (if Head in target_svd)
    auto isBoneOptimized = [&](const std::string& name) {
        return std::find(mBoneOrder.begin(), mBoneOrder.end(), name) != mBoneOrder.end();
    };
    if (isBoneOptimized("Head")) {
        copyScale("Head", "Neck");
        LOG_INFO("[CopyScale] Neck = Head");
    }

    if (isBoneOptimized("Torso")) {
        copyScale("Torso", "Spine");
        LOG_INFO("[CopyScale] Spine = Torso");
    }

    if (isBoneOptimized("TalusR")) {
        copyScale("TalusR", "FootPinkyR");
        copyScale("TalusR", "FootThumbR");
        LOG_INFO("[CopyScale] Spine = FootPinkyR, FootThumbR");
    }
    
    if (isBoneOptimized("TalusL")) {
        copyScale("TalusL", "FootPinkyL");
        copyScale("TalusL", "FootThumbL");
        LOG_INFO("[CopyScale] Spine = FootPinkyL, FootThumbL");
    }

    LOG_INFO("[CopyScale] HandR = ForeArmR, HandL = ForeArmL");
}

void C3D_Reader::interpolateDependent()
{
    // Interpolate Spine between Pelvis and Torso (insert before Torso)
    interpolateBoneTransforms("Spine", "Pelvis", "Torso");

    // Interpolate Neck between Torso and Head (insert before Head)
    interpolateBoneTransforms("Neck", "Torso", "Head");
}

void C3D_Reader::interpolateBoneTransforms(
    const std::string& newBone,
    const std::string& parentBone,
    const std::string& childBone)
{
    const double ratio = mFittingConfig.interpolateRatio;
    if (!mBoneR_frames.count(parentBone) || !mBoneR_frames.count(childBone)) return;

    const auto& R_parent = mBoneR_frames[parentBone];
    const auto& R_child = mBoneR_frames[childBone];
    const auto& t_parent = mBoneT_frames[parentBone];
    const auto& t_child = mBoneT_frames[childBone];

    int numFrames = std::min(R_parent.size(), R_child.size());
    std::vector<Eigen::Matrix3d> R_new(numFrames);
    std::vector<Eigen::Vector3d> t_new(numFrames);

    for (int i = 0; i < numFrames; ++i) {
        // Translation: linear interpolation
        t_new[i] = t_parent[i] * (1.0 - ratio) + t_child[i] * ratio;

        // Rotation: slerp
        Eigen::Quaterniond q_parent(R_parent[i]);
        Eigen::Quaterniond q_child(R_child[i]);
        Eigen::Quaterniond q_new = q_parent.slerp(ratio, q_child);
        R_new[i] = q_new.toRotationMatrix();
    }

    // Store in maps
    mBoneR_frames[newBone] = R_new;
    mBoneT_frames[newBone] = t_new;

    // Insert into mBoneOrder before childBone
    auto orderIt = std::find(mBoneOrder.begin(), mBoneOrder.end(), childBone);
    if (orderIt != mBoneOrder.end()) {
        mBoneOrder.insert(orderIt, newBone);
    } else {
        mBoneOrder.push_back(newBone);
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << ratio;
    LOG_INFO("[Interpolate] " << newBone << " = lerp(" << parentBone << ", " << childBone << ") ratio=" << oss.str());
}

// ============================================================================
// SECTION 11: Motion Skeleton Conversion (Free-Joint → Constrained Joint)
// ============================================================================

std::map<std::string, std::vector<Eigen::Isometry3d>> C3D_Reader::computeRelativeTransforms()
{
    std::map<std::string, std::vector<Eigen::Isometry3d>> relTransforms;

    if (mBoneOrder.empty() || mBoneR_frames.empty()) {
        LOG_WARN("[MotionConvert] No bone transforms available");
        return relTransforms;
    }

    auto skel = mFreeCharacter->getSkeleton();
    int numFrames = mBoneR_frames.begin()->second.size();

    for (const auto& boneName : mBoneOrder) {
        auto it_R = mBoneR_frames.find(boneName);
        auto it_t = mBoneT_frames.find(boneName);
        if (it_R == mBoneR_frames.end() || it_t == mBoneT_frames.end()) continue;

        auto* bn = skel->getBodyNode(boneName);
        if (!bn) continue;

        auto* parentBn = bn->getParentBodyNode();
        std::vector<Eigen::Isometry3d> relT(numFrames);

        for (int t = 0; t < numFrames; ++t) {
            // Build global transform for this bone
            Eigen::Isometry3d globalT = Eigen::Isometry3d::Identity();
            globalT.linear() = it_R->second[t];
            globalT.translation() = it_t->second[t];

            if (parentBn) {
                // Get parent global transform
                auto parentIt_R = mBoneR_frames.find(parentBn->getName());
                auto parentIt_t = mBoneT_frames.find(parentBn->getName());

                if (parentIt_R != mBoneR_frames.end() && parentIt_t != mBoneT_frames.end()) {
                    Eigen::Isometry3d parentGlobalT = Eigen::Isometry3d::Identity();
                    parentGlobalT.linear() = parentIt_R->second[t];
                    parentGlobalT.translation() = parentIt_t->second[t];

                    // Relative transform: pT_i = (W_Tp)^-1 * W_Ti
                    relT[t] = parentGlobalT.inverse() * globalT;
                } else {
                    // Parent not fitted, use global
                    relT[t] = globalT;
                }
            } else {
                // Root bone: relative = global
                relT[t] = globalT;
            }
        }

        relTransforms[boneName] = relT;
    }

    LOG_INFO("[MotionConvert] Computed relative transforms for " << relTransforms.size() << " bones, " << numFrames << " frames");
    return relTransforms;
}

JointOffsetResult C3D_Reader::estimateJointOffsets(
    const std::string& jointName,
    const std::vector<Eigen::Isometry3d>& relativeTransforms)
{
    JointOffsetResult result;

    if (relativeTransforms.empty()) {
        return result;
    }

    int numFrames = relativeTransforms.size();

    // Build linear system: A*x = b
    // For each frame t: [I -R(t)] * [c_p; c_i] = t(t)
    // A is (3*T x 6), b is (3*T x 1)
    Eigen::MatrixXd A(3 * numFrames, 6);
    Eigen::VectorXd b(3 * numFrames);

    for (int t = 0; t < numFrames; ++t) {
        A.block<3, 3>(3 * t, 0) = Eigen::Matrix3d::Identity();
        A.block<3, 3>(3 * t, 3) = -relativeTransforms[t].linear();
        b.segment<3>(3 * t) = relativeTransforms[t].translation();
    }

    // Solve via normal equations: x = (A^T A)^-1 A^T b
    Eigen::VectorXd x = (A.transpose() * A).ldlt().solve(A.transpose() * b);

    // Extract translation offsets
    result.parentOffset.setIdentity();
    result.childOffset.setIdentity();
    result.parentOffset.translation() = x.head<3>();  // c_p
    result.childOffset.translation() = x.tail<3>();   // c_i

    // Compute RMS residual
    Eigen::VectorXd residual = A * x - b;
    double rms = std::sqrt(residual.squaredNorm() / numFrames);

    result.valid = true;
    LOG_INFO("[JointOffset] " << jointName << ": parentOff=(" << x.head<3>().transpose()
             << "), childOff=(" << x.tail<3>().transpose() << "), RMS=" << std::fixed << std::setprecision(4) << rms);

    return result;
}

Eigen::Vector3d C3D_Reader::selectRevoluteAxis(
    const Eigen::Vector3d& xmlAxis,
    const std::vector<Eigen::Matrix3d>& /* rotations - unused, kept for API compatibility */)
{
    // Always use the XML axis - no PCA/blending
    //
    // Rationale:
    // 1. PCA axis depends on motion distribution and noise, causing instability
    // 2. For proper DOF allocation, off-axis motion should be absorbed by parent
    //    ball joint, not by changing the revolute axis
    //
    // The swing-twist decomposition in buildMotionFramePose handles off-axis
    // residuals by transferring them to the parent ball joint.

    if (xmlAxis.norm() < 0.5) {
        LOG_WARN("[RevoluteAxis] Invalid XML axis, defaulting to UnitX");
        return Eigen::Vector3d::UnitX();
    }

    LOG_INFO("[RevoluteAxis] Using XML axis: (" << xmlAxis.transpose() << ")");
    return xmlAxis.normalized();
}

Eigen::VectorXd C3D_Reader::buildMotionFramePose(
    int frameIdx,
    const std::map<std::string, JointOffsetResult>& offsets,
    const std::map<std::string, std::vector<Eigen::Isometry3d>>& relTransforms)
{
    if (!mMotionCharacter) {
        LOG_WARN("[MotionPose] No motion character set");
        return Eigen::VectorXd();
    }

    auto skel = mMotionCharacter->getSkeleton();
    Eigen::VectorXd pose = skel->getPositions();

    // Store swing residuals from revolute joints to transfer to parent ball joints
    // Key: parent joint name, Value: swing rotation to be absorbed
    std::map<std::string, Eigen::Matrix3d> swingResiduals;

    // First pass: Process all joints, collect swing residuals from revolute joints
    for (const auto& boneName : mFittingConfig.targetMotionJoint) {
        // Skip if not in mBoneOrder (was not fitted)
        if (std::find(mBoneOrder.begin(), mBoneOrder.end(), boneName) == mBoneOrder.end()) continue;

        auto* bn = skel->getBodyNode(boneName);
        if (!bn) continue;

        auto* joint = bn->getParentJoint();
        if (!joint) continue;

        // Get relative transform for this frame
        auto relIt = relTransforms.find(boneName);
        if (relIt == relTransforms.end() || frameIdx >= (int)relIt->second.size()) continue;

        const Eigen::Isometry3d& relT = relIt->second[frameIdx];

        // Get joint offset (if estimated)
        Eigen::Isometry3d parentOffset = Eigen::Isometry3d::Identity();
        Eigen::Isometry3d childOffset = Eigen::Isometry3d::Identity();
        Eigen::Vector3d revoluteAxis = Eigen::Vector3d::UnitX();

        auto offIt = offsets.find(joint->getName());
        if (offIt != offsets.end() && offIt->second.valid) {
            parentOffset = offIt->second.parentOffset;
            childOffset = offIt->second.childOffset;
            revoluteAxis = offIt->second.revoluteAxis;
        }

        // Compute observed joint transform: T_obs = parentOffset^-1 * relT * childOffset
        Eigen::Isometry3d T_obs = parentOffset.inverse() * relT * childOffset;

        // Project to joint angles based on joint type
        int jn_idx = joint->getIndexInSkeleton(0);
        int numDofs = joint->getNumDofs();

        if (joint->getType() == "FreeJoint") {
            if (numDofs == 6) {
                pose.segment<6>(jn_idx) = dart::dynamics::FreeJoint::convertToPositions(T_obs);
            }
        } else if (joint->getType() == "BallJoint") {
            if (numDofs == 3) {
                pose.segment<3>(jn_idx) = dart::dynamics::BallJoint::convertToPositions(T_obs.linear());
            }
        } else if (joint->getType() == "RevoluteJoint") {
            if (numDofs == 1) {
                std::string jointName = joint->getName();
                // For elbow joints, use cached cross-product angle directly
                // This ensures elbow axis coincides with bending normal n = u × l
                if ((jointName == "ForeArmR" || jointName == "ForeArmL") &&
                    mElbowAngle_frames.count(jointName) > 0 &&
                    frameIdx < (int)mElbowAngle_frames[jointName].size()) {
                    pose(jn_idx) = mElbowAngle_frames[jointName][frameIdx];
                } else {
                    // Swing-twist decomposition for other revolute joints (knees, etc.)
                    //
                    // Step 1: Compute twist (rotation about XML axis)
                    // Project log-rotation onto revoluteAxis: θ = ω · â
                    Eigen::Matrix3d R_rel = T_obs.linear();
                    Eigen::AngleAxisd aa(R_rel);
                    Eigen::Vector3d omega = aa.angle() * aa.axis();
                    double theta = omega.dot(revoluteAxis);

                    // R_twist = exp(θ * [â]×)
                    Eigen::Matrix3d R_twist = Eigen::AngleAxisd(theta, revoluteAxis).toRotationMatrix();

                    // Step 2: Compute swing (off-axis residual)
                    // R_swing = R_rel * R_twist^T
                    Eigen::Matrix3d R_swing = R_rel * R_twist.transpose();

                    // Store swing for parent ball joint compensation
                    auto* parentBn = bn->getParentBodyNode();
                    if (parentBn) {
                        auto* parentJoint = parentBn->getParentJoint();
                        if (parentJoint && parentJoint->getType() == "BallJoint") {
                            std::string parentJointName = parentJoint->getName();
                            // Accumulate swing (compose if multiple children)
                            if (swingResiduals.count(parentJointName) == 0) {
                                swingResiduals[parentJointName] = R_swing;
                            } else {
                                swingResiduals[parentJointName] = swingResiduals[parentJointName] * R_swing;
                            }
                        }
                    }

                    // Set revolute joint angle to twist only
                    pose(jn_idx) = theta;
                }
            }
        }
        // Other joint types: leave unchanged
    }

    // Second pass: Apply swing residuals to parent ball joints
    for (const auto& [parentJointName, R_swing] : swingResiduals) {
        auto* parentJoint = skel->getJoint(parentJointName);
        if (!parentJoint || parentJoint->getType() != "BallJoint") continue;

        int jn_idx = parentJoint->getIndexInSkeleton(0);
        int numDofs = parentJoint->getNumDofs();

        if (numDofs == 3) {
            // Get current ball joint rotation
            Eigen::Matrix3d R_current = dart::dynamics::BallJoint::convertToRotation(pose.segment<3>(jn_idx));

            // Apply swing: R'_parent = R_parent * R_swing
            Eigen::Matrix3d R_updated = R_current * R_swing;

            // Convert back to ball joint positions
            pose.segment<3>(jn_idx) = dart::dynamics::BallJoint::convertToPositions(R_updated);
        }
    }

    return pose;
}

void C3D_Reader::applyJointOffsetsToSkeleton(
    dart::dynamics::SkeletonPtr skel,
    const std::map<std::string, JointOffsetResult>& offsets)
{
    if (!skel) return;

    for (const auto& [jointName, offset] : offsets) {
        if (!offset.valid) continue;

        auto* joint = skel->getJoint(jointName);
        if (!joint) continue;

        // Apply parent offset (transform from parent body to joint)
        joint->setTransformFromParentBodyNode(offset.parentOffset);

        // Apply child offset (transform from joint to child body)
        joint->setTransformFromChildBodyNode(offset.childOffset);

        LOG_INFO("[ApplyOffset] " << jointName << ": applied offsets");
    }
}

MotionConversionResult C3D_Reader::convertToMotionSkeleton()
{
    MotionConversionResult result;

    if (!mMotionCharacter) {
        LOG_WARN("[MotionConvert] No motion character set, skipping conversion");
        return result;
    }

    if (mBoneOrder.empty() || mBoneR_frames.empty()) {
        LOG_WARN("[MotionConvert] No fitted bone transforms available");
        return result;
    }

    LOG_INFO("[MotionConvert] Starting conversion to motion skeleton...");

    // Step 1: Compute relative transforms
    auto relTransforms = computeRelativeTransforms();

    auto skel = mMotionCharacter->getSkeleton();
    int numFrames = mBoneR_frames.begin()->second.size();

    // Step 2: Estimate joint offsets for each target_motion_joint
    for (const auto& boneName : mFittingConfig.targetMotionJoint) {
        // Check if bone is in mBoneOrder (was fitted)
        if (std::find(mBoneOrder.begin(), mBoneOrder.end(), boneName) == mBoneOrder.end()) {
            LOG_WARN("[MotionConvert] target_motion_joint '" << boneName << "' not found in mBoneOrder, skipping");
            continue;
        }

        auto* bn = skel->getBodyNode(boneName);
        if (!bn) {
            LOG_WARN("[MotionConvert] Body node not found: " << boneName);
            continue;
        }

        auto* joint = bn->getParentJoint();
        if (!joint) {
            LOG_WARN("[MotionConvert] Joint not found: " << boneName);
            continue;
        }

        std::string jointName = joint->getName();

        auto relIt = relTransforms.find(boneName);
        if (relIt == relTransforms.end()) {
            LOG_WARN("[MotionConvert] Relative transforms not found: " << boneName);
            continue;
        }

        // Estimate translation offsets via least squares
        JointOffsetResult offset = estimateJointOffsets(jointName, relIt->second);

        // Initialize rotation offsets from skeleton definition
        offset.parentOffset.linear() = joint->getTransformFromParentBodyNode().linear();
        offset.childOffset.linear() = joint->getTransformFromChildBodyNode().linear();

        // For revolute joints, estimate axis using XML prior + PCA
        if (joint->getType() == "RevoluteJoint") {
            offset.parentOffset.translation() = joint->getTransformFromParentBodyNode().translation();
            offset.childOffset.translation() = joint->getTransformFromChildBodyNode().translation();
        }
        
        result.jointOffsets[jointName] = offset;
    }

    // Step 3: Apply offsets permanently to motion skeleton
    applyJointOffsetsToSkeleton(skel, result.jointOffsets);

    // Step 4: Build per-frame poses
    result.motionPoses.reserve(numFrames);
    for (int t = 0; t < numFrames; ++t) {
        Eigen::VectorXd pose = buildMotionFramePose(t, result.jointOffsets, relTransforms);
        result.motionPoses.push_back(pose);
    }

    // Step 5: Refine parent body node position for knee joints (TibiaL, TibiaR) using least-squares
    // to match child body node position between freeChar and motionChar
    //
    // Chain: ParentJoint (FemurL/R) → ParentBN (FemurL/R) → KneeJoint (TibiaL/R) → ChildBN (TibiaL/R)
    // We adjust ParentJoint's childOffset to move ParentBN such that the child body node aligns
    auto freeSkel = mFreeCharacter->getSkeleton();
    std::vector<std::string> kneeJoints = {"TibiaL", "TibiaR"};

    for (const auto& jointName : kneeJoints) {
        auto* joint = skel->getJoint(jointName);
        if (!joint || joint->getType() != "RevoluteJoint") continue;

        auto* childBn = joint->getChildBodyNode();
        if (!childBn) continue;

        auto* parentBn = childBn->getParentBodyNode();
        if (!parentBn) continue;

        // Get the PARENT joint (the joint that connects to parentBn)
        auto* parentJoint = parentBn->getParentJoint();
        if (!parentJoint) continue;

        // Get corresponding structures in freeChar
        auto* freeJoint = freeSkel->getJoint(jointName);
        if (!freeJoint) continue;
        auto* freeChildBn = freeJoint->getChildBodyNode();
        if (!freeChildBn) continue;

        // Build least-squares system: R(t) * Δt = d(t)
        // where d(t) = childBn_pos_free(t) - childBn_pos_motion(t)
        // and Δt adjusts parentJoint's childOffset (in parentBn's local frame)
        Eigen::MatrixXd A(3 * numFrames, 3);
        Eigen::VectorXd b_vec(3 * numFrames);

        for (int t = 0; t < numFrames; ++t) {
            // Apply poses to both skeletons
            skel->setPositions(result.motionPoses[t]);
            Eigen::VectorXd freePose = buildFramePose(t);
            freeSkel->setPositions(freePose);

            // Get child body node world position (origin of TibiaL/R body node)
            Eigen::Vector3d childBnPosMotion = childBn->getWorldTransform().translation();
            Eigen::Vector3d childBnPosFree = freeChildBn->getWorldTransform().translation();

            // The adjustment Δt is in parentBn's local frame
            // Moving parentBn by Δt moves childBn by R_parentBn * Δt in world
            Eigen::Matrix3d R = parentBn->getWorldTransform().linear();

            A.block<3, 3>(3 * t, 0) = R;
            b_vec.segment<3>(3 * t) = childBnPosFree - childBnPosMotion;
        }

        // Solve: Δt = (A^T A)^{-1} A^T b
        Eigen::Vector3d delta = (A.transpose() * A).ldlt().solve(A.transpose() * b_vec);

        // Compute RMS before and after
        Eigen::VectorXd residual = A * delta - b_vec;
        double rmsBefore = std::sqrt(b_vec.squaredNorm() / numFrames);
        double rmsAfter = std::sqrt(residual.squaredNorm() / numFrames);

        // Update parentJoint's childOffset (the offset from parentJoint to parentBn)
        std::string parentJointName = parentJoint->getName();
        auto parentIt = result.jointOffsets.find(parentJointName);
        if (parentIt != result.jointOffsets.end()) {
            parentIt->second.childOffset.translation() -= delta;
            LOG_INFO("[KneeRefine] " << jointName << " -> adjust " << parentJointName
                     << ".childOffset: Δt=(" << delta.transpose()
                     << "), RMS: " << std::fixed << std::setprecision(4) << rmsBefore << " -> " << rmsAfter);
        } else {
            LOG_WARN("[KneeRefine] " << jointName << ": parent joint " << parentJointName << " not in offsets map");
        }
    }

    applyJointOffsetsToSkeleton(skel, result.jointOffsets);

    result.valid = true;
    LOG_INFO("[MotionConvert] Conversion complete: " << result.jointOffsets.size()
             << " joints, " << result.motionPoses.size() << " frames");

    return result;
}

