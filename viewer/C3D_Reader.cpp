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

C3D_Reader::C3D_Reader(std::string marker_path, RenderCharacter *character)
{
    mCharacter = character;
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

    for (auto bn : mCharacter->getSkeleton()->getBodyNodes())
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
        m.bn = mCharacter->getSkeleton()->getBodyNode(bn);
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
            config.upperBody = sf["optimization"]["upper_body"].as<bool>(false);

            // SVD-based optimizer targets (3+ markers)
            if (sf["optimization"]["target_svd"]) {
                for (const auto& t : sf["optimization"]["target_svd"]) {
                    config.targetSvd.push_back(t.as<std::string>());
                }
            }

            config.lambdaRot = sf["optimization"]["lambda_rot"].as<double>(10.0);

            // Ceres-based optimizer targets (2+ markers with regularization)
            if (sf["optimization"]["target_ceres"]) {
                for (const auto& t : sf["optimization"]["target_ceres"]) {
                    config.targetCeres.push_back(t.as<std::string>());
                }
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
        LOG_INFO("  - upperBody: " << (config.upperBody ? "true" : "false"));
        LOG_INFO("  - markerMappings: " << config.markerMappings.size() << " markers");
        LOG_INFO("  - targetSvd: " << config.targetSvd.size() << " bones");
        LOG_INFO("  - lambdaRot: " << config.lambdaRot);
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
    mFittingConfig = loadSkeletonFittingConfig("data/config/skeleton_fitting.yaml");

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
    Eigen::VectorXd pos = mCharacter->getSkeleton()->getPositions();
    pos.setZero();
    mCharacter->getSkeleton()->setPositions(pos);

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
    if (mCharacter) {
        mCharacter->applySkeletonBodyNode(mSkelInfos, mCharacter->getSkeleton());
    }

    LOG_INFO("[C3D_Reader] Starting multi-stage skeleton fitting...");

    // Clear previous fitting results
    mBoneR_frames.clear();
    mBoneT_frames.clear();

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
    }

#ifdef USE_CERES
    // ============ STAGE 2b: Ceres-based bones (2+ markers with regularization) ============
    runCeresBoneFitting(mFittingConfig, boneToMarkers, mCurrentC3D->getAllMarkers(),
                        mCharacter, mSkelInfos, mBoneR_frames, mBoneT_frames);
#endif

    // ============ STAGE 2c: Upper Body Scaling (Torso/Spine/Neck/Shoulder only) ============
    // Arms/ForeArms are now handled by Ceres optimizer, only scale trunk/shoulder here
    if (mFittingConfig.upperBody) {
        // Get reference markers from marker set (T-pose)
        std::vector<Eigen::Vector3d> ref_markers;
        for (const auto& m : mMarkerSet) {
            ref_markers.push_back(m.getGlobalPos());
        }

        // Get initial frame markers
        const auto& init = mCurrentC3D->getMarkers(mFitFrameStart);

        // Lookup marker indices (C3D data)
        int RSHO = mFittingConfig.getDataIndexForMarker("RSHO");
        int LSHO = mFittingConfig.getDataIndexForMarker("LSHO");

        // Find skeleton marker indices for reference
        int rsho_ref = -1, lsho_ref = -1;
        for (size_t i = 0; i < mMarkerSet.size(); ++i) {
            if (mMarkerSet[i].name == "RSHO") rsho_ref = i;
            else if (mMarkerSet[i].name == "LSHO") lsho_ref = i;
        }

        double torso = 1.0;

        // Torso/Spine/Neck/Shoulder: use shoulder width ratio as proxy
        if (RSHO >= 0 && LSHO >= 0 && rsho_ref >= 0 && lsho_ref >= 0) {
            double init_dist = (init[LSHO] - init[RSHO]).norm();
            double ref_dist = (ref_markers[lsho_ref] - ref_markers[rsho_ref]).norm();
            torso = init_dist / ref_dist;

            for (const char* bone : {"Spine", "Torso", "Neck", "ShoulderR", "ShoulderL"}) {
                auto* bn = mCharacter->getSkeleton()->getBodyNode(bone);
                if (bn) {
                    int idx = bn->getIndexInSkeleton();
                    std::get<1>(mSkelInfos[idx]).value[3] = torso;
                }
            }
        }

        LOG_INFO("[Stage 2c] Upper body (trunk) scale: Torso=" << torso);
    }

    // ============ STAGE 3: Apply All Scales ============
    // LOG_INFO("[Stage 3] Applying scales to skeleton...");

    // Apply femur torsions from params
    BodyNode* femurRBn = mCharacter->getSkeleton()->getBodyNode("FemurR");
    BodyNode* femurLBn = mCharacter->getSkeleton()->getBodyNode("FemurL");
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
        mCharacter->applySkeletonBodyNode(mSkelInfos, mCharacter->getSkeleton());
    }

    // Print summary
    LOG_INFO("[C3D_Reader] Multi-stage skeleton fitting complete. Final scales:");
    // Combine SVD and Ceres targets for summary
    std::vector<std::string> allTargets;
    allTargets.insert(allTargets.end(), mFittingConfig.targetSvd.begin(), mFittingConfig.targetSvd.end());
    allTargets.insert(allTargets.end(), mFittingConfig.targetCeres.begin(), mFittingConfig.targetCeres.end());
    for (const auto& boneName : allTargets) {
        BodyNode* bn = mCharacter->getSkeleton()->getBodyNode(boneName);
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
    BodyNode* bn = mCharacter->getSkeleton()->getBodyNode(boneName);
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

// Use optimizer's global R, t for all fitted bones
// With SKEL_FREE_JOINTS, each bone is independent (6 DOF)
Eigen::VectorXd C3D_Reader::buildFramePose(int fitFrameIdx)
{
    // Initialize with zero pose
    Eigen::VectorXd pos = mCharacter->getSkeleton()->getPositions();
    pos.setZero();
    mCharacter->getSkeleton()->setPositions(pos);

    // Process SVD-fitted bones (pelvis first, then children)
    // This ensures parent transforms are set before computing child joint positions
    // Note: Ceres-fitted bones (arms) are handled in updateUpperBodyFromMarkers
    for (const auto& boneName : mFittingConfig.targetSvd) {

        // Check if we have transforms for this bone
        auto it = mBoneR_frames.find(boneName);
        if (it == mBoneR_frames.end()) continue;

        const auto& R_frames = it->second;
        if (fitFrameIdx >= (int)R_frames.size()) continue;

        auto* bn = mCharacter->getSkeleton()->getBodyNode(boneName);
        if (!bn) {
            LOG_WARN("[Fitting] Bone not found: " << boneName);
            continue;
        }

        auto* joint = bn->getParentJoint();
        if (!joint) {
            LOG_WARN("[Fitting] Joint not found: " << boneName);
            continue;
        }

        // Get stored global transform (bodynode world position/orientation from fitting)
        Eigen::Isometry3d bodynodeGlobalT = Eigen::Isometry3d::Identity();
        bodynodeGlobalT.linear() = R_frames[fitFrameIdx];
        bodynodeGlobalT.translation() = mBoneT_frames.at(boneName)[fitFrameIdx];

        Eigen::Isometry3d jointT;

        // Formula:
        // joint_global = parent_bn_global * parent_to_joint * joint_angle
        // child_global * child_to_joint = joint_global
        //
        // Therefore:
        // child_global * child_to_joint = parent_bn_global * parent_to_joint * joint_angle
        // joint_angle = parent_to_joint.inverse() * parent_bn_global.inverse() * child_global * child_to_joint

        Eigen::Isometry3d parentToJoint = joint->getTransformFromParentBodyNode();
        Eigen::Isometry3d childToJoint = joint->getTransformFromChildBodyNode();

        // Get parent bodynode global transform
        Eigen::Isometry3d parentBnGlobal = Eigen::Isometry3d::Identity();
        auto* parentBn = bn->getParentBodyNode();
        if (parentBn) {
            parentBnGlobal = parentBn->getTransform();
        }

        // Compute joint angle (local rotation/translation)
        jointT = parentToJoint.inverse() * parentBnGlobal.inverse() * bodynodeGlobalT * childToJoint;

        // Convert joint transform to FreeJoint positions and update skeleton
        int jn_idx = joint->getIndexInSkeleton(0);
        int jn_dof = joint->getNumDofs();
        if (jn_idx >= 0 && jn_idx + jn_dof <= pos.size()) {
            Eigen::VectorXd jointPos = FreeJoint::convertToPositions(jointT);
            pos.segment(jn_idx, jn_dof) = jointPos;

            // Update skeleton so next bone can get correct joint global position
            mCharacter->getSkeleton()->setPositions(pos);
        }
    }

    // ========== Upper body rotation from markers ==========
    if (mFittingConfig.upperBody && mCurrentC3D) {
        // Get current frame markers
        const auto& markers = mCurrentC3D->getMarkers(mFitFrameStart + fitFrameIdx);

        // Get reference markers from marker set (T-pose)
        std::vector<Eigen::Vector3d> refMarkers;
        for (const auto& m : mMarkerSet) {
            refMarkers.push_back(m.getGlobalPos());
        }

        // Update upper body joint positions from markers
        updateUpperBodyFromMarkers(fitFrameIdx, pos, markers, refMarkers);
        mCharacter->getSkeleton()->setPositions(pos);
    }

    return pos;
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
    auto skel = mCharacter->getSkeleton();

    // Marker indices from config
    int RSHO = mFittingConfig.getDataIndexForMarker("RSHO_shoul");
    int ROFF = mFittingConfig.getDataIndexForMarker("ROFF");
    int RELB = mFittingConfig.getDataIndexForMarker("RELB_arm");
    int RWRI = mFittingConfig.getDataIndexForMarker("RWRI");
    int LSHO = mFittingConfig.getDataIndexForMarker("LSHO_shoul");
    int LELB = mFittingConfig.getDataIndexForMarker("LELB_arm");
    int LWRI = mFittingConfig.getDataIndexForMarker("LWRI");

    if (RSHO < 0 || LSHO < 0) {
        LOG_WARN("[UpperBody] Missing shoulder markers, skipping upper body");
        return;
    }

    // ========== Torso/Spine rotation ==========
    // Compute rotation from shoulder triangle (RSHO, ROFF, LSHO)
    if (ROFF >= 0) {
        Eigen::Matrix3d origin = getRotationMatrixFromPoints(refMarkers[RSHO], refMarkers[ROFF], refMarkers[LSHO]);
        Eigen::Matrix3d current = getRotationMatrixFromPoints(markers[RSHO], markers[ROFF], markers[LSHO]);
        Eigen::Matrix3d T = current * origin.transpose();

        // Apply half rotation using slerp (damping)
        Eigen::Quaterniond q(T);
        Eigen::Quaterniond q_half = Eigen::Quaterniond::Identity().slerp(0.5, q);
        T = q_half.toRotationMatrix();

        // Apply to Torso and Spine (split rotation)
        for (const char* boneName : {"Torso", "Spine"}) {
            auto* bn = skel->getBodyNode(boneName);
            if (!bn) continue;

            auto* joint = bn->getParentJoint();
            if (!joint) continue;

            // Compute joint-local rotation
            Eigen::Isometry3d parentToJoint = joint->getTransformFromParentBodyNode();
            auto* parentBn = bn->getParentBodyNode();
            Eigen::Isometry3d parentBnGlobal = parentBn ? parentBn->getTransform() : Eigen::Isometry3d::Identity();

            // Get parent-local rotation
            Eigen::Matrix3d jointR = parentToJoint.linear().transpose() * parentBnGlobal.linear().transpose() * T;

            // Convert to joint positions
            int jn_idx = joint->getIndexInSkeleton(0);
            int jn_dof = joint->getNumDofs();

            if (jn_dof == 3) {
                // BallJoint
                Eigen::Vector3d jointPos = BallJoint::convertToPositions(jointR);
                pos.segment(jn_idx, 3) = jointPos;
            }
        }

        // Apply to Neck (same as Torso rotation)
        auto* neckBn = skel->getBodyNode("Neck");
        if (neckBn) {
            auto* joint = neckBn->getParentJoint();
            if (joint) {
                Eigen::Isometry3d parentToJoint = joint->getTransformFromParentBodyNode();
                auto* parentBn = neckBn->getParentBodyNode();
                Eigen::Isometry3d parentBnGlobal = parentBn ? parentBn->getTransform() : Eigen::Isometry3d::Identity();

                Eigen::Matrix3d jointR = parentToJoint.linear().transpose() * parentBnGlobal.linear().transpose() * T;

                int jn_idx = joint->getIndexInSkeleton(0);
                int jn_dof = joint->getNumDofs();
                if (jn_dof == 3) {
                    pos.segment(jn_idx, 3) = BallJoint::convertToPositions(jointR);
                }
            }
        }
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

    Eigen::VectorXd pos = mCharacter->getSkeleton()->getPositions();
    pos.setZero();
    mCharacter->getSkeleton()->setPositions(pos);

    // Pelvis
    jn_idx = mCharacter->getSkeleton()->getJoint("Pelvis")->getIndexInSkeleton(0);
    jn_dof = mCharacter->getSkeleton()->getJoint("Pelvis")->getNumDofs();

    Eigen::Matrix3d origin_pelvis = getRotationMatrixFromPoints(mRefMarkers[10], mRefMarkers[11], mRefMarkers[12]);
    Eigen::Matrix3d current_pelvis = getRotationMatrixFromPoints(_pos[10], _pos[11], _pos[12]);
    Eigen::Isometry3d current_pelvis_T = Eigen::Isometry3d::Identity();
    current_pelvis_T.linear() = current_pelvis * origin_pelvis.transpose();
    current_pelvis_T.translation() = (_pos[10] + _pos[11] + _pos[12]) / 3.0 - (mRefMarkers[10] + mRefMarkers[11] + mRefMarkers[12]) / 3.0;

    pos.segment(jn_idx, jn_dof) = FreeJoint::convertToPositions(current_pelvis_T);
    mCharacter->getSkeleton()->getJoint("Pelvis")->setPositions(FreeJoint::convertToPositions(current_pelvis_T));

    // Right Leg - FemurR
    jn_idx = mCharacter->getSkeleton()->getJoint("FemurR")->getIndexInSkeleton(0);
    jn_dof = mCharacter->getSkeleton()->getJoint("FemurR")->getNumDofs();
    Eigen::Matrix3d origin_femurR = getRotationMatrixFromPoints(mMarkerSet[25].getGlobalPos(), mMarkerSet[13].getGlobalPos(), mMarkerSet[14].getGlobalPos());
    Eigen::Matrix3d current_femurR = getRotationMatrixFromPoints(mMarkerSet[25].getGlobalPos(), _pos[13], _pos[14]);
    Eigen::Isometry3d pT = mCharacter->getSkeleton()->getJoint("FemurR")->getParentBodyNode()->getTransform() * mCharacter->getSkeleton()->getJoint("FemurR")->getTransformFromParentBodyNode();
    T = current_femurR * (origin_femurR.transpose());
    mCharacter->getSkeleton()->getJoint("FemurR")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    // TibiaR
    jn_idx = mCharacter->getSkeleton()->getJoint("TibiaR")->getIndexInSkeleton(0);
    jn_dof = mCharacter->getSkeleton()->getJoint("TibiaR")->getNumDofs();
    Eigen::Matrix3d origin_kneeR = getRotationMatrixFromPoints(mMarkerSet[14].getGlobalPos(), mMarkerSet[15].getGlobalPos(), mMarkerSet[16].getGlobalPos());
    Eigen::Matrix3d current_kneeR = getRotationMatrixFromPoints(_pos[14], _pos[15], _pos[16]);
    T = (current_kneeR * origin_kneeR.transpose());
    pT = mCharacter->getSkeleton()->getJoint("TibiaR")->getParentBodyNode()->getTransform() * mCharacter->getSkeleton()->getJoint("TibiaR")->getTransformFromParentBodyNode();
    Eigen::VectorXd kneeR_angles = pT.linear().transpose() * BallJoint::convertToPositions(T);
    mCharacter->getSkeleton()->getJoint("TibiaR")->setPosition(0, kneeR_angles[0]);

    // TalusR
    jn_idx = mCharacter->getSkeleton()->getJoint("TalusR")->getIndexInSkeleton(0);
    jn_dof = mCharacter->getSkeleton()->getJoint("TalusR")->getNumDofs();
    Eigen::Matrix3d origin_talusR = getRotationMatrixFromPoints(mMarkerSet[16].getGlobalPos(), mMarkerSet[17].getGlobalPos(), mMarkerSet[18].getGlobalPos());
    Eigen::Matrix3d current_talusR = getRotationMatrixFromPoints(_pos[16], _pos[17], _pos[18]);
    T = (current_talusR * origin_talusR.transpose());
    pT = mCharacter->getSkeleton()->getJoint("TalusR")->getParentBodyNode()->getTransform() * mCharacter->getSkeleton()->getJoint("TalusR")->getTransformFromParentBodyNode();
    mCharacter->getSkeleton()->getJoint("TalusR")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    // Left Leg - FemurL
    jn_idx = mCharacter->getSkeleton()->getJoint("FemurL")->getIndexInSkeleton(0);
    jn_dof = mCharacter->getSkeleton()->getJoint("FemurL")->getNumDofs();
    Eigen::Matrix3d origin_femurL = getRotationMatrixFromPoints(mMarkerSet[26].getGlobalPos(), mMarkerSet[19].getGlobalPos(), mMarkerSet[20].getGlobalPos());
    Eigen::Matrix3d current_femurL = getRotationMatrixFromPoints(mMarkerSet[26].getGlobalPos(), _pos[19], _pos[20]);
    T = current_femurL * origin_femurL.transpose();
    pT = mCharacter->getSkeleton()->getJoint("FemurL")->getParentBodyNode()->getTransform() * mCharacter->getSkeleton()->getJoint("FemurL")->getTransformFromParentBodyNode();
    mCharacter->getSkeleton()->getJoint("FemurL")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    // TibiaL
    jn_idx = mCharacter->getSkeleton()->getJoint("TibiaL")->getIndexInSkeleton(0);
    jn_dof = mCharacter->getSkeleton()->getJoint("TibiaL")->getNumDofs();
    Eigen::Matrix3d origin_kneeL = getRotationMatrixFromPoints(mMarkerSet[20].getGlobalPos(), mMarkerSet[21].getGlobalPos(), mMarkerSet[22].getGlobalPos());
    Eigen::Matrix3d current_kneeL = getRotationMatrixFromPoints(_pos[20], _pos[21], _pos[22]);
    T = current_kneeL * origin_kneeL.transpose();
    pT = mCharacter->getSkeleton()->getJoint("TibiaL")->getParentBodyNode()->getTransform() * mCharacter->getSkeleton()->getJoint("TibiaL")->getTransformFromParentBodyNode();
    Eigen::VectorXd kneeL_angles = pT.linear().transpose() * BallJoint::convertToPositions(T);
    mCharacter->getSkeleton()->getJoint("TibiaL")->setPosition(0, kneeL_angles[0]);

    // TalusL
    jn_idx = mCharacter->getSkeleton()->getJoint("TalusL")->getIndexInSkeleton(0);
    jn_dof = mCharacter->getSkeleton()->getJoint("TalusL")->getNumDofs();
    Eigen::Matrix3d origin_talusL = getRotationMatrixFromPoints(mMarkerSet[22].getGlobalPos(), mMarkerSet[23].getGlobalPos(), mMarkerSet[24].getGlobalPos());
    Eigen::Matrix3d current_talusL = getRotationMatrixFromPoints(_pos[22], _pos[23], _pos[24]);
    T = current_talusL * origin_talusL.transpose();
    pT = mCharacter->getSkeleton()->getJoint("TalusL")->getParentBodyNode()->getTransform() * mCharacter->getSkeleton()->getJoint("TalusL")->getTransformFromParentBodyNode();
    mCharacter->getSkeleton()->getJoint("TalusL")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    // Spine and Torso
    Eigen::Matrix3d origin_torso = getRotationMatrixFromPoints(mMarkerSet[3].getGlobalPos(), mMarkerSet[4].getGlobalPos(), mMarkerSet[7].getGlobalPos());
    Eigen::Matrix3d current_torso = getRotationMatrixFromPoints(_pos[3], _pos[4], _pos[7]);
    T = current_torso * origin_torso.transpose();
    pT = mCharacter->getSkeleton()->getJoint("Torso")->getParentBodyNode()->getTransform() * mCharacter->getSkeleton()->getJoint("Torso")->getTransformFromParentBodyNode();
    Eigen::Quaterniond tmp_T = Eigen::Quaterniond(T).slerp(0.5, Eigen::Quaterniond::Identity());
    T = tmp_T.toRotationMatrix();
    mCharacter->getSkeleton()->getJoint("Spine")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));
    mCharacter->getSkeleton()->getJoint("Torso")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    // Arms - ArmR
    Eigen::Vector3d v1 = _pos[3] - _pos[5];
    Eigen::Vector3d v2 = _pos[6] - _pos[5];
    double angle = abs(atan2(v1.cross(v2).norm(), v1.dot(v2)));
    if (angle > M_PI * 0.5) angle = M_PI - angle;
    jn_idx = mCharacter->getSkeleton()->getJoint("ForeArmR")->getIndexInSkeleton(0);
    pos[jn_idx] = angle;
    mCharacter->getSkeleton()->getJoint("ForeArmR")->setPosition(0, angle);
    Eigen::Matrix3d origin_armR = getRotationMatrixFromPoints(mMarkerSet[3].getGlobalPos(), mMarkerSet[5].getGlobalPos(), mMarkerSet[6].getGlobalPos());
    Eigen::Matrix3d current_armR = getRotationMatrixFromPoints(mMarkerSet[3].getGlobalPos(), _pos[5], _pos[6]);
    T = current_armR * origin_armR.transpose();
    pT = mCharacter->getSkeleton()->getJoint("ArmR")->getParentBodyNode()->getTransform() * mCharacter->getSkeleton()->getJoint("ArmR")->getTransformFromParentBodyNode();
    mCharacter->getSkeleton()->getJoint("ArmR")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    // ArmL
    v1 = _pos[8] - _pos[7];
    v2 = _pos[8] - _pos[9];
    angle = abs(atan2(v1.cross(v2).norm(), v1.dot(v2)));
    if (angle > M_PI * 0.5) angle = M_PI - angle;
    jn_idx = mCharacter->getSkeleton()->getJoint("ForeArmL")->getIndexInSkeleton(0);
    pos[jn_idx] = angle;
    mCharacter->getSkeleton()->getJoint("ForeArmL")->setPosition(0, angle);
    Eigen::Matrix3d origin_armL = getRotationMatrixFromPoints(mMarkerSet[7].getGlobalPos(), mMarkerSet[8].getGlobalPos(), mMarkerSet[9].getGlobalPos());
    Eigen::Matrix3d current_armL = getRotationMatrixFromPoints(mMarkerSet[7].getGlobalPos(), _pos[8], _pos[9]);
    T = current_armL * origin_armL.transpose();
    pT = mCharacter->getSkeleton()->getJoint("ArmL")->getParentBodyNode()->getTransform() * mCharacter->getSkeleton()->getJoint("ArmL")->getTransformFromParentBodyNode();
    mCharacter->getSkeleton()->getJoint("ArmL")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    return mCharacter->getSkeleton()->getPositions();
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
        mCharacter->applySkeletonBodyNode(mSkelInfos, mCharacter->getSkeleton());
    }

    std::vector<Eigen::Vector3d> ref_markers;
    for(auto m: mMarkerSet)
        ref_markers.push_back(m.getGlobalPos());

    // Pelvis size
    int idx = mCharacter->getSkeleton()->getBodyNode("Pelvis")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[11] - init_marker[10]).norm() / (ref_markers[11] - ref_markers[10]).norm();

    // FemurR size
    idx = mCharacter->getSkeleton()->getBodyNode("FemurR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[11] - init_marker[20]).norm() / (ref_markers[11] - ref_markers[20]).norm();

    // TibiaR size
    idx = mCharacter->getSkeleton()->getBodyNode("TibiaR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[16] - init_marker[14]).norm() / (ref_markers[16] - ref_markers[14]).norm();

    // TalusR size
    idx = mCharacter->getSkeleton()->getBodyNode("TalusR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[18] - init_marker[19]).norm() / (ref_markers[18] - ref_markers[19]).norm();

    // FemurL size
    idx = mCharacter->getSkeleton()->getBodyNode("FemurL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[11] - init_marker[20]).norm() / (ref_markers[11] - ref_markers[20]).norm();

    // TibiaL size
    idx = mCharacter->getSkeleton()->getBodyNode("TibiaL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[20] - init_marker[22]).norm() / (ref_markers[20] - ref_markers[22]).norm();

    // TalusL size
    idx = mCharacter->getSkeleton()->getBodyNode("TalusL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[18] - init_marker[19]).norm() / (ref_markers[18] - ref_markers[19]).norm();

    // Upper Body
    idx = mCharacter->getSkeleton()->getBodyNode("Spine")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mCharacter->getSkeleton()->getBodyNode("Torso")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mCharacter->getSkeleton()->getBodyNode("Neck")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mCharacter->getSkeleton()->getBodyNode("Head")->getIndexInSkeleton();
    // std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mCharacter->getSkeleton()->getBodyNode("ShoulderR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mCharacter->getSkeleton()->getBodyNode("ShoulderL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mCharacter->getSkeleton()->getBodyNode("ArmR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mCharacter->getSkeleton()->getBodyNode("ArmL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mCharacter->getSkeleton()->getBodyNode("ForeArmR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mCharacter->getSkeleton()->getBodyNode("ForeArmL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

    idx = mCharacter->getSkeleton()->getBodyNode("HandR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[4] - init_marker[3]).norm() / (ref_markers[4] - ref_markers[3]).norm();

    idx = mCharacter->getSkeleton()->getBodyNode("HandL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[4] - init_marker[3]).norm() / (ref_markers[4] - ref_markers[3]).norm();

    // Torsion
    idx = mCharacter->getSkeleton()->getBodyNode("FemurR")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[4] = torsionR;

    idx = mCharacter->getSkeleton()->getBodyNode("FemurL")->getIndexInSkeleton();
    std::get<1>(mSkelInfos[idx]).value[4] = torsionL;

    femurR_torsion = torsionR;
    femurL_torsion = torsionL;

    // Apply final skeleton modifications
    if (mCharacter) {
        mCharacter->applySkeletonBodyNode(mSkelInfos, mCharacter->getSkeleton());
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
    Eigen::VectorXd pos = mCharacter->getSkeleton()->getPositions();
    pos.setZero();

    // Set initial arm positions (forearms at 90 degrees)
    pos[mCharacter->getSkeleton()->getJoint("ForeArmR")->getIndexInSkeleton(0)] = M_PI * 0.5;
    pos[mCharacter->getSkeleton()->getJoint("ForeArmL")->getIndexInSkeleton(0)] = M_PI * 0.5;

    // Initialize knee joints (1-DOF revolute joints)
    pos[mCharacter->getSkeleton()->getJoint("TibiaR")->getIndexInSkeleton(0)] = 0.0;
    pos[mCharacter->getSkeleton()->getJoint("TibiaL")->getIndexInSkeleton(0)] = 0.0;

    mCharacter->getSkeleton()->setPositions(pos);

    // Fit skeleton to first frame markers (naive approach for ForeArm/Spine)
    fitSkeletonToMarker(firstFrameMarkers, params.femurTorsionL, params.femurTorsionR);
}

