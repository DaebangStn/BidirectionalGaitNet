// Read C3D Data and extract the gait and skeleton conditions
// This is naive implementation of C3D to BVH conversion, we recommend the usage of other softwars for conversion.
#ifndef C3D_READER_H
#define C3D_READER_H
#include "Environment.h"
#include "GLfunctions.h"
#include "RenderCharacter.h"
#include "C3D.h"
#include <ezc3d/ezc3d_all.h>
#include <yaml-cpp/yaml.h>
#include <map>
#include <unordered_map>


// C3D conversion parameters
struct C3DConversionParams
{
    double femurTorsionL = 0.0;
    double femurTorsionR = 0.0;
    bool doCalibration = false;  // Enable skeleton calibration (anisotropic bone scale fitting)
};

// Result of alternating bone fitting (anisotropic scale estimation)
struct BoneFitResult {
    Eigen::Vector3d scale;  // (sx, sy, sz) time-invariant anisotropic scales
    std::vector<Eigen::Matrix3d> R_frames;  // Per-frame rotations (joint pose)
    std::vector<Eigen::Vector3d> t_frames;  // Per-frame translations (joint pose)
    bool valid;
    int iterations;  // Convergence iterations used
    double finalRMS;  // Final reprojection RMS
};

// Marker Structure (moved before MarkerResolver to enable forward reference)
struct MocapMarker
{
    std::string name;
    Eigen::Vector3d offset;
    BodyNode *bn;

    Eigen::Vector3d getGlobalPos() const
    {
        // Body node size
        Eigen::Vector3d size = (dynamic_cast<const BoxShape *>(bn->getShapeNodeWith<VisualAspect>(0)->getShape().get()))->getSize();
        Eigen::Vector3d p = Eigen::Vector3d(std::abs(size[0]) * 0.5 * offset[0], std::abs(size[1]) * 0.5 * offset[1], std::abs(size[2]) * 0.5 * offset[2]);
        return bn->getTransform() * p;
    }

    void convertToOffset(Eigen::Vector3d pos)
    {
        Eigen::Vector3d size = (dynamic_cast<const BoxShape *>(bn->getShapeNodeWith<VisualAspect>(0)->getShape().get()))->getSize();
        Eigen::Vector3d p = bn->getTransform().inverse() * pos;
        offset = Eigen::Vector3d(p[0] / (std::abs(size[0]) * 0.5), p[1] / (std::abs(size[1]) * 0.5), p[2] / (std::abs(size[2]) * 0.5));
    }
};

// Marker reference for skeleton-to-data correspondence
// Contains both C3D data index and skeleton reference offset
struct MarkerReference {
    enum class Type { DataIndex, DataLabel };

    Type type = Type::DataIndex;
    std::string name;       // Skeleton marker name (e.g., "RASI")
    int dataIndex = -1;     // C3D data index (for measured position)
    std::string dataLabel;  // C3D label for Type::DataLabel (e.g., "R.ASIS")

    // Skeleton reference (populated during resolution from mMarkerSet)
    Eigen::Vector3d offset = Eigen::Vector3d::Zero();  // Bone-local offset
    std::string boneName;   // Associated bone name

    bool needsResolution() const { return type == Type::DataLabel && dataIndex < 0; }
    bool hasOffset() const { return !boneName.empty(); }

    static MarkerReference fromIndex(int idx) {
        MarkerReference ref;
        ref.type = Type::DataIndex;
        ref.dataIndex = idx;
        return ref;
    }

    static MarkerReference fromNameAndIndex(const std::string& n, int idx) {
        MarkerReference ref;
        ref.type = Type::DataIndex;
        ref.name = n;
        ref.dataIndex = idx;
        return ref;
    }

    static MarkerReference fromNameAndLabel(const std::string& n, const std::string& lbl) {
        MarkerReference ref;
        ref.type = Type::DataLabel;
        ref.name = n;
        ref.dataLabel = lbl;
        return ref;
    }
};

// Resolves marker labels to indices and skeleton offsets
class MarkerResolver {
public:
    void setC3DLabels(const std::vector<std::string>& labels);
    int resolve(const MarkerReference& ref) const;
    bool resolveAll(std::vector<MarkerReference>& refs, std::vector<int>& out) const;
    bool hasC3DLabels() const { return !mC3DLabels.empty(); }

    // Resolve skeleton offset from marker set
    void setMarkerSet(const std::vector<MocapMarker>& markerSet);
    bool resolveOffset(MarkerReference& ref) const;

private:
    std::vector<std::string> mC3DLabels;
    std::unordered_map<std::string, int> mLabelToIndex;

    // Skeleton marker info: name -> (offset, boneName)
    struct MarkerInfo {
        Eigen::Vector3d offset;
        std::string boneName;
    };
    std::unordered_map<std::string, MarkerInfo> mNameToInfo;
};

// SVD optimization target with per-bone settings
struct SvdTarget {
    std::string name;
    bool optimizeScale = true;  // false = only fit R,t, keep scale at 1.0
};

// Skeleton fitting configuration (loaded from YAML)
struct SkeletonFittingConfig {
    int frameStart = 0;
    int frameEnd = 0;  // -1 = all frames
    int maxIterations = 50;
    double convergenceThreshold = 1e-6;
    bool plotConvergence = true;

    // Flat marker mappings: skeleton marker name -> C3D data index/label
    std::vector<MarkerReference> markerMappings;

    // SVD-based optimizer targets (requires 3+ markers)
    std::vector<SvdTarget> targetSvd;

    double lambdaRot = 10.0;  // Ceres rotation regularization weight
    // Ceres-based optimizer targets (handles 2+ markers with regularization)
    std::vector<std::string> targetCeres;

    double interpolateRatio = 0.5;  // Interpolation ratio for dependent bones (Spine, Neck)
    double skelRatioBound = 1.5;    // Max scale ratio constraint: max(s)/min(s) <= skelRatioBound

    // Motion skeleton conversion targets (joint names)
    std::vector<std::string> targetMotionJoint;

    // Revolute axis selection mode: PCA, FIX (XML), BLEND
    enum class RevoluteAxisMode { PCA, FIX, BLEND };
    RevoluteAxisMode revoluteAxisMode = RevoluteAxisMode::BLEND;
    double revoluteAxisThresholdLow = 10.0;   // degrees
    double revoluteAxisThresholdHigh = 10.1;  // degrees

    // Inverse Kinematics refinement parameters (DLS + Line Search)
    struct IKParams {
        bool enabled = true;           // Enable/disable IK refinement
        int maxIterations = 15;
        double tolerance = 1e-4;
        double lambda = 0.01;          // DLS damping factor
        double alphaInit = 1.0;        // Initial line search step
        double beta = 0.5;             // Line search backtrack factor
        int maxLineSearch = 8;

        // Arm IK step limits (radians)
        double armMaxTwist = 0.15;
        double armMaxElbow = 0.2;

        // Leg IK step limits (radians)
        double legMaxHip = 0.15;
        double legMaxKnee = 0.2;
    };
    IKParams ik;

    // Symmetry enforcement configuration
    struct SymmetryConfig {
        bool enabled = true;
        double threshold = 1.5;           // ratio threshold for scales
        double torsionThreshold = 0.15;   // max angle diff in radians (~8.6°)

        // Common settings for all bones
        bool applyX = true;
        bool applyY = true;
        bool applyZ = true;
        bool applyUniform = true;
        bool applyTorsion = true;

        // List of bone base names (e.g., "Femur" -> FemurR/FemurL)
        std::vector<std::string> bones;
    };
    SymmetryConfig symmetry;

    // Plantar correction configuration (for refineTalus)
    struct PlantarCorrectionConfig {
        bool enabled = true;
        double velocityThreshold = 0.005;  // m/frame (5mm)
        int minLockFrames = 5;
        int blendFrames = 3;               // frames for smooth transition at boundaries
    };
    PlantarCorrectionConfig plantarCorrection;

    // Helper: get data index for a skeleton marker name (-1 if not found)
    int getDataIndexForMarker(const std::string& markerName) const {
        for (const auto& ref : markerMappings) {
            if (ref.name == markerName) {
                return ref.dataIndex;
            }
        }
        return -1;
    }

    // Helper: check if joint is in motion conversion target list (empty = all joints)
    bool isMotionJointTarget(const std::string& jointName) const {
        if (targetMotionJoint.empty()) return true;  // Empty = process all
        return std::find(targetMotionJoint.begin(), targetMotionJoint.end(), jointName)
               != targetMotionJoint.end();
    }

};

// Result of joint offset estimation for motion skeleton conversion
struct JointOffsetResult {
    Eigen::Isometry3d parentOffset;  // Parent body → joint pivot transform
    Eigen::Isometry3d childOffset;   // Joint pivot → child body transform
    Eigen::Vector3d revoluteAxis;    // For RevoluteJoint: estimated rotation axis
    bool valid = false;

    JointOffsetResult() {
        parentOffset.setIdentity();
        childOffset.setIdentity();
        revoluteAxis = Eigen::Vector3d::UnitX();
    }
};

// Result of converting free-joint poses to motion skeleton poses
struct MotionConversionResult {
    std::map<std::string, JointOffsetResult> jointOffsets;  // Per-joint fixed offsets
    std::vector<Eigen::VectorXd> motionPoses;               // Per-frame poses for motion skeleton
    bool valid = false;
};

// Foot lock phase for plantar correction (stance phase detection)
struct FootLockPhase {
    int startFrame;
    int endFrame;
    bool isLeft;  // true = left foot, false = right foot
};

// Result of static calibration (mFreeCharacter only, single frame)
// Outputs: bone scales + personalized marker offsets
struct StaticCalibrationResult {
    std::map<std::string, Eigen::Vector3d> boneScales;           // Bone name -> (sx, sy, sz) scale
    std::map<std::string, Eigen::Vector3d> personalizedOffsets;  // Marker name -> personalized offset
    std::map<std::string, double> talusHeightOffsets;            // Talus bone name -> shared y0 height offset
    bool success = false;
    std::string errorMessage;
};

// Result of dynamic calibration (both mFreeCharacter and mMotionCharacter)
// Outputs: bone scales + free poses + motion poses + joint offsets
struct DynamicCalibrationResult {
    std::map<std::string, Eigen::Vector3d> boneScales;      // Bone name -> (sx, sy, sz) scale
    std::vector<Eigen::VectorXd> freePoses;                 // Free skeleton poses (from mFreeCharacter)
    std::vector<Eigen::VectorXd> motionPoses;               // Articulated poses (from mMotionCharacter)
    MotionConversionResult motionResult;                    // Joint offsets etc.
    bool success = false;
    std::string errorMessage;
};

class C3D_Reader
{
    public:
        C3D_Reader(std::string fitting_config_path, std::string marker_path, RenderCharacter *free_character, RenderCharacter *motion_character);
        ~C3D_Reader();

        int getFrameRate() { return mFrameRate; }


        C3D* loadC3D(const std::string& path, const C3DConversionParams& params);
        C3D* loadC3DMarkersOnly(const std::string& path);  // Load markers only, no calibration/IK
        // buildFramePoseLegacy removed - kept as commented code in .cpp for reference
        Eigen::VectorXd buildFramePose(int fitFrameIdx);
        Eigen::VectorXd buildFramePoseLegacy(std::vector<Eigen::Vector3d>& _pos);

        // Arm rotation from marker heuristics (bending plane normal method)
        void computeArmRotations(int fitFrameIdx, Eigen::VectorXd& pos);

        // Interpolate dependent bone transforms (Spine, Neck)
        void interpolateDependent();
        void interpolateBoneTransforms(const std::string& newBone, const std::string& parentBone,
                                       const std::string& childBone);
        void fitSkeletonToMarker(std::vector<Eigen::Vector3d> init_marker, double torsionL = 0.0, double torsionR = 0.0);

        // New multi-frame anisotropic fitting methods
        void calibrateSkeleton(const C3DConversionParams& params);

        // Stage 1: Compute hip joint centers using Harrington method
        static Eigen::Vector3d computeHipJointCenter(
            const Eigen::Vector3d& LASI,
            const Eigen::Vector3d& RASI,
            const Eigen::Vector3d& SACR,
            bool isLeft);
        void computeHipJointCenters(std::vector<std::vector<Eigen::Vector3d>>& allMarkers);

        // Stage 2: Fit bones - extracts S (scale) and stores R, t (global transforms)
        void calibrateBone(const std::string& boneName,
                          const std::vector<const MarkerReference*>& markers,
                          const std::vector<std::vector<Eigen::Vector3d>>& allMarkers,
                          bool optimizeScale = true);

        // Core algorithm (SVD-based): uses MarkerReference for both offset and data index
        // Internally handles world-to-local transformation and returns GLOBAL transforms
        // Requires 3+ markers per bone
        // optimizeScale: if false, only fit R,t (pose), keep scale at 1.0
        BoneFitResult optimizeBoneScale(
            BodyNode* bn,                                    // BodyNode for coordinate transforms
            const std::vector<const MarkerReference*>& markers,  // Markers with offset and dataIndex
            const std::vector<std::vector<Eigen::Vector3d>>& globalP,  // Measured markers [K frames][N markers] in WORLD coords
            bool optimizeScale = true);

        // Note: Ceres-based optimizer (optimizeBoneScaleCeres) is in CeresOptimizer.h/cpp
        // and is only available when USE_CERES is defined

        Eigen::Vector3d getMarkerLocalPos(int markerIdx);

        const std::vector<MocapMarker>& getMarkerSet() { return mMarkerSet; }
        MotionData convertToMotion();
        std::vector<Eigen::VectorXd> mConvertedPos;

        // Config loading
        void loadSkeletonFittingConfig();
        const SkeletonFittingConfig& getFittingConfig() const { return mFittingConfig; }
        const std::string& getFittingConfigPath() const { return mFittingConfigPath; }

        // Marker label resolution
        void resolveMarkerReferences(const std::vector<std::string>& c3dLabels);

        // Accessors for fitted frame range
        int getFitFrameStart() const { return mFitFrameStart; }
        int getFitFrameEnd() const { return mFitFrameEnd; }

        // Motion skeleton conversion (free-joint poses → constrained joint angles)
        const MotionConversionResult& getMotionConversionResult() const { return mMotionResult; }
        MotionConversionResult convertToMotionSkeleton();

        // IK postprocessing methods (can be called independently via UI)
        void refineArmIK();      // Step 6: Adjust arm twist + elbow to match wrist markers
        void refineLegIK();      // Step 7: Adjust femur + tibia to match talus position
        void refineTalus();      // Step 8: Correct talus orientation during foot lock phases
        void enforceSymmetry();  // Step 9: Balance R/L bone scales based on threshold

        // ====== Static Calibration API ======
        // Uses mFreeCharacter ONLY - bone-by-bone SVD fitting, single frame
        // Outputs: bone scales + personalized marker offsets (THI, Shank, Heel, Toe)
        StaticCalibrationResult calibrateStatic(C3D* c3dData, const std::string& staticConfigPath);

        // Talus alternating optimization with shared body-frame height offset
        // Returns shared y0 height offset for heel/toe markers
        double calibrateTalusWithSharedHeight(
            const std::string& boneName,
            const std::vector<const MarkerReference*>& markers,
            const std::vector<Eigen::Vector3d>& frameMarkers);

        // Back-project marker position to bone-local offset (for THI, Shank)
        Eigen::Vector3d personalizeMarkerOffset(
            const std::string& markerName,
            const std::string& boneName,
            const Eigen::Vector3d& worldPos);

        // Export personalized markers and body scales to given directory
        void exportPersonalizedCalibration(
            const StaticCalibrationResult& result,
            const std::string& outputDir);

        // Dynamic calibration: multi-frame bone fitting + pose building
        // Input: C3D* with markers already loaded (from loadC3DMarkersOnly)
        // Uses both mFreeCharacter and mMotionCharacter
        DynamicCalibrationResult calibrateDynamic(C3D* c3dData);

        // Check if C3D has medial markers (for static calibration)
        static bool hasMedialMarkers(const std::vector<std::string>& labels);

    private:
        // Helper methods for loadC3D refactoring
        C3D* parseC3DFile(const std::string& path);
        void initializeSkeletonForIK(const std::vector<Eigen::Vector3d>& firstFrameMarkers,
                                      const C3DConversionParams& params);

        // Upper body helpers
        static Eigen::Matrix3d getRotationMatrixFromPoints(
            const Eigen::Vector3d& p1,
            const Eigen::Vector3d& p2,
            const Eigen::Vector3d& p3);
        void updateUpperBodyFromMarkers(
            int fitFrameIdx,
            Eigen::VectorXd& pos,
            const std::vector<Eigen::Vector3d>& markers,
            const std::vector<Eigen::Vector3d>& refMarkers);

        // Fallback scaling for USE_CERES=OFF
        void scaleArmsFallback();
        void copyDependentScales();

        RenderCharacter *mFreeCharacter;

        std::vector<BoneInfo> mSkelInfos;

        // Marker Set
        std::vector<MocapMarker> mMarkerSet;

        // Current C3D being processed (for access during calibration)
        C3D* mCurrentC3D = nullptr;
        int mFrameRate;

        double femurR_torsion;
        double femurL_torsion;

        std::vector<double> mCurrentPhi;

        // Skeleton fitting configuration
        SkeletonFittingConfig mFittingConfig;
        std::string mFittingConfigPath = "data/config/skeleton_fitting.yaml";  // Default path

        // Marker label resolver
        MarkerResolver mResolver;

        // Fitted frame range
        int mFitFrameStart = 0;
        int mFitFrameEnd = 0;

        // Per-bone per-frame transforms from optimizer (for all fitted bones)
        // Key: bone name, Value: per-frame transforms for fitted frame range
        std::map<std::string, std::vector<Eigen::Matrix3d>> mBoneR_frames;
        std::map<std::string, std::vector<Eigen::Vector3d>> mBoneT_frames;
        std::vector<std::string> mBoneOrder;  // Preserves insertion order from targetSvd

        // Arm rotation state for degeneracy handling (straight-arm case)
        Eigen::Vector3d mPrevArmNormalR = Eigen::Vector3d(0, 0, 1);
        Eigen::Vector3d mPrevArmNormalL = Eigen::Vector3d(0, 0, 1);

        // Cached elbow angles from cross product (for motion skeleton conversion)
        // Key: "ForeArmR" or "ForeArmL", Value: per-frame angles in [0, π]
        std::map<std::string, std::vector<double>> mElbowAngle_frames;

        // Motion skeleton conversion
        RenderCharacter* mMotionCharacter = nullptr;
        MotionConversionResult mMotionResult;
        std::vector<Eigen::VectorXd> mFreePoses;  // Free skeleton poses from buildFramePose

        // Helper functions for motion conversion
        std::map<std::string, std::vector<Eigen::Isometry3d>> computeRelativeTransforms();
        JointOffsetResult estimateJointOffsets(const std::string& jointName,
            const std::vector<Eigen::Isometry3d>& relativeTransforms);
        Eigen::Vector3d selectRevoluteAxis(const Eigen::Vector3d& xmlAxis,
            const std::vector<Eigen::Matrix3d>& rotations);
        Eigen::VectorXd buildMotionFramePose(int frameIdx,
            const std::map<std::string, JointOffsetResult>& offsets,
            const std::map<std::string, std::vector<Eigen::Isometry3d>>& relTransforms);
        void applyJointOffsetsToSkeleton(dart::dynamics::SkeletonPtr skel,
            const std::map<std::string, JointOffsetResult>& offsets);
};

#endif

