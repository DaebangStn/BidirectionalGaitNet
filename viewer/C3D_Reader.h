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

// Skeleton fitting configuration (loaded from YAML)
struct SkeletonFittingConfig {
    int frameStart = 0;
    int frameEnd = 0;  // -1 = all frames
    int maxIterations = 50;
    double convergenceThreshold = 1e-6;
    bool plotConvergence = true;
    bool upperBody = false;  // Enable upper body scaling/rotation (2-marker bones)

    // Flat marker mappings: skeleton marker name -> C3D data index/label
    std::vector<MarkerReference> markerMappings;

    // Optimization targets: list of bone names to optimize
    std::vector<std::string> optimizationTargets;

    // Helper: get data index for a skeleton marker name (-1 if not found)
    int getDataIndexForMarker(const std::string& markerName) const {
        for (const auto& ref : markerMappings) {
            if (ref.name == markerName) {
                return ref.dataIndex;
            }
        }
        return -1;
    }

};

class C3D_Reader
{
    public:
        C3D_Reader(std::string marker_path, RenderCharacter *character);
        ~C3D_Reader();

        int getFrameRate() { return mFrameRate; }


        C3D* loadC3D(const std::string& path, const C3DConversionParams& params);
        // buildFramePoseLegacy removed - kept as commented code in .cpp for reference
        Eigen::VectorXd buildFramePose(int fitFrameIdx);

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
                          const std::vector<std::vector<Eigen::Vector3d>>& allMarkers);

        // Core algorithm: uses MarkerReference for both offset and data index
        // Internally handles world-to-local transformation and returns GLOBAL transforms
        BoneFitResult optimizeBoneScale(
            BodyNode* bn,                                    // BodyNode for coordinate transforms
            const std::vector<const MarkerReference*>& markers,  // Markers with offset and dataIndex
            const std::vector<std::vector<Eigen::Vector3d>>& globalP);  // Measured markers [K frames][N markers] in WORLD coords
        Eigen::Vector3d getMarkerLocalPos(int markerIdx);

        const std::vector<MocapMarker>& getMarkerSet() { return mMarkerSet; }
        MotionData convertToMotion();
        std::vector<Eigen::VectorXd> mConvertedPos;

        // Config loading
        SkeletonFittingConfig loadSkeletonFittingConfig(const std::string& configPath);
        const SkeletonFittingConfig& getFittingConfig() const { return mFittingConfig; }

        // Marker label resolution
        void resolveMarkerReferences(const std::vector<std::string>& c3dLabels);

        // Accessors for fitted frame range
        int getFitFrameStart() const { return mFitFrameStart; }
        int getFitFrameEnd() const { return mFitFrameEnd; }

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

        RenderCharacter *mCharacter;

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

        // Marker label resolver
        MarkerResolver mResolver;

        // Fitted frame range
        int mFitFrameStart = 0;
        int mFitFrameEnd = 0;

        // Per-bone per-frame transforms from optimizer (for all fitted bones)
        // Key: bone name, Value: per-frame transforms for fitted frame range
        std::map<std::string, std::vector<Eigen::Matrix3d>> mBoneR_frames;
        std::map<std::string, std::vector<Eigen::Vector3d>> mBoneT_frames;
};

#endif

