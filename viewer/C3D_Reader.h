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


// C3D conversion parameters
struct C3DConversionParams
{
    double femurTorsionL = 0.0;
    double femurTorsionR = 0.0;
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

// Skeleton fitting configuration (loaded from YAML)
struct SkeletonFittingConfig {
    int frameStart = 0;
    int frameEnd = 0;  // -1 = all frames
    int maxIterations = 50;
    double convergenceThreshold = 1e-6;
    bool plotConvergence = true;

    struct BoneMapping {
        std::string boneName;
        std::vector<int> markerIndices;
        bool hasVirtualMarker = false;
    };
    std::vector<BoneMapping> boneMappings;

    // Load default bone mappings if config file not found
    void loadDefaults();
};

// Marker Structure
struct MocapMarker
{
    std::string name;
    Eigen::Vector3d offset;
    BodyNode *bn;

    Eigen::Vector3d getGlobalPos()
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

class C3D_Reader
{
    public:
        C3D_Reader(std::string marker_path, RenderCharacter *character);
        ~C3D_Reader();

        int getFrameRate() { return mFrameRate; }


        C3D* loadC3D(const std::string& path, const C3DConversionParams& params);
        Eigen::VectorXd getPoseFromC3D(std::vector<Eigen::Vector3d>& _pos);
        // New: Use optimizer's R, t for pelvis instead of marker centroid
        Eigen::VectorXd getPoseFromC3D_Optimized(int fitFrameIdx, std::vector<Eigen::Vector3d>& markers);
        // Eigen::VectorXd getPoseFromC3D_2(std::vector<Eigen::Vector3d> _pos);
        SkeletonPtr getBVHSkeleton() { return mVirtSkeleton; }

        // get Original Markers
        std::vector<Eigen::Vector3d> getMarkerPos(int idx) {return mOriginalMarkers[idx];}
        const std::vector<std::vector<Eigen::Vector3d>>& getAllOriginalMarkers() const { return mOriginalMarkers; }

        void fitSkeletonToMarker(std::vector<Eigen::Vector3d> init_marker, double torsionL = 0.0, double torsionR = 0.0);

        // New multi-frame anisotropic fitting methods
        void fitSkeletonMultiFrame(const std::vector<std::vector<Eigen::Vector3d>>& allMarkers,
                                   const C3DConversionParams& params,
                                   bool plotConvergence = false);

        // Stage 1: Compute hip joint centers using Harrington method
        static Eigen::Vector3d computeHipJointCenter(
            const Eigen::Vector3d& LASI,
            const Eigen::Vector3d& RASI,
            const Eigen::Vector3d& SACR,
            bool isLeft);
        void augmentMarkersWithHipJoints(std::vector<std::vector<Eigen::Vector3d>>& allMarkers);

        // Stage 2: Fit bones - extracts S (scale) and stores R, t (global transforms)
        void fitBoneLocal(const std::string& boneName,
                          const std::vector<int>& markerIndices,
                          bool hasVirtualMarker,
                          const std::vector<std::vector<Eigen::Vector3d>>& allMarkers,
                          bool plotConvergence = false);

        // Core algorithm: accepts BodyNode, marker indices, and GLOBAL marker positions
        // Internally handles world-to-local transformation and returns GLOBAL transforms
        BoneFitResult estimateScaleAlternating(
            BodyNode* bn,                                    // BodyNode for coordinate transforms
            const std::vector<int>& markerIndices,           // Marker indices to use
            const std::vector<std::vector<Eigen::Vector3d>>& globalP,  // Measured markers [K frames][N markers] in WORLD coords
            int maxIterations = 10,
            double convergenceThreshold = 1e-6,
            bool plotConvergence = false);
        Eigen::Vector3d getMarkerLocalPos(int markerIdx);

        const std::vector<MocapMarker>& getMarkerSet() { return mMarkerSet; }
        MotionData convertToMotion();
        std::vector<Eigen::VectorXd> mConvertedPos;

        // Config loading
        SkeletonFittingConfig loadSkeletonFittingConfig(const std::string& configPath);
        const SkeletonFittingConfig& getFittingConfig() const { return mFittingConfig; }
        void reloadFittingConfig();
        void resetSkeletonToDefault();

        // Accessors for fitted frame range
        int getFitFrameStart() const { return mFitFrameStart; }
        int getFitFrameEnd() const { return mFitFrameEnd; }

    private:
        // Helper methods for loadC3D refactoring
        C3D* loadMarkerData(const std::string& path);
        std::vector<Eigen::Vector3d> extractMarkersFromFrame(const ezc3d::c3d& c3d, size_t frameIdx);
        void initializeSkeletonForIK(const std::vector<Eigen::Vector3d>& firstFrameMarkers,
                                      const C3DConversionParams& params);
        std::vector<Eigen::VectorXd> convertFramesToSkeletonPoses(const ezc3d::c3d& c3d, size_t numFrames);
        void applyMotionPostProcessing(std::vector<Eigen::VectorXd>& motion, C3D* markerData);

        RenderCharacter *mCharacter;
        SkeletonPtr mVirtSkeleton;

        std::vector<BoneInfo> mSkelInfos;

        // Marker Set
        std::vector<MocapMarker> mMarkerSet;
        std::vector<Eigen::Vector3d> mRefMarkers;
        std::vector<Eigen::Isometry3d> mRefBnTransformation;

        // std::vector<Eigen::VectorXd> mMarkerPos;
        std::vector<std::vector<Eigen::Vector3d>> mOriginalMarkers;
        int mMarkerIdx;
        int mFrameRate;

        double femurR_torsion;
        double femurL_torsion;

        std::vector<Eigen::VectorXd> mCurrentMotion;
        std::vector<double> mCurrentPhi;

        // Skeleton fitting configuration
        SkeletonFittingConfig mFittingConfig;

        // Pelvis transform from Stage 0 fitting (frame 0 only - legacy)
        Eigen::Matrix3d mPelvisRotation = Eigen::Matrix3d::Identity();
        Eigen::Vector3d mPelvisTranslation = Eigen::Vector3d::Zero();

        // Per-frame pelvis transforms from optimizer (for fitted frame range only) - legacy
        std::vector<Eigen::Matrix3d> mPelvisR_frames;
        std::vector<Eigen::Vector3d> mPelvisT_frames;
        int mFitFrameStart = 0;
        int mFitFrameEnd = 0;

        // Per-bone per-frame transforms from optimizer (for all fitted bones)
        // Key: bone name, Value: per-frame transforms for fitted frame range
        std::map<std::string, std::vector<Eigen::Matrix3d>> mBoneR_frames;
        std::map<std::string, std::vector<Eigen::Vector3d>> mBoneT_frames;
};

#endif

