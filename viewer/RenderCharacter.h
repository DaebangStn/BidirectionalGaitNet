#ifndef RENDER_CHARACTER_H
#define RENDER_CHARACTER_H

#include <vector>
#include <string>
#include <tuple>
#include <map>
#include <Eigen/Dense>
#include "dart/dynamics/dynamics.hpp"
#include "Character.h"  // For ModifyInfo and BoneInfo types
#include "DARTHelper.h" // For skeleton flags

using namespace dart::dynamics;

struct RenderMarker {
    std::string name;
    Eigen::Vector3d offset;  // Normalized offset (-1 to 1)
    BodyNode* bodyNode;

    Eigen::Vector3d getGlobalPos() const;
};

class RenderCharacter {
public:
    RenderCharacter(const std::string& skelPath, int skelFlags = SKEL_COLLIDE_ALL);
    ~RenderCharacter();

    // Skeleton access (same interface as Character)
    SkeletonPtr getSkeleton() { return mSkeleton; }

    // Pose interpolation (same interface as Character)
    // Handles RevoluteJoint (linear), BallJoint (SLERP), FreeJoint (SLERP + linear)
    Eigen::VectorXd interpolatePose(const Eigen::VectorXd& pose1,
                                     const Eigen::VectorXd& pose2,
                                     double t,
                                     bool extrapolate_root = false);

    // Marker functionality
    void loadMarkers(const std::string& markerPath);
    std::vector<Eigen::Vector3d> getExpectedMarkerPositions() const;
    const std::vector<RenderMarker>& getMarkers() const { return mMarkers; }
    bool hasMarkers() const { return !mMarkers.empty(); }

    // Marker editing
    std::vector<RenderMarker>& getMarkersForEdit() { return mMarkers; }
    void addMarker(const std::string& name, const std::string& bodyNodeName, const Eigen::Vector3d& offset);
    void removeMarker(size_t index);
    void duplicateMarker(size_t index);
    bool saveMarkersToXml(const std::string& path) const;
    std::vector<std::string> getBodyNodeNames() const;

    // Skeleton modification (for C3D IK fitting)
    void applySkeletonBodyNode(const std::vector<BoneInfo>& info, SkeletonPtr skel);

    // Bone scale cache - single source of truth for scale values
    const std::vector<BoneInfo>& getSkelInfos() const { return mSkelInfos; }
    std::vector<BoneInfo>& getSkelInfos() { return mSkelInfos; }

    // Reset skeleton to default bone scales and zero pose
    void resetSkeletonToDefault();

    // Skeleton metadata accessors (for export)
    const std::map<std::string, std::string>& getContactFlags() const { return mContactFlags; }
    const std::map<std::string, std::string>& getObjFileLabels() const { return mObjFileLabels; }
    const std::map<std::string, std::vector<std::string>>& getBVHMap() const { return mBVHMap; }
    const std::vector<dart::dynamics::BodyNode*>& getEndEffectors() const { return mEndEffectors; }
    const Eigen::VectorXd& getKpVector() const { return mKp; }
    const Eigen::VectorXd& getKvVector() const { return mKv; }
    const std::string& getSkeletonPath() const { return mSkeletonPath; }

    // Reference skeleton (original, unmodified by calibration)
    SkeletonPtr getRefSkeleton() const { return mRefSkeleton; }

    // Skeleton export (bakes calibrated geometry)
    void exportSkeletonYAML(const std::string& path) const;

    // Body scale export/load (for static calibration persistence)
    void exportBodyScaleYAML(const std::string& path) const;
    bool loadBodyScaleYAML(const std::string& path);

private:
    // Skeleton metadata parsing
    void parseSkeletonMetadata(const std::string& path);
    void parseSkeletonMetadataFromXML(const std::string& path);
    void parseSkeletonMetadataFromYAML(const std::string& path);
    SkeletonPtr mSkeleton;
    SkeletonPtr mRefSkeleton;  // Reference skeleton for bone scaling
    std::vector<RenderMarker> mMarkers;
    std::vector<BoneInfo> mSkelInfos;  // Cached bone scale info

    // Skeleton metadata (parsed from XML/YAML)
    std::string mSkeletonPath;
    int mSkelFlags;  // Skeleton loading flags (SKEL_FREE_JOINTS, etc.)
    std::map<std::string, std::string> mContactFlags;      // body_name → "On"/"Off"
    std::map<std::string, std::string> mObjFileLabels;     // body_name → "mesh.obj"
    std::map<std::string, std::vector<std::string>> mBVHMap;  // joint_name → [bvh_channels]
    std::vector<dart::dynamics::BodyNode*> mEndEffectors;
    Eigen::VectorXd mKp;  // Joint Kp gains
    Eigen::VectorXd mKv;  // Joint Kv gains
};

#endif
