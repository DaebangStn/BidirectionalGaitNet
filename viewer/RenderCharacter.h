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

private:
    SkeletonPtr mSkeleton;
    SkeletonPtr mRefSkeleton;  // Reference skeleton for bone scaling
    std::vector<RenderMarker> mMarkers;
    std::vector<BoneInfo> mSkelInfos;  // Cached bone scale info
};

#endif
