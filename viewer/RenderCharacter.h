#ifndef RENDER_CHARACTER_H
#define RENDER_CHARACTER_H

#include <vector>
#include <string>
#include <tuple>
#include <map>
#include <Eigen/Dense>
#include "dart/dynamics/dynamics.hpp"
#include "Character.h"  // For ModifyInfo and BoneInfo types

using namespace dart::dynamics;

struct RenderMarker {
    std::string name;
    Eigen::Vector3d offset;  // Normalized offset (-1 to 1)
    BodyNode* bodyNode;

    Eigen::Vector3d getGlobalPos() const;
};

class RenderCharacter {
public:
    RenderCharacter(const std::string& skelPath);
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

    // Skeleton modification (for C3D IK fitting)
    void applySkeletonBodyNode(const std::vector<BoneInfo>& info, SkeletonPtr skel);

private:
    SkeletonPtr mSkeleton;
    SkeletonPtr mRefSkeleton;  // Reference skeleton for bone scaling
    std::vector<RenderMarker> mMarkers;
};

#endif
