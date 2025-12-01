// Ceres-based bone scale optimizer
// Handles 2+ marker bones with rotation regularization
#ifndef CERES_OPTIMIZER_H
#define CERES_OPTIMIZER_H

#include <Eigen/Dense>
#include <vector>
#include <map>
#include <dart/dynamics/dynamics.hpp>
#include "Character.h"  // For BoneInfo typedef

// Forward declarations
struct BoneFitResult;
struct MarkerReference;
struct SkeletonFittingConfig;

using dart::dynamics::BodyNode;
using dart::dynamics::BoxShape;
using dart::dynamics::VisualAspect;

// Ceres-based bone scale optimizer
// Used for bones with only 2 markers where SVD is underconstrained
// Uses rotation regularization to stabilize the optimization
BoneFitResult optimizeBoneScaleCeres(
    BodyNode* bn,
    const std::vector<const MarkerReference*>& markers,
    const std::vector<std::vector<Eigen::Vector3d>>& globalP,
    const SkeletonFittingConfig& config);

// Forward declarations for C3D_Reader types
class RenderCharacter;

// Stage 2b: Run Ceres-based bone fitting for all targetCeres bones
// Returns map of bone name -> (R_frames, t_frames) for use in buildFramePose
void runCeresBoneFitting(
    const SkeletonFittingConfig& config,
    const std::map<std::string, std::vector<const MarkerReference*>>& boneToMarkers,
    const std::vector<std::vector<Eigen::Vector3d>>& allMarkers,
    RenderCharacter* character,
    std::vector<BoneInfo>& skelInfos,
    std::map<std::string, std::vector<Eigen::Matrix3d>>& boneR_frames,
    std::map<std::string, std::vector<Eigen::Vector3d>>& boneT_frames);

// Apply Ceres optimizer results in buildFramePose (arm rotations)
void applyCeresArmRotations(
    const SkeletonFittingConfig& config,
    const std::map<std::string, std::vector<Eigen::Matrix3d>>& boneR_frames,
    const std::map<std::string, std::vector<Eigen::Vector3d>>& boneT_frames,
    int fitFrameIdx,
    dart::dynamics::SkeletonPtr skel,
    Eigen::VectorXd& pos);

#endif // CERES_OPTIMIZER_H
