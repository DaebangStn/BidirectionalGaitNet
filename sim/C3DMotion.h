#ifndef __C3D_MOTION_H__
#define __C3D_MOTION_H__

#include "Motion.h"
#include "C3D.h"
#include <memory>
#include <vector>

/**
 * @brief C3DMotion combines C3D marker data with skeleton pose data
 *
 * This class stores both raw marker positions (from C3D file) and
 * skeleton poses (converted via inverse kinematics). It implements
 * the Motion interface using skeleton poses while providing access
 * to marker data for visualization.
 */
class C3DMotion : public Motion {
private:
    std::unique_ptr<C3D> mMarkerData;           // Raw marker positions
    std::vector<Eigen::VectorXd> mSkeletonPoses; // IK-converted skeleton poses
    std::string mName;
    std::string mSourceFile;                    // Track which C3D file this came from
    double mFrameRate;
    double mFrameTime;

public:
    /**
     * @brief Construct C3DMotion from marker data and skeleton poses
     * @param markerData Raw C3D marker data (takes ownership)
     * @param skeletonPoses IK-converted skeleton poses
     * @param sourceFile Path to source C3D file
     */
    C3DMotion(C3D* markerData,
              const std::vector<Eigen::VectorXd>& skeletonPoses,
              const std::string& sourceFile);

    ~C3DMotion() = default;

    // Motion interface - returns skeleton poses
    Eigen::VectorXd getTargetPose(double phase) override;
    Eigen::VectorXd getPose(int frameIdx) override;
    Eigen::VectorXd getPose(double phase) override;
    double getMaxTime() const override;
    int getNumFrames() const override;
    double getFrameTime() const override;
    std::string getName() const override;
    void setRefMotion(Character* character, dart::simulation::WorldPtr world) override;

    // Type identification
    std::string getSourceType() const override { return "C3DMotion"; }
    std::string getSourceFile() const { return mSourceFile; }
    std::string getLogHeader() const override { return "[C3D]"; }

    // Motion data access for motionPoseEval
    Eigen::VectorXd getRawMotionData() const override;
    int getValuesPerFrame() const override;
    int getTotalTimesteps() const override { return getNumFrames(); }

    // Marker access methods (delegate to mMarkerData)
    const std::vector<Eigen::Vector3d>& getMarkers(int frameIdx) const;
    std::vector<Eigen::Vector3d> getInterpolatedMarkers(double frameFloat) const;
    Eigen::Vector3d getCentroid(int frameIdx) const;
    Eigen::Vector3d getCentroid(double frameFloat) const;

    // Direct access to marker data object
    C3D* getMarkerData() const { return mMarkerData.get(); }

    // Helper for frame index calculation
    int getFrameIndex(double phase) const;
};

#endif // __C3D_MOTION_H__
