#ifndef SIM_C3D_H
#define SIM_C3D_H

#include "Motion.h"
#include <vector>
#include <string>
#include <Eigen/Core>

/**
 * @brief Unified C3D class for marker data and skeleton poses
 *
 * This class stores both raw marker positions (from C3D file) and
 * optionally skeleton poses (converted via inverse kinematics).
 * When skeleton poses are set, the Motion interface methods return
 * skeleton data; otherwise they return flattened marker data.
 */
class C3D : public Motion
{
public:
    C3D();
    explicit C3D(const std::string& path);
    ~C3D() override = default;

    bool load(const std::string& path);
    bool isLoaded() const { return !mMarkers.empty(); }

    // Marker access methods
    const std::vector<Eigen::Vector3d>& getMarkers(int frameIdx) const;
    void setMarkers(int frameIdx, const std::vector<Eigen::Vector3d>& markers);
    std::vector<Eigen::Vector3d> getInterpolatedMarkers(double frameFloat) const;
    Eigen::Vector3d getCentroid(int frameIdx) const;
    Eigen::Vector3d getCentroid(double frameFloat) const;
    int getFrameIndex(double phase) const;

    // Skeleton pose support (from IK conversion)
    void setSkeletonPoses(const std::vector<Eigen::VectorXd>& poses);
    const std::vector<Eigen::VectorXd>& getSkeletonPoses() const { return mSkeletonPoses; }
    bool hasSkeletonPoses() const { return !mSkeletonPoses.empty(); }

    // Source file tracking
    void setSourceFile(const std::string& path) { mSourceFile = path; }
    std::string getSourceFile() const { return mSourceFile; }

    // Static utility for computing centroid from any marker vector
    static bool computeCentroid(const std::vector<Eigen::Vector3d>& markers, Eigen::Vector3d& centroid);

    // Static utility for detecting and correcting backward walking in marker data
    static bool detectAndCorrectBackwardWalking(std::vector<std::vector<Eigen::Vector3d>>& allFrameMarkers);

    double getFrameRate() const { return mFrameRate; }

    // Label access (from C3D POINT/LABELS parameter)
    const std::vector<std::string>& getLabels() const { return mLabels; }
    int getNumLabels() const { return static_cast<int>(mLabels.size()); }

    // Motion interface --------------------------------------------------
    Eigen::VectorXd getTargetPose(double phase) override;
    Eigen::VectorXd getPose(int frameIdx) override;
    Eigen::VectorXd getPose(double phase) override;
    double getMaxTime() const override;
    int getNumFrames() const override;
    double getFrameTime() const override;
    std::string getName() const override;
    void setRefMotion(Character* character, dart::simulation::WorldPtr world) override;
    std::string getSourceType() const override { return "c3d"; }
    std::string getLogHeader() const override { return "[C3D]"; }
    int getValuesPerFrame() const override;
    int getTotalTimesteps() const override { return getNumFrames(); }
    int getTimestepsPerCycle() const override { return getNumFrames(); }
    std::vector<double> getTimestamps() const override;
    Eigen::VectorXd getRawMotionData() const override;

private:
    Eigen::Vector3d blendMarker(const Eigen::Vector3d& a,
                                 const Eigen::Vector3d& b,
                                 double weight) const;

    // Marker data (always present after load)
    std::vector<std::vector<Eigen::Vector3d>> mMarkers;
    std::vector<std::string> mLabels;

    // Skeleton poses (optional, from IK conversion)
    std::vector<Eigen::VectorXd> mSkeletonPoses;
    std::string mSourceFile;

    double mFrameRate;
    double mFrameTime;
    std::string mName;
};

#endif // SIM_C3D_H
