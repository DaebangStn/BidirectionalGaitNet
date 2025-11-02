#ifndef SIM_C3D_H
#define SIM_C3D_H

#include "Motion.h"
#include <vector>
#include <string>
#include <Eigen/Core>

class C3D : public Motion
{
public:
    C3D();
    explicit C3D(const std::string& path);
    ~C3D() override = default;

    bool load(const std::string& path);
    bool isLoaded() const { return !mMarkers.empty(); }

    const std::vector<Eigen::Vector3d>& getMarkers(int frameIdx) const;
    void setMarkers(int frameIdx, const std::vector<Eigen::Vector3d>& markers);
    std::vector<Eigen::Vector3d> getInterpolatedMarkers(double frameFloat) const;
    Eigen::Vector3d getCentroid(int frameIdx) const;
    Eigen::Vector3d getCentroid(double frameFloat) const;
    int getFrameIndex(double phase) const;

    // Static utility for computing centroid from any marker vector
    static bool computeCentroid(const std::vector<Eigen::Vector3d>& markers, Eigen::Vector3d& centroid);

    // Static utility for detecting and correcting backward walking in marker data
    static bool detectAndCorrectBackwardWalking(std::vector<std::vector<Eigen::Vector3d>>& allFrameMarkers);

    double getFrameRate() const { return mFrameRate; }

    // Motion interface --------------------------------------------------
    Eigen::VectorXd getTargetPose(double phase) override;
    Eigen::VectorXd getPose(int frameIdx) override;
    Eigen::VectorXd getPose(double phase) override;
    double getMaxTime() const override;
    int getNumFrames() const override;
    double getFrameTime() const override;
    std::string getName() const override;
    void setRefMotion(Character* character, dart::simulation::WorldPtr world) override;
    std::string getSourceType() const override { return "c3dMarkers"; }
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

    std::vector<std::vector<Eigen::Vector3d>> mMarkers;
    std::vector<std::string> mLabels;
    double mFrameRate;
    double mFrameTime;
    std::string mName;
};

#endif // SIM_C3D_H
