#include "C3DMotion.h"
#include <cmath>
#include <algorithm>

C3DMotion::C3DMotion(C3D* markerData,
                     const std::vector<Eigen::VectorXd>& skeletonPoses,
                     const std::string& sourceFile)
    : mMarkerData(markerData)
    , mSkeletonPoses(skeletonPoses)
    , mSourceFile(sourceFile)
{
    if (mMarkerData) {
        mName = mMarkerData->getName();
        mFrameTime = mMarkerData->getFrameTime();
        mFrameRate = (mFrameTime > 0.0) ? (1.0 / mFrameTime) : 60.0;
    } else {
        mName = "C3DMotion";
        mFrameTime = 1.0 / 60.0;
        mFrameRate = 60.0;
    }
}

Eigen::VectorXd C3DMotion::getTargetPose(double phase)
{
    return getPose(phase);
}

Eigen::VectorXd C3DMotion::getPose(int frameIdx)
{
    if (frameIdx < 0 || frameIdx >= static_cast<int>(mSkeletonPoses.size()))
        return Eigen::VectorXd();
    return mSkeletonPoses[frameIdx];
}

Eigen::VectorXd C3DMotion::getPose(double phase)
{
    return getPose(getFrameIndex(phase));
}

double C3DMotion::getMaxTime() const
{
    if (mSkeletonPoses.empty())
        return 0.0;
    return static_cast<double>(mSkeletonPoses.size()) * mFrameTime;
}

int C3DMotion::getNumFrames() const
{
    return static_cast<int>(mSkeletonPoses.size());
}

double C3DMotion::getFrameTime() const
{
    return mFrameTime;
}

std::string C3DMotion::getName() const
{
    return mName;
}

void C3DMotion::setRefMotion(Character* character, dart::simulation::WorldPtr world)
{
    // C3DMotion doesn't need reference motion setup
    // Skeleton poses are already computed via IK
}

const std::vector<Eigen::Vector3d>& C3DMotion::getMarkers(int frameIdx) const
{
    if (!mMarkerData || mMarkerData->getNumFrames() == 0) {
        static const std::vector<Eigen::Vector3d> empty;
        return empty;
    }

    // Clamp frameIdx to valid range
    frameIdx = std::clamp(frameIdx, 0, mMarkerData->getNumFrames() - 1);
    return mMarkerData->getMarkers(frameIdx);
}

std::vector<Eigen::Vector3d> C3DMotion::getInterpolatedMarkers(double frameFloat) const
{
    if (mMarkerData)
        return mMarkerData->getInterpolatedMarkers(frameFloat);
    return std::vector<Eigen::Vector3d>();
}

Eigen::Vector3d C3DMotion::getCentroid(int frameIdx) const
{
    if (mMarkerData)
        return mMarkerData->getCentroid(frameIdx);
    return Eigen::Vector3d::Zero();
}

Eigen::Vector3d C3DMotion::getCentroid(double frameFloat) const
{
    if (mMarkerData)
        return mMarkerData->getCentroid(frameFloat);
    return Eigen::Vector3d::Zero();
}

int C3DMotion::getFrameIndex(double phase) const
{
    if (mSkeletonPoses.empty())
        return 0;

    double wrapped = std::fmod(std::max(phase, 0.0), 1.0);
    int idx = static_cast<int>(std::round(wrapped * (mSkeletonPoses.size() - 1)));
    idx = std::clamp(idx, 0, static_cast<int>(mSkeletonPoses.size()) - 1);
    return idx;
}

Eigen::VectorXd C3DMotion::getRawMotionData() const
{
    if (mSkeletonPoses.empty())
        return Eigen::VectorXd();

    int valuesPerFrame = getValuesPerFrame();
    int totalFrames = getNumFrames();

    Eigen::VectorXd rawData(valuesPerFrame * totalFrames);

    for (int i = 0; i < totalFrames; ++i) {
        rawData.segment(i * valuesPerFrame, valuesPerFrame) = mSkeletonPoses[i];
    }

    return rawData;
}

int C3DMotion::getValuesPerFrame() const
{
    if (mSkeletonPoses.empty())
        return 56; // Default skeleton DOF

    return mSkeletonPoses[0].size();
}
