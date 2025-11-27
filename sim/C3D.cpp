#include "C3D.h"

#include <ezc3d/ezc3d_all.h>
#include "Log.h"
#include <filesystem>
#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;

C3D::C3D()
    : mFrameRate(60.0),
      mFrameTime(1.0 / 60.0)
{
}

C3D::C3D(const std::string& path)
    : C3D()
{
    load(path);
}

bool C3D::load(const std::string& path)
{
    try
    {
        ezc3d::c3d c3d(path);
        mName = fs::path(path).filename().string();

        mFrameRate = c3d.header().frameRate();
        if (mFrameRate <= 0.0)
            mFrameRate = 60.0;
        mFrameTime = 1.0 / mFrameRate;

        mLabels.clear();
        try
        {
            mLabels = c3d.parameters().group("POINT").parameter("LABELS").valuesAsString();
        }
        catch (...)
        {
            // Some files omit labels - ignore.
        }

        const auto& data = c3d.data();
        const size_t numFrames = data.nbFrames();
        mMarkers.clear();
        mMarkers.reserve(numFrames);

        for (size_t frameIdx = 0; frameIdx < numFrames; ++frameIdx)
        {
            const auto& frame = data.frame(frameIdx);
            const auto& points = frame.points();
            const size_t numPoints = points.nbPoints();

            std::vector<Eigen::Vector3d> frameMarkers;
            frameMarkers.reserve(numPoints);

            for (size_t pointIdx = 0; pointIdx < numPoints; ++pointIdx)
            {
                const auto& point = points.point(pointIdx);
                Eigen::Vector3d marker;
                marker[0] = 0.001 * point.y();
                marker[1] = 0.001 * point.z();
                marker[2] = 0.001 * point.x();
                frameMarkers.emplace_back(marker);
            }

            mMarkers.emplace_back(std::move(frameMarkers));
        }

        if (mMarkers.empty())
        {
            LOG_WARN("[C3D] Warning: No markers loaded from " << path);
            return false;
        }
        return true;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("[C3D] Failed to load " << path << ": " << e.what());
        mMarkers.clear();
        return false;
    }
}

const std::vector<Eigen::Vector3d>& C3D::getMarkers(int frameIdx) const
{
    static const std::vector<Eigen::Vector3d> empty;
    if (frameIdx < 0 || frameIdx >= static_cast<int>(mMarkers.size()))
        return empty;
    return mMarkers[frameIdx];
}

void C3D::setMarkers(int frameIdx, const std::vector<Eigen::Vector3d>& markers)
{
    if (frameIdx >= 0 && frameIdx < static_cast<int>(mMarkers.size())) {
        mMarkers[frameIdx] = markers;
    }
}

std::vector<Eigen::Vector3d> C3D::getInterpolatedMarkers(double frameFloat) const
{
    std::vector<Eigen::Vector3d> result;
    if (mMarkers.empty())
        return result;

    const double totalFrames = static_cast<double>(mMarkers.size());
    if (frameFloat < 0.0)
        frameFloat = 0.0;
    if (frameFloat >= totalFrames)
        frameFloat = std::fmod(frameFloat, totalFrames);
    if (frameFloat < 0.0)
        frameFloat += totalFrames;

    int idx0 = static_cast<int>(std::floor(frameFloat));
    int idx1 = (idx0 + 1) % static_cast<int>(mMarkers.size());
    double weight = frameFloat - static_cast<double>(idx0);

    const auto& markers0 = mMarkers[idx0];
    const auto& markers1 = mMarkers[idx1];
    const size_t count = std::min(markers0.size(), markers1.size());
    result.resize(count);
    for (size_t i = 0; i < count; ++i)
        result[i] = blendMarker(markers0[i], markers1[i], weight);
    return result;
}

Eigen::Vector3d C3D::getCentroid(int frameIdx) const
{
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    const auto& markers = getMarkers(frameIdx);
    if (markers.empty())
        return centroid;

    int valid = 0;
    for (const auto& marker : markers)
    {
        if (!marker.array().isFinite().all())
            continue;
        centroid += marker;
        valid++;
    }
    if (valid == 0)
        return Eigen::Vector3d::Zero();
    return centroid / static_cast<double>(valid);
}

Eigen::Vector3d C3D::getCentroid(double frameFloat) const
{
    if (mMarkers.empty())
        return Eigen::Vector3d::Zero();
    auto markers = getInterpolatedMarkers(frameFloat);
    if (markers.empty())
        return Eigen::Vector3d::Zero();
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    int valid = 0;
    for (const auto& marker : markers)
    {
        if (!marker.array().isFinite().all())
            continue;
        centroid += marker;
        valid++;
    }
    if (valid == 0)
        return Eigen::Vector3d::Zero();
    return centroid / static_cast<double>(valid);
}

int C3D::getFrameIndex(double phase) const
{
    if (mMarkers.empty())
        return 0;
    double wrapped = std::fmod(std::max(phase, 0.0), 1.0);
    int idx = static_cast<int>(std::round(wrapped * (mMarkers.size() - 1)));
    idx = std::clamp(idx, 0, static_cast<int>(mMarkers.size()) - 1);
    return idx;
}

Eigen::VectorXd C3D::getTargetPose(double phase)
{
    return getPose(phase);
}

Eigen::VectorXd C3D::getPose(int frameIdx)
{
    // If skeleton poses are available, return those
    if (!mSkeletonPoses.empty()) {
        if (frameIdx < 0 || frameIdx >= static_cast<int>(mSkeletonPoses.size()))
            return Eigen::VectorXd();
        return mSkeletonPoses[frameIdx];
    }

    // Fallback: return flattened marker data
    if (mMarkers.empty())
        return Eigen::VectorXd();
    const auto& markers = getMarkers(frameIdx);
    const int valuesPerFrame = static_cast<int>(markers.size() * 3);
    Eigen::VectorXd flattened(valuesPerFrame);
    for (size_t i = 0; i < markers.size(); ++i)
        flattened.segment<3>(static_cast<Eigen::Index>(i) * 3) = markers[i];
    return flattened;
}

Eigen::VectorXd C3D::getPose(double phase)
{
    if (mMarkers.empty())
        return Eigen::VectorXd();
    return getPose(getFrameIndex(phase));
}

double C3D::getMaxTime() const
{
    if (mMarkers.empty())
        return 0.0;
    return static_cast<double>(mMarkers.size()) * mFrameTime;
}

int C3D::getNumFrames() const
{
    // When skeleton poses are available, use their count (matches getRawMotionData behavior)
    if (!mSkeletonPoses.empty()) {
        return static_cast<int>(mSkeletonPoses.size());
    }
    return static_cast<int>(mMarkers.size());
}

double C3D::getFrameTime() const
{
    return mFrameTime;
}

std::string C3D::getName() const
{
    return mName;
}

void C3D::setRefMotion(Character*, dart::simulation::WorldPtr)
{
    // Markers-only; nothing to do.
}

int C3D::getValuesPerFrame() const
{
    // If skeleton poses are available, return skeleton DOF
    if (!mSkeletonPoses.empty()) {
        return static_cast<int>(mSkeletonPoses.front().size());
    }

    // Fallback: marker count * 3
    if (mMarkers.empty())
        return 0;
    return static_cast<int>(mMarkers.front().size() * 3);
}

std::vector<double> C3D::getTimestamps() const
{
    std::vector<double> timestamps;
    const int frames = getNumFrames();
    if (frames == 0 || mFrameRate <= 0.0)
        return timestamps;
    timestamps.reserve(frames);
    const double dt = 1.0 / mFrameRate;
    for (int i = 0; i < frames; ++i)
        timestamps.push_back(i * dt);
    return timestamps;
}

Eigen::VectorXd C3D::getRawMotionData() const
{
    // If skeleton poses are available, return those
    if (!mSkeletonPoses.empty()) {
        const int valuesPerFrame = getValuesPerFrame();
        const int totalFrames = static_cast<int>(mSkeletonPoses.size());
        Eigen::VectorXd rawData(valuesPerFrame * totalFrames);
        for (int i = 0; i < totalFrames; ++i) {
            rawData.segment(i * valuesPerFrame, valuesPerFrame) = mSkeletonPoses[i];
        }
        return rawData;
    }

    // Fallback: return flattened marker data
    if (mMarkers.empty())
        return Eigen::VectorXd();
    const int valuesPerFrame = getValuesPerFrame();
    if (valuesPerFrame == 0)
        return Eigen::VectorXd();
    Eigen::VectorXd flattened(valuesPerFrame * getNumFrames());
    for (int frame = 0; frame < getNumFrames(); ++frame)
    {
        const auto& markers = mMarkers[frame];
        for (size_t i = 0; i < markers.size(); ++i)
            flattened.segment<3>(frame * valuesPerFrame + static_cast<int>(i) * 3) = markers[i];
    }
    return flattened;
}

Eigen::Vector3d C3D::blendMarker(const Eigen::Vector3d& a,
                                 const Eigen::Vector3d& b,
                                 double weight) const
{
    const bool aValid = a.array().isFinite().all();
    const bool bValid = b.array().isFinite().all();
    if (aValid && bValid)
        return (1.0 - weight) * a + weight * b;
    if (aValid)
        return a;
    if (bValid)
        return b;
    return Eigen::Vector3d::Zero();
}

bool C3D::computeCentroid(const std::vector<Eigen::Vector3d>& markers, Eigen::Vector3d& centroid)
{
    centroid.setZero();
    if (markers.empty())
        return false;

    int valid = 0;
    for (const auto& marker : markers)
    {
        if (!marker.array().isFinite().all())
            continue;
        centroid += marker;
        valid++;
    }
    if (valid == 0)
        return false;

    centroid /= static_cast<double>(valid);
    return true;
}

bool C3D::detectAndCorrectBackwardWalking(std::vector<std::vector<Eigen::Vector3d>>& allFrameMarkers)
{
    if (allFrameMarkers.size() < 2)
        return false;

    // Compute centroids for first and last frames
    Eigen::Vector3d firstCentroid, lastCentroid;
    if (!computeCentroid(allFrameMarkers.front(), firstCentroid) ||
        !computeCentroid(allFrameMarkers.back(), lastCentroid))
    {
        return false;
    }

    // Detect backward walking: last Z < first Z
    if (lastCentroid.z() >= firstCentroid.z())
        return false; // Forward walking, no correction needed

    LOG_WARN("[C3D] Detected backward walking (last Z < first Z)");

    // Negate both Z and X coordinates to reverse walking direction
    // while preserving left/right orientation
    for (auto& frameMarkers : allFrameMarkers)
    {
        for (auto& marker : frameMarkers)
        {
            marker.z() = -marker.z();  // Reverse forward/backward
            marker.x() = -marker.x();  // Mirror left/right to compensate
        }
    }

    return true;
}

void C3D::setSkeletonPoses(const std::vector<Eigen::VectorXd>& poses)
{
    mSkeletonPoses = poses;
}
