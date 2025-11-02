#include <cmath>
#include <utility>
#include <tinyxml2.h>
#include <ezc3d/ezc3d_all.h>
#include "C3D_Reader.h"
#include "C3DMotion.h"
#include "C3D.h"
#include "Log.h"

Eigen::MatrixXd getRotationMatrixFromPoints(Eigen::Vector3d p0, Eigen::Vector3d p1, Eigen::Vector3d p2)
{
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

    Eigen::Vector3d _axis = p1 - p0;
    Eigen::Vector3d axis1 = p2 - p0;
    Eigen::Vector3d axis2 = _axis.cross(axis1);
    Eigen::Vector3d axis3 = axis1.cross(axis2);

    axis1.normalize();
    axis2.normalize();
    axis3.normalize();

    R.col(0) = axis1;
    R.col(1) = axis2;
    R.col(2) = axis3;

    return R;
}

C3D_Reader::C3D_Reader(std::string skel_path, std::string marker_path, Environment *env)
{
    mEnv = env;

    mFrameRate = 60;

    tinyxml2::XMLDocument doc;
    doc.LoadFile(marker_path.c_str());
    if (doc.Error())
    {
        std::cout << "Error loading marker set file: " << marker_path << std::endl;
        std::cout << doc.ErrorName() << std::endl;
        std::cout << doc.ErrorStr() << std::endl;
        return;
    }

    mSkelInfos.clear();
    mMarkerSet.clear();

    mVirtSkeleton = BuildFromFile(skel_path, SKEL_COLLIDE_ALL);

    for (auto bn : mVirtSkeleton->getBodyNodes())
    {
        ModifyInfo SkelInfo;
        mSkelInfos.push_back(std::make_pair(bn->getName(), SkelInfo));
    }

    auto marker = doc.FirstChildElement("Markers");
    for (TiXmlElement *s = marker->FirstChildElement(); s != NULL; s = s->NextSiblingElement())
    {
        std::string name = std::string(s->Attribute("name"));
        std::string bn = std::string(s->Attribute("bn"));
        Eigen::Vector3d offset = string_to_vector3d(s->Attribute("offset"));

        MocapMarker m;
        m.name = name;
        m.bn = mVirtSkeleton->getBodyNode(bn);
        m.offset = offset;

        mMarkerSet.push_back(m);
    }

    femurL_torsion = 0.0;
    femurR_torsion = 0.0;
}

C3D_Reader::~C3D_Reader()
{
}

// ============================================================================
// Helper Methods for loadC3D
// ============================================================================

C3D* C3D_Reader::loadMarkerData(const std::string& path)
{
    LOG_VERBOSE("[C3D_Reader] Loading marker data from: " << path);

    C3D* markerData = new C3D(path);
    if (!markerData || markerData->getNumFrames() == 0) {
        LOG_ERROR("[C3D_Reader] Failed to load marker data or no frames found");
        delete markerData;
        return nullptr;
    }

    LOG_VERBOSE("[C3D_Reader] Marker data loaded: " << markerData->getNumFrames() << " frames");
    return markerData;
}

std::vector<Eigen::Vector3d> C3D_Reader::extractMarkersFromFrame(const ezc3d::c3d& c3d, size_t frameIdx)
{
    const auto& frame = c3d.data().frame(frameIdx);
    const auto& points = frame.points();
    const size_t numPoints = points.nbPoints();

    std::vector<Eigen::Vector3d> markers;
    markers.reserve(numPoints);

    for (size_t i = 0; i < numPoints; ++i)
    {
        const auto& point = points.point(i);
        Eigen::Vector3d marker;
        // Convert from C3D coordinate system (mm) to meters with axis reordering
        marker[0] = 0.001 * point.y();
        marker[1] = 0.001 * point.z();
        marker[2] = 0.001 * point.x();
        markers.emplace_back(marker);
    }

    return markers;
}

void C3D_Reader::initializeSkeletonForIK(const std::vector<Eigen::Vector3d>& firstFrameMarkers,
                                          const C3DConversionParams& params)
{
    LOG_VERBOSE("[C3D_Reader] Initializing skeleton pose for IK");

    // Initialize skeleton to zero pose
    Eigen::VectorXd pos = mVirtSkeleton->getPositions();
    pos.setZero();

    // Set initial arm positions (forearms at 90 degrees)
    pos[mVirtSkeleton->getJoint("ForeArmR")->getIndexInSkeleton(0)] = M_PI * 0.5;
    pos[mVirtSkeleton->getJoint("ForeArmL")->getIndexInSkeleton(0)] = M_PI * 0.5;

    // Initialize knee joints (1-DOF revolute joints)
    pos[mVirtSkeleton->getJoint("TibiaR")->getIndexInSkeleton(0)] = 0.0;
    pos[mVirtSkeleton->getJoint("TibiaL")->getIndexInSkeleton(0)] = 0.0;

    mVirtSkeleton->setPositions(pos);

    // Fit skeleton to first frame markers
    fitSkeletonToMarker(firstFrameMarkers, params.femurTorsionL, params.femurTorsionR);

    // Store reference markers and transformations
    mRefMarkers.clear();
    for (auto m : mMarkerSet)
        mRefMarkers.push_back(m.getGlobalPos());

    mRefBnTransformation.clear();
    for (auto bn : mVirtSkeleton->getBodyNodes())
        mRefBnTransformation.push_back(bn->getTransform());
}

std::vector<Eigen::VectorXd> C3D_Reader::convertFramesToSkeletonPoses(const ezc3d::c3d& c3d, size_t numFrames)
{
    LOG_VERBOSE("[C3D_Reader] Converting " << numFrames << " frames to skeleton poses");

    std::vector<Eigen::VectorXd> motion;
    motion.reserve(numFrames);

    mOriginalMarkers.clear();
    mOriginalMarkers.reserve(numFrames);

    for (size_t frameIdx = 0; frameIdx < numFrames; ++frameIdx)
    {
        std::vector<Eigen::Vector3d> markers = extractMarkersFromFrame(c3d, frameIdx);
        Eigen::VectorXd pose = getPoseFromC3D(markers);

        // Log first 3 frames for debugging marker-skeleton alignment (before moving markers)
        if (frameIdx < 3) {
            Eigen::Vector3d rootPos(pose[3], pose[4], pose[5]);
            Eigen::Vector3d pelvisCenter = (markers[10] + markers[11] + markers[12]) / 3.0;

            LOG_VERBOSE("[C3D_Reader] Frame " << frameIdx << ":");
            LOG_VERBOSE("  - Root position (skeleton): [" << rootPos[0] << ", " << rootPos[1] << ", " << rootPos[2] << "]");
            LOG_VERBOSE("  - Pelvis center (markers): [" << pelvisCenter[0] << ", " << pelvisCenter[1] << ", " << pelvisCenter[2] << "]");
            LOG_VERBOSE("  - Difference (root - pelvis): [" << (rootPos[0] - pelvisCenter[0]) << ", "
                        << (rootPos[1] - pelvisCenter[1]) << ", " << (rootPos[2] - pelvisCenter[2]) << "]");
        }

        motion.push_back(pose);
        mOriginalMarkers.push_back(std::move(markers));
    }

    LOG_VERBOSE("[C3D_Reader] Conversion complete: " << motion.size() << " poses");
    return motion;
}

// deprecated: it is required only for give offset to 3/8 of total frames
void C3D_Reader::applyMotionPostProcessing(std::vector<Eigen::VectorXd>& motion, C3D* markerData)
{
    LOG_VERBOSE("[C3D_Reader] Applying post-processing to motion and markers");

    if (motion.empty()) {
        LOG_WARN("[C3D_Reader] Empty motion data, skipping post-processing");
        return;
    }

    // Calculate frame offset for walking cycle removal (3/8 of total frames)
    int offset = motion.size() * 3 / 8;

    // Step 1: Zero skeleton positions relative to frame 0
    Eigen::Vector3d initial_skeleton_offset(motion[0][3], 0.0, motion[0][5]);

    for (size_t i = 1; i < motion.size(); i++)
    {
        motion[i][3] -= motion[0][3];
        motion[i][5] -= motion[0][5];
    }

    motion[0][3] = 0;
    motion[0][5] = 0;

    // Step 2: Reorder frames (walking offset removal) for both skeleton and markers
    std::vector<Eigen::VectorXd> new_motion;
    new_motion.reserve(motion.size());

    std::vector<std::vector<Eigen::Vector3d>> new_markers;
    new_markers.reserve(motion.size());

    mCurrentMotion.clear();
    mCurrentMotion.reserve(motion.size());

    // Copy frames from offset to end
    for (size_t i = offset; i < motion.size(); i++)
    {
        new_motion.push_back(motion[i]);
        new_markers.push_back(mOriginalMarkers[i]);
        mCurrentMotion.push_back(motion[i]);
    }

    // Calculate position offset for continuity
    Eigen::Vector3d offset_pos = new_motion.back().segment(3,3);
    offset_pos[1] = 0.0;

    // Copy frames from start to offset with position adjustment
    for (int i = 0; i < offset; i++)
    {
        motion[i].segment(3,3) += offset_pos;
        new_motion.push_back(motion[i]);
        new_markers.push_back(mOriginalMarkers[i]);
        mCurrentMotion.push_back(motion[i]);
    }

    // Step 3: Zero positions relative to new frame 0
    Eigen::Vector3d final_skeleton_offset(new_motion[0][3], 0.0, new_motion[0][5]);

    for (size_t i = 1; i < new_motion.size(); i++)
    {
        new_motion[i][3] -= new_motion[0][3];
        new_motion[i][5] -= new_motion[0][5];

        mCurrentMotion[i][3] -= mCurrentMotion[0][3];
        mCurrentMotion[i][5] -= mCurrentMotion[0][5];
    }

    new_motion[0][3] = 0;
    new_motion[0][5] = 0;
    mCurrentMotion[0][3] = 0;
    mCurrentMotion[0][5] = 0;

    // Step 4: Apply same offset to markers for alignment AND reorder them
    Eigen::Vector3d total_marker_offset = initial_skeleton_offset + final_skeleton_offset;

    LOG_VERBOSE("[C3D_Reader] initial_skeleton_offset: [" << initial_skeleton_offset[0] << ", " << initial_skeleton_offset[1] << ", " << initial_skeleton_offset[2] << "]");
    LOG_VERBOSE("[C3D_Reader] final_skeleton_offset: [" << final_skeleton_offset[0] << ", " << final_skeleton_offset[1] << ", " << final_skeleton_offset[2] << "]");
    LOG_VERBOSE("[C3D_Reader] total_marker_offset: [" << total_marker_offset[0] << ", " << total_marker_offset[1] << ", " << total_marker_offset[2] << "]");

    // Apply offset to reordered markers and update markerData
    for (size_t frameIdx = 0; frameIdx < new_markers.size(); ++frameIdx) {
        const auto& original = new_markers[frameIdx];
        std::vector<Eigen::Vector3d> aligned_markers;
        aligned_markers.reserve(original.size());

        for (const auto& marker : original) {
            Eigen::Vector3d aligned = marker;
            aligned[0] -= total_marker_offset[0];
            aligned[2] -= total_marker_offset[2];
            aligned_markers.push_back(aligned);
        }

        // Update markerData with aligned and reordered markers using proper setter
        markerData->setMarkers(frameIdx, aligned_markers);

        // Log first frame to verify update
        if (frameIdx == 0) {
            Eigen::Vector3d pelvisAfterAlign = (aligned_markers[10] + aligned_markers[11] + aligned_markers[12]) / 3.0;
            LOG_VERBOSE("[C3D_Reader] Frame 0 pelvis center AFTER alignment: [" << pelvisAfterAlign[0] << ", " << pelvisAfterAlign[1] << ", " << pelvisAfterAlign[2] << "]");
        }
    }

    // Replace motion with post-processed version
    motion = std::move(new_motion);
}

// ============================================================================
// Main loadC3D Function
// ============================================================================

C3DMotion* C3D_Reader::loadC3D(const std::string& path, const C3DConversionParams& params)
{
    LOG_VERBOSE("[C3D_Reader] loadC3D started for: " << path);

    // Step 1: Load marker data from C3D file
    C3D* markerData = loadMarkerData(path);
    if (!markerData) {
        return nullptr;
    }

    // Step 2: Load ezc3d data and setup
    ezc3d::c3d c3d(path);
    mFrameRate = static_cast<int>(std::lround(c3d.header().frameRate()));

    const size_t numFrames = c3d.data().nbFrames();
    if (numFrames == 0) {
        LOG_ERROR("[C3D_Reader] No frames found in C3D file");
        delete markerData;
        return nullptr;
    }

    // Step 2.5: Extract all markers and apply backward walking correction before IK
    std::vector<std::vector<Eigen::Vector3d>> allMarkers;
    allMarkers.reserve(numFrames);
    for (size_t frameIdx = 0; frameIdx < numFrames; ++frameIdx) {
        allMarkers.push_back(extractMarkersFromFrame(c3d, frameIdx));
    }

    // Detect and correct backward walking (matches sim/C3D.cpp behavior)
    C3D::detectAndCorrectBackwardWalking(allMarkers);

    // Step 3: Initialize skeleton pose for IK using corrected first frame
    initializeSkeletonForIK(allMarkers[0], params);

    // Step 4: Convert all frames to skeleton poses via IK using corrected markers
    mCurrentMotion.clear();
    std::vector<Eigen::VectorXd> motion;
    motion.reserve(numFrames);

    mOriginalMarkers.clear();
    mOriginalMarkers.reserve(numFrames);

    LOG_VERBOSE("[C3D_Reader] Converting " << numFrames << " frames to skeleton poses");
    for (size_t frameIdx = 0; frameIdx < numFrames; ++frameIdx)
    {
        Eigen::VectorXd pose = getPoseFromC3D(allMarkers[frameIdx]);

        // Log first 3 frames for debugging marker-skeleton alignment
        if (frameIdx < 3) {
            Eigen::Vector3d rootPos(pose[3], pose[4], pose[5]);
            Eigen::Vector3d pelvisCenter = (allMarkers[frameIdx][10] + allMarkers[frameIdx][11] + allMarkers[frameIdx][12]) / 3.0;

            LOG_VERBOSE("[C3D_Reader] Frame " << frameIdx << ":");
            LOG_VERBOSE("  - Root position (skeleton): [" << rootPos[0] << ", " << rootPos[1] << ", " << rootPos[2] << "]");
            LOG_VERBOSE("  - Pelvis center (markers): [" << pelvisCenter[0] << ", " << pelvisCenter[1] << ", " << pelvisCenter[2] << "]");
            LOG_VERBOSE("  - Difference (root - pelvis): [" << (rootPos[0] - pelvisCenter[0]) << ", "
                        << (rootPos[1] - pelvisCenter[1]) << ", " << (rootPos[2] - pelvisCenter[2]) << "]");
        }

        motion.push_back(pose);
        mOriginalMarkers.push_back(allMarkers[frameIdx]);
    }
    LOG_VERBOSE("[C3D_Reader] Conversion complete: " << motion.size() << " poses");

    // Step 5: Apply post-processing (reordering, zeroing, marker alignment)
    // - deprecated: it is required only for give offset to 3/8 of total frames
    // applyMotionPostProcessing(motion, markerData);

    // Step 6: Create and return C3DMotion object
    LOG_VERBOSE("[C3D_Reader] Creating C3DMotion with " << motion.size() << " frames");
    LOG_VERBOSE("[C3D_Reader] First frame DOF: " << (motion.empty() ? 0 : motion[0].size()));

    C3DMotion* result = new C3DMotion(markerData, motion, path);

    LOG_VERBOSE("[C3D_Reader] C3DMotion created successfully");
    LOG_VERBOSE("[C3D_Reader] - NumFrames: " << result->getNumFrames());
    LOG_VERBOSE("[C3D_Reader] - ValuesPerFrame: " << result->getValuesPerFrame());
    LOG_VERBOSE("[C3D_Reader] - RawMotionData size: " << result->getRawMotionData().size());

    return result;
}

// std::vector<Eigen::VectorXd>
MotionData
C3D_Reader::convertToMotion()
{
    MotionData motion;
    motion.name = "C3D";
    motion.motion = Eigen::VectorXd::Zero(6060);
    motion.param = mEnv->getParamState(0);
    motion.param.setOnes();

    double times = 1.0 / mFrameRate * mCurrentMotion.size();  // mMotion.size();

    // Global Ratio 를 알아내야함

    double globalRatio = 0.0;

    // for(auto m : mSkelInfos)
    for (int i = 0; i < mSkelInfos.size(); i++)
    {
        auto m = mSkelInfos[i];
        if (i < 13 && std::get<0>(m).find("Foot") == std::string::npos && std::get<0>(m).find("Talus") == std::string::npos)
            if (globalRatio < std::get<1>(m).value[3])
            {   
                std::cout << std::get<0>(m) << " : " << std::get<1>(m).value[3] << std::endl;
                globalRatio = std::get<1>(m).value[3];
            }
    }

    double abs_stride = mCurrentMotion.back()[5] - mCurrentMotion.front()[5];
    
    abs_stride /= (globalRatio * mEnv->getRefStride());

    // Set Stride
    motion.param[0] = abs_stride * 0.5;
    // Set Cadence
    motion.param[1] = (mEnv->getRefCadence() * sqrt(globalRatio) / (times * 0.5));

    motion.param[2] = globalRatio;

    // Femur L/R
    std::cout << "Femur L : " << std::get<1>(mSkelInfos[mVirtSkeleton->getBodyNode("FemurL")->getIndexInSkeleton()]).value[3] << std::endl;
    std::cout << "Femur R : " << std::get<1>(mSkelInfos[mVirtSkeleton->getBodyNode("FemurR")->getIndexInSkeleton()]).value[3] << std::endl;

    motion.param[3] = std::get<1>(mSkelInfos[mVirtSkeleton->getBodyNode("FemurL")->getIndexInSkeleton()]).value[3] / globalRatio;
    motion.param[4] = std::get<1>(mSkelInfos[mVirtSkeleton->getBodyNode("FemurR")->getIndexInSkeleton()]).value[3] / globalRatio;
    
    motion.param[3] = dart::math::clip(motion.param[3], 0.0, 1.0);
    motion.param[4] = dart::math::clip(motion.param[4], 0.0, 1.0);

    // Tibia L/R
    // std::cout << "Tibia L : " << std::get<1>(mSkelInfos[mEnv->getCharacter()->getSkeleton()->getJoint("TibiaL")->getIndexInSkeleton(0)]).value[3] << std::endl;
    // std::cout << "Tibia R : " << std::get<1>(mSkelInfos[mEnv->getCharacter()->getSkeleton()->getJoint("TibiaR")->getIndexInSkeleton(0)]).value[3] << std::endl;

    motion.param[5] = std::get<1>(mSkelInfos[mEnv->getCharacter()->getSkeleton()->getBodyNode("TibiaL")->getIndexInSkeleton()]).value[3] / globalRatio;
    motion.param[6] = std::get<1>(mSkelInfos[mEnv->getCharacter()->getSkeleton()->getBodyNode("TibiaR")->getIndexInSkeleton()]).value[3] / globalRatio;

    motion.param[5] = dart::math::clip(motion.param[5], 0.0, 1.0);
    motion.param[6] = dart::math::clip(motion.param[6], 0.0, 1.0);

    std::cout << "Tibia L : " << std::get<1>(mSkelInfos[mVirtSkeleton->getBodyNode("TibiaL")->getIndexInSkeleton()]).value[3] << std::endl;
    std::cout << "Tibia R : " << std::get<1>(mSkelInfos[mVirtSkeleton->getBodyNode("TibiaR")->getIndexInSkeleton()]).value[3] << std::endl;

    // // Arm L/R
    // motion.param[7] = std::get<1>(mSkelInfos[mEnv->getCharacter()->getSkeleton()->getJoint("ArmL")->getIndexInSkeleton(0)]).value[3] / globalRatio;
    // motion.param[8] = std::get<1>(mSkelInfos[mEnv->getCharacter()->getSkeleton()->getJoint("ArmR")->getIndexInSkeleton(0)]).value[3] / globalRatio;

    // // ForArm L/R
    // motion.param[9] = std::get<1>(mSkelInfos[mEnv->getCharacter()->getSkeleton()->getJoint("ForeArmL")->getIndexInSkeleton(0)]).value[3] / globalRatio;
    // motion.param[10] = std::get<1>(mSkelInfos[mEnv->getCharacter()->getSkeleton()->getJoint("ForeArmR")->getIndexInSkeleton(0)]).value[3] / globalRatio;

    std::cout << "global ratio" << globalRatio << std::endl;

    // Set Skeleton Parameter
    // 가장 큰 것 기준으로 줄이고 나머지 Length 로 하나하나 추가
    motion.param[11] = femurL_torsion;
    motion.param[12] = femurR_torsion;

    // std::vector<Eigen::VectorXd> mConvertedPos;
    mConvertedPos.clear();
    Eigen::VectorXd pos_backup = mEnv->getCharacter()->getSkeleton()->getPositions();
    Eigen::VectorXd pos = pos_backup;
    pos.setZero();
    for(int i = 0; i < mCurrentMotion.size(); i++)
    {
        mVirtSkeleton->setPositions(mCurrentMotion[i]);
        for (auto jn : mVirtSkeleton->getJoints())
        {
            auto skel_jn = mEnv->getCharacter()->getSkeleton()->getJoint(jn->getName());
            if(jn->getNumDofs() > skel_jn->getNumDofs())
                skel_jn->setPosition(0, jn->getPositions()[0]);
            else if (jn->getNumDofs() == skel_jn->getNumDofs())
                skel_jn->setPositions(jn->getPositions());
        }

        pos = mEnv->getCharacter()->getSkeleton()->getPositions();
        mConvertedPos.push_back(mEnv->getCharacter()->posToSixDof(pos));
    }
    std::cout << "Converted Positions : " << mConvertedPos.size() << std::endl;

    // Converting
    int current_idx = 0;
    std::vector<double> cur_phis;
    for (int i = 0; i < mConvertedPos.size(); i++)
        cur_phis.push_back(2.0 * i / mConvertedPos.size());
    cur_phis[0] = -1E-6;

    int phi_idx = 0;
    std::vector<double> ref_phis;
    for (int i = 0; i < 60; i++)
        ref_phis.push_back(2.0 * i / 60.0);



    // Converting pos to motion
    while (phi_idx < ref_phis.size() && current_idx < mConvertedPos.size() - 1)
    {
        if (cur_phis[current_idx] <= ref_phis[phi_idx] && ref_phis[phi_idx] <= cur_phis[current_idx + 1])
        {
            Eigen::VectorXd motion_pos = mConvertedPos[current_idx];
            double w0 = (ref_phis[phi_idx] - cur_phis[current_idx]) / (cur_phis[current_idx + 1] - cur_phis[current_idx]);
            double w1 = (cur_phis[current_idx + 1] - ref_phis[phi_idx]) / (cur_phis[current_idx + 1] - cur_phis[current_idx]);

            motion_pos.setZero();
            motion_pos += w0 * mConvertedPos[current_idx + 1];
            motion_pos += w1 * mConvertedPos[current_idx];

            Eigen::Vector3d v0 = ((current_idx == 0) ? mConvertedPos[current_idx + 1].segment(6, 3) - mConvertedPos[current_idx].segment(6, 3) : mConvertedPos[current_idx].segment(6, 3) - mConvertedPos[current_idx - 1].segment(6, 3)) * mFrameRate / 30.0;
            Eigen::Vector3d v1 = ((current_idx == mConvertedPos.size() - 1) ? mConvertedPos[current_idx].segment(6, 3) - mConvertedPos[current_idx - 1].segment(6, 3) : mConvertedPos[current_idx+1].segment(6, 3) - mConvertedPos[current_idx ].segment(6, 3)) * mFrameRate / 30.0;
            Eigen::Vector3d v = w0 * v0 + w1 * v1;

            motion_pos[6] = v[0];
            motion_pos[8] = v[2];

            motion.motion.segment(phi_idx * motion_pos.rows(), motion_pos.rows()) = motion_pos;


            // std::cout << phi_idx * motion_pos.rows() << "\t" << motion_pos.rows() << std::endl;
            phi_idx++;
        }
        else
            current_idx++;
    }
    mEnv->getCharacter()->getSkeleton()->setPositions(pos_backup);
    return motion;
}

// Eigen::VectorXd
// C3D_Reader::getPoseFromC3D_2(std::vector<Eigen::Vector3d> _pos)
// {
//     // It assumes that there is no torsion of tibia.

//     int jn_idx = 0;
//     int jn_dof = 0;
//     Eigen::Matrix3d T = Eigen::Matrix3d::Identity();

//     Eigen::VectorXd pos = mBVHSkeleton->getPositions();
//     pos.setZero();

//     // Pelvis

//     jn_idx = mBVHSkeleton->getJoint("Pelvis")->getIndexInSkeleton(0);
//     jn_dof = mBVHSkeleton->getJoint("Pelvis")->getNumDofs();

//     Eigen::Matrix3d origin_pelvis = getRotationMatrixFromPoints(mRefMarkers[10], mRefMarkers[11], mRefMarkers[12]);
//     Eigen::Matrix3d current_pelvis = getRotationMatrixFromPoints(_pos[10], _pos[11], _pos[12]);
//     Eigen::Isometry3d current_pelvis_T = Eigen::Isometry3d::Identity();
//     current_pelvis_T.linear() = current_pelvis * origin_pelvis.transpose();
//     current_pelvis_T.translation() = (_pos[10] + _pos[11] + _pos[12]) / 3.0 - (mRefMarkers[10] + mRefMarkers[11] + mRefMarkers[12]) / 3.0;

//     pos.segment(jn_idx, jn_dof) = FreeJoint::convertToPositions(current_pelvis_T);

//     return pos;
// }

Eigen::VectorXd C3D_Reader::getPoseFromC3D(std::vector<Eigen::Vector3d>& _pos)
{
    int jn_idx = 0;
    int jn_dof = 0;
    Eigen::Matrix3d T = Eigen::Matrix3d::Identity();

    Eigen::VectorXd pos = mVirtSkeleton->getPositions();
    pos.setZero();
    mVirtSkeleton->setPositions(pos);

    // Pelvis

    jn_idx = mVirtSkeleton->getJoint("Pelvis")->getIndexInSkeleton(0);
    jn_dof = mVirtSkeleton->getJoint("Pelvis")->getNumDofs();

    Eigen::Matrix3d origin_pelvis = getRotationMatrixFromPoints(mRefMarkers[10], mRefMarkers[11], mRefMarkers[12]);
    Eigen::Matrix3d current_pelvis = getRotationMatrixFromPoints(_pos[10], _pos[11], _pos[12]);
    Eigen::Isometry3d current_pelvis_T = Eigen::Isometry3d::Identity();
    current_pelvis_T.linear() = current_pelvis * origin_pelvis.transpose();
    current_pelvis_T.translation() = (_pos[10] + _pos[11] + _pos[12]) / 3.0 - (mRefMarkers[10] + mRefMarkers[11] + mRefMarkers[12]) / 3.0;

    pos.segment(jn_idx, jn_dof) = FreeJoint::convertToPositions(current_pelvis_T);

    mVirtSkeleton->getJoint("Pelvis")->setPositions(FreeJoint::convertToPositions(current_pelvis_T));
    // Right Leg

    // FemurR
    jn_idx = mVirtSkeleton->getJoint("FemurR")->getIndexInSkeleton(0);
    jn_dof = mVirtSkeleton->getJoint("FemurR")->getNumDofs();

    Eigen::Matrix3d origin_femurR = getRotationMatrixFromPoints(mMarkerSet[25].getGlobalPos() , mMarkerSet[13].getGlobalPos() , mMarkerSet[14].getGlobalPos() );
    Eigen::Matrix3d current_femurR = getRotationMatrixFromPoints(mMarkerSet[25].getGlobalPos() , _pos[13] , _pos[14] );
    Eigen::Isometry3d pT = mVirtSkeleton->getJoint("FemurR")->getParentBodyNode()->getTransform() * mVirtSkeleton->getJoint("FemurR")->getTransformFromParentBodyNode();

    T = current_femurR * (origin_femurR.transpose());
    mVirtSkeleton->getJoint("FemurR")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));


    // TibiaR
    jn_idx = mVirtSkeleton->getJoint("TibiaR")->getIndexInSkeleton(0);
    jn_dof = mVirtSkeleton->getJoint("TibiaR")->getNumDofs();

    Eigen::Matrix3d origin_kneeR = getRotationMatrixFromPoints(mMarkerSet[14].getGlobalPos(), mMarkerSet[15].getGlobalPos(), mMarkerSet[16].getGlobalPos());
    Eigen::Matrix3d current_kneeR = getRotationMatrixFromPoints(_pos[14], _pos[15], _pos[16]);
    T = (current_kneeR * origin_kneeR.transpose());

    pT = mVirtSkeleton->getJoint("TibiaR")->getParentBodyNode()->getTransform() * mVirtSkeleton->getJoint("TibiaR")->getTransformFromParentBodyNode();
    // Extract only the first component for 1-DOF knee joint
    Eigen::VectorXd kneeR_angles = pT.linear().transpose() * BallJoint::convertToPositions(T);
    mVirtSkeleton->getJoint("TibiaR")->setPosition(0, kneeR_angles[0]);
    
    // TalusR
    jn_idx = mVirtSkeleton->getJoint("TalusR")->getIndexInSkeleton(0);
    jn_dof = mVirtSkeleton->getJoint("TalusR")->getNumDofs();

    Eigen::Matrix3d origin_talusR = getRotationMatrixFromPoints(mMarkerSet[16].getGlobalPos(), mMarkerSet[17].getGlobalPos(), mMarkerSet[18].getGlobalPos());
    Eigen::Matrix3d current_talusR = getRotationMatrixFromPoints(_pos[16], _pos[17], _pos[18]);
    T = (current_talusR * origin_talusR.transpose());
    pT = mVirtSkeleton->getJoint("TalusR")->getParentBodyNode()->getTransform() * mVirtSkeleton->getJoint("TalusR")->getTransformFromParentBodyNode();
    mVirtSkeleton->getJoint("TalusR")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));
    // FemurL 
    jn_idx = mVirtSkeleton->getJoint("FemurL")->getIndexInSkeleton(0);
    jn_dof = mVirtSkeleton->getJoint("FemurL")->getNumDofs();

    Eigen::Matrix3d origin_femurL = getRotationMatrixFromPoints(mMarkerSet[26].getGlobalPos(), mMarkerSet[19].getGlobalPos(), mMarkerSet[20].getGlobalPos());
    Eigen::Matrix3d current_femurL = getRotationMatrixFromPoints(mMarkerSet[26].getGlobalPos(), _pos[19], _pos[20]);
    T = current_femurL * origin_femurL.transpose();
    pT = mVirtSkeleton->getJoint("FemurL")->getParentBodyNode()->getTransform() * mVirtSkeleton->getJoint("FemurL")->getTransformFromParentBodyNode();

    mVirtSkeleton->getJoint("FemurL")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    // TibiaL
    jn_idx = mVirtSkeleton->getJoint("TibiaL")->getIndexInSkeleton(0);
    jn_dof = mVirtSkeleton->getJoint("TibiaL")->getNumDofs();

    Eigen::Matrix3d origin_kneeL = getRotationMatrixFromPoints(mMarkerSet[20].getGlobalPos(), mMarkerSet[21].getGlobalPos(), mMarkerSet[22].getGlobalPos());
    Eigen::Matrix3d current_kneeL = getRotationMatrixFromPoints(_pos[20], _pos[21], _pos[22]);
    T = current_kneeL * origin_kneeL.transpose();
    pT = mVirtSkeleton->getJoint("TibiaL")->getParentBodyNode()->getTransform() * mVirtSkeleton->getJoint("TibiaL")->getTransformFromParentBodyNode();

    // Extract only the first component for 1-DOF knee joint
    Eigen::VectorXd kneeL_angles = pT.linear().transpose() * BallJoint::convertToPositions(T);
    mVirtSkeleton->getJoint("TibiaL")->setPosition(0, kneeL_angles[0]);

    // TalusL
    jn_idx = mVirtSkeleton->getJoint("TalusL")->getIndexInSkeleton(0);
    jn_dof = mVirtSkeleton->getJoint("TalusL")->getNumDofs();

    Eigen::Matrix3d origin_talusL = getRotationMatrixFromPoints(mMarkerSet[22].getGlobalPos(), mMarkerSet[23].getGlobalPos(), mMarkerSet[24].getGlobalPos());
    Eigen::Matrix3d current_talusL = getRotationMatrixFromPoints(_pos[22], _pos[23], _pos[24]);
    T = current_talusL * origin_talusL.transpose();
    pT = mVirtSkeleton->getJoint("TalusL")->getParentBodyNode()->getTransform() * mVirtSkeleton->getJoint("TalusL")->getTransformFromParentBodyNode();

    mVirtSkeleton->getJoint("TalusL")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));


    // Spine and Torso
    Eigen::Matrix3d origin_torso = getRotationMatrixFromPoints(mMarkerSet[3].getGlobalPos(), mMarkerSet[4].getGlobalPos(), mMarkerSet[7].getGlobalPos());
    Eigen::Matrix3d current_torso = getRotationMatrixFromPoints(_pos[3], _pos[4], _pos[7]);
    T = current_torso * origin_torso.transpose();
    pT = mVirtSkeleton->getJoint("Torso")->getParentBodyNode()->getTransform() * mVirtSkeleton->getJoint("Torso")->getTransformFromParentBodyNode();
    Eigen::Quaterniond tmp_T = Eigen::Quaterniond(T).slerp(0.5, Eigen::Quaterniond::Identity());
    T = tmp_T.toRotationMatrix();

    // Spine
    jn_idx = mVirtSkeleton->getJoint("Spine")->getIndexInSkeleton(0);
    jn_dof = mVirtSkeleton->getJoint("Spine")->getNumDofs();
    mVirtSkeleton->getJoint("Spine")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));
    // Torso
    jn_idx = mVirtSkeleton->getJoint("Torso")->getIndexInSkeleton(0);
    jn_dof = mVirtSkeleton->getJoint("Torso")->getNumDofs();
    mVirtSkeleton->getJoint("Torso")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    // Neck and Head
    Eigen::Matrix3d origin_head = getRotationMatrixFromPoints(mMarkerSet[0].getGlobalPos(), mMarkerSet[1].getGlobalPos(), mMarkerSet[2].getGlobalPos());
    Eigen::Matrix3d current_head = getRotationMatrixFromPoints(_pos[0], _pos[1], _pos[2]);
    T = current_head * origin_head.transpose();
    pT = mVirtSkeleton->getJoint("Head")->getParentBodyNode()->getTransform() * mVirtSkeleton->getJoint("Head")->getTransformFromParentBodyNode();
    tmp_T = Eigen::Quaterniond(T).slerp(0.5, Eigen::Quaterniond::Identity());
    T = tmp_T.toRotationMatrix();

    // // Neck
    // jn_idx = mBVHSkeleton->getJoint("Neck")->getIndexInSkeleton(0);
    // jn_dof = mBVHSkeleton->getJoint("Neck")->getNumDofs();
    // mBVHSkeleton->getJoint("Neck")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));
    // // Head
    // jn_idx = mBVHSkeleton->getJoint("Head")->getIndexInSkeleton(0);
    // jn_dof = mBVHSkeleton->getJoint("Head")->getNumDofs();
    // mBVHSkeleton->getJoint("Head")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    // Arm

    // ArmR

    // Elbow Angle
    Eigen::Vector3d v1 = _pos[3] - _pos[5];
    Eigen::Vector3d v2 = _pos[6] - _pos[5];

    double angle = abs(atan2(v1.cross(v2).norm(), v1.dot(v2)));
    
    if(angle > M_PI * 0.5)
        angle = M_PI - angle;

    jn_idx = mVirtSkeleton->getJoint("ForeArmR")->getIndexInSkeleton(0);
    pos[jn_idx] = angle;
    mVirtSkeleton->getJoint("ForeArmR")->setPosition(0, angle);

    Eigen::Matrix3d origin_armR = getRotationMatrixFromPoints(mMarkerSet[3].getGlobalPos(), mMarkerSet[5].getGlobalPos(), mMarkerSet[6].getGlobalPos());
    Eigen::Matrix3d current_armR = getRotationMatrixFromPoints(mMarkerSet[3].getGlobalPos(), _pos[5], _pos[6]);
    T = current_armR * origin_armR.transpose();
    pT = mVirtSkeleton->getJoint("ArmR")->getParentBodyNode()->getTransform() * mVirtSkeleton->getJoint("ArmR")->getTransformFromParentBodyNode();

    mVirtSkeleton->getJoint("ArmR")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    // ArmL

    // Elbow Angle
    v1 = _pos[8] - _pos[7];
    v2 = _pos[8] - _pos[9];

    angle = abs(atan2(v1.cross(v2).norm(), v1.dot(v2)));

    if(angle > M_PI * 0.5)
        angle = M_PI - angle;

    jn_idx = mVirtSkeleton->getJoint("ForeArmL")->getIndexInSkeleton(0);
    pos[jn_idx] = angle;
    mVirtSkeleton->getJoint("ForeArmL")->setPosition(0, angle);

    Eigen::Matrix3d origin_armL = getRotationMatrixFromPoints(mMarkerSet[7].getGlobalPos(), mMarkerSet[8].getGlobalPos(), mMarkerSet[9].getGlobalPos());
    Eigen::Matrix3d current_armL = getRotationMatrixFromPoints(mMarkerSet[7].getGlobalPos(), _pos[8], _pos[9]);
    T = current_armL * origin_armL.transpose();
    pT = mVirtSkeleton->getJoint("ArmL")->getParentBodyNode()->getTransform() * mVirtSkeleton->getJoint("ArmL")->getTransformFromParentBodyNode();

    mVirtSkeleton->getJoint("ArmL")->setPositions(pT.linear().transpose() * BallJoint::convertToPositions(T));

    return mVirtSkeleton->getPositions();
}
