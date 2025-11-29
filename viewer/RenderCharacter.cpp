#include "RenderCharacter.h"
#include "DARTHelper.h"
#include "UriResolver.h"
#include "Log.h"
#include <tinyxml2.h>
#include <iomanip>
#include <fstream>

using namespace dart::dynamics;

// Static helper map for skeleton bone axes (same as Character.cpp)
static std::map<std::string, int> skeletonAxis = {
    {"Pelvis", 1},
    {"FemurR", 1},
    {"TibiaR", 1},
    {"TalusR", 2},
    {"FootThumbR", 2},
    {"FootPinkyR", 2},
    {"FemurL", 1},
    {"TibiaL", 1},
    {"TalusL", 2},
    {"FootThumbL", 2},
    {"FootPinkyL", 2},
    {"Spine", 1},
    {"Torso", 1},
    {"Neck", 1},
    {"Head", 1},
    {"ShoulderR", 0},
    {"ArmR", 0},
    {"ForeArmR", 0},
    {"HandR", 0},
    {"ShoulderL", 0},
    {"ArmL", 0},
    {"ForeArmL", 0},
    {"HandL", 0},
};

// Helper to unfold ModifyInfo
static std::tuple<Eigen::Vector3d, double, double> UnfoldModifyInfo(const ModifyInfo &info)
{
    return std::make_tuple(Eigen::Vector3d(info[0], info[1], info[2]), info[3], info[4]);
}

// Helper for modifying isometry (same as Character.cpp)
static Eigen::Isometry3d modifyIsometry3d(const Eigen::Isometry3d &iso, const ModifyInfo &info, int axis, bool rotate = true)
{
    Eigen::Vector3d l;
    double s, t;
    std::tie(l, s, t) = UnfoldModifyInfo(info);
    Eigen::Vector3d translation = iso.translation();
    translation = translation.cwiseProduct(l);
    translation *= s;
    auto tmp = Eigen::Isometry3d(Eigen::Translation3d(translation));
    tmp.linear() = iso.linear();
    Eigen::Vector3d angle_axis = Eigen::Vector3d::Zero();
    angle_axis[axis] = 1;
    if (rotate)
        tmp.rotate(Eigen::AngleAxisd(t, angle_axis));
    return tmp;
}

// Helper for modifying shape nodes (same as Character.cpp)
static void modifyShapeNode(BodyNode *rtgBody, BodyNode *stdBody, const ModifyInfo &info, int axis)
{
    Eigen::Vector3d l;
    double s, t;
    std::tie(l, s, t) = UnfoldModifyInfo(info);
    double la = l[axis], lb = l[(axis + 1) % 3], lc = l[(axis + 2) % 3];

    for (size_t i = 0; i < rtgBody->getNumShapeNodes(); i++)
    {
        ShapeNode *rtgShape = rtgBody->getShapeNode(i), *stdShape = stdBody->getShapeNode(i);
        ShapePtr newShape;
        if (auto rtg = std::dynamic_pointer_cast<CapsuleShape>(rtgShape->getShape()))
        {
            auto std = std::dynamic_pointer_cast<CapsuleShape>(stdShape->getShape());
            double radius = std->getRadius() * s * (lb + lc) / 2, height = std->getHeight() * s * la;
            newShape = ShapePtr(new CapsuleShape(radius, height));
        }
        else if (auto rtg = std::dynamic_pointer_cast<SphereShape>(rtgShape->getShape()))
        {
            auto std = std::dynamic_pointer_cast<SphereShape>(stdShape->getShape());
            double radius = std->getRadius() * s * (la + lb + lc) / 3;
            newShape = ShapePtr(new SphereShape(radius));
        }
        else if (auto rtg = std::dynamic_pointer_cast<CylinderShape>(rtgShape->getShape()))
        {
            auto std = std::dynamic_pointer_cast<CylinderShape>(stdShape->getShape());
            double radius = std->getRadius() * s * (lb + lc) / 2, height = std->getHeight() * s * la;
            newShape = ShapePtr(new CylinderShape(radius, height));
        }
        else if (std::dynamic_pointer_cast<BoxShape>(rtgShape->getShape()))
        {
            auto std = std::dynamic_pointer_cast<BoxShape>(stdShape->getShape());
            Eigen::Vector3d size = std->getSize() * s;
            size = size.cwiseProduct(l);
            newShape = ShapePtr(new BoxShape(size));
        }
        else if (auto rtg = std::dynamic_pointer_cast<MeshShape>(rtgShape->getShape()))
        {
            auto std = std::dynamic_pointer_cast<MeshShape>(stdShape->getShape());
            Eigen::Vector3d scale = std->getScale();
            scale *= s;
            scale = scale.cwiseProduct(l);
            rtg->setScale(scale);
            Eigen::Isometry3d s = stdShape->getRelativeTransform(), r = modifyIsometry3d(s.inverse(), info, axis, false).inverse();
            rtgShape->setRelativeTransform(r);
            newShape = rtg;
        }
        rtgShape->setShape(newShape);
    }
    ShapePtr shape;
    rtgBody->eachShapeNodeWith<DynamicsAspect>([&shape](dart::dynamics::ShapeNode* sn) {
        shape = sn->getShape();
        return false;
    });
    double mass = stdBody->getMass() * l[0] * l[1] * l[2] * s * s * s;
    dart::dynamics::Inertia inertia;
    inertia.setMass(mass);
    inertia.setMoment(shape->computeInertia(mass));
    rtgBody->setInertia(inertia);
}

RenderCharacter::RenderCharacter(const std::string& skelPath, int skelFlags)
{
    mSkeleton = BuildFromFile(skelPath, skelFlags);
    // Also create reference skeleton for bone scaling operations
    mRefSkeleton = BuildFromFile(skelPath, skelFlags);
    mRefSkeleton->setPositions(Eigen::VectorXd::Zero(mRefSkeleton->getNumDofs()));

    // Initialize mSkelInfos with default scale for all bones
    for (size_t i = 0; i < mSkeleton->getNumBodyNodes(); ++i)
    {
        auto* bn = mSkeleton->getBodyNode(i);
        mSkelInfos.push_back(std::make_tuple(bn->getName(), ModifyInfo()));
    }
}

RenderCharacter::~RenderCharacter() {}

Eigen::VectorXd RenderCharacter::interpolatePose(
    const Eigen::VectorXd& pose1,
    const Eigen::VectorXd& pose2,
    double t,
    bool extrapolate_root)
{
    if (t <= 0.000001) return pose1;
    if (t >= 0.999999) return pose2;
    if (t < 0 || t > 1) {
        LOG_WARN("[RenderCharacter] interpolatePose: t is out of range: " << t);
        return pose1;
    }
    Eigen::VectorXd interpolated = Eigen::VectorXd::Zero(pose1.rows());

    for (const auto jn : mSkeleton->getJoints())
    {
        int dof = jn->getNumDofs();
        if (dof == 0) continue;
        int idx = jn->getIndexInSkeleton(0);

        if (dof == 1)
        {
            // RevoluteJoint: linear interpolation
            interpolated[idx] = pose1[idx] * (1.0 - t) + pose2[idx] * t;
        }
        else if (dof == 3)
        {
            // BallJoint: quaternion SLERP
            Eigen::Quaterniond q1 = Eigen::Quaterniond(BallJoint::convertToRotation(pose1.segment(idx, dof)));
            Eigen::Quaterniond q2 = Eigen::Quaterniond(BallJoint::convertToRotation(pose2.segment(idx, dof)));
            Eigen::Quaterniond q = q1.slerp(t, q2);
            interpolated.segment(idx, dof) = BallJoint::convertToPositions(q.toRotationMatrix());
        }
        else if (dof == 6)
        {
            // FreeJoint: SLERP for rotation + linear/extrapolated translation
            Eigen::Quaterniond q1 = Eigen::Quaterniond(BallJoint::convertToRotation(pose1.segment(idx, 3)));
            Eigen::Quaterniond q2 = Eigen::Quaterniond(BallJoint::convertToRotation(pose2.segment(idx, 3)));
            Eigen::Quaterniond q = q1.slerp(t, q2);
            interpolated.segment(idx, 3) = BallJoint::convertToPositions(q.toRotationMatrix());

            // Root position with optional extrapolation
            if (extrapolate_root && idx == 0)
            {
                // Extrapolate with velocity from pose1 to pose2
                Eigen::Vector3d velocity = pose2.segment(idx + 3, 3) - pose1.segment(idx + 3, 3);
                interpolated.segment(idx + 3, 3) = pose1.segment(idx + 3, 3) + velocity * t;
            }
            else
            {
                // Standard linear interpolation
                interpolated.segment(idx + 3, 3) = pose1.segment(idx + 3, 3) * (1.0 - t) + pose2.segment(idx + 3, 3) * t;
            }
        }
    }

    return interpolated;
}

void RenderCharacter::loadMarkers(const std::string& markerPath)
{
    mMarkers.clear();

    // Resolve URI scheme (e.g., @data/ -> absolute path)
    std::string resolvedPath = PMuscle::URIResolver::getInstance().resolve(markerPath);

    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(resolvedPath.c_str()) != tinyxml2::XML_SUCCESS) {
        LOG_WARN("[RenderCharacter] Failed to load marker file: " << markerPath);
        return;
    }

    auto* markersElem = doc.FirstChildElement("Markers");
    if (!markersElem) {
        LOG_WARN("[RenderCharacter] No <Markers> element found in: " << markerPath);
        return;
    }

    for (auto* markerElem = markersElem->FirstChildElement("marker");
         markerElem != nullptr;
         markerElem = markerElem->NextSiblingElement("marker"))
    {
        const char* name = markerElem->Attribute("name");
        const char* bn = markerElem->Attribute("bn");
        const char* offsetStr = markerElem->Attribute("offset");

        if (!name || !bn || !offsetStr) {
            LOG_WARN("[RenderCharacter] Incomplete marker definition, skipping");
            continue;
        }

        // Parse offset string "x y z"
        Eigen::Vector3d offset;
        std::istringstream iss(offsetStr);
        iss >> offset[0] >> offset[1] >> offset[2];

        // Find body node
        BodyNode* bodyNode = mSkeleton->getBodyNode(bn);
        if (!bodyNode) {
            LOG_WARN("[RenderCharacter] Body node not found: " << bn);
            continue;
        }

        RenderMarker marker;
        marker.name = name;
        marker.offset = offset;
        marker.bodyNode = bodyNode;
        mMarkers.push_back(marker);
    }

    LOG_INFO("[RenderCharacter] Loaded " << mMarkers.size() << " markers from: " << markerPath);
}

Eigen::Vector3d RenderMarker::getGlobalPos() const
{
    if (!bodyNode) return Eigen::Vector3d::Zero();

    // Get body bounding box size from visual shape
    auto* shapeNode = bodyNode->getShapeNodeWith<VisualAspect>(0);
    if (!shapeNode) return bodyNode->getTransform().translation();

    const auto* boxShape = dynamic_cast<const BoxShape*>(shapeNode->getShape().get());
    if (!boxShape) return bodyNode->getTransform().translation();

    Eigen::Vector3d size = boxShape->getSize();

    // Scale normalized offset by half-body dimensions
    Eigen::Vector3d p(
        std::abs(size[0]) * 0.5 * offset[0],
        std::abs(size[1]) * 0.5 * offset[1],
        std::abs(size[2]) * 0.5 * offset[2]
    );

    // Transform to world frame
    return bodyNode->getTransform() * p;
}

std::vector<Eigen::Vector3d> RenderCharacter::getExpectedMarkerPositions() const
{
    std::vector<Eigen::Vector3d> positions;
    positions.reserve(mMarkers.size());
    for (const auto& marker : mMarkers) {
        positions.push_back(marker.getGlobalPos());
    }
    return positions;
}

void RenderCharacter::applySkeletonBodyNode(const std::vector<BoneInfo>& info, SkeletonPtr skel)
{
    // Cache the scale info
    mSkelInfos = info;

    for (const auto& bone : info)
    {
        std::string name;
        ModifyInfo modInfo;
        std::tie(name, modInfo) = bone;

        auto axisIt = skeletonAxis.find(name);
        if (axisIt == skeletonAxis.end())
            continue;
        int axis = axisIt->second;

        BodyNode *rtgBody = skel->getBodyNode(name);
        BodyNode *stdBody = mRefSkeleton->getBodyNode(name);
        if (rtgBody == nullptr || stdBody == nullptr)
            continue;

        modifyShapeNode(rtgBody, stdBody, modInfo, axis);

        if (Joint *rtgParent = rtgBody->getParentJoint())
        {
            Joint *stdParent = stdBody->getParentJoint();
            Eigen::Isometry3d up = stdParent->getTransformFromChildBodyNode();
            rtgParent->setTransformFromChildBodyNode(modifyIsometry3d(up, modInfo, axis));
        }

        for (size_t i = 0; i < rtgBody->getNumChildJoints(); i++)
        {
            Joint *rtgJoint = rtgBody->getChildJoint(i);
            Joint *stdJoint = stdBody->getChildJoint(i);
            Eigen::Isometry3d down = stdJoint->getTransformFromParentBodyNode();
            rtgJoint->setTransformFromParentBodyNode(modifyIsometry3d(down, modInfo, axis, false));
        }
    }
}

// Marker editing methods
void RenderCharacter::addMarker(const std::string& name, const std::string& bodyNodeName, const Eigen::Vector3d& offset)
{
    BodyNode* bn = mSkeleton->getBodyNode(bodyNodeName);
    if (!bn) {
        LOG_WARN("[RenderCharacter] Cannot add marker - body node not found: " << bodyNodeName);
        return;
    }

    RenderMarker marker;
    marker.name = name;
    marker.offset = offset;
    marker.bodyNode = bn;
    mMarkers.push_back(marker);
}

void RenderCharacter::removeMarker(size_t index)
{
    if (index < mMarkers.size()) {
        mMarkers.erase(mMarkers.begin() + index);
    }
}

void RenderCharacter::duplicateMarker(size_t index)
{
    if (index < mMarkers.size()) {
        RenderMarker copy = mMarkers[index];
        copy.name = copy.name + "_copy";
        mMarkers.insert(mMarkers.begin() + index + 1, copy);
    }
}

bool RenderCharacter::saveMarkersToXml(const std::string& path) const
{
    std::ofstream file(path);
    if (!file.is_open()) {
        LOG_WARN("[RenderCharacter] Failed to open file for writing: " << path);
        return false;
    }

    // Find max lengths for alignment
    size_t maxNameLen = 4;  // minimum "name"
    size_t maxBnLen = 2;    // minimum "bn"
    for (const auto& marker : mMarkers) {
        maxNameLen = std::max(maxNameLen, marker.name.length());
        maxBnLen = std::max(maxBnLen, marker.bodyNode->getName().length());
    }

    // Write XML with aligned columns
    file << "<?xml version=\"1.0\"?>\n";
    file << "<!-- Marker set - offset is relative scale (not meter) -->\n";
    file << "<Markers>\n";

    for (const auto& marker : mMarkers) {
        // Format offset with fixed width for each component (handle negative sign)
        std::ostringstream ossX, ossY, ossZ;
        ossX << std::fixed << std::setprecision(6) << std::setw(10) << marker.offset[0];
        ossY << std::fixed << std::setprecision(6) << std::setw(10) << marker.offset[1];
        ossZ << std::fixed << std::setprecision(6) << std::setw(10) << marker.offset[2];

        // Build padded name and bn strings (padding after closing quote)
        std::string nameAttr = "name=\"" + marker.name + "\"";
        std::string bnAttr = "bn=\"" + marker.bodyNode->getName() + "\"";
        size_t nameAttrLen = 7 + maxNameLen + 1;  // name="..."
        size_t bnAttrLen = 4 + maxBnLen + 1;      // bn="..."

        file << "  <marker " << std::left << std::setw(nameAttrLen) << nameAttr << "  "
             << std::left << std::setw(bnAttrLen) << bnAttr << "  "
             << "offset=\"" << ossX.str() << " " << ossY.str() << " " << ossZ.str() << "\"/>\n";
    }

    file << "</Markers>\n";
    file.close();

    LOG_INFO("[RenderCharacter] Saved " << mMarkers.size() << " markers to: " << path);
    return true;
}

std::vector<std::string> RenderCharacter::getBodyNodeNames() const
{
    std::vector<std::string> names;
    for (size_t i = 0; i < mSkeleton->getNumBodyNodes(); ++i) {
        names.push_back(mSkeleton->getBodyNode(i)->getName());
    }
    return names;
}


void RenderCharacter::resetSkeletonToDefault()
{
    // Reset all bone parameters to default (scale = 1.0)
    for (auto& info : mSkelInfos) {
        std::get<1>(info) = ModifyInfo();
    }
    applySkeletonBodyNode(mSkelInfos, mSkeleton);
    // Set skeleton to zero pose
    if (mSkeleton) {
        mSkeleton->setPositions(Eigen::VectorXd::Zero(mSkeleton->getNumDofs()));
    }
    LOG_INFO("[RenderCharacter] Skeleton reset to default scales and zero pose");
}
