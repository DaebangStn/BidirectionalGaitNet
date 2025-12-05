#include "RenderCharacter.h"
#include "DARTHelper.h"
#include "UriResolver.h"
#include "Log.h"
#include <tinyxml2.h>
#include <yaml-cpp/yaml.h>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>
#include <ctime>

using namespace dart::dynamics;

// Default PD gains (same as Character.cpp)
static constexpr double kDefaultKp = 300.0;
static constexpr double kDefaultKv = 25.0;

// Helper: parse space-separated string to VectorXd (split_string is already in DARTHelper.h)
static Eigen::VectorXd string_to_vectorXd(const char* str, int expected_size) {
    std::vector<double> values;
    std::istringstream iss(str);
    double val;
    while (iss >> val) {
        values.push_back(val);
    }
    Eigen::VectorXd result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = values[i];
    }
    return result;
}

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
    mSkeletonPath = skelPath;
    mSkelFlags = skelFlags;  // Store flags for later use
    mSkeleton = BuildFromFile(skelPath, skelFlags);
    // Also create reference skeleton for bone scaling operations
    mRefSkeleton = BuildFromFile(skelPath, skelFlags);
    mRefSkeleton->setPositions(Eigen::VectorXd::Zero(mRefSkeleton->getNumDofs()));

    // Initialize Kp/Kv with defaults
    int numDofs = mSkeleton->getNumDofs();
    LOG_INFO("[RenderCharacter] Skeleton has " << numDofs << " DOFs");
    mKp = Eigen::VectorXd::Ones(numDofs) * kDefaultKp;
    mKv = Eigen::VectorXd::Ones(numDofs) * kDefaultKv;

    // Initialize mSkelInfos with default scale for all bones
    for (size_t i = 0; i < mSkeleton->getNumBodyNodes(); ++i)
    {
        auto* bn = mSkeleton->getBodyNode(i);
        mSkelInfos.push_back(std::make_tuple(bn->getName(), ModifyInfo()));
    }

    // Parse skeleton metadata (contact flags, obj labels, bvh map, end effectors, kp/kv)
    parseSkeletonMetadata(skelPath);
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

// ═══════════════════════════════════════════════════════════════════════════
// Skeleton Metadata Parsing
// ═══════════════════════════════════════════════════════════════════════════

void RenderCharacter::parseSkeletonMetadata(const std::string& path)
{
    // Resolve URI scheme
    std::string resolvedPath = PMuscle::URIResolver::getInstance().resolve(path);

    // Detect format by extension
    std::string ext;
    size_t dot_pos = resolvedPath.find_last_of('.');
    if (dot_pos != std::string::npos) {
        ext = resolvedPath.substr(dot_pos);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }

    if (ext == ".yaml" || ext == ".yml") {
        parseSkeletonMetadataFromYAML(resolvedPath);
    } else {
        parseSkeletonMetadataFromXML(resolvedPath);
    }

    // Reset root DOF gains (root joint should not be controlled by PD)
    auto rootJoint = mSkeleton->getRootJoint();
    if (rootJoint && rootJoint->getNumDofs() > 0) {
        int rootIdx = rootJoint->getIndexInSkeleton(0);
        int rootDofs = rootJoint->getNumDofs();
        for (int i = 0; i < rootDofs; ++i) {
            mKp[rootIdx + i] = 0.0;
            mKv[rootIdx + i] = 0.0;
        }
    }
}

void RenderCharacter::parseSkeletonMetadataFromXML(const std::string& resolvedPath)
{
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(resolvedPath.c_str()) != tinyxml2::XML_SUCCESS) {
        LOG_WARN("[RenderCharacter] Failed to parse XML skeleton metadata: " << resolvedPath);
        return;
    }

    tinyxml2::XMLElement* skel_elem = doc.FirstChildElement("Skeleton");
    if (!skel_elem) {
        LOG_WARN("[RenderCharacter] No <Skeleton> element found in: " << resolvedPath);
        return;
    }

    for (tinyxml2::XMLElement* node = skel_elem->FirstChildElement("Node");
         node != nullptr;
         node = node->NextSiblingElement("Node"))
    {
        const char* node_name_attr = node->Attribute("name");
        if (!node_name_attr) continue;
        std::string node_name = node_name_attr;

        // End effector flag
        const char* ee_attr = node->Attribute("endeffector");
        if (ee_attr && std::string(ee_attr) == "True") {
            auto bn = mSkeleton->getBodyNode(node_name);
            if (bn) {
                mEndEffectors.push_back(bn);
            }
        }

        // Body element: contact and obj
        tinyxml2::XMLElement* body_elem = node->FirstChildElement("Body");
        if (body_elem) {
            const char* contact_attr = body_elem->Attribute("contact");
            if (contact_attr) {
                mContactFlags[node_name] = std::string(contact_attr);
            }
            const char* obj_attr = body_elem->Attribute("obj");
            if (obj_attr) {
                mObjFileLabels[node_name] = std::string(obj_attr);
            }
        }

        // Joint element: bvh, kp, kv
        tinyxml2::XMLElement* joint_elem = node->FirstChildElement("Joint");
        if (!joint_elem) continue;

        // BVH mapping
        const char* bvh_attr = joint_elem->Attribute("bvh");
        if (bvh_attr) {
            auto bvh_list = split_string(bvh_attr);
            mBVHMap[node_name] = bvh_list;
        }

        // Skip Kp/Kv parsing for SKEL_FREE_JOINTS skeleton
        // (all joints become 6-DOF FreeJoints, so XML DOF values don't match)
        if (mSkelFlags & SKEL_FREE_JOINTS) {
            continue;
        }

        // Get joint DOF info
        auto jnt = mSkeleton->getJoint(node_name);
        if (!jnt) continue;
        int dof = jnt->getNumDofs();
        if (dof <= 0) continue;
        int idx = jnt->getIndexInSkeleton(0);

        // Bounds check
        if (idx < 0 || idx + dof > static_cast<int>(mKp.size())) {
            LOG_WARN("[RenderCharacter] XML Joint " << node_name << " idx=" << idx << " dof=" << dof
                     << " exceeds mKp size=" << mKp.size());
            continue;
        }

        // Kp/Kv parsing
        const char* kp_attr = joint_elem->Attribute("kp");
        if (kp_attr) {
            Eigen::VectorXd kp_vals = string_to_vectorXd(kp_attr, dof);
            for (int i = 0; i < dof; ++i) {
                if (idx + i >= static_cast<int>(mKp.size())) {
                    LOG_ERROR("[RenderCharacter] Index out of range: idx=" << idx << " i=" << i << " mKp.size=" << mKp.size());
                    break;
                }
                mKp[idx + i] = kp_vals[i];
            }
            const char* kv_attr = joint_elem->Attribute("kv");
            if (kv_attr) {
                Eigen::VectorXd kv_vals = string_to_vectorXd(kv_attr, dof);
                for (int i = 0; i < dof; ++i) {
                    if (idx + i >= static_cast<int>(mKv.size())) {
                        LOG_ERROR("[RenderCharacter] Index out of range for Kv: idx=" << idx << " i=" << i);
                        break;
                    }
                    mKv[idx + i] = kv_vals[i];
                }
            } else {
                for (int i = 0; i < dof; i++) {
                    if (idx + i >= static_cast<int>(mKv.size()) || idx + i >= static_cast<int>(mKp.size())) {
                        LOG_ERROR("[RenderCharacter] Index out of range for derived Kv");
                        break;
                    }
                    mKv[idx + i] = sqrt(2 * mKp[idx + i]);
                }
            }
        }
    }

    LOG_INFO("[RenderCharacter] Parsed XML metadata: " << mEndEffectors.size() << " end-effectors, "
             << mContactFlags.size() << " contact flags, " << mBVHMap.size() << " BVH mappings");
}

void RenderCharacter::parseSkeletonMetadataFromYAML(const std::string& resolvedPath)
{
    YAML::Node yaml_doc;
    try {
        yaml_doc = YAML::LoadFile(resolvedPath);
    } catch (const YAML::Exception& e) {
        LOG_WARN("[RenderCharacter] Failed to parse YAML skeleton metadata: " << resolvedPath);
        return;
    }

    if (!yaml_doc["skeleton"] || !yaml_doc["skeleton"]["nodes"]) {
        LOG_WARN("[RenderCharacter] No skeleton/nodes found in YAML: " << resolvedPath);
        return;
    }

    const YAML::Node& nodes = yaml_doc["skeleton"]["nodes"];
    for (size_t i = 0; i < nodes.size(); i++) {
        const YAML::Node& node = nodes[i];

        if (!node["name"]) continue;
        std::string node_name = node["name"].as<std::string>();

        // End effector flag
        if (node["ee"] && node["ee"].as<bool>()) {
            auto bn = mSkeleton->getBodyNode(node_name);
            if (bn) {
                mEndEffectors.push_back(bn);
            }
        }

        // Body element: contact and obj
        if (node["body"]) {
            const YAML::Node& body = node["body"];
            if (body["contact"]) {
                bool contact = body["contact"].as<bool>();
                mContactFlags[node_name] = contact ? "On" : "Off";
            }
            if (body["obj"]) {
                mObjFileLabels[node_name] = body["obj"].as<std::string>();
            }
        }

        // Joint element
        if (!node["joint"]) continue;
        const YAML::Node& joint = node["joint"];

        // BVH mapping
        if (joint["bvh"]) {
            std::vector<std::string> bvh_list;
            if (joint["bvh"].IsSequence()) {
                for (size_t j = 0; j < joint["bvh"].size(); j++) {
                    bvh_list.push_back(joint["bvh"][j].as<std::string>());
                }
            } else {
                bvh_list.push_back(joint["bvh"].as<std::string>());
            }
            mBVHMap[node_name] = bvh_list;
        }

        // Skip Kp/Kv parsing for SKEL_FREE_JOINTS skeleton
        // (all joints become 6-DOF FreeJoints, so YAML DOF values don't match)
        if (mSkelFlags & SKEL_FREE_JOINTS) {
            continue;
        }

        // Get joint DOF info
        auto jnt = mSkeleton->getJoint(node_name);
        if (!jnt) continue;
        int dof = jnt->getNumDofs();
        if (dof <= 0) continue;
        int idx = jnt->getIndexInSkeleton(0);

        // Bounds check
        if (idx + dof > static_cast<int>(mKp.size())) {
            LOG_WARN("[RenderCharacter] Joint " << node_name << " idx=" << idx << " dof=" << dof
                     << " exceeds mKp size=" << mKp.size());
            continue;
        }

        // Kp/Kv parsing
        if (joint["kp"]) {
            if (joint["kp"].IsSequence()) {
                for (int d = 0; d < dof && d < static_cast<int>(joint["kp"].size()); d++) {
                    mKp[idx + d] = joint["kp"][d].as<double>();
                }
            } else {
                double kp_val = joint["kp"].as<double>();
                for (int d = 0; d < dof; d++) {
                    mKp[idx + d] = kp_val;
                }
            }

            if (joint["kv"]) {
                if (joint["kv"].IsSequence()) {
                    for (int d = 0; d < dof && d < static_cast<int>(joint["kv"].size()); d++) {
                        mKv[idx + d] = joint["kv"][d].as<double>();
                    }
                } else {
                    double kv_val = joint["kv"].as<double>();
                    for (int d = 0; d < dof; d++) {
                        mKv[idx + d] = kv_val;
                    }
                }
            } else {
                for (int d = 0; d < dof; d++) {
                    mKv[idx + d] = sqrt(2 * mKp[idx + d]);
                }
            }
        }
    }

    LOG_INFO("[RenderCharacter] Parsed YAML metadata: " << mEndEffectors.size() << " end-effectors, "
             << mContactFlags.size() << " contact flags, " << mBVHMap.size() << " BVH mappings");
}

// ═══════════════════════════════════════════════════════════════════════════
// Skeleton Export (YAML)
// ═══════════════════════════════════════════════════════════════════════════

// Helper: format 3x3 matrix as YAML array
static std::string formatMatrixYAML(const Eigen::Matrix3d& M) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "[[";
    for (int i = 0; i < 3; i++) {
        if (i > 0) oss << ", ";
        oss << std::setw(7) << M(0, i);
    }
    oss << "], [";
    for (int i = 0; i < 3; i++) {
        if (i > 0) oss << ", ";
        oss << std::setw(7) << M(1, i);
    }
    oss << "], [";
    for (int i = 0; i < 3; i++) {
        if (i > 0) oss << ", ";
        oss << std::setw(7) << M(2, i);
    }
    oss << "]]";
    return oss.str();
}

// Helper: format Vector3d as YAML array
static std::string formatVectorYAML(const Eigen::Vector3d& v) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "[";
    for (int i = 0; i < 3; i++) {
        if (i > 0) oss << ", ";
        oss << std::setw(7) << v[i];
    }
    oss << "]";
    return oss.str();
}

// Helper: get shape type and size
static std::pair<std::string, Eigen::Vector3d> getShapeInfo(dart::dynamics::ShapePtr shape) {
    if (auto box = std::dynamic_pointer_cast<BoxShape>(shape)) {
        return {"Box", box->getSize()};
    } else if (auto sphere = std::dynamic_pointer_cast<SphereShape>(shape)) {
        double r = sphere->getRadius();
        return {"Sphere", Eigen::Vector3d(r, r, r)};
    } else if (auto capsule = std::dynamic_pointer_cast<CapsuleShape>(shape)) {
        double r = capsule->getRadius();
        double h = capsule->getHeight();
        return {"Capsule", Eigen::Vector3d(r, h, r)};
    } else if (auto cylinder = std::dynamic_pointer_cast<CylinderShape>(shape)) {
        double r = cylinder->getRadius();
        double h = cylinder->getHeight();
        return {"Cylinder", Eigen::Vector3d(r, h, r)};
    } else if (auto mesh = std::dynamic_pointer_cast<MeshShape>(shape)) {
        return {"Mesh", Eigen::Vector3d(0, 0, 0)};
    }
    return {"Box", Eigen::Vector3d(0.1, 0.1, 0.1)};
}

// Helper: get joint type string
static std::string getJointTypeString(Joint* joint) {
    if (dynamic_cast<FreeJoint*>(joint)) {
        return "Free";
    } else if (dynamic_cast<BallJoint*>(joint)) {
        return "Ball";
    } else if (dynamic_cast<RevoluteJoint*>(joint)) {
        return "Revolute";
    } else if (dynamic_cast<dart::dynamics::PrismaticJoint*>(joint)) {
        return "Prismatic";
    } else if (dynamic_cast<dart::dynamics::WeldJoint*>(joint)) {
        return "Weld";
    }
    return "Ball";
}

// Helper: format joint limits as YAML array
static std::string formatJointLimitsYAML(Joint* joint, bool isLower) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "[";
    size_t numDofs = joint->getNumDofs();
    for (size_t i = 0; i < numDofs; ++i) {
        if (i > 0) oss << ", ";
        oss << std::setw(7);
        if (isLower) {
            oss << joint->getPositionLowerLimit(i);
        } else {
            oss << joint->getPositionUpperLimit(i);
        }
    }
    oss << "]";
    return oss.str();
}

// Helper: format joint params (kp/kv) as YAML array
static std::string formatJointParamsYAML(Joint* joint, const std::string& param,
                                          const Eigen::VectorXd& kpVec, const Eigen::VectorXd& kvVec) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    oss << "[";

    size_t numDofs = joint->getNumDofs();
    Eigen::Index baseIndex = (numDofs > 0 && joint->getSkeleton()) ?
        static_cast<Eigen::Index>(joint->getIndexInSkeleton(0)) : 0;

    for (size_t i = 0; i < numDofs; ++i) {
        if (i > 0) oss << ", ";
        oss << std::setw(5);
        Eigen::Index gainIndex = baseIndex + static_cast<Eigen::Index>(i);
        double value = 0.0;
        if (param == "kp" && gainIndex < kpVec.size()) {
            value = kpVec[gainIndex];
        } else if (param == "kv" && gainIndex < kvVec.size()) {
            value = kvVec[gainIndex];
        }
        oss << value;
    }
    oss << "]";
    return oss.str();
}

// Helper: get current timestamp
static std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

void RenderCharacter::exportSkeletonYAML(const std::string& path) const
{
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        LOG_ERROR("[RenderCharacter] Failed to open file for export: " << path);
        return;
    }

    // Save current skeleton state and move to zero pose
    Eigen::VectorXd saved_positions = mSkeleton->getPositions();
    mSkeleton->setPositions(Eigen::VectorXd::Zero(mSkeleton->getNumDofs()));

    // Write metadata section
    ofs << "metadata:" << std::endl;
    ofs << "  generator: \"C3DProcessorApp\"" << std::endl;
    ofs << "  timestamp: \"" << getCurrentTimestamp() << "\"" << std::endl;
    ofs << "  version: v1" << std::endl;
    ofs << "  skeleton_from: \"" << mSkeletonPath << "\"" << std::endl;
    ofs << std::endl;

    // Write skeleton section
    ofs << "skeleton:" << std::endl;
    ofs << "  name: \"" << mSkeleton->getName() << "\"" << std::endl;
    ofs << "  nodes:" << std::endl;

    // Iterate through all body nodes
    auto bodyNodes = mSkeleton->getBodyNodes();
    for (auto bn : bodyNodes) {
        std::string nodeName = bn->getName();
        auto parent = bn->getParentBodyNode();
        std::string parentName = parent ? parent->getName() : "None";

        // Start node entry
        ofs << "    - {name: " << nodeName << ", parent: " << parentName;

        // End effector flag
        bool isEndEffector = std::find(mEndEffectors.begin(), mEndEffectors.end(), bn) != mEndEffectors.end();
        ofs << ", ee: " << (isEndEffector ? "True" : "false");

        // Body properties
        if (bn->getNumShapeNodes() > 0) {
            auto shapeNode = bn->getShapeNode(0);
            auto shape = shapeNode->getShape();
            auto [shapeType, shapeSize] = getShapeInfo(shape);
            double mass = bn->getMass();

            // Get local transform (relative to parent joint frame)
            Eigen::Isometry3d bodyTransform = bn->getRelativeTransform();

            // Contact flag
            std::string contact_label = "On";
            if (mContactFlags.count(nodeName)) {
                contact_label = mContactFlags.at(nodeName);
            }
            bool contact_bool = (contact_label != "Off");

            // Ensure mass is at least 0.01 to prevent simulation errors
            double exportMass = (mass < 0.01) ? 0.01 : mass;
            ofs << ", " << std::endl << "       body: {type: " << shapeType
                << ", mass: " << std::fixed << std::setprecision(2) << exportMass
                << ", size: " << formatVectorYAML(shapeSize)
                << ", contact: " << (contact_bool ? "true" : "false");

            // OBJ filename
            if (mObjFileLabels.count(nodeName)) {
                ofs << ", obj: \"" << mObjFileLabels.at(nodeName) << "\"";
            } else if (auto meshShape = std::dynamic_pointer_cast<MeshShape>(shape)) {
                std::string meshPath = meshShape->getMeshUri();
                if (!meshPath.empty()) {
                    size_t lastSlash = meshPath.find_last_of("/\\");
                    std::string meshFilename = (lastSlash != std::string::npos) ?
                        meshPath.substr(lastSlash + 1) : meshPath;
                    ofs << ", obj: \"" << meshFilename << "\"";
                }
            }

            ofs << "," << std::endl << "       R: " << formatMatrixYAML(bodyTransform.linear())
                << "," << std::endl << "       t: " << formatVectorYAML(bodyTransform.translation()) << "}";
        }

        // Joint properties
        auto joint = bn->getParentJoint();
        if (joint) {
            std::string jointType = getJointTypeString(joint);

            ofs << ", " << std::endl << std::endl << "       joint: {type: " << jointType;

            // BVH mapping
            if (mBVHMap.count(nodeName)) {
                const auto& bvhList = mBVHMap.at(nodeName);
                ofs << ", bvh: ";
                for (size_t i = 0; i < bvhList.size(); ++i) {
                    ofs << bvhList[i];
                    if (i < bvhList.size() - 1) ofs << " ";
                }
            }

            // Joint axis for Revolute/Prismatic
            if (auto revJoint = dynamic_cast<RevoluteJoint*>(joint)) {
                Eigen::Vector3d axis = revJoint->getAxis();
                ofs << ", axis: " << formatVectorYAML(axis);
            } else if (auto prisJoint = dynamic_cast<dart::dynamics::PrismaticJoint*>(joint)) {
                Eigen::Vector3d axis = prisJoint->getAxis();
                ofs << ", axis: " << formatVectorYAML(axis);
            }

            // Joint limits and gains for non-Free joints
            if (jointType != "Free" && joint->getNumDofs() > 0) {
                ofs << ", " << std::endl << "       lower: " << formatJointLimitsYAML(joint, true);
                ofs << "," << " upper: " << formatJointLimitsYAML(joint, false);
                ofs << "," << std::endl << "       kp: " << formatJointParamsYAML(joint, "kp", mKp, mKv);
                ofs << "," << " kv: " << formatJointParamsYAML(joint, "kv", mKp, mKv);
            }

            // Joint transform (local: relative to parent body frame)
            Eigen::Isometry3d jointTransform = joint->getTransformFromParentBodyNode();
            ofs << "," << std::endl << "       R: " << formatMatrixYAML(jointTransform.linear())
                << "," << std::endl << "       t: " << formatVectorYAML(jointTransform.translation()) << "}";
        }

        ofs << "}" << std::endl << std::endl;
    }

    ofs.close();

    // Restore original skeleton state
    mSkeleton->setPositions(saved_positions);

    LOG_INFO("[RenderCharacter] Exported skeleton with " << bodyNodes.size() << " nodes to: " << path);
}
