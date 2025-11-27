#ifndef __MS_DARTHELPER_H__
#define __MS_DARTHELPER_H__
#include "dart/dart.hpp"
#include <tinyxml2.h>
#include <yaml-cpp/yaml.h>

#include <experimental/filesystem>
#include "UriResolver.h"

using namespace dart::dynamics;

typedef tinyxml2::XMLElement TiXmlElement;
typedef tinyxml2::XMLDocument TiXmlDocument;

namespace dart_fs = std::experimental::filesystem;

namespace Eigen
{
    using Vector1d = Matrix<double, 1, 1>;
    using Matrix1d = Matrix<double, 1, 1>;
}

std::vector<std::string> split_string(const std::string &input);

std::vector<double> split_to_double(const std::string &input, int num);
Eigen::Vector1d string_to_vector1d(const std::string &input);
Eigen::Vector3d string_to_vector3d(const std::string &input);
Eigen::Vector4d string_to_vector4d(const std::string &input);
Eigen::VectorXd string_to_vectorXd(const std::string &input, int n);
Eigen::Matrix3d string_to_matrix3d(const std::string &input);

dart::dynamics::ShapePtr MakeSphereShape(double radius);
dart::dynamics::ShapePtr MakeBoxShape(const Eigen::Vector3d &size);
dart::dynamics::ShapePtr MakeCapsuleShape(double radius, double height);
dart::dynamics::ShapePtr MakeCylinderShape(double radius, double height);
dart::dynamics::Inertia MakeInertia(const dart::dynamics::ShapePtr &shape, double mass);
dart::dynamics::FreeJoint::Properties *MakeFreeJointProperties(const std::string &name, const Eigen::Isometry3d &parent_to_joint = Eigen::Isometry3d::Identity(), const Eigen::Isometry3d &child_to_joint = Eigen::Isometry3d::Identity(), const double damping = 0.4);
dart::dynamics::PlanarJoint::Properties *MakePlanarJointProperties(const std::string &name, const Eigen::Isometry3d &parent_to_joint = Eigen::Isometry3d::Identity(), const Eigen::Isometry3d &child_to_joint = Eigen::Isometry3d::Identity());
dart::dynamics::BallJoint::Properties* MakeBallJointProperties(const std::string &name, const Eigen::Isometry3d &parent_to_joint = Eigen::Isometry3d::Identity(), const Eigen::Isometry3d &child_to_joint = Eigen::Isometry3d::Identity(), const Eigen::Vector3d &lower = Eigen::Vector3d::Constant(-2.0), const Eigen::Vector3d &upper = Eigen::Vector3d::Constant(2.0), const double damping = 0.4, const double friction = 0.0, const Eigen::Vector3d &stiffness = Eigen::Vector3d::Zero());
dart::dynamics::RevoluteJoint::Properties *MakeRevoluteJointProperties(const std::string &name, const Eigen::Vector3d &axis, const Eigen::Isometry3d &parent_to_joint = Eigen::Isometry3d::Identity(), const Eigen::Isometry3d &child_to_joint = Eigen::Isometry3d::Identity(), const Eigen::Vector1d &lower = Eigen::Vector1d::Constant(-2.0), const Eigen::Vector1d &upper = Eigen::Vector1d::Constant(2.0), const double damping = 0.4, const double friction = 0.0, const double stiffness = 0.0);
dart::dynamics::WeldJoint::Properties *MakeWeldJointProperties(const std::string &name, const Eigen::Isometry3d &parent_to_joint = Eigen::Isometry3d::Identity(), const Eigen::Isometry3d &child_to_joint = Eigen::Isometry3d::Identity());
dart::dynamics::BodyNode *MakeBodyNode(const dart::dynamics::SkeletonPtr &skeleton, dart::dynamics::BodyNode *parent, dart::dynamics::Joint::Properties *joint_properties, const std::string &joint_type, dart::dynamics::Inertia inertia);

// Skeleton loading flags (bit flags)
enum SkeletonLoadFlags {
	SKEL_DEFAULT = 0,                      // Default: collision enabled, respects XML
	SKEL_NO_COLLISION = 1 << 0,            // Disable all collision detection
	SKEL_COLLIDE_ALL = 1 << 1,             // Force all bodies to collide (self-collision)
	SKEL_REMOVE_JOINT_LIMIT = 1 << 2,      // BVH mode - remove joint limits
	SKEL_FREE_JOINTS = 1 << 3              // Debug mode - all joints become FreeJoint (6 DOF)
};

dart::dynamics::SkeletonPtr BuildFromXML(const std::string &path, int flags = SKEL_DEFAULT);
dart::dynamics::SkeletonPtr BuildFromFile(const std::string &path, int flags = SKEL_DEFAULT);

std::string Trim(std::string str);
#endif
