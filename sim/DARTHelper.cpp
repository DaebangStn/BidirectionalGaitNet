#include "DARTHelper.h"
#include <tinyxml2.h>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include "Log.h"

std::vector<std::string>
split_string(const std::string &input)
{
	std::vector<std::string> result;
	std::vector<int> blank_idx;
	blank_idx.push_back(-1);
	for (int i = 0; i < input.size(); i++)
	{
		if (input[i] == ' ')
			blank_idx.push_back(i);
	}
	blank_idx.push_back(input.size());

	for (int i = 0; i < blank_idx.size() - 1; i++)
		result.push_back(input.substr(blank_idx[i] + 1, blank_idx[i + 1] - blank_idx[i] - 1));
	return result;
}

ShapePtr
MakeSphereShape(double radius)
{
	return std::shared_ptr<SphereShape>(new SphereShape(radius));
}
ShapePtr
MakeBoxShape(const Eigen::Vector3d &size)
{
	return std::shared_ptr<BoxShape>(new BoxShape(size));
}
ShapePtr MakeCapsuleShape(double radius, double height)
{
	return std::shared_ptr<CapsuleShape>(new CapsuleShape(radius, height));
}
ShapePtr MakeCylinderShape(double radius, double height)
{
	return std::shared_ptr<CylinderShape>(new CylinderShape(radius, height));
}

dart::dynamics::Inertia MakeInertia(const dart::dynamics::ShapePtr &shape, double mass)
{
	dart::dynamics::Inertia inertia;

	inertia.setMass(mass);
	inertia.setMoment(shape->computeInertia(mass));
	return inertia;
}

FreeJoint::Properties *
MakeFreeJointProperties(const std::string &name, const Eigen::Isometry3d &parent_to_joint, const Eigen::Isometry3d &child_to_joint, const double damping)
{
	FreeJoint::Properties *props = new FreeJoint::Properties();
	props->mName = name;
	props->mT_ParentBodyToJoint = parent_to_joint;
	props->mT_ChildBodyToJoint = child_to_joint;
	props->mIsPositionLimitEnforced = false;
	props->mVelocityLowerLimits = Eigen::Vector6d::Constant(-100.0);
	props->mVelocityUpperLimits = Eigen::Vector6d::Constant(100.0);
	props->mDampingCoefficients = Eigen::Vector6d::Constant(damping);
	return props;
}
PlanarJoint::Properties *
MakePlanarJointProperties(const std::string &name, const Eigen::Isometry3d &parent_to_joint, const Eigen::Isometry3d &child_to_joint)
{
	PlanarJoint::Properties *props = new PlanarJoint::Properties();
	props->mName = name;
	props->mT_ParentBodyToJoint = parent_to_joint;
	props->mT_ChildBodyToJoint = child_to_joint;
	props->mIsPositionLimitEnforced = false;
	props->mVelocityLowerLimits = Eigen::Vector3d::Constant(-100.0);
	props->mVelocityUpperLimits = Eigen::Vector3d::Constant(100.0);
	props->mDampingCoefficients = Eigen::Vector3d::Constant(0.4);
	return props;
}

BallJoint::Properties* MakeBallJointProperties(const std::string &name, 
                                               const Eigen::Isometry3d &parent_to_joint, 
                                               const Eigen::Isometry3d &child_to_joint, 
                                               const Eigen::Vector3d &lower, 
                                               const Eigen::Vector3d &upper, 
                                               const double damping, 
                                               const double friction, 
                                               const Eigen::Vector3d &stiffness)
{
	BallJoint::Properties *props = new BallJoint::Properties();
	props->mName = name;
	props->mT_ParentBodyToJoint = parent_to_joint;
	props->mT_ChildBodyToJoint = child_to_joint;
	props->mIsPositionLimitEnforced = true;
	props->mPositionLowerLimits = lower;
	props->mPositionUpperLimits = upper;
	props->mVelocityLowerLimits = Eigen::Vector3d::Constant(-100.0);
	props->mVelocityUpperLimits = Eigen::Vector3d::Constant(100.0);
	props->mForceLowerLimits = Eigen::Vector3d::Constant(-1000.0);
	props->mForceUpperLimits = Eigen::Vector3d::Constant(1000.0);
	props->mDampingCoefficients = Eigen::Vector3d::Constant(damping);
	props->mFrictions = Eigen::Vector3d::Constant(friction);
	props->mSpringStiffnesses = stiffness;
	return props;
}

RevoluteJoint::Properties *
MakeRevoluteJointProperties(const std::string &name, const Eigen::Vector3d &axis, const Eigen::Isometry3d &parent_to_joint, const Eigen::Isometry3d &child_to_joint, const Eigen::Vector1d &lower, const Eigen::Vector1d &upper, const double damping, const double friction, const double stiffness)
{
	RevoluteJoint::Properties *props = new RevoluteJoint::Properties();
	props->mName = name;
	props->mT_ParentBodyToJoint = parent_to_joint;
	props->mT_ChildBodyToJoint = child_to_joint;
	props->mIsPositionLimitEnforced = true;
	props->mPositionLowerLimits = lower;
	props->mPositionUpperLimits = upper;
	props->mAxis = axis;
	props->mVelocityLowerLimits = Eigen::Vector1d::Constant(-100.0);
	props->mVelocityUpperLimits = Eigen::Vector1d::Constant(100.0);
	props->mForceLowerLimits = Eigen::Vector1d::Constant(-1000.0);
	props->mForceUpperLimits = Eigen::Vector1d::Constant(1000.0);
	props->mDampingCoefficients = Eigen::Vector1d::Constant(damping);
	props->mFrictions = Eigen::Vector1d::Constant(friction);
	props->mSpringStiffnesses = Eigen::Vector1d::Constant(stiffness);
	return props;
}
WeldJoint::Properties *
MakeWeldJointProperties(const std::string &name, const Eigen::Isometry3d &parent_to_joint, const Eigen::Isometry3d &child_to_joint)
{
	WeldJoint::Properties *props = new WeldJoint::Properties();
	props->mName = name;
	props->mT_ParentBodyToJoint = parent_to_joint;
	props->mT_ChildBodyToJoint = child_to_joint;
	return props;
}
BodyNode *
MakeBodyNode(const SkeletonPtr &skeleton, BodyNode *parent, Joint::Properties *joint_properties, const std::string &joint_type, dart::dynamics::Inertia inertia)
{
	BodyNode *bn;

	if (joint_type == "Free")
	{
		FreeJoint::Properties *prop = dynamic_cast<FreeJoint::Properties *>(joint_properties);
		bn = skeleton->createJointAndBodyNodePair<FreeJoint>(parent, (*prop), BodyNode::AspectProperties(joint_properties->mName)).second;
	}
	else if (joint_type == "Planar")
	{
		PlanarJoint::Properties *prop = dynamic_cast<PlanarJoint::Properties *>(joint_properties);
		bn = skeleton->createJointAndBodyNodePair<PlanarJoint>(parent, (*prop), BodyNode::AspectProperties(joint_properties->mName)).second;
	}
	else if (joint_type == "Ball")
	{
		BallJoint::Properties *prop = dynamic_cast<BallJoint::Properties *>(joint_properties);
		bn = skeleton->createJointAndBodyNodePair<BallJoint>(parent, (*prop), BodyNode::AspectProperties(joint_properties->mName)).second;
	}
	else if (joint_type == "Revolute")
	{
		RevoluteJoint::Properties *prop = dynamic_cast<RevoluteJoint::Properties *>(joint_properties);
		bn = skeleton->createJointAndBodyNodePair<RevoluteJoint>(parent, (*prop), BodyNode::AspectProperties(joint_properties->mName)).second;
	}
	else if (joint_type == "Weld")
	{
		WeldJoint::Properties *prop = dynamic_cast<WeldJoint::Properties *>(joint_properties);
		bn = skeleton->createJointAndBodyNodePair<WeldJoint>(parent, (*prop), BodyNode::AspectProperties(joint_properties->mName)).second;
	}

	bn->setInertia(inertia);
	return bn;
}
Eigen::Vector3d Proj(const Eigen::Vector3d &u, const Eigen::Vector3d &v)
{
	Eigen::Vector3d proj;
	proj = u.dot(v) / u.dot(u) * u;
	return proj;
}
Eigen::Isometry3d Orthonormalize(const Eigen::Isometry3d &T_old)
{
	Eigen::Isometry3d T;
	Eigen::Vector3d v0, v1, v2;
	Eigen::Vector3d u0, u1, u2;

	T.translation() = T_old.translation();

	v0 = T_old.linear().col(0);
	v1 = T_old.linear().col(1);
	v2 = T_old.linear().col(2);

	u0 = v0;
	u1 = v1 - Proj(u0, v1);
	u2 = v2 - Proj(u0, v2) - Proj(u1, v2);

	u0.normalize();
	u1.normalize();
	u2.normalize();

	T.linear().col(0) = u0;
	T.linear().col(1) = u1;
	T.linear().col(2) = u2;
	return T;
}
std::vector<double> split_to_double(const std::string &input, int num)
{
	std::vector<double> result;
	std::string::size_type sz = 0, nsz = 0;
	for (int i = 0; i < num; i++)
	{
		result.push_back(std::stof(input.substr(sz), &nsz));
		sz += nsz;
	}
	return result;
}
Eigen::Vector1d string_to_vector1d(const std::string &input)
{
	std::vector<double> v = split_to_double(input, 1);
	Eigen::Vector1d res;
	res << v[0];

	return res;
}
Eigen::Vector3d string_to_vector3d(const std::string &input)
{
	std::vector<double> v = split_to_double(input, 3);
	Eigen::Vector3d res;
	res << v[0], v[1], v[2];

	return res;
}
Eigen::Vector4d string_to_vector4d(const std::string &input)
{
	std::vector<double> v = split_to_double(input, 4);
	Eigen::Vector4d res;
	res << v[0], v[1], v[2], v[3];

	return res;
}
Eigen::VectorXd string_to_vectorXd(const std::string &input, int n)
{
	std::vector<double> v = split_to_double(input, n);
	Eigen::VectorXd res(n);
	for (int i = 0; i < n; i++)
		res[i] = v[i];

	return res;
}
Eigen::Matrix3d string_to_matrix3d(const std::string &input)
{
	std::vector<double> v = split_to_double(input, 9);
	Eigen::Matrix3d res;
	res << v[0], v[1], v[2],
		v[3], v[4], v[5],
		v[6], v[7], v[8];

	return res;
}

std::string
Trim(std::string str)
{
	str.erase(remove(str.begin(), str.end(), ' '), str.end());
	return str;
}

template <typename Derived>
static std::string formatVector(const Eigen::MatrixBase<Derived> &vec)
{
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(6) << "[";
	for (int i = 0; i < vec.size(); ++i)
	{
		if (i > 0)
			oss << ", ";
		oss << vec(i);
	}
	oss << "]";
	return oss.str();
}

static std::string formatMatrix3(const Eigen::Matrix3d &mat)
{
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(6) << "[";
	for (int r = 0; r < 3; ++r)
	{
		if (r > 0)
			oss << "; ";
		oss << "[";
		for (int c = 0; c < 3; ++c)
		{
			if (c > 0)
				oss << ", ";
			oss << mat(r, c);
		}
		oss << "]";
	}
	oss << "]";
	return oss.str();
}

static std::string formatIsometry(const Eigen::Isometry3d &T)
{
	std::ostringstream oss;
	oss << "{t=" << formatVector(T.translation()) << ", R=" << formatMatrix3(T.linear()) << "}";
	return oss.str();
}

static void logBodyNodeConfiguration(const dart::dynamics::BodyNode *bn)
{
	if (bn == nullptr)
		return;

	const auto *joint = bn->getParentJoint();
	Eigen::Isometry3d world = bn->getWorldTransform();
	std::string message = "[DARTHelper] Node '" + bn->getName() + "' world=" + formatIsometry(world);

	if (joint != nullptr)
	{
		message += " parent_to_joint=" + formatIsometry(joint->getTransformFromParentBodyNode());
		message += " child_to_joint=" + formatIsometry(joint->getTransformFromChildBodyNode());
		message += " q=" + formatVector(joint->getPositions());
	}
	else
	{
		message += " (no parent joint)";
	}

	LOG_INFO(message);
}

// Forward declaration
dart::dynamics::SkeletonPtr BuildFromYAML(const std::string &path, int flags);

// YAML helper functions for skeleton loading
static Eigen::Vector3d yaml_to_vector3d(const YAML::Node& node) {
	if (!node.IsSequence() || node.size() != 3) {
		throw std::runtime_error("Expected 3-element array for Vector3d");
	}
	return Eigen::Vector3d(node[0].as<double>(),
	                       node[1].as<double>(),
	                       node[2].as<double>());
}

static Eigen::Matrix3d yaml_to_matrix3d(const YAML::Node& node) {
	if (!node.IsSequence() || node.size() != 3) {
		throw std::runtime_error("Expected 3x3 array for Matrix3d");
	}
	Eigen::Matrix3d mat;
	for (int i = 0; i < 3; i++) {
		if (!node[i].IsSequence() || node[i].size() != 3) {
			throw std::runtime_error("Expected 3 elements per row in Matrix3d");
		}
		for (int j = 0; j < 3; j++) {
			mat(i, j) = node[i][j].as<double>();
		}
	}
	return mat;
}

static Eigen::Isometry3d yaml_to_transform(const YAML::Node& R, const YAML::Node& t) {
	Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
	T.linear() = yaml_to_matrix3d(R);
	T.translation() = yaml_to_vector3d(t);
	return Orthonormalize(T);
}

dart::dynamics::SkeletonPtr BuildFromXML(const std::string &path, int flags)
{
	std::string resolvedPath = PMuscle::URIResolver::getInstance().resolve(path);
	
	// Hardcoded default damping
	const double defaultDamping = 0.4;

	// Extract flags
	bool isContact = (flags & SKEL_NO_COLLISION) == 0;  // Collision enabled unless NO_COLLISION flag set
	bool collide_all = (flags & SKEL_COLLIDE_ALL) != 0;
	bool isBVH = (flags & SKEL_REMOVE_JOINT_LIMIT) != 0;

	TiXmlDocument doc;
	LOG_VERBOSE("[DARTHelper] Building skeleton from file : " << resolvedPath);
	if (doc.LoadFile(resolvedPath.c_str()))
	{
		std::cerr << "[DARTHelper] Failed to load file" << std::endl;
		std::cerr << "[DARTHelper] Error : " << doc.ErrorName() << std::endl;
		std::cerr << "[DARTHelper] Error Str : " << doc.ErrorStr() << std::endl;
		return nullptr;
	}

	TiXmlElement *skeleton_elem = doc.FirstChildElement("Skeleton");
	std::string skel_name = skeleton_elem->Attribute("name");
	SkeletonPtr skel = Skeleton::create(skel_name);

	for (TiXmlElement *node = skeleton_elem->FirstChildElement("Node"); node != nullptr; node = node->NextSiblingElement("Node"))
	{
		std::string name = node->Attribute("name");
		std::string parent_str = node->Attribute("parent");
		BodyNode *parent = nullptr;
		if (parent_str != "None")
			parent = skel->getBodyNode(parent_str);

		ShapePtr shape;
		Eigen::Isometry3d T_body = Eigen::Isometry3d::Identity();
		TiXmlElement *body = node->FirstChildElement("Body");
		std::string type = body->Attribute("type");
		std::string obj_file = "None";
		if (body->Attribute("obj"))
			obj_file = body->Attribute("obj");

		double mass = std::stod(body->Attribute("mass"));

		if (type == "Box")
		{
			Eigen::Vector3d size = string_to_vector3d(body->Attribute("size"));
			shape = MakeBoxShape(size);
		}
		else if (type == "Sphere")
		{
			double radius = std::stod(body->Attribute("radius"));
			shape = MakeSphereShape(radius);
		}
		else if (type == "Capsule")
		{
			double radius = std::stod(body->Attribute("radius"));
			double height = std::stod(body->Attribute("height"));
			shape = MakeCapsuleShape(radius, height);
		}
		else if (type == "Cylinder")
		{
			double radius = std::stod(body->Attribute("radius"));
			double height = std::stod(body->Attribute("height"));
			shape = MakeCylinderShape(radius, height);
		}

		bool contact = false;
		if (body->Attribute("contact") != nullptr)
		{
			std::string c = body->Attribute("contact");
			if (c == "On") contact = true & isContact;
		}		
		// Apply collide_all override - this should enable collision for ALL body nodes when collide_all=true
		contact |= collide_all;

		Eigen::Vector4d color = Eigen::Vector4d::Constant(0.2);
		if (body->Attribute("color") != nullptr)
			color = string_to_vector4d(body->Attribute("color"));

		dart::dynamics::Inertia inertia = MakeInertia(shape, mass);
		T_body.linear() = string_to_matrix3d(body->FirstChildElement("Transformation")->Attribute("linear"));
		T_body.translation() = string_to_vector3d(body->FirstChildElement("Transformation")->Attribute("translation"));
		T_body = Orthonormalize(T_body);
		TiXmlElement *joint = node->FirstChildElement("Joint");
		type = joint->Attribute("type");
		Joint::Properties *props;

		Eigen::Isometry3d T_joint = Eigen::Isometry3d::Identity();
		T_joint.linear() = string_to_matrix3d(joint->FirstChildElement("Transformation")->Attribute("linear"));
		T_joint.translation() = string_to_vector3d(joint->FirstChildElement("Transformation")->Attribute("translation"));
		T_joint = Orthonormalize(T_joint);

		Eigen::Isometry3d parent_to_joint;
		if (parent == nullptr)
			parent_to_joint = T_joint;
		else
			parent_to_joint = parent->getTransform().inverse() * T_joint;

		Eigen::Isometry3d child_to_joint = T_body.inverse() * T_joint;
		if (type == "Free")
		{
			double damping = defaultDamping;
			if (joint->Attribute("damping") != NULL)
				damping = std::stod(joint->Attribute("damping"));
			props = MakeFreeJointProperties(name, parent_to_joint, child_to_joint, damping);
		}
		else if (type == "Planar")
		{
			props = MakePlanarJointProperties(name, parent_to_joint, child_to_joint);
		}
		else if (type == "Weld")
		{
			props = MakeWeldJointProperties(name, parent_to_joint, child_to_joint);
		}
		else if (type == "Ball" || (isBVH && name.find("ForeArm") == std::string::npos))
		{
			Eigen::Vector3d lower; // = string_to_vector3d(joint->Attribute("lower"));
			Eigen::Vector3d upper; // = string_to_vector3d(joint->Attribute("upper"));
			if (isBVH)
			{
				lower = Eigen::Vector3d(-3.14, -3.14, -3.14);
				upper = Eigen::Vector3d(3.14, 3.14, 3.14);
				type = "Ball";
			}
			else
			{
				lower = string_to_vector3d(joint->Attribute("lower"));
				upper = string_to_vector3d(joint->Attribute("upper"));
			}
			double damping = defaultDamping;
			double friction = 0;
			Eigen::Vector3d stiffness = Eigen::Vector3d::Zero(3);
			if (joint->Attribute("damping") != NULL)
				damping = std::stod(joint->Attribute("damping"));

			if (joint->Attribute("friction") != NULL)
				friction = std::stod(joint->Attribute("friction"));

			if (joint->Attribute("stiffness") != NULL)
				stiffness = string_to_vector3d(joint->Attribute("stiffness"));

			props = MakeBallJointProperties(name, parent_to_joint, child_to_joint, lower, upper, damping, friction, stiffness);
		}
		else if (type == "Revolute")
		{

			Eigen::Vector1d lower = string_to_vector1d(joint->Attribute("lower"));
			Eigen::Vector1d upper = string_to_vector1d(joint->Attribute("upper"));
			Eigen::Vector3d axis = string_to_vector3d(joint->Attribute("axis"));
			double damping = defaultDamping;
			double friction = 0;
			double stiffness = 0;
			if (joint->Attribute("damping") != NULL)
				damping = std::stod(joint->Attribute("damping"));

			if (joint->Attribute("friction") != NULL)
				friction = std::stod(joint->Attribute("friction"));

			if (joint->Attribute("stiffness") != NULL)
				stiffness = std::stod(joint->Attribute("stiffness"));

			props = MakeRevoluteJointProperties(name, axis, parent_to_joint, child_to_joint, lower, upper, damping, friction, stiffness);
		}

		auto bn = MakeBodyNode(skel, parent, props, type, inertia);
		if (contact) bn->createShapeNodeWith<VisualAspect, CollisionAspect, DynamicsAspect>(shape);
		else bn->createShapeNodeWith<VisualAspect, DynamicsAspect>(shape);

		dart::dynamics::ShapeNode* lastShapeNode = nullptr;
		bn->eachShapeNodeWith<VisualAspect>([&lastShapeNode](dart::dynamics::ShapeNode* sn) {
			lastShapeNode = sn;
			return true;
		});
		if (lastShapeNode) lastShapeNode->getVisualAspect()->setColor(color);

		if (obj_file != "None")
		{
			std::string obj_uri = "@data/skeleton/OBJ/" + obj_file;
			std::string obj_path = PMuscle::URIResolver::getInstance().resolve(obj_uri);
			const aiScene *scene = MeshShape::loadMesh(std::string(obj_path));

			MeshShapePtr visual_shape = std::shared_ptr<MeshShape>(new MeshShape(Eigen::Vector3d(0.01, 0.01, 0.01), scene));
			visual_shape->setColorMode(MeshShape::ColorMode::SHAPE_COLOR);
			auto vsn = bn->createShapeNodeWith<VisualAspect>(visual_shape);

			Eigen::Isometry3d T_obj;
			T_obj.setIdentity();
			T_obj = T_body.inverse();
			vsn->setRelativeTransform(T_obj);
		}

		// logBodyNodeConfiguration(bn);
	}
	return skel;
}

dart::dynamics::SkeletonPtr BuildFromFile(const std::string &path, int flags)
{
	std::string resolvedPath = PMuscle::URIResolver::getInstance().resolve(path);

	// Detect format from file extension
	std::string ext;
	size_t dot_pos = resolvedPath.find_last_of('.');
	if (dot_pos != std::string::npos) {
		ext = resolvedPath.substr(dot_pos);
		std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
	}

	// Route to appropriate parser
	if (ext == ".yaml" || ext == ".yml") {
		return BuildFromYAML(path, flags);
	} else if (ext == ".xml") {
		return BuildFromXML(path, flags);
	}

	LOG_ERROR("[DARTHelper] Unsupported file extension: " << ext);
	return nullptr;
}

dart::dynamics::SkeletonPtr BuildFromYAML(const std::string &path, int flags)
{
	// Hardcoded default damping
	const double defaultDamping = 0.4;

	// Extract flags
	bool isContact = (flags & SKEL_NO_COLLISION) == 0;
	bool collide_all = (flags & SKEL_COLLIDE_ALL) != 0;
	bool isBVH = (flags & SKEL_REMOVE_JOINT_LIMIT) != 0;

	std::string resolvedPath = PMuscle::URIResolver::getInstance().resolve(path);
	LOG_VERBOSE("[DARTHelper] Building skeleton from YAML file : " << resolvedPath);

	YAML::Node doc = YAML::LoadFile(resolvedPath);

	YAML::Node skel_node = doc["skeleton"];
	std::string skel_name = skel_node["name"].as<std::string>();
	SkeletonPtr skel = Skeleton::create(skel_name);

	YAML::Node nodes = skel_node["nodes"];
	for (const auto& node : nodes) {
		std::string name = node["name"].as<std::string>();
		std::string parent_str = node["parent"].as<std::string>();
		BodyNode *parent = nullptr;
		if (parent_str != "None")
			parent = skel->getBodyNode(parent_str);

		ShapePtr shape;
		Eigen::Isometry3d T_body = Eigen::Isometry3d::Identity();
		YAML::Node body = node["body"];
		std::string type = body["type"].as<std::string>();
		double mass = body["mass"].as<double>();

		if (type == "Box") {
			Eigen::Vector3d size = yaml_to_vector3d(body["size"]);
			shape = MakeBoxShape(size);
		}
		else if (type == "Sphere") {
			double radius = body["radius"].as<double>();
			shape = MakeSphereShape(radius);
		}
		else if (type == "Capsule") {
			double radius = body["radius"].as<double>();
			double height = body["height"].as<double>();
			shape = MakeCapsuleShape(radius, height);
		}
		else if (type == "Cylinder") {
			double radius = body["radius"].as<double>();
			double height = body["height"].as<double>();
			shape = MakeCylinderShape(radius, height);
		}

		bool contact = false;
		if (body["contact"]) {
			contact = body["contact"].as<bool>() & isContact;
		}
		contact |= collide_all;

		Eigen::Vector4d color = Eigen::Vector4d::Constant(0.2);

		dart::dynamics::Inertia inertia = MakeInertia(shape, mass);
		T_body = yaml_to_transform(body["R"], body["t"]);

		YAML::Node joint = node["joint"];
		type = joint["type"].as<std::string>();
		Joint::Properties *props;

		Eigen::Isometry3d T_joint = yaml_to_transform(joint["R"], joint["t"]);

		// T_joint and T_body are LOCAL transforms (relative to parent)
		// parent_to_joint: joint transform relative to parent body (already in T_joint)
		// child_to_joint: joint transform relative to child body
		// Formula T_body.inverse() * T_joint works for both GLOBAL (XML) and LOCAL (YAML) cases
		Eigen::Isometry3d parent_to_joint = T_joint;
		Eigen::Isometry3d child_to_joint = T_body.inverse() * T_joint;

		if (type == "Free") {
			double damping = defaultDamping;
			if (joint["kv"]) {
				damping = joint["kv"][0].as<double>();
			}
			props = MakeFreeJointProperties(name, parent_to_joint, child_to_joint, damping);
		}
		else if (type == "Planar") {
			props = MakePlanarJointProperties(name, parent_to_joint, child_to_joint);
		}
		else if (type == "Weld") {
			props = MakeWeldJointProperties(name, parent_to_joint, child_to_joint);
		}
		else if (type == "Ball" || (isBVH && name.find("ForeArm") == std::string::npos)) {
			Eigen::Vector3d lower;
			Eigen::Vector3d upper;
			if (isBVH) {
				lower = Eigen::Vector3d(-3.14, -3.14, -3.14);
				upper = Eigen::Vector3d(3.14, 3.14, 3.14);
				type = "Ball";
			}
			else {
				lower = yaml_to_vector3d(joint["lower"]);
				upper = yaml_to_vector3d(joint["upper"]);
			}
			double damping = defaultDamping;
			double friction = 0;
			Eigen::Vector3d stiffness = Eigen::Vector3d::Zero(3);
			if (joint["damping"]) {
				damping = joint["damping"][0].as<double>();
			}

			props = MakeBallJointProperties(name, parent_to_joint, child_to_joint, lower, upper, damping, friction, stiffness);
		}
		else if (type == "Revolute") {
			Eigen::Vector1d lower(joint["lower"][0].as<double>());
			Eigen::Vector1d upper(joint["upper"][0].as<double>());
			Eigen::Vector3d axis = yaml_to_vector3d(joint["axis"]);
			double damping = defaultDamping;
			double friction = 0;
			double stiffness = 0;
			if (joint["damping"]) {
				damping = joint["damping"][0].as<double>();
			}

			props = MakeRevoluteJointProperties(name, axis, parent_to_joint, child_to_joint, lower, upper, damping, friction, stiffness);
		}

		auto bn = MakeBodyNode(skel, parent, props, type, inertia);
		if (contact) bn->createShapeNodeWith<VisualAspect, CollisionAspect, DynamicsAspect>(shape);
		else bn->createShapeNodeWith<VisualAspect, DynamicsAspect>(shape);

		dart::dynamics::ShapeNode* lastShapeNode = nullptr;
		bn->eachShapeNodeWith<VisualAspect>([&lastShapeNode](dart::dynamics::ShapeNode* sn) {
			lastShapeNode = sn;
			return true;
		});
		if (lastShapeNode) lastShapeNode->getVisualAspect()->setColor(color);

		if (body["obj"]) {
			std::string obj_file = body["obj"].as<std::string>();
			std::string obj_uri = "@data/skeleton/OBJ/" + obj_file;
			std::string obj_path = PMuscle::URIResolver::getInstance().resolve(obj_uri);
			const aiScene *scene = MeshShape::loadMesh(std::string(obj_path));

			MeshShapePtr visual_shape = std::shared_ptr<MeshShape>(new MeshShape(Eigen::Vector3d(0.01, 0.01, 0.01), scene));
			visual_shape->setColorMode(MeshShape::ColorMode::SHAPE_COLOR);
			auto vsn = bn->createShapeNodeWith<VisualAspect>(visual_shape);

			// Visual mesh OBJ files are modeled in zero pose world coordinates
			// We need to transform from zero pose world to body-local frame
			// Get body's world transform in zero pose (just computed by DART)
			Eigen::Isometry3d T_body_world = bn->getWorldTransform();
			Eigen::Isometry3d T_obj = T_body_world.inverse();
			vsn->setRelativeTransform(T_obj);
		}

		// logBodyNodeConfiguration(bn);
	}
	return skel;
}
