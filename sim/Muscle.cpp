#include "Muscle.h"
#include "Log.h"

using namespace dart::dynamics;
std::vector<int> sort_indices(const std::vector<double> &val)
{
    std::vector<int> idx(val.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(), [&val](int i1, int i2)
              { return val[i1] < val[i2]; });

    return idx;
}
Anchor::Anchor(std::vector<BodyNode *> bns, std::vector<Eigen::Vector3d> lps, std::vector<double> ws)
    : bodynodes(bns), local_positions(lps), weights(ws), num_related_bodies(bns.size())
{
}

Eigen::Vector3d Anchor::GetPoint() const
{
    Eigen::Vector3d p;
    p.setZero();
    for (int i = 0; i < num_related_bodies; i++) p += weights[i] * (bodynodes[i]->getTransform() * local_positions[i]);
    return p;
}

Muscle::Muscle(std::string _name, double _f0, double _lm0, double _lt0, double _pen_angle, 
    double _type1_fraction, bool useVelocityForce)
    : selected(false), mUseVelocityForce(useVelocityForce), name(_name), 
    f0_base(_f0), f0(_f0), pen_angle(_pen_angle), lm_opt(_lm0), lt_rel(_lt0), 
    v_m(0.0), lmt_ref(0.0), lmt_rel(1.0), activation(0.0), f_toe(0.33), k_toe(3.0), k_lin(51.878788), 
    e_toe(0.02), e_t0(0.033), k_pe(5.5), e_mo(0.3), gamma(0.45), type1_fraction(_type1_fraction), l_ratio(1.0), f_ratio(1.0),
    lt_rel_base(_lt0), lt_rel_ofs(0.0)
{
    lm_rel = lmt_rel - lt_rel;
}

void Muscle::AddAnchor(const dart::dynamics::SkeletonPtr &skel, dart::dynamics::BodyNode *bn, const Eigen::Vector3d &glob_pos, int num_related_bodies, bool meshLbsWeight)
{
    std::vector<double> distance;
    std::vector<Eigen::Vector3d> local_positions;
    distance.resize(skel->getNumBodyNodes(), 0.0);
    local_positions.resize(skel->getNumBodyNodes());
    for (int i = 0; i < skel->getNumBodyNodes(); i++)
    {
        Eigen::Isometry3d T;
        T = skel->getBodyNode(i)->getTransform() * skel->getBodyNode(i)->getParentJoint()->getTransformFromChildBodyNode();
        local_positions[i] = skel->getBodyNode(i)->getTransform().inverse() * glob_pos;
        distance[i] = (glob_pos - T.translation()).norm();
    }

    std::vector<int> index_sort_by_distance = sort_indices(distance);
    std::vector<dart::dynamics::BodyNode *> lbs_body_nodes;
    std::vector<Eigen::Vector3d> lbs_local_positions;
    std::vector<double> lbs_weights;

    double total_weight = 0.0;

    if (distance[index_sort_by_distance[0]] < 0.08)
    {
        // Calculate Distance From Mesh
        if (meshLbsWeight)
        {
            double min_distance = 100000000.0;
            BodyNode *bn_current = skel->getBodyNode(index_sort_by_distance[0]);
            bn_current->eachShapeNodeWith<VisualAspect>([&](dart::dynamics::ShapeNode* sn) {
                if (sn->getShape()->is<MeshShape>())
                {
                    auto shape = std::dynamic_pointer_cast<MeshShape>(sn->getShape());
                    for (int idx = 0; idx < shape->getMesh()->mMeshes[0]->mNumVertices; idx++)
                    {
                        Eigen::Vector3d pos = 0.01 *
                                              Eigen::Vector3d(
                                                  shape->getMesh()->mMeshes[0]->mVertices[idx][0],
                                                  shape->getMesh()->mMeshes[0]->mVertices[idx][1],
                                                  shape->getMesh()->mMeshes[0]->mVertices[idx][2]);
                        if ((glob_pos - pos).norm() < min_distance)
                            min_distance = (glob_pos - pos).norm(); // pow((glob_pos - pos).squaredNorm(), 2);
                    }
                }
                return true;
            });
            distance[bn_current->getIndexInSkeleton()] = min_distance;
        }

        lbs_weights.push_back(1.0 / sqrt(distance[index_sort_by_distance[0]]));
        total_weight += lbs_weights[0];
        lbs_body_nodes.push_back(skel->getBodyNode(index_sort_by_distance[0]));
        lbs_local_positions.push_back(local_positions[index_sort_by_distance[0]]);

        if (lbs_body_nodes[0]->getParentBodyNode() != nullptr)
        {
            BodyNode *bn_current = lbs_body_nodes[0]->getParentBodyNode();

            // Calculate Distance From Mesh
            if (meshLbsWeight)
            {
                double min_distance = 100000000.0;
                bn_current->eachShapeNodeWith<VisualAspect>([&](dart::dynamics::ShapeNode* sn) {
                    if (sn->getShape()->is<MeshShape>())
                    {
                        auto shape = std::dynamic_pointer_cast<MeshShape>(sn->getShape());
                        for (int idx = 0; idx < shape->getMesh()->mMeshes[0]->mNumVertices; idx++)
                        {
                            Eigen::Vector3d pos = 0.01 *
                                                  Eigen::Vector3d(
                                                      shape->getMesh()->mMeshes[0]->mVertices[idx][0],
                                                      shape->getMesh()->mMeshes[0]->mVertices[idx][1],
                                                      shape->getMesh()->mMeshes[0]->mVertices[idx][2]);
                            if ((glob_pos - pos).norm() < min_distance)
                                min_distance = (glob_pos - pos).norm();
                        }
                    }
                    return true;
                });
                distance[bn_current->getIndexInSkeleton()] = min_distance;
            }

            lbs_weights.push_back(1.0 / sqrt(distance[bn_current->getIndexInSkeleton()]));
            total_weight += lbs_weights[1];
            lbs_body_nodes.push_back(bn_current);
            lbs_local_positions.push_back(local_positions[bn_current->getIndexInSkeleton()]);
        }
    }
    else
    {
        total_weight = 1.0;
        lbs_weights.push_back(1.0);
        lbs_body_nodes.push_back(bn);
        lbs_local_positions.push_back(bn->getTransform().inverse() * glob_pos);
    }

    for (int i = 0; i < lbs_body_nodes.size(); i++)
        lbs_weights[i] /= total_weight;
    mAnchors.push_back(new Anchor(lbs_body_nodes, lbs_local_positions, lbs_weights));
}
void Muscle::AddAnchor(dart::dynamics::BodyNode *bn, const Eigen::Vector3d &glob_pos)
{
    std::vector<dart::dynamics::BodyNode *> lbs_body_nodes;
    std::vector<Eigen::Vector3d> lbs_local_positions;
    std::vector<double> lbs_weights;

    lbs_body_nodes.push_back(bn);
    lbs_local_positions.push_back(bn->getTransform().inverse() * glob_pos);
    lbs_weights.push_back(1.0);

    mAnchors.push_back(new Anchor(lbs_body_nodes, lbs_local_positions, lbs_weights));
}
void Muscle::SetMuscle()
{
    int n = mAnchors.size();
    mCachedAnchorPositions.resize(n);

    lmt_ref = 0;
    for (int i = 1; i < n; i++) lmt_ref += (mAnchors[i]->GetPoint() - mAnchors[i - 1]->GetPoint()).norm();
    lmt_base = lmt_ref;
    lmt = lmt_ref;

    UpdateGeometry();
    Eigen::MatrixXd Jt = GetJacobianTranspose();
    auto Ap = GetForceJacobianAndPassive();
    Eigen::VectorXd JtA = Jt * Ap.first;
    num_related_dofs = 0;
    related_dof_indices.clear();

    related_vec = Eigen::VectorXd::Zero(JtA.rows());

    for (int i = 0; i < JtA.rows(); i++)
    {
        if (std::abs(JtA[i]) > 1E-10)
        {
            num_related_dofs++;
            related_dof_indices.push_back(i);
        }
        if (JtA[i] > 1E-10) related_vec[i] = 1;
        else if (JtA[i] < -1E-10) related_vec[i] = -1;
        else related_vec[i] = 0;
    }
    bool isValid = true;
    if (original_related_dof_indices.size() > 0)
    {
        for (int i = 0; i < related_dof_indices.size(); i++)
        {
            bool isExist = false;
            for (auto idx : original_related_dof_indices)
            {
                if (related_dof_indices[i] == idx) isExist = true;
            }
            if (isExist == false)
            {
                isValid = false;
                break;
            }
        }
    }
    else for (auto i : related_dof_indices) original_related_dof_indices.push_back(i);

    if (isValid == false)
    {
        std::cout << "MUSCLE RELATED DOF CHANGED " << name << std::endl;
        exit(-1);
    }
}

void Muscle::RefreshMuscleParams()
{
    lt_rel = lt_rel_base + lt_rel_ofs;
    lmt_ref = lmt_base * l_ratio;
    f0 = f0_base * f_ratio;
}

void Muscle::ApplyForceToBody()
{
    double f = GetForce();

    for (int i = 0; i < mAnchors.size() - 1; i++)
    {
        Eigen::Vector3d dir = mCachedAnchorPositions[i + 1] - mCachedAnchorPositions[i];
        dir.normalize();
        dir = f * dir;
        mAnchors[i]->bodynodes[0]->addExtForce(dir, mCachedAnchorPositions[i], false, false);
    }

    for (int i = 1; i < mAnchors.size(); i++)
    {
        Eigen::Vector3d dir = mCachedAnchorPositions[i - 1] - mCachedAnchorPositions[i];
        dir.normalize();
        dir = f * dir;
        mAnchors[i]->bodynodes[0]->addExtForce(dir, mCachedAnchorPositions[i], false, false);
    }
}

bool Muscle::UpdateGeometry()
{
    for (int i = 0; i < mAnchors.size(); i++) mCachedAnchorPositions[i] = mAnchors[i]->GetPoint();
    lmt = 0.0;
    for (int i = 1; i < mAnchors.size(); i++) lmt += (mCachedAnchorPositions[i] - mCachedAnchorPositions[i - 1]).norm();
    lmt_rel = lmt / lmt_ref;
    lm_rel = lmt_rel - lt_rel;
    if (mUseVelocityForce) UpdateVelocities();
    lm_norm = lm_rel / (lm_opt + 1e-3);
    return lm_norm < 1.5; // Return whether the joint angle is in ROM
}

// Should be called after Update called
void Muscle::UpdateVelocities()
{
    const auto &skel = mAnchors[0]->bodynodes[0]->getSkeleton();
    // ComputeJacobians();
    mCachedAnchorVelocities.clear();
    for (int i = 0; i < mAnchors.size(); i++)
    {
        Eigen::Vector3d a_v = Eigen::Vector3d::Zero();
        for (int j = 0; j < mAnchors[i]->bodynodes.size(); j++)
            a_v += mAnchors[i]->weights[j] * (mAnchors[i]->bodynodes[j]->getLinearVelocity(mAnchors[i]->local_positions[j]));
        mCachedAnchorVelocities.push_back(a_v);
    }
    v_m = 0;
    for (int i = 0; i < mAnchors.size() - 1; i++)
    {
        double l = (mCachedAnchorPositions[i + 1] - mCachedAnchorPositions[i]).norm();
        v_m += (1 / l) * (mCachedAnchorPositions[i + 1] - mCachedAnchorPositions[i]).dot((mCachedAnchorVelocities[i + 1] - mCachedAnchorVelocities[i]));
    }
}
double Muscle::GetForce()
{
    return Getf_A() * activation + Getf_p();
}
double Muscle::Getf_A()
{
    double f_a = F_L(lm_norm) * (mUseVelocityForce ? F_V(v_m * 0.1 / lmt_ref * lm_opt) : 1.0) * cos(pen_angle);
    return f0 * f_a;
}
double Muscle::Getf_p()
{
    // if (lm_norm > 1.4) LOG_WARN("[Muscle] Getf_p: " + name + " - lm_norm=" + std::to_string(lm_norm) + " is too high");
    // double f_p = F_psv(std::min(lm_norm, 1.4)) * cos(pen_angle);
    double f_p = F_psv(lm_norm) * cos(pen_angle);
    return f0 * f_p;
}
Eigen::VectorXd Muscle::GetRelatedJtA()
{
    Eigen::MatrixXd Jt_reduced = GetReducedJacobianTranspose();
    Eigen::VectorXd A = GetForceJacobianAndPassive().first;
    Eigen::VectorXd JtA_reduced = Jt_reduced * A;
    return JtA_reduced;
}

void Muscle::RelaxPassiveForce()
{
    double old_coeff = lm_norm;
    SetLmNorm(1.0);
    std::cout << "[Muscle] RelaxPassiveForce: " << name
              << " | old_coeff=" << old_coeff
              << " → new_coeff=1.0" << std::endl;
}

void Muscle::SetLmNorm(double target_lm_norm)
{
    double old_lt_rel_ofs = lt_rel_ofs;
    double target_lt_rel = lmt_rel - target_lm_norm * lm_opt;
    lt_rel_ofs = target_lt_rel - lt_rel_base;

    // Guard: ensure l_t0 doesn't go below minimum threshold
    const double MIN_LT = 0.001;
    double new_lt = lt_rel_base + lt_rel_ofs;

    if (new_lt < MIN_LT) {
        double clamped_offset = MIN_LT - lt_rel_base;
        std::cout << "[Muscle] WARNING: " << name
                  << " - l_t0 would be " << new_lt
                  << " (below minimum " << MIN_LT << ")"
                  << " | Clamping l_t0_offset: " << lt_rel_ofs
                  << " → " << clamped_offset << std::endl;
        lt_rel_ofs = clamped_offset;
    }

    RefreshMuscleParams();

    std::cout << "[Muscle] SetLmNorm: " << name
              << " | target_lm_norm=" << target_lm_norm
              << " | lt_rel_ofs: " << old_lt_rel_ofs << " → " << lt_rel_ofs
              << " | lmt_rel=" << lmt_rel << std::endl;
}

Eigen::MatrixXd Muscle::GetReducedJacobianTranspose()
{
    const auto &skel = mAnchors[0]->bodynodes[0]->getSkeleton();
    Eigen::MatrixXd Jt(num_related_dofs, 3 * mAnchors.size());

    Jt.setZero();
    for (int i = 0; i < mAnchors.size(); i++)
    {
        auto bn = mAnchors[i]->bodynodes[0];
        dart::math::Jacobian J = dart::math::Jacobian::Zero(6, num_related_dofs);
        for (int j = 0; j < num_related_dofs; j++)
        {
            auto &indices = bn->getDependentGenCoordIndices();
            int idx = std::find(indices.begin(), indices.end(), related_dof_indices[j]) - indices.begin();
            if (idx != indices.size())
                J.col(j) = bn->getJacobian().col(idx);
        }
        Eigen::Vector3d offset = mAnchors[i]->bodynodes[0]->getTransform().inverse() * mCachedAnchorPositions[i];
        dart::math::LinearJacobian JLinear = J.bottomRows<3>() + J.topRows<3>().colwise().cross(offset);
        Jt.block(0, i * 3, num_related_dofs, 3) = (bn->getTransform().linear() * JLinear).transpose();
    }
    return Jt;
}

Eigen::VectorXd Muscle::GetRelatedJtp()
{
    Eigen::MatrixXd Jt_reduced = GetReducedJacobianTranspose();
    Eigen::VectorXd P = GetForceJacobianAndPassive().second;
    Eigen::VectorXd JtP_reduced = Jt_reduced * P;
    return JtP_reduced;
}

Eigen::MatrixXd Muscle::GetJacobianTranspose()
{
    const auto &skel = mAnchors[0]->bodynodes[0]->getSkeleton();
    int dof = skel->getNumDofs();

    Eigen::MatrixXd Jt(dof, 3 * mAnchors.size());

    Jt.setZero();
    for (int i = 0; i < mAnchors.size(); i++)
        Jt.block(0, i * 3, dof, 3) = skel->getLinearJacobian(mAnchors[i]->bodynodes[0], mAnchors[i]->bodynodes[0]->getTransform().inverse() * mCachedAnchorPositions[i]).transpose();

    return Jt;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> Muscle::GetForceJacobianAndPassive()
{
    double f_a = Getf_A();
    double f_p = Getf_p();

    std::vector<Eigen::Vector3d> force_dir;
    for (int i = 0; i < mAnchors.size(); i++)
    {
        force_dir.push_back(Eigen::Vector3d::Zero());
    }

    for (int i = 0; i < mAnchors.size() - 1; i++)
    {
        Eigen::Vector3d dir = mCachedAnchorPositions[i + 1] - mCachedAnchorPositions[i];
        dir.normalize();
        force_dir[i] += dir;
    }

    for (int i = 1; i < mAnchors.size(); i++)
    {
        Eigen::Vector3d dir = mCachedAnchorPositions[i - 1] - mCachedAnchorPositions[i];
        dir.normalize();
        force_dir[i] += dir;
    }

    Eigen::VectorXd A(3 * mAnchors.size());
    Eigen::VectorXd p(3 * mAnchors.size());
    A.setZero();
    p.setZero();

    for (int i = 0; i < mAnchors.size(); i++)
    {
        A.segment<3>(i * 3) = force_dir[i] * f_a;
        p.segment<3>(i * 3) = force_dir[i] * f_p;
    }
    return std::make_pair(A, p);
}

std::vector<dart::dynamics::Joint *> Muscle::GetRelatedJoints()
{
    auto skel = mAnchors[0]->bodynodes[0]->getSkeleton();
    std::map<dart::dynamics::Joint *, int> jns;
    std::vector<dart::dynamics::Joint *> jns_related;
    for (int i = 0; i < skel->getNumJoints(); i++)
        jns.insert(std::make_pair(skel->getJoint(i), 0));

    Eigen::VectorXd dl_dtheta = Getdl_dtheta();

    for (int i = 0; i < dl_dtheta.rows(); i++)
        if (std::abs(dl_dtheta[i]) > 1E-10)
            jns[skel->getDof(i)->getJoint()] += 1;

    for (auto jn : jns)
        if (jn.second > 0)
            jns_related.push_back(jn.first);
    return jns_related;
}
std::vector<dart::dynamics::BodyNode *> Muscle::GetRelatedBodyNodes()
{
    std::vector<dart::dynamics::BodyNode *> bns_related;
    auto rjs = GetRelatedJoints();
    for (auto joint : rjs)
    {
        bns_related.push_back(joint->getChildBodyNode());
    }

    return bns_related;
}
void Muscle::ComputeJacobians()
{
    const auto &skel = mAnchors[0]->bodynodes[0]->getSkeleton();
    int dof = skel->getNumDofs();
    mCachedJs.resize(mAnchors.size());
    for (int i = 0; i < mAnchors.size(); i++)
    {
        mCachedJs[i].resize(3, skel->getNumDofs());
        mCachedJs[i].setZero();

        for (int j = 0; j < mAnchors[i]->num_related_bodies; j++)
        {
            mCachedJs[i] += mAnchors[i]->weights[j] * skel->getLinearJacobian(mAnchors[i]->bodynodes[j], mAnchors[i]->local_positions[j]);
        }
    }
}
Eigen::VectorXd Muscle::Getdl_dtheta()
{
    ComputeJacobians();
    const auto &skel = mAnchors[0]->bodynodes[0]->getSkeleton();
    Eigen::VectorXd dl_dtheta(skel->getNumDofs());
    dl_dtheta.setZero();
    for (int i = 0; i < mAnchors.size() - 1; i++)
    {
        Eigen::Vector3d pi = mCachedAnchorPositions[i + 1] - mCachedAnchorPositions[i];
        Eigen::MatrixXd dpi_dtheta = mCachedJs[i + 1] - mCachedJs[i];
        Eigen::VectorXd dli_d_theta = (dpi_dtheta.transpose() * pi) / (lmt_ref * pi.norm());
        dl_dtheta += dli_d_theta;
    }

    for (int i = 0; i < dl_dtheta.rows(); i++) if (std::abs(dl_dtheta[i]) < 1E-10) dl_dtheta[i] = 0.0;
    return dl_dtheta;
}

double Muscle::F_psv(double _l_m)
{
    double f_pl = (exp(k_pe * (_l_m - 1) / e_mo) - 1.0) / (exp(k_pe) - 1.0);
    if (_l_m < 1.0) return 0.0;
    else return f_pl;
}
double Muscle::F_L(double _l_m)
{
    return exp(-(_l_m - 1.0) * (_l_m - 1.0) / gamma);
}
double Muscle::F_V(double _v_m)
{
    double f_av = 0;
    if (_v_m <= -1) f_av = 0;
    else if (-1 < _v_m && _v_m <= 0) f_av = (1 + _v_m) / (1 - _v_m / 0.25);
    else f_av = (1 + _v_m * 1.6 / 0.06) / (1 + _v_m / 0.06);
    return f_av;
}

double Muscle::GetMass()
{
    return f0 * (1.0 - lt_rel) * lmt_ref / 100.0;
}

double Muscle::Getdl_velocity()
{
    ComputeJacobians();
    const auto &skel = mAnchors[0]->bodynodes[0]->getSkeleton();
    double dl_velocity = 0.0;
    for (int i = 0; i < mAnchors.size() - 1; i++)
    {
        Eigen::Vector3d dist = mCachedAnchorPositions[i + 1] - mCachedAnchorPositions[i];
        Eigen::Vector3d d_dist = (mCachedJs[i + 1] - mCachedJs[i]) * skel->getVelocities();
        dl_velocity += dist.dot(d_dist) / dist.norm();
    }
    return dl_velocity;
}

std::vector<std::vector<double>> Muscle::GetGraphData()
{
    std::vector<std::vector<double>> result;
    std::vector<double> x;
    std::vector<double> a;
    std::vector<double> a_f;
    std::vector<double> p;
    std::vector<double> current;

    UpdateGeometry();

    result.clear();
    x.clear();
    a.clear();
    a_f.clear();
    p.clear();
    current.clear();

    for (int i = 0; i < 250; i++)
    {
        x.push_back(i * 0.01);
        a.push_back(f0 * F_L(i * 0.01));
        a_f.push_back(f0 * F_L(i * 0.01) * activation);
        p.push_back(f0 * F_psv(i * 0.01));
    }
    current.push_back(lm_norm);
    result.push_back(current);
    result.push_back(x);
    result.push_back(a);
    result.push_back(a_f);
    result.push_back(p);

    return result;
}

double Muscle::GetBHAR04_EnergyRate() // It assume that activation == excitation (i.e. no delay)
{
    double e_dot = 0.0;
    double a_dot = 0.0;
    double m_dot = 0.0;
    double s_dot = 0.0;
    double w_dot = 0.0;

    double mass = GetMass();
    double f_a_u = 40 * type1_fraction * sin(M_PI * 0.5 * activation) + 133 * (1 - type1_fraction) * (1 - cos(M_PI * 0.5 * activation));
    double f_m_a = 74 * type1_fraction * sin(M_PI * 0.5 * activation) + 111 * (1 - type1_fraction) * (1 - cos(M_PI * 0.5 * activation));
    double g_l = 0.0;
    double l_m_ratio = lm_rel / lm_opt;

    if (l_m_ratio < 0.5)
        g_l = 0.5;
    else if (l_m_ratio < 1.0)
        g_l = l_m_ratio;
    else if (l_m_ratio < 1.5)
        g_l = -2 * l_m_ratio + 3;
    else
        g_l = 0;

    a_dot = mass * f_a_u;
    m_dot = mass * g_l * f_m_a;

    double alpha = 0.0;
    double dl_velocity = Getdl_velocity();
    if (dl_velocity <= 0)
        alpha = 0.16 * Getf_A() + 0.18 * GetForce();
    else
        alpha = 0.157 * GetForce();

    s_dot = -alpha * dl_velocity;

    w_dot = -dl_velocity * GetForce();

    e_dot = a_dot + m_dot + s_dot + w_dot;

    return e_dot;
}
