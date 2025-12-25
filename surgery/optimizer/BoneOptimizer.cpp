// Bone Optimizer - Ceres-based bone scale optimization
// Moved from viewer/CeresOptimizer.cpp to surgery/optimizer/

#include <cmath>
#include <algorithm>
#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <dart/dynamics/BallJoint.hpp>
#include <dart/dynamics/FreeJoint.hpp>
#include "BoneOptimizer.h"
#include "C3D_Reader.h"
#include "RenderCharacter.h"
#include "Log.h"

// ============================================================================
// Ceres Cost Functors for Bone Scale Optimization
// ============================================================================

// Marker residual: ||s * R * q + t - p||
// q: reference marker position (bone-local, from skeleton T-pose)
// p: measured marker position (bone-local, from C3D data)
struct MarkerResidual {
    Eigen::Vector3d q;  // reference (bone-local)
    Eigen::Vector3d p;  // measured (bone-local)

    MarkerResidual(const Eigen::Vector3d& q_, const Eigen::Vector3d& p_)
        : q(q_), p(p_) {}

    template <typename T>
    bool operator()(const T* const alpha,
                    const T* const pose_k,
                    T* residuals) const {
        const T s = ceres::exp(alpha[0]);
        const T aa[3] = {pose_k[3], pose_k[4], pose_k[5]};
        const T q_T[3] = {T(q.x()), T(q.y()), T(q.z())};

        T Rq[3];
        ceres::AngleAxisRotatePoint(aa, q_T, Rq);

        residuals[0] = s * Rq[0] + pose_k[0] - T(p.x());
        residuals[1] = s * Rq[1] + pose_k[1] - T(p.y());
        residuals[2] = s * Rq[2] + pose_k[2] - T(p.z());
        return true;
    }
};

// Rotation regularizer: w * ||ω - ω_ref||
// Penalizes deviation from reference rotation (typically zero)
struct RotationRegularizer {
    Eigen::Vector3d omega_ref;
    double weight;

    RotationRegularizer(const Eigen::Vector3d& ref, double w)
        : omega_ref(ref), weight(w) {}

    template <typename T>
    bool operator()(const T* const pose_k, T* residuals) const {
        const T w = T(weight);
        residuals[0] = w * (pose_k[3] - T(omega_ref.x()));
        residuals[1] = w * (pose_k[4] - T(omega_ref.y()));
        residuals[2] = w * (pose_k[5] - T(omega_ref.z()));
        return true;
    }
};

// ============================================================================
// optimizeBoneScaleCeres - Ceres-based optimizer for 2+ marker bones
//   Uses rotation regularization to handle underconstrained N=2 case
// ============================================================================

BoneFitResult optimizeBoneScaleCeres(
    BodyNode* bn,
    const std::vector<const MarkerReference*>& markers,
    const std::vector<std::vector<Eigen::Vector3d>>& globalP,
    const SkeletonFittingConfig& config)
{
    BoneFitResult out;
    out.valid = false;
    out.scale = Eigen::Vector3d::Ones();
    out.iterations = 0;
    out.finalRMS = 0.0;

    if (!bn) return out;

    const int N = (int)markers.size();
    const int K = (int)globalP.size();
    if (N < 2 || K == 0) {
        LOG_WARN("[Ceres] Insufficient markers or frames: N=" << N << ", K=" << K);
        return out;
    }

    // ========== Setup coordinate transforms ==========
    Eigen::Isometry3d bnTransform = bn->getTransform();
    Eigen::Isometry3d invTransform = bnTransform.inverse();

    // Get bone size for computing q from marker offsets
    auto* shapeNode = bn->getShapeNodeWith<VisualAspect>(0);
    if (!shapeNode) {
        LOG_WARN("[Ceres] Bone shape node not found: " << bn->getName());
        return out;
    }
    const auto* boxShape = dynamic_cast<const BoxShape*>(shapeNode->getShape().get());
    if (!boxShape) {
        LOG_WARN("[Ceres] Bone shape not found: " << bn->getName());
        return out;
    }
    Eigen::Vector3d size = boxShape->getSize();

    // ========== Build q (reference) in bone-local coords ==========
    // q[i]: reference marker position (from skeleton T-pose, computed from offset * size)
    std::vector<Eigen::Vector3d> q(N);
    for (int i = 0; i < N; ++i) {
        const Eigen::Vector3d& offset = markers[i]->offset;
        q[i] = Eigen::Vector3d(
            std::abs(size[0]) * 0.5 * offset[0],
            std::abs(size[1]) * 0.5 * offset[1],
            std::abs(size[2]) * 0.5 * offset[2]
        );
    }

    // ========== Build p (measured) in bone-local coords ==========
    // p[k][i]: measured marker position per frame (transform world -> bone-local)
    std::vector<std::vector<Eigen::Vector3d>> p(K);
    for (int k = 0; k < K; ++k) {
        p[k].resize(N);
        for (int i = 0; i < N; ++i) {
            int idx = markers[i]->dataIndex;
            if (idx >= 0 && idx < (int)globalP[k].size()) {
                p[k][i] = invTransform * globalP[k][idx];
            } else {
                LOG_WARN("[Ceres] Invalid marker dataIndex: " << markers[i]->name);
                return out;
            }
        }
    }

    // ========== Parameter Setup ==========
    double alpha = 0.0;  // log(scale), init to log(1) = 0
    std::vector<double> pose(6 * K, 0.0);  // [tx,ty,tz,rx,ry,rz] per frame

    // Initialize translation from centroid difference
    for (int k = 0; k < K; ++k) {
        Eigen::Vector3d centroid_q = Eigen::Vector3d::Zero();
        Eigen::Vector3d centroid_p = Eigen::Vector3d::Zero();
        for (int i = 0; i < N; ++i) {
            centroid_q += q[i];
            centroid_p += p[k][i];
        }
        centroid_q /= N;
        centroid_p /= N;
        Eigen::Vector3d t_init = centroid_p - centroid_q;
        pose[6*k + 0] = t_init.x();
        pose[6*k + 1] = t_init.y();
        pose[6*k + 2] = t_init.z();
    }

    // ========== Build Ceres Problem ==========
    ceres::Problem problem;
    problem.AddParameterBlock(&alpha, 1);
    for (int k = 0; k < K; ++k) {
        problem.AddParameterBlock(&pose[6*k], 6);
    }

    // (1) Marker residuals: ||s * R * q[i] + t - p[k][i]||
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < N; ++i) {
            ceres::CostFunction* cost =
                new ceres::AutoDiffCostFunction<MarkerResidual, 3, 1, 6>(
                    new MarkerResidual(q[i], p[k][i]));
            problem.AddResidualBlock(cost, nullptr, &alpha, &pose[6*k]);
        }
    }

    // (2) Rotation regularizer: λ_rot * ||ω_k - 0||²
    double lambda_rot = config.lambdaRot;
    if (lambda_rot > 0) {
        double w_rot = std::sqrt(lambda_rot);
        Eigen::Vector3d omega_ref = Eigen::Vector3d::Zero();
        for (int k = 0; k < K; ++k) {
            ceres::CostFunction* reg =
                new ceres::AutoDiffCostFunction<RotationRegularizer, 3, 6>(
                    new RotationRegularizer(omega_ref, w_rot));
            problem.AddResidualBlock(reg, nullptr, &pose[6*k]);
        }
    }

    // ========== Solve with iteration callback for loss tracking ==========
    std::vector<double> lossHistory;

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = config.maxIterations;
    options.update_state_every_iteration = true;

    // Custom callback to track loss
    struct LossCallback : public ceres::IterationCallback {
        std::vector<double>* history;
        ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
            history->push_back(summary.cost);
            return ceres::SOLVER_CONTINUE;
        }
    };
    LossCallback callback;
    callback.history = &lossHistory;
    options.callbacks.push_back(&callback);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // ========== Plot loss in ASCII chart ==========
    if (config.plotConvergence && !lossHistory.empty()) {
        const int chartWidth = 60;
        const int chartHeight = 10;
        double maxLoss = *std::max_element(lossHistory.begin(), lossHistory.end());
        double minLoss = *std::min_element(lossHistory.begin(), lossHistory.end());

        std::cout << "\n[Ceres " << bn->getName() << "] Loss convergence:" << std::endl;
        std::cout << std::string(chartWidth + 10, '-') << std::endl;

        for (int row = chartHeight - 1; row >= 0; --row) {
            double threshold = minLoss + (maxLoss - minLoss) * row / (chartHeight - 1);
            printf("%8.2e |", threshold);
            for (int col = 0; col < std::min(chartWidth, (int)lossHistory.size()); ++col) {
                int idx = col * (int)lossHistory.size() / chartWidth;
                std::cout << (lossHistory[idx] >= threshold ? "*" : " ");
            }
            std::cout << std::endl;
        }
        std::cout << std::string(10, ' ') << "+" << std::string(chartWidth, '-') << std::endl;
        std::cout << "           0" << std::string(chartWidth - 10, ' ')
                  << lossHistory.size() - 1 << " iter" << std::endl;
    }

    // ========== Extract Results ==========
    out.valid = (summary.termination_type == ceres::CONVERGENCE ||
                 summary.termination_type == ceres::NO_CONVERGENCE);
    out.scale = Eigen::Vector3d::Constant(std::exp(alpha));  // Uniform scale
    out.iterations = (int)summary.iterations.size();
    out.finalRMS = std::sqrt(summary.final_cost / (K * N));

    out.R_frames.resize(K);
    out.t_frames.resize(K);
    for (int k = 0; k < K; ++k) {
        // Extract rotation matrix from angle-axis
        double aa[3] = {pose[6*k + 3], pose[6*k + 4], pose[6*k + 5]};
        double R[9];
        ceres::AngleAxisToRotationMatrix(aa, R);

        // Ceres stores column-major: R = [R00,R10,R20, R01,R11,R21, R02,R12,R22]
        Eigen::Matrix3d Rmat;
        Rmat << R[0], R[3], R[6],
                R[1], R[4], R[7],
                R[2], R[5], R[8];

        Eigen::Vector3d t_local(pose[6*k + 0], pose[6*k + 1], pose[6*k + 2]);

        // Convert bone-local to global
        Eigen::Isometry3d localT = Eigen::Isometry3d::Identity();
        localT.linear() = Rmat;
        localT.translation() = t_local;

        Eigen::Isometry3d globalT = bnTransform * localT;
        out.R_frames[k] = globalT.linear();
        out.t_frames[k] = globalT.translation();
    }

    LOG_INFO("[Ceres] " << bn->getName() << ": scale=" << out.scale[0]
             << " RMS=" << out.finalRMS << " iter=" << out.iterations);

    return out;
}

// ============================================================================
// Stage 2b: Run Ceres-based bone fitting for all targetCeres bones
// ============================================================================

void runCeresBoneFitting(
    const SkeletonFittingConfig& config,
    const std::map<std::string, std::vector<const MarkerReference*>>& boneToMarkers,
    const std::vector<std::vector<Eigen::Vector3d>>& allMarkers,
    RenderCharacter* character,
    std::vector<BoneInfo>& skelInfos,
    std::map<std::string, std::vector<Eigen::Matrix3d>>& boneR_frames,
    std::map<std::string, std::vector<Eigen::Vector3d>>& boneT_frames)
{
    if (config.targetCeres.empty()) {
        return;
    }

    std::cout << "\n=== Stage 2b: Ceres-based bone fitting (2+ markers) ===\n" << std::endl;

    for (const auto& boneName : config.targetCeres) {
        auto it = boneToMarkers.find(boneName);
        if (it == boneToMarkers.end() || it->second.empty()) {
            LOG_WARN("[C3D_Reader] No marker mappings found for bone: " << boneName);
            continue;
        }

        auto* bn = character->getSkeleton()->getBodyNode(boneName);
        if (!bn) {
            LOG_WARN("[C3D_Reader] Bone not found: " << boneName);
            continue;
        }

        BoneFitResult fitResult = optimizeBoneScaleCeres(bn, it->second, allMarkers, config);
        if (fitResult.valid) {
            // Store scale
            int idx = bn->getIndexInSkeleton();
            std::get<1>(skelInfos[idx]).value[3] = fitResult.scale[0];

            // Store per-frame transforms (same as SVD version)
            boneR_frames[boneName] = fitResult.R_frames;
            boneT_frames[boneName] = fitResult.t_frames;
        }
    }
}

// ============================================================================
// Apply Ceres optimizer results in buildFramePose (arm rotations)
// ============================================================================

void applyCeresArmRotations(
    const SkeletonFittingConfig& config,
    const std::map<std::string, std::vector<Eigen::Matrix3d>>& boneR_frames,
    const std::map<std::string, std::vector<Eigen::Vector3d>>& boneT_frames,
    int fitFrameIdx,
    dart::dynamics::SkeletonPtr skel,
    Eigen::VectorXd& pos)
{
    using dart::dynamics::FreeJoint;

    for (const auto& boneName : config.targetCeres) {
        auto it = boneR_frames.find(boneName);
        if (it == boneR_frames.end()) continue;

        const auto& R_frames = it->second;
        if (fitFrameIdx >= (int)R_frames.size()) continue;

        auto* bn = skel->getBodyNode(boneName);
        if (!bn) continue;

        auto* joint = bn->getParentJoint();
        if (!joint) continue;

        // Get stored global transform from Ceres optimizer
        Eigen::Isometry3d bodynodeGlobalT = Eigen::Isometry3d::Identity();
        bodynodeGlobalT.linear() = R_frames[fitFrameIdx];
        bodynodeGlobalT.translation() = boneT_frames.at(boneName)[fitFrameIdx];

        // Same joint angle computation as buildFramePose main loop
        Eigen::Isometry3d parentToJoint = joint->getTransformFromParentBodyNode();
        Eigen::Isometry3d childToJoint = joint->getTransformFromChildBodyNode();

        Eigen::Isometry3d parentBnGlobal = Eigen::Isometry3d::Identity();
        auto* parentBn = bn->getParentBodyNode();
        if (parentBn) {
            parentBnGlobal = parentBn->getTransform();
        }

        Eigen::Isometry3d jointT = parentToJoint.inverse() * parentBnGlobal.inverse() * bodynodeGlobalT * childToJoint;

        int jn_idx = joint->getIndexInSkeleton(0);
        int jn_dof = joint->getNumDofs();
        if (jn_idx >= 0 && jn_idx + jn_dof <= pos.size()) {
            Eigen::VectorXd jointPos = FreeJoint::convertToPositions(jointT);
            pos.segment(jn_idx, jn_dof) = jointPos;
            skel->setPositions(pos);  // Update for next bone's parent transform
        }
    }
}
