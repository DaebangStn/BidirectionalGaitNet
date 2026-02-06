// Waypoint Optimizer - Ceres-based muscle waypoint optimization
// Migrated from SkelGen/generator/MuscleGenerator.cpp

#include "WaypointOptimizer.h"
#include "Log.h"
#include <ceres/ceres.h>
#include <algorithm>
#include <iomanip>
#include <memory>

namespace PMuscle {

static constexpr double kEpsilon = 1e-8;
static constexpr double kSqrtEpsilon = 1e-10;

static double applyLoss(double diff, int power) {
    return std::pow(std::abs(diff), power);
}

// ============================================================================
// DOF Sweep Configuration
// ============================================================================

struct DOFSweepConfig {
    int dof_idx = -1;
    double lower = -M_PI;
    double upper = M_PI;

    bool isValid() const { return dof_idx >= 0; }
};

/**
 * Find the DOF with largest influence on muscle length.
 * Uses Getdl_dtheta() to compute dL/dtheta and picks the max.
 */
static DOFSweepConfig findBestDOF(
    Muscle* muscle,
    dart::dynamics::SkeletonPtr skeleton)
{
    DOFSweepConfig config;

    muscle->UpdateGeometry();
    Eigen::VectorXd dl_dtheta = muscle->Getdl_dtheta();

    double max_influence = 0.0;
    for (int i = 0; i < dl_dtheta.size(); ++i) {
        double influence = std::abs(dl_dtheta[i]);
        if (influence > max_influence) {
            max_influence = influence;
            config.dof_idx = i;
        }
    }

    if (!config.isValid() || config.dof_idx >= static_cast<int>(skeleton->getNumDofs())) {
        config.dof_idx = -1;
        return config;
    }

    config.lower = skeleton->getPositionLowerLimit(config.dof_idx);
    config.upper = skeleton->getPositionUpperLimit(config.dof_idx);
    if (config.lower < -M_PI) config.lower = -M_PI;
    if (config.upper > M_PI) config.upper = M_PI;

    return config;
}

// ============================================================================
// Optimization Context
// ============================================================================

struct OptimizationContext {
    Muscle* subject_muscle;
    Muscle* reference_muscle;
    dart::dynamics::SkeletonPtr subject_skeleton;
    dart::dynamics::SkeletonPtr reference_skeleton;

    int anchor_index;
    int num_samples;
    DOFSweepConfig dof_config;
    Eigen::VectorXd ref_pose;
    LengthCurveCharacteristics ref_chars;
    bool verbose = false;
    double weightPhase = 1.0;
    double weightDelta = 50.0;
    double weightSamples = 1.0;
    int numPhaseSamples = 3;
    int lossPower = 2;
    LengthCurveType lengthType = LengthCurveType::MTU_LENGTH;
    bool adaptiveSampleWeight = false;

    static std::shared_ptr<OptimizationContext> create(
        Muscle* subj_muscle, Muscle* ref_muscle,
        dart::dynamics::SkeletonPtr subj_skel, dart::dynamics::SkeletonPtr ref_skel,
        int anchor_idx, int samples,
        const DOFSweepConfig& dof_cfg, const Eigen::VectorXd& pose,
        const LengthCurveCharacteristics& chars, bool is_verbose = false,
        double weight_phase = 1.0, double weight_delta = 50.0,
        double weight_samples = 1.0, int num_phase_samples = 3,
        int loss_power = 2, LengthCurveType length_type = LengthCurveType::MTU_LENGTH,
        bool adaptive_sample_weight = false)
    {
        auto ctx = std::make_shared<OptimizationContext>();
        ctx->subject_muscle = subj_muscle;
        ctx->reference_muscle = ref_muscle;
        ctx->subject_skeleton = subj_skel;
        ctx->reference_skeleton = ref_skel;
        ctx->anchor_index = anchor_idx;
        ctx->num_samples = samples;
        ctx->dof_config = dof_cfg;
        ctx->ref_pose = pose;
        ctx->ref_chars = chars;
        ctx->verbose = is_verbose;
        ctx->weightPhase = weight_phase;
        ctx->weightDelta = weight_delta;
        ctx->weightSamples = weight_samples;
        ctx->numPhaseSamples = num_phase_samples;
        ctx->lossPower = loss_power;
        ctx->lengthType = length_type;
        ctx->adaptiveSampleWeight = adaptive_sample_weight;
        return ctx;
    }
};

// ============================================================================
// DOF Sweep
// ============================================================================

template<typename Func>
static void sweepDOF(const OptimizationContext& ctx, Func&& func) {
    for (int sample = 0; sample <= ctx.num_samples; ++sample) {
        double t = static_cast<double>(sample) / ctx.num_samples;
        double dof_val = ctx.dof_config.lower + t * (ctx.dof_config.upper - ctx.dof_config.lower);

        Eigen::VectorXd pose = ctx.ref_pose;
        pose[ctx.dof_config.dof_idx] = dof_val;

        ctx.subject_skeleton->setPositions(pose);
        ctx.reference_skeleton->setPositions(pose);
        ctx.subject_muscle->UpdateGeometry();
        ctx.reference_muscle->UpdateGeometry();

        func(sample);
    }
}

// ============================================================================
// Length Curve Computation
// ============================================================================

int WaypointOptimizer::getMotionDOF(const std::string& hdf_filepath)
{
    try {
        HDF motion(hdf_filepath);
        return motion.getValuesPerFrame();
    } catch (const std::exception& e) {
        LOG_ERROR("[WaypointOpt] Failed to read HDF file: " << e.what());
        return -1;
    }
}

/**
 * Compute muscle length curve by sweeping a specific DOF.
 *
 * lmt_ref must be initialized before calling this function.
 * This function only calls UpdateGeometry() (not SetMuscle/updateLmtRef)
 * to keep the normalization base stable during optimization.
 *
 * @param length_type MTU_LENGTH uses lmt, NORMALIZED uses lm_norm
 */
static std::vector<double> computeMuscleLengthCurveWithDOF(
    Muscle* muscle,
    dart::dynamics::SkeletonPtr skeleton,
    int num_samples,
    const DOFSweepConfig& dof_config,
    LengthCurveType length_type = LengthCurveType::MTU_LENGTH)
{
    std::vector<double> lengths;
    lengths.reserve(num_samples);

    if (!dof_config.isValid()) {
        double val = (length_type == LengthCurveType::NORMALIZED) ? muscle->lm_norm : muscle->lmt;
        lengths.push_back(val);
        return lengths;
    }

    Eigen::VectorXd saved_pose = skeleton->getPositions();

    for (int i = 0; i < num_samples; ++i) {
        double t = static_cast<double>(i) / (num_samples - 1);
        double dof_value = dof_config.lower + t * (dof_config.upper - dof_config.lower);

        Eigen::VectorXd pose = saved_pose;
        pose[dof_config.dof_idx] = dof_value;
        skeleton->setPositions(pose);

        muscle->UpdateGeometry();
        double val = (length_type == LengthCurveType::NORMALIZED) ? muscle->lm_norm : muscle->lmt;
        lengths.push_back(val);
    }

    skeleton->setPositions(saved_pose);
    return lengths;
}

std::vector<double> WaypointOptimizer::computeMuscleLengthCurve(
    Muscle* muscle,
    dart::dynamics::SkeletonPtr skeleton,
    int num_samples,
    LengthCurveType length_type)
{
    DOFSweepConfig dof_config = findBestDOF(muscle, skeleton);
    return computeMuscleLengthCurveWithDOF(muscle, skeleton, num_samples, dof_config, length_type);
}

LengthCurveCharacteristics WaypointOptimizer::analyzeLengthCurve(
    const std::vector<double>& lengths,
    int numPhaseSamples)
{
    LengthCurveCharacteristics chars{};

    if (lengths.empty()) {
        return chars;
    }

    auto [min_it, max_it] = std::minmax_element(lengths.begin(), lengths.end());

    chars.min_length = *min_it;
    chars.max_length = *max_it;
    chars.delta = chars.max_length - chars.min_length;

    int min_idx = std::distance(lengths.begin(), min_it);
    int max_idx = std::distance(lengths.begin(), max_it);

    double denom = static_cast<double>(lengths.size() - 1);
    chars.min_phase = min_idx / denom;
    chars.max_phase = max_idx / denom;

    int n = static_cast<int>(lengths.size());
    if (n > 1 && numPhaseSamples > 0) {
        chars.phase_samples.reserve(numPhaseSamples);
        for (int i = 0; i < numPhaseSamples; ++i) {
            double phase = (numPhaseSamples > 1)
                           ? static_cast<double>(i) / (numPhaseSamples - 1)
                           : 0.0;
            double float_idx = phase * (n - 1);
            int idx_lo = static_cast<int>(float_idx);
            int idx_hi = std::min(idx_lo + 1, n - 1);
            double t = float_idx - idx_lo;
            double value = (1.0 - t) * lengths[idx_lo] + t * lengths[idx_hi];
            chars.phase_samples.push_back(value);
        }
    }

    return chars;
}

// ============================================================================
// Shape Energy
// ============================================================================

/**
 * Compute shape energy for segments adjacent to the current anchor.
 * Each anchor's cost function is independent of other anchors' state.
 *
 * Energy = avg((1-dot)) with exponential penalty 5^(1-dot) when (1-dot) > 0.3
 */
static double computeShapeEnergy(const OptimizationContext& ctx) {
    double energy = 0.0;
    int count = 0;
    int idx = ctx.anchor_index;

    sweepDOF(ctx, [&](int) {
        auto& subj_anchors = ctx.subject_muscle->GetAnchors();
        auto& ref_anchors = ctx.reference_muscle->GetAnchors();

        // Segment before: anchor is END point
        if (idx > 0) {
            Eigen::Vector3d subj_seg = subj_anchors[idx]->GetPoint() - subj_anchors[idx-1]->GetPoint();
            Eigen::Vector3d ref_seg = ref_anchors[idx]->GetPoint() - ref_anchors[idx-1]->GetPoint();

            double subj_len = subj_seg.norm();
            double ref_len = ref_seg.norm();

            if (subj_len > kEpsilon && ref_len > kEpsilon) {
                double dot = (subj_seg / subj_len).dot(ref_seg / ref_len);
                double e = 1.0 - dot;
                if (e > 0.3) e = std::pow(5.0, e);
                energy += e;
                ++count;
            }
        }

        // Segment after: anchor is START point
        if (idx < static_cast<int>(subj_anchors.size()) - 1) {
            Eigen::Vector3d subj_seg = subj_anchors[idx+1]->GetPoint() - subj_anchors[idx]->GetPoint();
            Eigen::Vector3d ref_seg = ref_anchors[idx+1]->GetPoint() - ref_anchors[idx]->GetPoint();

            double subj_len = subj_seg.norm();
            double ref_len = ref_seg.norm();

            if (subj_len > kEpsilon && ref_len > kEpsilon) {
                double dot = (subj_seg / subj_len).dot(ref_seg / ref_len);
                double e = 1.0 - dot;
                if (e > 0.3) e = std::pow(5.0, e);
                energy += e;
                ++count;
            }
        }
    });

    return (count > 0) ? energy / count : 0.0;
}

/**
 * Per-phase shape metrics across ALL segments (for visualization/logging).
 */
static void computePerPhaseShapeMetrics(
    WaypointOptResult& result,
    Muscle* subject_muscle,
    Muscle* reference_muscle,
    dart::dynamics::SkeletonPtr subject_skeleton,
    dart::dynamics::SkeletonPtr reference_skeleton,
    int num_samples,
    const DOFSweepConfig& dof_config,
    const Eigen::VectorXd& ref_pose,
    bool is_before)
{
    auto& angles = is_before ? result.shape_angle_before : result.shape_angle_after;
    auto& energies = is_before ? result.shape_energy_before : result.shape_energy_after;
    angles.clear();
    energies.clear();
    angles.reserve(num_samples);
    energies.reserve(num_samples);

    for (int sample = 0; sample < num_samples; ++sample) {
        double t = static_cast<double>(sample) / (num_samples - 1);
        double dof_val = dof_config.lower + t * (dof_config.upper - dof_config.lower);

        Eigen::VectorXd pose = ref_pose;
        pose[dof_config.dof_idx] = dof_val;

        subject_skeleton->setPositions(pose);
        reference_skeleton->setPositions(pose);
        subject_muscle->UpdateGeometry();
        reference_muscle->UpdateGeometry();

        auto& subj_anchors = subject_muscle->GetAnchors();
        auto& ref_anchors = reference_muscle->GetAnchors();

        double dot_sum = 0.0;
        double energy_sum = 0.0;
        int segment_count = 0;

        for (size_t i = 0; i + 1 < subj_anchors.size(); ++i) {
            Eigen::Vector3d subj_seg = subj_anchors[i+1]->GetPoint() - subj_anchors[i]->GetPoint();
            Eigen::Vector3d ref_seg = ref_anchors[i+1]->GetPoint() - ref_anchors[i]->GetPoint();

            double subj_len = subj_seg.norm();
            double ref_len = ref_seg.norm();

            if (subj_len > kEpsilon && ref_len > kEpsilon) {
                double dot = (subj_seg / subj_len).dot(ref_seg / ref_len);
                dot_sum += dot;
                energy_sum += 1.0 - dot;
                ++segment_count;
            }
        }

        double avg_dot = (segment_count > 0) ? dot_sum / segment_count : 1.0;
        avg_dot = std::clamp(avg_dot, -1.0, 1.0);
        angles.push_back(std::acos(avg_dot) * 180.0 / M_PI);
        energies.push_back((segment_count > 0) ? energy_sum / segment_count : 0.0);
    }
}

static Eigen::Vector3d computeShapeGradientNumeric(const OptimizationContext& ctx) {
    constexpr double h = 1e-6;
    Eigen::Vector3d gradient;

    auto& anchor = ctx.subject_muscle->GetAnchors()[ctx.anchor_index];
    Eigen::Vector3d orig_local = anchor->local_positions[0];

    for (int axis = 0; axis < 3; ++axis) {
        anchor->local_positions[0] = orig_local;
        anchor->local_positions[0][axis] += h;
        ctx.subject_muscle->UpdateGeometry();
        double f_plus = computeShapeEnergy(ctx);

        anchor->local_positions[0] = orig_local;
        anchor->local_positions[0][axis] -= h;
        ctx.subject_muscle->UpdateGeometry();
        double f_minus = computeShapeEnergy(ctx);

        gradient[axis] = (f_plus - f_minus) / (2.0 * h);
    }

    anchor->local_positions[0] = orig_local;
    ctx.subject_muscle->UpdateGeometry();
    return gradient;
}

// ============================================================================
// Length Curve Energy
// ============================================================================

static double computeLengthCurveEnergy(const OptimizationContext& ctx) {
    ctx.subject_skeleton->setPositions(ctx.ref_pose);

    auto lengths = computeMuscleLengthCurveWithDOF(
        ctx.subject_muscle, ctx.subject_skeleton, ctx.num_samples, ctx.dof_config, ctx.lengthType);
    auto subj_chars = WaypointOptimizer::analyzeLengthCurve(lengths, ctx.numPhaseSamples);

    double energy = 0.0;

    // Phase matching
    energy += ctx.weightPhase * applyLoss(subj_chars.min_phase - ctx.ref_chars.min_phase, ctx.lossPower);
    energy += ctx.weightPhase * applyLoss(subj_chars.max_phase - ctx.ref_chars.max_phase, ctx.lossPower);

    // Delta matching
    energy += ctx.weightDelta * applyLoss(subj_chars.delta - ctx.ref_chars.delta, ctx.lossPower);

    // Sample matching
    if (ctx.weightSamples > kEpsilon &&
        subj_chars.phase_samples.size() == ctx.ref_chars.phase_samples.size()) {
        for (size_t i = 0; i < subj_chars.phase_samples.size(); ++i) {
            double sample_weight = 1.0;
            if (ctx.adaptiveSampleWeight) {
                double norm_len = ctx.ref_chars.phase_samples[i];
                if (norm_len >= 1.0) {
                    sample_weight = std::pow(5.0, norm_len);
                }
            }
            energy += ctx.weightSamples * sample_weight *
                      applyLoss(subj_chars.phase_samples[i] - ctx.ref_chars.phase_samples[i], ctx.lossPower);
        }
    }

    return energy;
}

static Eigen::Vector3d computeLengthCurveGradientNumeric(const OptimizationContext& ctx) {
    constexpr double h = 1e-6;
    Eigen::Vector3d gradient;

    auto& anchor = ctx.subject_muscle->GetAnchors()[ctx.anchor_index];
    Eigen::Vector3d orig_local = anchor->local_positions[0];

    for (int axis = 0; axis < 3; ++axis) {
        anchor->local_positions[0] = orig_local;
        anchor->local_positions[0][axis] += h;
        ctx.subject_muscle->UpdateGeometry();
        double f_plus = computeLengthCurveEnergy(ctx);

        anchor->local_positions[0] = orig_local;
        anchor->local_positions[0][axis] -= h;
        ctx.subject_muscle->UpdateGeometry();
        double f_minus = computeLengthCurveEnergy(ctx);

        gradient[axis] = (f_plus - f_minus) / (2.0 * h);
    }

    anchor->local_positions[0] = orig_local;
    ctx.subject_muscle->UpdateGeometry();
    return gradient;
}

// ============================================================================
// Unified Ceres Cost Functions
// ============================================================================

/**
 * Unified shape cost: single parameter block of size 3*N_opt containing all
 * optimizable anchor positions. Ceres passes trial values for ALL anchors
 * simultaneously, eliminating stale-value artifacts during trust-region steps.
 *
 * cost = 0.5 * r² = lambda_S * sum_i(E_shape_i)
 */
class UnifiedShapeCost : public ceres::CostFunction {
public:
    UnifiedShapeCost(
        std::vector<std::shared_ptr<OptimizationContext>> contexts,
        std::vector<int> opt_indices,
        double lambda_S)
        : contexts_(std::move(contexts))
        , opt_indices_(std::move(opt_indices))
        , lambda_S_(lambda_S)
    {
        int N_opt = static_cast<int>(opt_indices_.size());
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(3 * N_opt);
    }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        const double* params = parameters[0];
        auto& anchors = contexts_[0]->subject_muscle->GetAnchors();

        // Set all optimizable anchors from unified parameter block
        for (size_t k = 0; k < opt_indices_.size(); ++k) {
            int ai = opt_indices_[k];
            anchors[ai]->local_positions[0] = Eigen::Vector3d(
                params[3*k+0], params[3*k+1], params[3*k+2]);
        }
        contexts_[0]->subject_muscle->UpdateGeometry();

        // Compute total shape energy across all optimizable anchors
        double E_total = 0.0;
        for (const auto& ctx : contexts_) {
            E_total += computeShapeEnergy(*ctx);
        }

        double sqrt_E = std::sqrt(E_total + kSqrtEpsilon);
        double sqrt_2w = std::sqrt(2.0 * lambda_S_);
        residuals[0] = sqrt_2w * sqrt_E;

        if (jacobians && jacobians[0]) {
            constexpr double h = 1e-6;
            double scale = sqrt_2w / (2.0 * sqrt_E + kEpsilon);
            int N_opt = static_cast<int>(opt_indices_.size());

            for (int k = 0; k < N_opt; ++k) {
                int ai = opt_indices_[k];
                Eigen::Vector3d orig = anchors[ai]->local_positions[0];

                for (int axis = 0; axis < 3; ++axis) {
                    anchors[ai]->local_positions[0] = orig;
                    anchors[ai]->local_positions[0][axis] += h;
                    contexts_[0]->subject_muscle->UpdateGeometry();
                    double E_plus = 0.0;
                    for (const auto& ctx : contexts_)
                        E_plus += computeShapeEnergy(*ctx);

                    anchors[ai]->local_positions[0] = orig;
                    anchors[ai]->local_positions[0][axis] -= h;
                    contexts_[0]->subject_muscle->UpdateGeometry();
                    double E_minus = 0.0;
                    for (const auto& ctx : contexts_)
                        E_minus += computeShapeEnergy(*ctx);

                    double dE = (E_plus - E_minus) / (2.0 * h);
                    jacobians[0][3*k + axis] = scale * dE;

                    anchors[ai]->local_positions[0] = orig;
                }
            }
            contexts_[0]->subject_muscle->UpdateGeometry();
        }
        return true;
    }

private:
    std::vector<std::shared_ptr<OptimizationContext>> contexts_;
    std::vector<int> opt_indices_;
    double lambda_S_;
};

/**
 * Unified length curve cost: single parameter block of size 3*N_opt.
 * Length curve energy is the same regardless of which context computes it
 * (all contexts share the same muscle), so we call it once.
 *
 * cost = 0.5 * r² = N_opt * lambda_L * E_length
 */
class UnifiedLengthCurveCost : public ceres::CostFunction {
public:
    UnifiedLengthCurveCost(
        std::shared_ptr<OptimizationContext> ctx,
        std::vector<int> opt_indices,
        int N_opt,
        double lambda_L)
        : ctx_(std::move(ctx))
        , opt_indices_(std::move(opt_indices))
        , N_opt_(N_opt)
        , lambda_L_(lambda_L)
    {
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(3 * N_opt);
    }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        const double* params = parameters[0];
        auto& anchors = ctx_->subject_muscle->GetAnchors();

        // Set all optimizable anchors from unified parameter block
        for (size_t k = 0; k < opt_indices_.size(); ++k) {
            int ai = opt_indices_[k];
            anchors[ai]->local_positions[0] = Eigen::Vector3d(
                params[3*k+0], params[3*k+1], params[3*k+2]);
        }
        ctx_->subject_muscle->UpdateGeometry();

        double E_length = computeLengthCurveEnergy(*ctx_);

        double sqrt_E = std::sqrt(E_length + kSqrtEpsilon);
        double sqrt_2w = std::sqrt(2.0 * N_opt_ * lambda_L_);
        residuals[0] = sqrt_2w * sqrt_E;

        if (jacobians && jacobians[0]) {
            constexpr double h = 1e-6;
            double scale = sqrt_2w / (2.0 * sqrt_E + kEpsilon);

            for (int k = 0; k < N_opt_; ++k) {
                int ai = opt_indices_[k];
                Eigen::Vector3d orig = anchors[ai]->local_positions[0];

                for (int axis = 0; axis < 3; ++axis) {
                    anchors[ai]->local_positions[0] = orig;
                    anchors[ai]->local_positions[0][axis] += h;
                    ctx_->subject_muscle->UpdateGeometry();
                    double E_plus = computeLengthCurveEnergy(*ctx_);

                    anchors[ai]->local_positions[0] = orig;
                    anchors[ai]->local_positions[0][axis] -= h;
                    ctx_->subject_muscle->UpdateGeometry();
                    double E_minus = computeLengthCurveEnergy(*ctx_);

                    double dE = (E_plus - E_minus) / (2.0 * h);
                    jacobians[0][3*k + axis] = scale * dE;

                    anchors[ai]->local_positions[0] = orig;
                }
            }
            ctx_->subject_muscle->UpdateGeometry();
        }
        return true;
    }

private:
    std::shared_ptr<OptimizationContext> ctx_;
    std::vector<int> opt_indices_;
    int N_opt_;
    double lambda_L_;
};

// ============================================================================
// Main Optimization
// ============================================================================

WaypointOptResult WaypointOptimizer::optimizeMuscle(
    Muscle* subject_muscle,
    Muscle* reference_muscle,
    dart::dynamics::SkeletonPtr reference_skeleton,
    dart::dynamics::SkeletonPtr subject_skeleton,
    const Config& config)
{
    WaypointOptResult result;
    result.muscle_name = subject_muscle->name;
    result.success = false;

    // Save poses for restoration
    Eigen::VectorXd initial_subject_pose = subject_skeleton->getPositions();
    Eigen::VectorXd initial_reference_pose = reference_skeleton->getPositions();

    auto restorePoses = [&]() {
        subject_skeleton->setPositions(initial_subject_pose);
        reference_skeleton->setPositions(initial_reference_pose);
    };

    // 1. Early-out checks
    auto& anchors = subject_muscle->GetAnchors();
    if (config.fixOriginInsertion && anchors.size() <= 2) {
        result.subject_before_lengths = computeMuscleLengthCurve(
            subject_muscle, subject_skeleton, config.numSampling);
        result.subject_before_chars = analyzeLengthCurve(result.subject_before_lengths, config.numPhaseSamples);
        result.subject_after_lengths = result.subject_before_lengths;
        result.subject_after_chars = result.subject_before_chars;
        result.success = true;
        restorePoses();
        return result;
    }

    // 2. Find best DOF
    DOFSweepConfig dof_config = findBestDOF(subject_muscle, subject_skeleton);
    if (!dof_config.isValid()) {
        LOG_WARN("[WaypointOpt] No valid DOF found for " << subject_muscle->name);
        result.subject_before_lengths = computeMuscleLengthCurve(
            subject_muscle, subject_skeleton, config.numSampling);
        result.subject_before_chars = analyzeLengthCurve(result.subject_before_lengths, config.numPhaseSamples);
        result.subject_after_lengths = result.subject_before_lengths;
        result.subject_after_chars = result.subject_before_chars;
        result.success = true;
        restorePoses();
        return result;
    }

    result.dof_idx = dof_config.dof_idx;
    result.dof_name = subject_skeleton->getDof(dof_config.dof_idx)->getName();
    result.length_type = config.lengthType;

    // 3. Set common base pose for all curve computations
    Eigen::VectorXd ref_pose = reference_skeleton->getPositions();
    subject_skeleton->setPositions(ref_pose);
    reference_skeleton->setPositions(ref_pose);

    // For NORMALIZED mode: initialize lmt_ref at zero pose ONCE.
    // lmt_ref must not change during optimization; UpdateGeometry() uses
    // this fixed value to compute lm_norm at each DOF sweep position.
    if (config.lengthType == LengthCurveType::NORMALIZED) {
        subject_skeleton->setPositions(Eigen::VectorXd::Zero(subject_skeleton->getNumDofs()));
        subject_muscle->SetMuscle();
        reference_skeleton->setPositions(Eigen::VectorXd::Zero(reference_skeleton->getNumDofs()));
        reference_muscle->SetMuscle();
        subject_skeleton->setPositions(ref_pose);
        reference_skeleton->setPositions(ref_pose);
    }

    // 4. Compute initial curves
    result.reference_lengths = computeMuscleLengthCurveWithDOF(
        reference_muscle, reference_skeleton, config.numSampling, dof_config, config.lengthType);
    if (result.reference_lengths.empty()) {
        LOG_ERROR("[WaypointOpt] Failed to compute reference curve for " << subject_muscle->name);
        restorePoses();
        return result;
    }
    result.reference_chars = analyzeLengthCurve(result.reference_lengths, config.numPhaseSamples);

    result.subject_before_lengths = computeMuscleLengthCurveWithDOF(
        subject_muscle, subject_skeleton, config.numSampling, dof_config, config.lengthType);
    if (result.subject_before_lengths.empty()) {
        LOG_ERROR("[WaypointOpt] Failed to compute subject curve for " << subject_muscle->name);
        restorePoses();
        return result;
    }
    result.subject_before_chars = analyzeLengthCurve(result.subject_before_lengths, config.numPhaseSamples);

    result.phases.reserve(config.numSampling);
    for (int i = 0; i < config.numSampling; ++i) {
        result.phases.push_back(static_cast<double>(i) / (config.numSampling - 1));
    }

    if (config.verbose) {
        LOG_INFO("[WaypointOpt] " << subject_muscle->name << ": "
                 << anchors.size() << " anchors, best DOF=" << result.dof_name
                 << " (" << dof_config.dof_idx << ")"
                 << " [" << dof_config.lower << ", " << dof_config.upper << "]");
    }

    computePerPhaseShapeMetrics(result, subject_muscle, reference_muscle,
        subject_skeleton, reference_skeleton,
        config.numSampling, dof_config, ref_pose, true);

    // 5. Setup unified parameter block
    // Build list of optimizable anchor indices and the unified parameter vector
    std::vector<int> opt_indices;
    for (size_t i = 0; i < anchors.size(); ++i) {
        bool is_origin_insertion = (i == 0 || i == anchors.size() - 1);
        bool is_fixed = config.fixOriginInsertion && is_origin_insertion;
        if (!is_fixed) {
            opt_indices.push_back(static_cast<int>(i));
        }
    }
    int N_opt = static_cast<int>(opt_indices.size());

    // Contiguous parameter block: [x0,y0,z0, x1,y1,z1, ..., x_{N-1},y_{N-1},z_{N-1}]
    std::vector<double> unified_params(3 * N_opt);
    for (int k = 0; k < N_opt; ++k) {
        int ai = opt_indices[k];
        unified_params[3*k+0] = anchors[ai]->local_positions[0].x();
        unified_params[3*k+1] = anchors[ai]->local_positions[0].y();
        unified_params[3*k+2] = anchors[ai]->local_positions[0].z();
    }

    // Save initial positions for bounds and revert
    std::vector<std::array<double, 3>> initial_positions(N_opt);
    for (int k = 0; k < N_opt; ++k) {
        initial_positions[k] = {unified_params[3*k+0], unified_params[3*k+1], unified_params[3*k+2]};
    }

    if (config.verbose) {
        LOG_INFO("[WaypointOpt] Initial anchor positions:");
        for (size_t i = 0; i < anchors.size(); ++i) {
            LOG_INFO("  [" << i << "] " << anchors[i]->local_positions[0].transpose());
        }
        LOG_INFO("[WaypointOpt] Optimizable anchors: " << N_opt << " indices: ");
        for (int k = 0; k < N_opt; ++k) {
            LOG_INFO("  opt[" << k << "] -> anchor " << opt_indices[k]);
        }
    }

    // 6. Build Ceres problem with unified parameter block
    ceres::Solver::Options options;
    options.max_num_iterations = config.maxIterations;
    options.linear_solver_type = ceres::DENSE_QR;
    options.function_tolerance = config.functionTolerance;
    options.gradient_tolerance = config.gradientTolerance;
    options.parameter_tolerance = config.parameterTolerance;

    ceres::Problem problem;
    std::vector<std::shared_ptr<OptimizationContext>> contexts;

    // Create one OptimizationContext per optimizable anchor
    for (int k = 0; k < N_opt; ++k) {
        int ai = opt_indices[k];
        auto ctx = OptimizationContext::create(
            subject_muscle, reference_muscle,
            subject_skeleton, reference_skeleton,
            ai, config.numSampling,
            dof_config, ref_pose, result.reference_chars, config.verbose,
            config.weightPhase, config.weightDelta,
            config.weightSamples, config.numPhaseSamples, config.lossPower, config.lengthType,
            config.adaptiveSampleWeight);
        contexts.push_back(ctx);
    }

    if (N_opt > 0) {
        double* param_block = unified_params.data();
        problem.AddParameterBlock(param_block, 3 * N_opt);

        // Set per-element bounds
        for (int k = 0; k < N_opt; ++k) {
            int ai = opt_indices[k];
            bool is_origin_insertion = (ai == 0 || ai == static_cast<int>(anchors.size()) - 1);
            double max_disp = is_origin_insertion
                ? config.maxDisplacementOriginInsertion
                : config.maxDisplacement;
            for (int dim = 0; dim < 3; ++dim) {
                double current_val = unified_params[3*k + dim];
                problem.SetParameterLowerBound(param_block, 3*k + dim, current_val - max_disp);
                problem.SetParameterUpperBound(param_block, 3*k + dim, current_val + max_disp);
            }
        }

        // Add 2 residual blocks (shape + length) pointing to unified parameter block
        if (config.lambdaShape > 0) {
            problem.AddResidualBlock(
                new UnifiedShapeCost(contexts, opt_indices, config.lambdaShape),
                nullptr, param_block);
        }
        if (config.lambdaLengthCurve > 0) {
            problem.AddResidualBlock(
                new UnifiedLengthCurveCost(contexts[0], opt_indices, N_opt, config.lambdaLengthCurve),
                nullptr, param_block);
        }
    }

    // Helper: sync anchors from unified_params
    auto syncAnchorsFromParams = [&]() {
        for (int k = 0; k < N_opt; ++k) {
            int ai = opt_indices[k];
            anchors[ai]->local_positions[0] = Eigen::Vector3d(
                unified_params[3*k+0], unified_params[3*k+1], unified_params[3*k+2]);
        }
        subject_muscle->UpdateGeometry();
    };

    // Compute energies from current unified_params state
    auto computeEnergies = [&](double& shape_out, double& length_out, double& total_out, const char* label) {
        syncAnchorsFromParams();

        shape_out = 0.0;
        length_out = 0.0;
        for (const auto& ctx : contexts) {
            double ctx_shape = computeShapeEnergy(*ctx);
            double ctx_length = computeLengthCurveEnergy(*ctx);
            if (config.verbose) {
                LOG_INFO("[WaypointOpt] " << label << " ctx[anchor=" << ctx->anchor_index
                         << "] shape=" << ctx_shape << " length=" << ctx_length);
            }
            shape_out += ctx_shape;
            length_out += ctx_length;
        }
        total_out = config.lambdaShape * shape_out + config.lambdaLengthCurve * length_out;
        if (config.verbose) {
            LOG_INFO("[WaypointOpt] " << label << " TOTAL: shape=" << shape_out
                     << " length=" << length_out << " weighted=" << total_out
                     << " (contexts=" << contexts.size() << ")");
        }
    };

    computeEnergies(result.initial_shape_energy, result.initial_length_energy, result.initial_total_cost, "Initial");

    // ========================================================================
    // DIAGNOSTIC 1: Purity check — call f(x) 5 times at same state
    // ========================================================================
    if (config.verbose && !contexts.empty()) {
        LOG_INFO("[WaypointOpt] === PURITY CHECK (5 repeated calls at same x) ===");

        syncAnchorsFromParams();

        int purity_fail_count = 0;

        for (const auto& ctx : contexts) {
            // Shape purity: 5 calls WITHOUT re-syncing
            double shape_vals[5];
            for (int trial = 0; trial < 5; ++trial) {
                shape_vals[trial] = computeShapeEnergy(*ctx);
            }
            bool shape_pure = true;
            for (int i = 1; i < 5; ++i) {
                if (shape_vals[i] != shape_vals[0]) { shape_pure = false; break; }
            }
            if (!shape_pure) ++purity_fail_count;
            LOG_INFO("[WaypointOpt] Shape purity[anchor=" << ctx->anchor_index << "]: "
                     << (shape_pure ? "PASS" : "**FAIL**")
                     << " vals=[" << shape_vals[0] << ", " << shape_vals[1] << ", "
                     << shape_vals[2] << ", " << shape_vals[3] << ", " << shape_vals[4] << "]");

            // Length purity: 5 calls WITHOUT re-syncing
            double length_vals[5];
            for (int trial = 0; trial < 5; ++trial) {
                length_vals[trial] = computeLengthCurveEnergy(*ctx);
            }
            bool length_pure = true;
            for (int i = 1; i < 5; ++i) {
                if (length_vals[i] != length_vals[0]) { length_pure = false; break; }
            }
            if (!length_pure) ++purity_fail_count;
            LOG_INFO("[WaypointOpt] Length purity[anchor=" << ctx->anchor_index << "]: "
                     << (length_pure ? "PASS" : "**FAIL**")
                     << " vals=[" << length_vals[0] << ", " << length_vals[1] << ", "
                     << length_vals[2] << ", " << length_vals[3] << ", " << length_vals[4] << "]");
        }

        // Cross-context purity: A→B→A should give same result for A
        if (contexts.size() >= 2) {
            double a1 = computeShapeEnergy(*contexts[0]);
            computeShapeEnergy(*contexts[1]);  // intervening call to different context
            double a2 = computeShapeEnergy(*contexts[0]);
            if (a1 != a2) ++purity_fail_count;
            LOG_INFO("[WaypointOpt] Cross-context shape purity: "
                     << (a1 == a2 ? "PASS" : "**FAIL**")
                     << " A1=" << a1 << " A2=" << a2 << " diff=" << (a2 - a1));

            double la1 = computeLengthCurveEnergy(*contexts[0]);
            computeLengthCurveEnergy(*contexts[1]);
            double la2 = computeLengthCurveEnergy(*contexts[0]);
            if (la1 != la2) ++purity_fail_count;
            LOG_INFO("[WaypointOpt] Cross-context length purity: "
                     << (la1 == la2 ? "PASS" : "**FAIL**")
                     << " A1=" << la1 << " A2=" << la2 << " diff=" << (la2 - la1));
        }

        LOG_INFO("[WaypointOpt] Purity check: "
                 << (purity_fail_count == 0 ? "ALL PASS" :
                     (std::to_string(purity_fail_count) + " FAIL(s)")));
    }

    // ========================================================================
    // DIAGNOSTIC 2: Epsilon sensitivity — gradients at h=1e-4, 1e-6, 1e-8
    // ========================================================================
    if (config.verbose && !contexts.empty()) {
        LOG_INFO("[WaypointOpt] === EPSILON SENSITIVITY CHECK ===");

        double max_angle = 0.0;

        auto angleDeg = [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) -> double {
            double na = a.norm(), nb = b.norm();
            if (na < 1e-15 || nb < 1e-15) return -1.0;  // undefined
            return std::acos(std::clamp(a.dot(b) / (na * nb), -1.0, 1.0)) * 180.0 / M_PI;
        };

        for (const auto& ctx : contexts) {
            auto& anchor = subject_muscle->GetAnchors()[ctx->anchor_index];
            Eigen::Vector3d orig_local = anchor->local_positions[0];

            double eps_values[] = {1e-4, 1e-6, 1e-8};
            Eigen::Vector3d shape_grads[3], length_grads[3];

            for (int e = 0; e < 3; ++e) {
                double h = eps_values[e];

                // Shape gradient at this epsilon
                for (int axis = 0; axis < 3; ++axis) {
                    anchor->local_positions[0] = orig_local;
                    anchor->local_positions[0][axis] += h;
                    subject_muscle->UpdateGeometry();
                    double fp = computeShapeEnergy(*ctx);

                    anchor->local_positions[0] = orig_local;
                    anchor->local_positions[0][axis] -= h;
                    subject_muscle->UpdateGeometry();
                    double fm = computeShapeEnergy(*ctx);

                    shape_grads[e][axis] = (fp - fm) / (2.0 * h);
                }

                // Length gradient at this epsilon
                for (int axis = 0; axis < 3; ++axis) {
                    anchor->local_positions[0] = orig_local;
                    anchor->local_positions[0][axis] += h;
                    subject_muscle->UpdateGeometry();
                    double fp = computeLengthCurveEnergy(*ctx);

                    anchor->local_positions[0] = orig_local;
                    anchor->local_positions[0][axis] -= h;
                    subject_muscle->UpdateGeometry();
                    double fm = computeLengthCurveEnergy(*ctx);

                    length_grads[e][axis] = (fp - fm) / (2.0 * h);
                }

                anchor->local_positions[0] = orig_local;
                subject_muscle->UpdateGeometry();
            }

            // Track max angle across all anchors
            for (int i = 0; i < 2; ++i) {
                double sa = angleDeg(shape_grads[i], shape_grads[i+1]);
                double la = angleDeg(length_grads[i], length_grads[i+1]);
                if (sa > 0 && sa > max_angle) max_angle = sa;
                if (la > 0 && la > max_angle) max_angle = la;
            }

            LOG_INFO("[WaypointOpt] Shape grad eps[anchor=" << ctx->anchor_index << "]:"
                     << "\n    h=1e-4: " << shape_grads[0].transpose() << " |g|=" << shape_grads[0].norm()
                     << "\n    h=1e-6: " << shape_grads[1].transpose() << " |g|=" << shape_grads[1].norm()
                     << "\n    h=1e-8: " << shape_grads[2].transpose() << " |g|=" << shape_grads[2].norm()
                     << "\n    angle(1e-4,1e-6)=" << angleDeg(shape_grads[0], shape_grads[1]) << "°"
                     << " angle(1e-6,1e-8)=" << angleDeg(shape_grads[1], shape_grads[2]) << "°");

            LOG_INFO("[WaypointOpt] Length grad eps[anchor=" << ctx->anchor_index << "]:"
                     << "\n    h=1e-4: " << length_grads[0].transpose() << " |g|=" << length_grads[0].norm()
                     << "\n    h=1e-6: " << length_grads[1].transpose() << " |g|=" << length_grads[1].norm()
                     << "\n    h=1e-8: " << length_grads[2].transpose() << " |g|=" << length_grads[2].norm()
                     << "\n    angle(1e-4,1e-6)=" << angleDeg(length_grads[0], length_grads[1]) << "°"
                     << " angle(1e-6,1e-8)=" << angleDeg(length_grads[1], length_grads[2]) << "°");
        }

        LOG_INFO("[WaypointOpt] Epsilon sensitivity: max angle = "
                 << std::fixed << std::setprecision(2) << max_angle << "°");
    }

    // 7. Solve with iteration callback for verification
    class VerificationCallback : public ceres::IterationCallback {
    public:
        VerificationCallback(
            const std::vector<std::shared_ptr<OptimizationContext>>& ctxs,
            std::vector<double>& params, const std::vector<int>& opt_idx,
            const std::vector<Anchor*>& anch,
            Muscle* muscle, double ls, double ll, int n_opt, bool verb)
            : contexts_(ctxs), unified_params_(params), opt_indices_(opt_idx)
            , anchors_(anch), muscle_(muscle)
            , lambdaShape_(ls), lambdaLength_(ll), N_opt_(n_opt), verbose_(verb) {}

        ceres::CallbackReturnType operator()(const ceres::IterationSummary& iter_summary) override {
            if (!verbose_) return ceres::SOLVER_CONTINUE;

            // Sync anchors from current Ceres parameter state
            for (int k = 0; k < N_opt_; ++k) {
                int ai = opt_indices_[k];
                anchors_[ai]->local_positions[0] = Eigen::Vector3d(
                    unified_params_[3*k+0], unified_params_[3*k+1], unified_params_[3*k+2]);
            }
            muscle_->UpdateGeometry();

            double shape_total = 0.0, length_total = 0.0;
            for (const auto& ctx : contexts_) {
                shape_total += computeShapeEnergy(*ctx);
                length_total += computeLengthCurveEnergy(*ctx);
            }

            // Expected Ceres cost with unified blocks:
            // cost = lambda_S * shape_total + N_opt * lambda_L * length_total
            double expected_ceres = lambdaShape_ * shape_total + N_opt_ * lambdaLength_ * length_total;
            double ratio = (expected_ceres > 1e-12) ? iter_summary.cost / expected_ceres : 0.0;

            LOG_INFO("[WaypointOpt] iter=" << iter_summary.iteration
                     << " ceres=" << iter_summary.cost
                     << " expected=" << expected_ceres
                     << " ratio=" << ratio
                     << " shape=" << shape_total
                     << " length=" << length_total);

            return ceres::SOLVER_CONTINUE;
        }

    private:
        const std::vector<std::shared_ptr<OptimizationContext>>& contexts_;
        std::vector<double>& unified_params_;
        const std::vector<int>& opt_indices_;
        const std::vector<Anchor*>& anchors_;
        Muscle* muscle_;
        double lambdaShape_, lambdaLength_;
        int N_opt_;
        bool verbose_;
    };

    VerificationCallback verification_cb(
        contexts, unified_params, opt_indices, anchors, subject_muscle,
        config.lambdaShape, config.lambdaLengthCurve, N_opt, config.verbose);
    options.callbacks.push_back(&verification_cb);
    options.update_state_every_iteration = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    result.num_iterations = static_cast<int>(summary.iterations.size());

    if (config.verbose) {
        LOG_INFO("[WaypointOpt] " << subject_muscle->name
                 << ": " << summary.iterations.size() << " iters, "
                 << ceres::TerminationTypeToString(summary.termination_type));
    }

    // ========================================================================
    // DIAGNOSTIC 3: Post-solve comparison — Ceres.Evaluate vs our verification
    // ========================================================================
    if (config.verbose && N_opt > 0) {
        // A) Ceres' own re-evaluation at final parameters
        double ceres_eval_cost = 0.0;
        std::vector<double> ceres_residuals;
        ceres::Problem::EvaluateOptions eval_opts;
        problem.Evaluate(eval_opts, &ceres_eval_cost, &ceres_residuals, nullptr, nullptr);

        // B) Our independent verification at the same parameter state
        double verify_shape = 0.0, verify_length = 0.0;
        syncAnchorsFromParams();
        for (const auto& ctx : contexts) {
            verify_shape += computeShapeEnergy(*ctx);
            verify_length += computeLengthCurveEnergy(*ctx);
        }

        // C) Expected Ceres cost with unified blocks (2 residual blocks, not 2*N_opt):
        // cost = lambda_S * shape_total + N_opt * lambda_L * length_total + epsilon_terms
        int num_residual_blocks = 0;
        if (config.lambdaShape > 0) ++num_residual_blocks;
        if (config.lambdaLengthCurve > 0) ++num_residual_blocks;
        double eps_term = 0.0;
        if (config.lambdaShape > 0) eps_term += config.lambdaShape * kSqrtEpsilon;
        if (config.lambdaLengthCurve > 0) eps_term += N_opt * config.lambdaLengthCurve * kSqrtEpsilon;
        double expected_ceres = config.lambdaShape * verify_shape
                              + N_opt * config.lambdaLengthCurve * verify_length
                              + eps_term;

        double ratio_eval_expected = (expected_ceres > 1e-15) ? ceres_eval_cost / expected_ceres : 0.0;
        double ratio_eval_summary = (summary.final_cost > 1e-15) ? ceres_eval_cost / summary.final_cost : 0.0;

        LOG_INFO("[WaypointOpt] POST-SOLVE DIAGNOSTIC:"
                 << "\n    summary.final_cost  = " << summary.final_cost
                 << "\n    ceres.Evaluate()    = " << ceres_eval_cost
                 << "\n    our expected        = " << expected_ceres
                 << "\n    our shape_total     = " << verify_shape
                 << "\n    our length_total    = " << verify_length
                 << "\n    ratio(eval/expected)= " << ratio_eval_expected
                 << "\n    ratio(eval/summary) = " << ratio_eval_summary);

        // D) Print individual residuals for detailed comparison
        LOG_INFO("[WaypointOpt] Ceres residuals (" << ceres_residuals.size() << " total):");
        for (size_t i = 0; i < ceres_residuals.size(); ++i) {
            LOG_INFO("    r[" << i << "] = " << ceres_residuals[i]
                     << " r²=" << ceres_residuals[i] * ceres_residuals[i]);
        }

        if (std::abs(ratio_eval_expected - 1.0) > 0.01) {
            LOG_WARN("[WaypointOpt] *** MISMATCH: Ceres.Evaluate disagrees with our verification by "
                     << std::abs(ratio_eval_expected - 1.0) * 100.0 << "% ***");
        }
    }

    // Apply results: write from unified_params back to anchors
    for (int k = 0; k < N_opt; ++k) {
        int ai = opt_indices[k];
        anchors[ai]->local_positions[0] = Eigen::Vector3d(
            unified_params[3*k+0], unified_params[3*k+1], unified_params[3*k+2]);
    }

    if (config.verbose) {
        LOG_INFO("[WaypointOpt] Final anchor positions:");
        for (size_t i = 0; i < anchors.size(); ++i) {
            LOG_INFO("  [" << i << "] " << anchors[i]->local_positions[0].transpose());
        }
    }

    subject_skeleton->setPositions(ref_pose);
    reference_skeleton->setPositions(ref_pose);
    subject_muscle->UpdateGeometry();
    reference_muscle->UpdateGeometry();

    // Detect bound hits
    constexpr double kBoundTolerance = 1e-4;
    result.num_bound_hits = 0;
    for (int k = 0; k < N_opt; ++k) {
        int ai = opt_indices[k];
        bool is_origin_insertion = (ai == 0 || ai == static_cast<int>(anchors.size()) - 1);
        double max_disp = is_origin_insertion
            ? config.maxDisplacementOriginInsertion
            : config.maxDisplacement;

        for (int dim = 0; dim < 3; ++dim) {
            double disp = std::abs(unified_params[3*k + dim] - initial_positions[k][dim]);
            if (disp >= max_disp - kBoundTolerance) {
                result.num_bound_hits++;
                if (config.verbose) {
                    LOG_WARN("[WaypointOpt] Anchor " << ai << " dim " << dim
                             << " hit bound: disp=" << disp << " max=" << max_disp);
                }
                break;
            }
        }
    }

    // Check convergence
    const bool converged = (summary.termination_type == ceres::CONVERGENCE);
    if (!converged) {
        std::string reason;
        switch (summary.termination_type) {
            case ceres::NO_CONVERGENCE:   reason = "NO_CONVERGENCE"; break;
            case ceres::FAILURE:          reason = "FAILURE"; break;
            case ceres::USER_FAILURE:     reason = "USER_FAILURE"; break;
            default:                      reason = "Unknown"; break;
        }
        LOG_WARN("[WaypointOpt] Did not converge: " << reason
                 << " (cost " << summary.initial_cost << " -> " << summary.final_cost << ")");
    }

    // 8. Compute final energies
    computeEnergies(result.final_shape_energy, result.final_length_energy, result.final_total_cost, "Final");

    // 9. Compute "after" curves for visualization
    subject_skeleton->setPositions(ref_pose);
    result.subject_after_lengths = computeMuscleLengthCurveWithDOF(
        subject_muscle, subject_skeleton, config.numSampling, dof_config, config.lengthType);
    result.subject_after_chars = analyzeLengthCurve(result.subject_after_lengths, config.numPhaseSamples);

    computePerPhaseShapeMetrics(result, subject_muscle, reference_muscle,
        subject_skeleton, reference_skeleton,
        config.numSampling, dof_config, ref_pose, false);

    // 10. Revert if optimization made things worse
    bool ceres_cost_increased = summary.final_cost > summary.initial_cost;
    bool total_energy_increased = result.final_total_cost > result.initial_total_cost;

    if (ceres_cost_increased || total_energy_increased) {
        LOG_WARN("[WaypointOpt] " << subject_muscle->name << " FAILED:"
                 << " Ceres(" << summary.initial_cost << "->" << summary.final_cost << ")"
                 << " Shape(" << result.initial_shape_energy << "->" << result.final_shape_energy << ")"
                 << " Length(" << result.initial_length_energy << "->" << result.final_length_energy << ")"
                 << " — reverting");

        for (int k = 0; k < N_opt; ++k) {
            int ai = opt_indices[k];
            anchors[ai]->local_positions[0] = Eigen::Vector3d(
                initial_positions[k][0], initial_positions[k][1], initial_positions[k][2]);
        }

        // Re-initialize muscle state with restored positions
        subject_skeleton->setPositions(Eigen::VectorXd::Zero(subject_skeleton->getNumDofs()));
        subject_muscle->SetMuscle();
        subject_skeleton->setPositions(ref_pose);
    }

    result.success = converged && !ceres_cost_increased && !total_energy_increased;
    restorePoses();
    return result;
}

} // namespace PMuscle
