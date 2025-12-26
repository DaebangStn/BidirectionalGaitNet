// Waypoint Optimizer - Ceres-based implementation with analytical gradients
// Migrated from SkelGen/generator/MuscleGenerator.cpp

#include "WaypointOptimizer.h"
#include "Log.h"
#include <ceres/ceres.h>
#include <algorithm>
#include <memory>
#include <iomanip>

namespace PMuscle {

// ============================================================================
// Constants
// ============================================================================

static constexpr double kEpsilon = 1e-8;
static constexpr double kSqrtEpsilon = 1e-10;

// Default weights for length curve energy (can be overridden via Config)
static constexpr double kDefaultPhaseWeight = 1.0;
static constexpr double kDefaultDeltaWeight = 50.0;

// Solver tolerances
static constexpr double kFunctionTolerance = 1e-4;
static constexpr double kGradientTolerance = 1e-5;
static constexpr double kParameterTolerance = 1e-5;

// Maximum anchor displacement from initial position (30cm)
static constexpr double kMaxDisplacement = 0.3;

// ============================================================================
// Loss Function Helper
// ============================================================================

/**
 * @brief Apply power loss: |x|^power
 */
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
 * @brief Find the DOF with largest influence on muscle length
 *
 * Uses Getdl_dtheta() to compute ∂L/∂θ for each DOF.
 * Selects the DOF with largest absolute derivative magnitude.
 */
static DOFSweepConfig findBestDOF(
    Muscle* muscle,
    dart::dynamics::SkeletonPtr skeleton)
{
    DOFSweepConfig config;

    // UpdateGeometry to refresh mCachedAnchorPositions before computing Jacobians
    muscle->UpdateGeometry();

    // Compute length-DOF Jacobian (calls ComputeJacobians internally)
    Eigen::VectorXd dl_dtheta = muscle->Getdl_dtheta();

    // Find DOF with largest influence on muscle length
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

    // Get and clamp position limits
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
    bool useAnalyticalGradient = true;
    double weightPhase = kDefaultPhaseWeight;
    double weightDelta = kDefaultDeltaWeight;
    double weightSamples = 1.0;
    int numPhaseSamples = 3;
    int lossPower = 2;
    LengthCurveType lengthType = LengthCurveType::MTU_LENGTH;

    // Pointers to ALL parameter blocks (for cross-anchor sync)
    std::vector<double*> all_param_blocks;

    // Sync all anchors EXCEPT the current one from parameter blocks
    // The current anchor should already be set from the params passed to Evaluate
    void syncOtherAnchors() const {
        auto& anchors = subject_muscle->GetAnchors();
        for (size_t i = 0; i < all_param_blocks.size() && i < anchors.size(); ++i) {
            // Skip the current anchor - it's already set from Evaluate's params
            if (static_cast<int>(i) == anchor_index) continue;

            if (all_param_blocks[i]) {
                anchors[i]->local_positions[0] = Eigen::Vector3d(
                    all_param_blocks[i][0],
                    all_param_blocks[i][1],
                    all_param_blocks[i][2]);
            }
        }
        subject_muscle->UpdateGeometry();
    }

    // Factory method
    static std::shared_ptr<OptimizationContext> create(
        Muscle* subj_muscle,
        Muscle* ref_muscle,
        dart::dynamics::SkeletonPtr subj_skel,
        dart::dynamics::SkeletonPtr ref_skel,
        int anchor_idx,
        int samples,
        const DOFSweepConfig& dof_cfg,
        const Eigen::VectorXd& pose,
        const LengthCurveCharacteristics& chars,
        bool is_verbose = false,
        bool analytical_grad = true,
        double weight_phase = kDefaultPhaseWeight,
        double weight_delta = kDefaultDeltaWeight,
        double weight_samples = 1.0,
        int num_phase_samples = 3,
        int loss_power = 2,
        LengthCurveType length_type = LengthCurveType::MTU_LENGTH)
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
        ctx->useAnalyticalGradient = analytical_grad;
        ctx->weightPhase = weight_phase;
        ctx->weightDelta = weight_delta;
        ctx->weightSamples = weight_samples;
        ctx->numPhaseSamples = num_phase_samples;
        ctx->lossPower = loss_power;
        ctx->lengthType = length_type;
        return ctx;
    }
};

// ============================================================================
// DOF Sweep Helper
// ============================================================================

/**
 * @brief Execute a function at each DOF sweep position
 */
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
// Helper Functions
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
 * @brief Compute muscle length curve using provided DOF config (for internal use during optimization)
 *
 * IMPORTANT: Use this version during optimization to avoid recomputing DOF selection.
 *
 * @param length_type MTU_LENGTH uses lmt (raw MTU length), NORMALIZED uses lm_norm
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

    Eigen::VectorXd ref_pose = skeleton->getPositions();

    // For NORMALIZED mode, recompute lmt_ref at zero pose before sweeping
    // This ensures lm_norm is correctly normalized for the current anchor positions
    if (length_type == LengthCurveType::NORMALIZED) {
        skeleton->setPositions(Eigen::VectorXd::Zero(skeleton->getNumDofs()));
        muscle->UpdateGeometry();
        muscle->updateLmtRef();  // Safe: only updates lmt_ref, not related_dof_indices
    }

    for (int i = 0; i < num_samples; ++i) {
        double t = static_cast<double>(i) / (num_samples - 1);
        double dof_value = dof_config.lower + t * (dof_config.upper - dof_config.lower);

        Eigen::VectorXd pose = ref_pose;
        pose[dof_config.dof_idx] = dof_value;
        skeleton->setPositions(pose);

        muscle->UpdateGeometry();
        double val = (length_type == LengthCurveType::NORMALIZED) ? muscle->lm_norm : muscle->lmt;
        lengths.push_back(val);
    }

    skeleton->setPositions(ref_pose);
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

    // Sample at evenly-spaced phases (e.g., 0, 0.5, 1.0 for N=3)
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
            // Linear interpolation
            double value = (1.0 - t) * lengths[idx_lo] + t * lengths[idx_hi];
            chars.phase_samples.push_back(value);
        }
    }

    return chars;
}

// ============================================================================
// Segment Gradient Computation
// ============================================================================

/**
 * @brief Compute gradient contribution from a single segment for fShape
 *
 * @param seg Subject segment vector (from start to end)
 * @param ref_seg Reference segment vector
 * @param R Rotation matrix of anchor's body
 * @param is_end_point True if anchor is at end of segment, false if at start
 * @return Gradient contribution in local coordinates
 */
static Eigen::Vector3d computeSegmentShapeGradient(
    const Eigen::Vector3d& seg,
    const Eigen::Vector3d& ref_seg,
    const Eigen::Matrix3d& R,
    bool is_end_point)
{
    double len = seg.norm();
    if (len < kEpsilon) return Eigen::Vector3d::Zero();

    Eigen::Vector3d d = seg / len;
    double ref_len = ref_seg.norm();
    if (ref_len < kEpsilon) return Eigen::Vector3d::Zero();

    Eigen::Vector3d ref_d = ref_seg / ref_len;
    Eigen::Vector3d cross = d.cross(ref_d);
    double cross_norm = cross.norm();

    if (cross_norm < kEpsilon) return Eigen::Vector3d::Zero();

    // ∂||a×b||/∂a = b×(a×b) / ||a×b|| = -(a×b)×b / ||a×b||
    Eigen::Vector3d dcross_dd = ref_d.cross(cross) / cross_norm;

    // ∂d/∂p = ±(I - d*d^T) / len
    // + for end point (moving end increases segment in d direction)
    // - for start point (moving start decreases segment in d direction)
    double sign = is_end_point ? 1.0 : -1.0;
    Eigen::Matrix3d dd_dp = sign * (Eigen::Matrix3d::Identity() - d * d.transpose()) / len;

    return R.transpose() * dd_dp.transpose() * dcross_dd;
}

// ============================================================================
// Energy Functions - fShape
// ============================================================================

/**
 * @brief Compute shape energy for segments ADJACENT to the current anchor only.
 *
 * This ensures each anchor's cost function is independent of other anchors'
 * optimization state, avoiding shared mutable state issues in Ceres.
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
                Eigen::Vector3d subj_dir = subj_seg / subj_len;
                Eigen::Vector3d ref_dir = ref_seg / ref_len;
                energy += subj_dir.cross(ref_dir).norm();
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
                Eigen::Vector3d subj_dir = subj_seg / subj_len;
                Eigen::Vector3d ref_dir = ref_seg / ref_len;
                energy += subj_dir.cross(ref_dir).norm();
                ++count;
            }
        }
    });

    return (count > 0) ? energy / count : 0.0;
}

/**
 * @brief Compute per-phase shape energy for all segments
 *
 * Unlike computeShapeEnergy which focuses on segments adjacent to a single anchor,
 * this computes the total direction misalignment across ALL muscle segments at each phase.
 *
 * @return Vector of angles (in degrees) at each phase, showing direction misalignment
 */
static std::vector<double> computePerPhaseShapeEnergy(
    Muscle* subject_muscle,
    Muscle* reference_muscle,
    dart::dynamics::SkeletonPtr subject_skeleton,
    dart::dynamics::SkeletonPtr reference_skeleton,
    int num_samples,
    const DOFSweepConfig& dof_config,
    const Eigen::VectorXd& ref_pose)
{
    std::vector<double> per_phase_energy;
    per_phase_energy.reserve(num_samples);

    for (int sample = 0; sample < num_samples; ++sample) {
        double t = static_cast<double>(sample) / (num_samples - 1);
        double dof_val = dof_config.lower + t * (dof_config.upper - dof_config.lower);

        Eigen::VectorXd pose = ref_pose;
        pose[dof_config.dof_idx] = dof_val;

        subject_skeleton->setPositions(pose);
        reference_skeleton->setPositions(pose);
        subject_muscle->UpdateGeometry();
        reference_muscle->UpdateGeometry();

        // Compute cross product sum for all segments at this phase
        auto& subj_anchors = subject_muscle->GetAnchors();
        auto& ref_anchors = reference_muscle->GetAnchors();

        double cross_sum = 0.0;
        int segment_count = 0;

        for (size_t i = 0; i + 1 < subj_anchors.size(); ++i) {
            Eigen::Vector3d subj_seg = subj_anchors[i+1]->GetPoint() - subj_anchors[i]->GetPoint();
            Eigen::Vector3d ref_seg = ref_anchors[i+1]->GetPoint() - ref_anchors[i]->GetPoint();

            double subj_len = subj_seg.norm();
            double ref_len = ref_seg.norm();

            if (subj_len > kEpsilon && ref_len > kEpsilon) {
                Eigen::Vector3d subj_dir = subj_seg / subj_len;
                Eigen::Vector3d ref_dir = ref_seg / ref_len;
                cross_sum += subj_dir.cross(ref_dir).norm();
                ++segment_count;
            }
        }

        // Convert to angle in degrees: asin(||cross||) * 180 / PI
        double avg_cross = (segment_count > 0) ? cross_sum / segment_count : 0.0;
        // Clamp to [0, 1] for numerical safety
        avg_cross = std::min(1.0, std::max(0.0, avg_cross));
        double angle_deg = std::asin(avg_cross) * 180.0 / M_PI;
        per_phase_energy.push_back(angle_deg);
    }

    return per_phase_energy;
}

static Eigen::Vector3d computeShapeGradient(const OptimizationContext& ctx) {
    Eigen::Vector3d gradient = Eigen::Vector3d::Zero();
    int count = 0;  // Same counting as energy function
    int idx = ctx.anchor_index;

    auto& subj_anchors = ctx.subject_muscle->GetAnchors();
    auto& ref_anchors = ctx.reference_muscle->GetAnchors();

    // Store original pose for restoration
    Eigen::VectorXd original_pose = ctx.subject_skeleton->getPositions();

    sweepDOF(ctx, [&](int) {
        // R must be computed at EACH pose since skeleton rotates during sweep
        // For blended anchors: ∂global/∂local[0] = weights[0] * R[0]
        Eigen::Matrix3d R = subj_anchors[idx]->weights[0] *
                            subj_anchors[idx]->bodynodes[0]->getTransform().linear();

        // Segment before: anchor is END point
        if (idx > 0) {
            Eigen::Vector3d seg = subj_anchors[idx]->GetPoint() - subj_anchors[idx-1]->GetPoint();
            Eigen::Vector3d ref_seg = ref_anchors[idx]->GetPoint() - ref_anchors[idx-1]->GetPoint();
            gradient += computeSegmentShapeGradient(seg, ref_seg, R, true);
            ++count;  // Same counting as energy
        }

        // Segment after: anchor is START point
        if (idx < static_cast<int>(subj_anchors.size()) - 1) {
            Eigen::Vector3d seg = subj_anchors[idx+1]->GetPoint() - subj_anchors[idx]->GetPoint();
            Eigen::Vector3d ref_seg = ref_anchors[idx+1]->GetPoint() - ref_anchors[idx]->GetPoint();
            gradient += computeSegmentShapeGradient(seg, ref_seg, R, false);
            ++count;  // Same counting as energy
        }
    });

    // Restore original pose
    ctx.subject_skeleton->setPositions(original_pose);
    ctx.subject_muscle->UpdateGeometry();

    if (count > 0) {
        return gradient / static_cast<double>(count);
    }
    return Eigen::Vector3d::Zero();
}

/**
 * @brief Compute shape gradient using numeric differentiation for verification
 */
static Eigen::Vector3d computeShapeGradientNumeric(const OptimizationContext& ctx) {
    constexpr double h = 1e-6;  // Finite difference step
    Eigen::Vector3d gradient;

    int idx = ctx.anchor_index;
    auto& anchor = ctx.subject_muscle->GetAnchors()[idx];
    Eigen::Vector3d orig_local = anchor->local_positions[0];

    for (int axis = 0; axis < 3; ++axis) {
        // f(x + h)
        anchor->local_positions[0] = orig_local;
        anchor->local_positions[0][axis] += h;
        ctx.subject_muscle->UpdateGeometry();
        double f_plus = computeShapeEnergy(ctx);

        // f(x - h)
        anchor->local_positions[0] = orig_local;
        anchor->local_positions[0][axis] -= h;
        ctx.subject_muscle->UpdateGeometry();
        double f_minus = computeShapeEnergy(ctx);

        gradient[axis] = (f_plus - f_minus) / (2.0 * h);
    }

    // Restore original position
    anchor->local_positions[0] = orig_local;
    ctx.subject_muscle->UpdateGeometry();

    return gradient;
}

// ============================================================================
// Energy Functions - fLengthCurve
// ============================================================================

static double computeLengthCurveEnergy(const OptimizationContext& ctx) {
    // Use cached DOF config from context - DO NOT call findBestDOF here!
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
            energy += ctx.weightSamples *
                      applyLoss(subj_chars.phase_samples[i] - ctx.ref_chars.phase_samples[i], ctx.lossPower);
        }
    }

    return energy;
}

/**
 * @brief Helper to compute ∂L/∂local at the current skeleton pose
 */
static Eigen::Vector3d computeLengthGradientAtPose(const OptimizationContext& ctx) {
    int idx = ctx.anchor_index;
    auto& anchors = ctx.subject_muscle->GetAnchors();

    ctx.subject_muscle->UpdateGeometry();

    Eigen::Matrix3d R = anchors[idx]->bodynodes[0]->getTransform().linear();
    Eigen::Vector3d p_i = anchors[idx]->GetPoint();
    Eigen::Vector3d dL_dp = Eigen::Vector3d::Zero();

    // ∂L/∂p[i] = (p[i] - p[i-1])/len_prev + (p[i] - p[i+1])/len_next
    if (idx > 0) {
        Eigen::Vector3d diff = p_i - anchors[idx - 1]->GetPoint();
        double len = diff.norm();
        if (len > kEpsilon) {
            dL_dp += diff / len;
        }
    }

    if (idx < static_cast<int>(anchors.size()) - 1) {
        Eigen::Vector3d diff = p_i - anchors[idx + 1]->GetPoint();
        double len = diff.norm();
        if (len > kEpsilon) {
            dL_dp += diff / len;
        }
    }

    // Transform to local coordinates
    return R.transpose() * dL_dp;
}

/**
 * @brief Compute phase energy only (for numeric differentiation)
 *
 * Uses cached DOF config from context - DO NOT call findBestDOF here!
 */
static double computePhaseEnergy(const OptimizationContext& ctx) {
    // Use cached DOF config from context
    auto lengths = computeMuscleLengthCurveWithDOF(
        ctx.subject_muscle, ctx.subject_skeleton, ctx.num_samples, ctx.dof_config, ctx.lengthType);
    auto subj_chars = WaypointOptimizer::analyzeLengthCurve(lengths, ctx.numPhaseSamples);

    double energy = ctx.weightPhase * applyLoss(subj_chars.min_phase - ctx.ref_chars.min_phase, ctx.lossPower)
                  + ctx.weightPhase * applyLoss(subj_chars.max_phase - ctx.ref_chars.max_phase, ctx.lossPower);

    // Include sample matching in phase energy for gradient computation
    if (ctx.weightSamples > kEpsilon &&
        subj_chars.phase_samples.size() == ctx.ref_chars.phase_samples.size()) {
        for (size_t i = 0; i < subj_chars.phase_samples.size(); ++i) {
            energy += ctx.weightSamples *
                      applyLoss(subj_chars.phase_samples[i] - ctx.ref_chars.phase_samples[i], ctx.lossPower);
        }
    }

    return energy;
}

/**
 * @brief Compute phase gradient numerically (argmin/argmax are non-differentiable)
 */
static Eigen::Vector3d computePhaseGradientNumeric(const OptimizationContext& ctx) {
    constexpr double h = 1e-6;
    Eigen::Vector3d gradient;

    int idx = ctx.anchor_index;
    auto& anchor = ctx.subject_muscle->GetAnchors()[idx];
    Eigen::Vector3d orig_local = anchor->local_positions[0];

    for (int axis = 0; axis < 3; ++axis) {
        anchor->local_positions[0] = orig_local;
        anchor->local_positions[0][axis] += h;
        ctx.subject_muscle->UpdateGeometry();
        double f_plus = computePhaseEnergy(ctx);

        anchor->local_positions[0] = orig_local;
        anchor->local_positions[0][axis] -= h;
        ctx.subject_muscle->UpdateGeometry();
        double f_minus = computePhaseEnergy(ctx);

        gradient[axis] = (f_plus - f_minus) / (2.0 * h);
    }

    anchor->local_positions[0] = orig_local;
    ctx.subject_muscle->UpdateGeometry();

    return gradient;
}

/**
 * @brief Compute delta gradient analytically
 *
 * Uses cached DOF config from context - DO NOT call findBestDOF here!
 */
static Eigen::Vector3d computeDeltaGradient(const OptimizationContext& ctx) {
    // Use cached DOF config from context
    const DOFSweepConfig& dof_config = ctx.dof_config;
    if (!dof_config.isValid()) {
        return Eigen::Vector3d::Zero();
    }

    // Compute length curve to find min/max sample indices
    std::vector<double> lengths;
    lengths.reserve(ctx.num_samples + 1);

    Eigen::VectorXd ref_pose = ctx.subject_skeleton->getPositions();

    // For NORMALIZED mode, recompute lmt_ref at zero pose first
    if (ctx.lengthType == LengthCurveType::NORMALIZED) {
        ctx.subject_skeleton->setPositions(Eigen::VectorXd::Zero(ctx.subject_skeleton->getNumDofs()));
        ctx.subject_muscle->UpdateGeometry();
        ctx.subject_muscle->updateLmtRef();
    }

    for (int sample = 0; sample <= ctx.num_samples; ++sample) {
        double t = static_cast<double>(sample) / ctx.num_samples;
        double dof_val = dof_config.lower + t * (dof_config.upper - dof_config.lower);

        Eigen::VectorXd pose = ref_pose;
        pose[dof_config.dof_idx] = dof_val;
        ctx.subject_skeleton->setPositions(pose);
        ctx.subject_muscle->UpdateGeometry();

        double val = (ctx.lengthType == LengthCurveType::NORMALIZED)
            ? ctx.subject_muscle->lm_norm : ctx.subject_muscle->lmt;
        lengths.push_back(val);
    }

    auto [min_it, max_it] = std::minmax_element(lengths.begin(), lengths.end());
    int min_idx = std::distance(lengths.begin(), min_it);
    int max_idx = std::distance(lengths.begin(), max_it);

    double delta_subj = *max_it - *min_it;
    double delta_ref = ctx.ref_chars.delta;

    // ∂L/∂local at max_pose
    double t_max = static_cast<double>(max_idx) / ctx.num_samples;
    double dof_max = dof_config.lower + t_max * (dof_config.upper - dof_config.lower);
    Eigen::VectorXd pose_max = ref_pose;
    pose_max[dof_config.dof_idx] = dof_max;
    ctx.subject_skeleton->setPositions(pose_max);
    Eigen::Vector3d dL_dlocal_max = computeLengthGradientAtPose(ctx);

    // ∂L/∂local at min_pose
    double t_min = static_cast<double>(min_idx) / ctx.num_samples;
    double dof_min = dof_config.lower + t_min * (dof_config.upper - dof_config.lower);
    Eigen::VectorXd pose_min = ref_pose;
    pose_min[dof_config.dof_idx] = dof_min;
    ctx.subject_skeleton->setPositions(pose_min);
    Eigen::Vector3d dL_dlocal_min = computeLengthGradientAtPose(ctx);

    // ∂delta/∂local = ∂max_length/∂local - ∂min_length/∂local
    Eigen::Vector3d dDelta_dlocal = dL_dlocal_max - dL_dlocal_min;

    // Chain rule: ∂E/∂local = 2 * weightDelta * (delta_subj - delta_ref) * ∂delta/∂local
    double dE_ddelta = 2.0 * ctx.weightDelta * (delta_subj - delta_ref);

    ctx.subject_skeleton->setPositions(ref_pose);
    ctx.subject_muscle->UpdateGeometry();

    return dDelta_dlocal * dE_ddelta;
}

/**
 * @brief Compute length curve gradient: analytical for delta, numeric for phase
 */
static Eigen::Vector3d computeLengthCurveGradient(const OptimizationContext& ctx) {
    Eigen::Vector3d gradient = Eigen::Vector3d::Zero();

    // Delta term: analytical gradient
    if (ctx.weightDelta > kEpsilon) {
        gradient += computeDeltaGradient(ctx);
    }

    // Phase terms: numeric gradient (argmin/argmax are non-differentiable)
    if (ctx.weightPhase > kEpsilon) {
        gradient += computePhaseGradientNumeric(ctx);
    }

    return gradient;
}

/**
 * @brief Compute length curve gradient using numeric differentiation
 */
static Eigen::Vector3d computeLengthCurveGradientNumeric(const OptimizationContext& ctx) {
    constexpr double h = 1e-6;  // Finite difference step
    Eigen::Vector3d gradient;

    int idx = ctx.anchor_index;
    auto& anchor = ctx.subject_muscle->GetAnchors()[idx];
    Eigen::Vector3d orig_local = anchor->local_positions[0];

    for (int axis = 0; axis < 3; ++axis) {
        // f(x + h)
        anchor->local_positions[0] = orig_local;
        anchor->local_positions[0][axis] += h;
        ctx.subject_muscle->UpdateGeometry();
        double f_plus = computeLengthCurveEnergy(ctx);

        // f(x - h)
        anchor->local_positions[0] = orig_local;
        anchor->local_positions[0][axis] -= h;
        ctx.subject_muscle->UpdateGeometry();
        double f_minus = computeLengthCurveEnergy(ctx);

        gradient[axis] = (f_plus - f_minus) / (2.0 * h);
    }

    // Restore original position
    anchor->local_positions[0] = orig_local;
    ctx.subject_muscle->UpdateGeometry();

    return gradient;
}

// ============================================================================
// Ceres Cost Functions - Base Class with Template Method Pattern
// ============================================================================

class WaypointCostFunction : public ceres::SizedCostFunction<1, 3> {
public:
    WaypointCostFunction(std::shared_ptr<OptimizationContext> ctx, double weight)
        : ctx_(std::move(ctx)), weight_(weight), eval_count_(0) {}

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        updateAnchorPosition(parameters[0]);

        double energy = computeEnergy();
        double sqrt_energy = std::sqrt(energy + kSqrtEpsilon);
        residuals[0] = weight_ * sqrt_energy;

        if (jacobians && jacobians[0]) {
            Eigen::Vector3d grad = computeGradient();
            double scale = weight_ / (2.0 * sqrt_energy + kEpsilon);
            jacobians[0][0] = scale * grad.x();
            jacobians[0][1] = scale * grad.y();
            jacobians[0][2] = scale * grad.z();

            // Gradient verification logging (only when using analytical gradient)
            if (ctx_->verbose && ctx_->useAnalyticalGradient && eval_count_ < 3) {
                Eigen::Vector3d num_grad = computeNumericGradientForVerification();
                double scale_num = weight_ / (2.0 * sqrt_energy + kEpsilon);
                Eigen::Vector3d scaled_num = scale_num * num_grad;
                Eigen::Vector3d scaled_ana(jacobians[0][0], jacobians[0][1], jacobians[0][2]);

                double error = (scaled_ana - scaled_num).norm();
                double rel_error = error / (scaled_num.norm() + kEpsilon);

                if (rel_error > 0.1) {  // >10% relative error
                    LOG_WARN("[GradCheck] " << getName() << " anchor=" << ctx_->anchor_index
                             << " rel_err=" << std::fixed << std::setprecision(2) << (rel_error * 100) << "%"
                             << " ana=" << scaled_ana.transpose()
                             << " num=" << scaled_num.transpose());
                }
                ++eval_count_;
            }
        }
        return true;
    }

protected:
    virtual double computeEnergy() const = 0;
    virtual Eigen::Vector3d computeGradient() const = 0;
    virtual Eigen::Vector3d computeNumericGradientForVerification() const = 0;
    virtual const char* getName() const = 0;

    const OptimizationContext& context() const { return *ctx_; }

private:
    void updateAnchorPosition(const double* params) const {
        // Update THIS anchor's position from the passed params (Ceres may pass test values)
        auto& anchor = ctx_->subject_muscle->GetAnchors()[ctx_->anchor_index];
        anchor->local_positions[0] = Eigen::Vector3d(params[0], params[1], params[2]);

        // Sync OTHER anchors from their parameter blocks
        // This ensures neighbor anchors have correct positions for segment computation
        ctx_->syncOtherAnchors();
    }

    std::shared_ptr<OptimizationContext> ctx_;
    double weight_;
    mutable int eval_count_;
};

class ShapeCostFunction final : public WaypointCostFunction {
public:
    using WaypointCostFunction::WaypointCostFunction;

protected:
    double computeEnergy() const override { return computeShapeEnergy(context()); }
    Eigen::Vector3d computeGradient() const override {
        if (context().useAnalyticalGradient) {
            return computeShapeGradient(context());
        } else {
            return computeShapeGradientNumeric(context());
        }
    }
    Eigen::Vector3d computeNumericGradientForVerification() const override {
        return computeShapeGradientNumeric(context());
    }
    const char* getName() const override { return "Shape"; }
};

class LengthCurveCostFunction final : public WaypointCostFunction {
public:
    using WaypointCostFunction::WaypointCostFunction;

protected:
    double computeEnergy() const override { return computeLengthCurveEnergy(context()); }
    Eigen::Vector3d computeGradient() const override {
        // computeLengthCurveGradient now uses:
        // - Analytical gradient for delta term (differentiable)
        // - Numeric gradient for phase terms (argmin/argmax are non-differentiable)
        if (context().useAnalyticalGradient) {
            return computeLengthCurveGradient(context());
        } else {
            return computeLengthCurveGradientNumeric(context());
        }
    }
    Eigen::Vector3d computeNumericGradientForVerification() const override {
        return computeLengthCurveGradientNumeric(context());
    }
    const char* getName() const override { return "LengthCurve"; }
};
// ============================================================================
// Waypoint Parameters RAII Wrapper
// ============================================================================

class WaypointParameters {
public:
    explicit WaypointParameters(const std::vector<Anchor*>& anchors) {
        params_.reserve(anchors.size());
        for (const auto* anchor : anchors) {
            auto pos = std::make_unique<double[]>(3);
            pos[0] = anchor->local_positions[0].x();
            pos[1] = anchor->local_positions[0].y();
            pos[2] = anchor->local_positions[0].z();
            params_.push_back(std::move(pos));
        }
    }

    double* operator[](size_t i) { return params_[i].get(); }
    size_t size() const { return params_.size(); }

    void applyTo(const std::vector<Anchor*>& anchors) const {
        for (size_t i = 0; i < params_.size(); ++i) {
            anchors[i]->local_positions[0] << params_[i][0], params_[i][1], params_[i][2];
        }
    }

private:
    std::vector<std::unique_ptr<double[]>> params_;
};

// ============================================================================
// Main Optimization Function
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

    // Store initial poses for restoration before return
    Eigen::VectorXd initial_subject_pose = subject_skeleton->getPositions();
    Eigen::VectorXd initial_reference_pose = reference_skeleton->getPositions();

    auto restorePoses = [&]() {
        subject_skeleton->setPositions(initial_subject_pose);
        reference_skeleton->setPositions(initial_reference_pose);
    };

    // 1. Check if optimization is needed
    auto& anchors = subject_muscle->GetAnchors();
    if (config.fixOriginInsertion && anchors.size() <= 2) {
        // Only two anchors, so no optimization needed
        result.subject_before_lengths = computeMuscleLengthCurve(
            subject_muscle, subject_skeleton, config.numSampling);
        result.subject_before_chars = analyzeLengthCurve(result.subject_before_lengths, config.numPhaseSamples);
        result.subject_after_lengths = result.subject_before_lengths;
        result.subject_after_chars = result.subject_before_chars;
        result.success = true;
        restorePoses();
        return result;
    }

    // 2. Find best DOF FIRST - use this for all curve computations
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

    // Store DOF info for display
    result.dof_idx = dof_config.dof_idx;
    result.dof_name = subject_skeleton->getDof(dof_config.dof_idx)->getName();

    // 3. Compute curves using SAME DOF for consistency
    // Store length type in result for plot labeling
    result.length_type = config.lengthType;

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

    // Generate phase data
    result.phases.reserve(config.numSampling);
    for (int i = 0; i < config.numSampling; ++i) {
        result.phases.push_back(static_cast<double>(i) / (config.numSampling - 1));
    }

    // DOF already validated above at line 833, just log if verbose
    if (config.verbose) {
        std::string dof_name = subject_skeleton->getDof(dof_config.dof_idx)->getName();
        LOG_INFO("[WaypointOpt] " << subject_muscle->name << ": "
                 << anchors.size() << " anchors, best DOF=" << dof_name
                 << " (" << dof_config.dof_idx << ")"
                 << " [" << dof_config.lower << ", " << dof_config.upper << "]");
    }

    Eigen::VectorXd ref_pose = subject_skeleton->getPositions();

    // Compute per-phase shape energy for "before" state
    result.shape_energy_before = computePerPhaseShapeEnergy(
        subject_muscle, reference_muscle,
        subject_skeleton, reference_skeleton,
        config.numSampling, dof_config, ref_pose);

    // 4. Setup Ceres problem
    ceres::Problem problem;
    WaypointParameters waypoint_params(anchors);

    if (config.verbose) {
        LOG_INFO("[WaypointOpt] Initial anchor positions:");
        for (size_t i = 0; i < anchors.size(); ++i) {
            auto& pos = anchors[i]->local_positions[0];
            LOG_INFO("  [" << i << "] " << pos.transpose());
        }
    }

    // Build vector of all parameter block pointers for cross-anchor sync
    std::vector<double*> all_param_ptrs;
    all_param_ptrs.reserve(waypoint_params.size());
    for (size_t i = 0; i < waypoint_params.size(); ++i) {
        all_param_ptrs.push_back(waypoint_params[i]);
    }

    // Store contexts for setting up all_param_blocks after creation
    std::vector<std::shared_ptr<OptimizationContext>> contexts;

    for (size_t i = 0; i < waypoint_params.size(); ++i) {
        problem.AddParameterBlock(waypoint_params[i], 3);

        bool is_fixed = config.fixOriginInsertion && (i == 0 || i == anchors.size() - 1);
        if (is_fixed) {
            problem.SetParameterBlockConstant(waypoint_params[i]);
            continue;
        }

        // Add bounds to prevent unrealistic anchor movements
        for (int dim = 0; dim < 3; ++dim) {
            double initial_val = waypoint_params[i][dim];
            problem.SetParameterLowerBound(waypoint_params[i], dim, initial_val - kMaxDisplacement);
            problem.SetParameterUpperBound(waypoint_params[i], dim, initial_val + kMaxDisplacement);
        }

        auto ctx = OptimizationContext::create(
            subject_muscle, reference_muscle,
            subject_skeleton, reference_skeleton,
            static_cast<int>(i), config.numSampling,
            dof_config, ref_pose, result.reference_chars, config.verbose,
            config.analyticalGradient, config.weightPhase, config.weightDelta,
            config.weightSamples, config.numPhaseSamples, config.lossPower, config.lengthType);

        // Set up cross-anchor sync pointers
        ctx->all_param_blocks = all_param_ptrs;

        if (config.lambdaShape > 0) {
            problem.AddResidualBlock(
                new ShapeCostFunction(ctx, config.lambdaShape),
                nullptr, waypoint_params[i]);
        }
        if (config.lambdaLengthCurve > 0) {
            problem.AddResidualBlock(
                new LengthCurveCostFunction(ctx, config.lambdaLengthCurve),
                nullptr, waypoint_params[i]);
        }

        contexts.push_back(ctx);
    }

    // 5. Compute initial energies (before optimization)
    // Always compute both energies for display, but total only includes weighted terms
    auto computeEnergies = [&](double& shape_out, double& length_out, double& total_out) {
        shape_out = 0.0;
        length_out = 0.0;
        for (const auto& ctx : contexts) {
            shape_out += computeShapeEnergy(*ctx);
            length_out += computeLengthCurveEnergy(*ctx);
        }
        // Total cost only includes weighted terms (matches Ceres cost)
        total_out = config.lambdaShape * shape_out + config.lambdaLengthCurve * length_out;
    };

    computeEnergies(result.initial_shape_energy, result.initial_length_energy, result.initial_total_cost);

    // 6. Solve
    ceres::Solver::Options options;
    options.max_num_iterations = config.maxIterations;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = config.verbose;
    options.logging_type = config.verbose ? ceres::PER_MINIMIZER_ITERATION : ceres::SILENT;
    options.function_tolerance = kFunctionTolerance;
    options.gradient_tolerance = kGradientTolerance;
    options.parameter_tolerance = kParameterTolerance;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (config.verbose) {
        LOG_INFO("[WaypointOpt] " << summary.BriefReport());
        LOG_INFO("[WaypointOpt] Final anchor positions:");
        for (size_t i = 0; i < waypoint_params.size(); ++i) {
            LOG_INFO("  [" << i << "] " << waypoint_params[i][0] << " "
                     << waypoint_params[i][1] << " " << waypoint_params[i][2]);
        }
    }

    // 6. Check termination type - only CONVERGENCE is success
    const bool converged = (summary.termination_type == ceres::CONVERGENCE);
    const bool hit_max_iter = (summary.termination_type == ceres::NO_CONVERGENCE);

    if (!converged) {
        std::string reason;
        switch (summary.termination_type) {
            case ceres::NO_CONVERGENCE:
                reason = "NO_CONVERGENCE (hit max iterations without converging)";
                break;
            case ceres::FAILURE:
                reason = "FAILURE (solver encountered irrecoverable error)";
                break;
            case ceres::USER_FAILURE:
                reason = "USER_FAILURE (callback indicated failure)";
                break;
            default:
                reason = "Unknown termination type: " + std::to_string(summary.termination_type);
                break;
        }
        LOG_WARN("[WaypointOpt] Optimization did NOT converge: " << reason);
        LOG_WARN("[WaypointOpt] Final cost: " << summary.final_cost
                 << ", Initial cost: " << summary.initial_cost);
    }

    // 7. Apply results (even if not converged, to see what happened)
    waypoint_params.applyTo(anchors);

    // SetMuscle() computes lmt_ref from current pose - must be zero pose
    subject_skeleton->setPositions(Eigen::VectorXd::Zero(subject_skeleton->getNumDofs()));
    subject_muscle->SetMuscle();

    // Restore to base pose for energy computation
    subject_skeleton->setPositions(ref_pose);

    // 8. Compute final energies
    computeEnergies(result.final_shape_energy, result.final_length_energy, result.final_total_cost);

    // 9. Compute final curve using SAME DOF as optimization (not findBestDOF again!)
    result.subject_after_lengths = computeMuscleLengthCurveWithDOF(
        subject_muscle, subject_skeleton, config.numSampling, dof_config, config.lengthType);
    result.subject_after_chars = analyzeLengthCurve(result.subject_after_lengths, config.numPhaseSamples);

    // Compute per-phase shape energy for "after" state
    result.shape_energy_after = computePerPhaseShapeEnergy(
        subject_muscle, reference_muscle,
        subject_skeleton, reference_skeleton,
        config.numSampling, dof_config, ref_pose);

    // Only CONVERGENCE is considered success
    result.success = converged;
    restorePoses();
    return result;
}

} // namespace PMuscle
