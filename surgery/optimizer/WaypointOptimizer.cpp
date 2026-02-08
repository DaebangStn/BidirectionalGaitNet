// Waypoint Optimizer - Ceres-based muscle waypoint optimization
// Migrated from SkelGen/generator/MuscleGenerator.cpp

#include "WaypointOptimizer.h"
#include "Log.h"
#include <ceres/ceres.h>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <unordered_map>

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

    struct JointDOF { int dof_idx; double lower, upper; };
    std::vector<JointDOF> joint_dofs;  // all DOFs of the best DOF's parent joint

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

static void findBestJointDOFs(DOFSweepConfig& config, dart::dynamics::SkeletonPtr skeleton) {
    auto* joint = skeleton->getDof(config.dof_idx)->getJoint();
    for (int d = 0; d < static_cast<int>(joint->getNumDofs()); ++d) {
        int idx = static_cast<int>(joint->getIndexInSkeleton(d));
        double lo = std::max(skeleton->getPositionLowerLimit(idx), -M_PI);
        double hi = std::min(skeleton->getPositionUpperLimit(idx),  M_PI);
        config.joint_dofs.push_back({idx, lo, hi});
    }
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
    bool multiDofJointSweep = false;

    static std::shared_ptr<OptimizationContext> create(
        Muscle* subj_muscle, Muscle* ref_muscle,
        dart::dynamics::SkeletonPtr subj_skel, dart::dynamics::SkeletonPtr ref_skel,
        int anchor_idx, int samples,
        const DOFSweepConfig& dof_cfg, const Eigen::VectorXd& pose,
        const LengthCurveCharacteristics& chars, bool is_verbose = false,
        double weight_phase = 1.0, double weight_delta = 50.0,
        double weight_samples = 1.0, int num_phase_samples = 3,
        int loss_power = 2, LengthCurveType length_type = LengthCurveType::MTU_LENGTH,
        bool adaptive_sample_weight = false, bool multi_dof_joint_sweep = false)
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
        ctx->multiDofJointSweep = multi_dof_joint_sweep;
        return ctx;
    }
};

// Cached reference anchor world positions per sweep sample (for Jacobian optimization)
struct CachedSweepRef {
    // ref_positions[sample_idx] maps anchor_index -> world position
    std::vector<std::unordered_map<int, Eigen::Vector3d>> ref_positions;
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

template<typename Func>
static void sweepMultiDOF(const OptimizationContext& ctx, Func&& func) {
    int sample_idx = 0;
    for (const auto& jdof : ctx.dof_config.joint_dofs) {
        for (int s = 0; s <= ctx.num_samples; ++s) {
            double t = static_cast<double>(s) / ctx.num_samples;
            double dof_val = jdof.lower + t * (jdof.upper - jdof.lower);
            Eigen::VectorXd pose = ctx.ref_pose;
            pose[jdof.dof_idx] = dof_val;
            ctx.subject_skeleton->setPositions(pose);
            ctx.reference_skeleton->setPositions(pose);
            ctx.subject_muscle->UpdateGeometry();
            ctx.reference_muscle->UpdateGeometry();
            func(sample_idx++);
        }
    }
}

// Subject-only sweep variants — skip reference skeleton/muscle update (for Jacobian with cached ref)
template <typename Func>
static void sweepDOFSubjectOnly(const OptimizationContext& ctx, Func&& func) {
    for (int sample = 0; sample <= ctx.num_samples; ++sample) {
        double t = static_cast<double>(sample) / ctx.num_samples;
        double dof_val = ctx.dof_config.lower + t * (ctx.dof_config.upper - ctx.dof_config.lower);

        Eigen::VectorXd pose = ctx.ref_pose;
        pose[ctx.dof_config.dof_idx] = dof_val;

        ctx.subject_skeleton->setPositions(pose);
        ctx.subject_muscle->UpdateGeometry();

        func(sample);
    }
}

template <typename Func>
static void sweepMultiDOFSubjectOnly(const OptimizationContext& ctx, Func&& func) {
    int sample_idx = 0;
    for (const auto& jdof : ctx.dof_config.joint_dofs) {
        for (int s = 0; s <= ctx.num_samples; ++s) {
            double t = static_cast<double>(s) / ctx.num_samples;
            double dof_val = jdof.lower + t * (jdof.upper - jdof.lower);
            Eigen::VectorXd pose = ctx.ref_pose;
            pose[jdof.dof_idx] = dof_val;
            ctx.subject_skeleton->setPositions(pose);
            ctx.subject_muscle->UpdateGeometry();
            func(sample_idx++);
        }
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
 * For NORMALIZED mode, recomputes lmt_ref at zero pose before sweeping.
 * This creates a moving normalization target that tracks the current
 * anchor positions, which improves optimization results.
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

    Eigen::VectorXd saved_pose = skeleton->getPositions();

    // For NORMALIZED mode, recompute lmt_ref at zero pose before sweeping
    // This ensures lm_norm is correctly normalized for the current anchor positions
    if (length_type == LengthCurveType::NORMALIZED) {
        skeleton->setPositions(Eigen::VectorXd::Zero(skeleton->getNumDofs()));
        muscle->SetMuscle();
    }

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

    auto callback = [&](int) {
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
    };

    if (ctx.multiDofJointSweep && !ctx.dof_config.joint_dofs.empty()) {
        sweepMultiDOF(ctx, callback);
    } else {
        sweepDOF(ctx, callback);
    }

    return (count > 0) ? energy / count : 0.0;
}

// Compute shape energy AND populate reference position cache (for forward evaluation)
static double computeShapeEnergyWithCache(const OptimizationContext& ctx, CachedSweepRef& cache) {
    double energy = 0.0;
    int count = 0;
    int idx = ctx.anchor_index;
    int sample_counter = 0;

    // Determine total samples for cache sizing
    int total_samples;
    if (ctx.multiDofJointSweep && !ctx.dof_config.joint_dofs.empty()) {
        total_samples = static_cast<int>(ctx.dof_config.joint_dofs.size()) * (ctx.num_samples + 1);
    } else {
        total_samples = ctx.num_samples + 1;
    }
    cache.ref_positions.resize(total_samples);

    auto callback = [&](int sample_idx) {
        auto& subj_anchors = ctx.subject_muscle->GetAnchors();
        auto& ref_anchors = ctx.reference_muscle->GetAnchors();

        // Cache reference positions for this sample
        auto& ref_map = cache.ref_positions[sample_counter];
        if (idx > 0) {
            ref_map[idx] = ref_anchors[idx]->GetPoint();
            ref_map[idx - 1] = ref_anchors[idx - 1]->GetPoint();
        }
        if (idx < static_cast<int>(ref_anchors.size()) - 1) {
            ref_map[idx] = ref_anchors[idx]->GetPoint();
            ref_map[idx + 1] = ref_anchors[idx + 1]->GetPoint();
        }
        ++sample_counter;

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
    };

    if (ctx.multiDofJointSweep && !ctx.dof_config.joint_dofs.empty()) {
        sweepMultiDOF(ctx, callback);
    } else {
        sweepDOF(ctx, callback);
    }

    return (count > 0) ? energy / count : 0.0;
}

// Compute shape energy using cached reference positions (subject-only sweep for Jacobian)
static double computeShapeEnergyCached(const OptimizationContext& ctx, const CachedSweepRef& cache) {
    double energy = 0.0;
    int count = 0;
    int idx = ctx.anchor_index;
    int sample_counter = 0;

    auto callback = [&](int sample_idx) {
        auto& subj_anchors = ctx.subject_muscle->GetAnchors();
        const auto& ref_map = cache.ref_positions[sample_counter];
        ++sample_counter;

        // Segment before: anchor is END point
        if (idx > 0) {
            auto it_cur = ref_map.find(idx);
            auto it_prev = ref_map.find(idx - 1);
            if (it_cur != ref_map.end() && it_prev != ref_map.end()) {
                Eigen::Vector3d subj_seg = subj_anchors[idx]->GetPoint() - subj_anchors[idx-1]->GetPoint();
                Eigen::Vector3d ref_seg = it_cur->second - it_prev->second;

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
        }

        // Segment after: anchor is START point
        if (idx < static_cast<int>(subj_anchors.size()) - 1) {
            auto it_cur = ref_map.find(idx);
            auto it_next = ref_map.find(idx + 1);
            if (it_cur != ref_map.end() && it_next != ref_map.end()) {
                Eigen::Vector3d subj_seg = subj_anchors[idx+1]->GetPoint() - subj_anchors[idx]->GetPoint();
                Eigen::Vector3d ref_seg = it_next->second - it_cur->second;

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
        }
    };

    if (ctx.multiDofJointSweep && !ctx.dof_config.joint_dofs.empty()) {
        sweepMultiDOFSubjectOnly(ctx, callback);
    } else {
        sweepDOFSubjectOnly(ctx, callback);
    }

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
    // For NORMALIZED mode, recompute lmt_ref at zero pose first
    // This creates a moving normalization target that tracks current anchor positions
    if (ctx.lengthType == LengthCurveType::NORMALIZED) {
        ctx.subject_skeleton->setPositions(Eigen::VectorXd::Zero(ctx.subject_skeleton->getNumDofs()));
        ctx.subject_muscle->SetMuscle();
    }

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
 * Combined per-anchor cost: shape + length in a single CostFunction.
 * 2 residuals sharing one forward eval and one set of 3 perturbations.
 *
 * residuals[0] = sqrt(2 * lambda_S) * sqrt(E_shape)
 * residuals[1] = sqrt(2 * lambda_L) * sqrt(E_length)
 */
class PerAnchorCombinedCost : public ceres::CostFunction {
public:
    PerAnchorCombinedCost(
        std::shared_ptr<OptimizationContext> ctx,
        int anchor_index,
        double lambda_S,
        double lambda_L)
        : ctx_(std::move(ctx))
        , anchor_index_(anchor_index)
        , lambda_S_(lambda_S)
        , lambda_L_(lambda_L)
    {
        set_num_residuals(2);
        mutable_parameter_block_sizes()->push_back(3);
    }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        const double* params = parameters[0];
        auto& anchors = ctx_->subject_muscle->GetAnchors();

        anchors[anchor_index_]->local_positions[0] = Eigen::Vector3d(
            params[0], params[1], params[2]);
        ctx_->subject_muscle->UpdateGeometry();

        // Forward evaluation: compute both energies in one pass
        CachedSweepRef cache;
        double E_shape, E_length;
        if (jacobians && jacobians[0]) {
            E_shape = computeShapeEnergyWithCache(*ctx_, cache);
        } else {
            E_shape = computeShapeEnergy(*ctx_);
        }
        E_length = computeLengthCurveEnergy(*ctx_);

        double sqrt_Es = std::sqrt(E_shape + kSqrtEpsilon);
        double sqrt_El = std::sqrt(E_length + kSqrtEpsilon);
        double sqrt_2s = std::sqrt(2.0 * lambda_S_);
        double sqrt_2l = std::sqrt(2.0 * lambda_L_);
        residuals[0] = sqrt_2s * sqrt_Es;
        residuals[1] = sqrt_2l * sqrt_El;

        if (jacobians && jacobians[0]) {
            constexpr double h = 1e-6;
            double scale_s = sqrt_2s / (2.0 * sqrt_Es + kEpsilon);
            double scale_l = sqrt_2l / (2.0 * sqrt_El + kEpsilon);
            Eigen::Vector3d orig(params[0], params[1], params[2]);

            // Single set of 3 perturbations for both residuals
            for (int axis = 0; axis < 3; ++axis) {
                anchors[anchor_index_]->local_positions[0] = orig;
                anchors[anchor_index_]->local_positions[0][axis] += h;
                ctx_->subject_muscle->UpdateGeometry();

                double Es_plus = computeShapeEnergyCached(*ctx_, cache);
                double El_plus = computeLengthCurveEnergy(*ctx_);

                // Jacobian layout: jacobians[0] is 2×3 row-major
                // Row 0 = shape residual, Row 1 = length residual
                jacobians[0][0 * 3 + axis] = scale_s * (Es_plus - E_shape) / h;
                jacobians[0][1 * 3 + axis] = scale_l * (El_plus - E_length) / h;
            }

            anchors[anchor_index_]->local_positions[0] = orig;
            ctx_->subject_muscle->UpdateGeometry();
        }
        return true;
    }

private:
    std::shared_ptr<OptimizationContext> ctx_;
    int anchor_index_;
    double lambda_S_;
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

    if (config.multiDofJointSweep) {
        findBestJointDOFs(dof_config, subject_skeleton);
        if (config.verbose) {
            auto* joint = subject_skeleton->getDof(dof_config.dof_idx)->getJoint();
            LOG_INFO("[WaypointOpt] Multi-DOF joint sweep: joint '"
                     << joint->getName() << "' has "
                     << dof_config.joint_dofs.size() << " DOFs");
        }
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

    // Save initial positions for bounds and revert
    std::vector<Eigen::Vector3d> initial_positions(N_opt);
    for (int k = 0; k < N_opt; ++k) {
        int ai = opt_indices[k];
        initial_positions[k] = anchors[ai]->local_positions[0];
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

    // Capture initial world-space segment directions at ref_pose for force direction diagnostic
    std::vector<Eigen::Vector3d> initial_world_points(anchors.size());
    if (config.verbose) {
        subject_skeleton->setPositions(ref_pose);
        subject_muscle->UpdateGeometry();
        for (size_t i = 0; i < anchors.size(); ++i) {
            initial_world_points[i] = anchors[i]->GetPoint();
        }
    }

    // 6. Create optimization contexts and Ceres solver options
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.function_tolerance = config.functionTolerance;
    options.gradient_tolerance = config.gradientTolerance;
    options.parameter_tolerance = config.parameterTolerance;
    options.minimizer_progress_to_stdout = false;

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
            config.adaptiveSampleWeight, config.multiDofJointSweep);
        contexts.push_back(ctx);
    }

    // Compute energies from current anchor state
    auto computeEnergies = [&](double& shape_out, double& length_out, double& total_out, const char* label) {
        subject_muscle->UpdateGeometry();

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

    // Early-skip: if initial energy is negligible, muscle is already well-aligned
    constexpr double kEarlySkipThreshold = 1e-8;
    if (result.initial_total_cost < kEarlySkipThreshold) {
        if (config.verbose) {
            LOG_INFO("[WaypointOpt] " << subject_muscle->name
                     << ": skipping (initial_total_cost=" << result.initial_total_cost << " < threshold)");
        }
        result.final_shape_energy = result.initial_shape_energy;
        result.final_length_energy = result.initial_length_energy;
        result.final_total_cost = result.initial_total_cost;
        result.num_iterations = 0;
        result.num_bound_hits = 0;

        subject_skeleton->setPositions(ref_pose);
        result.subject_after_lengths = computeMuscleLengthCurveWithDOF(
            subject_muscle, subject_skeleton, config.numSampling, dof_config, config.lengthType);
        result.subject_after_chars = analyzeLengthCurve(result.subject_after_lengths, config.numPhaseSamples);

        computePerPhaseShapeMetrics(result, subject_muscle, reference_muscle,
            subject_skeleton, reference_skeleton,
            config.numSampling, dof_config, ref_pose, false);

        result.success = true;
        restorePoses();
        return result;
    }

    // ========================================================================
    // DIAGNOSTIC 1: Purity check — call f(x) 5 times at same state
    // ========================================================================
    if (config.verbose && !contexts.empty()) {
        LOG_INFO("[WaypointOpt] === PURITY CHECK (5 repeated calls at same x) ===");

        subject_muscle->UpdateGeometry();

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

    // 7. Sequential per-anchor optimization loop
    //    config.maxIterations = max outer passes
    //    Stop when ALL anchors converge in the same pass (parameter displacement check)
    int total_outer_iters = 0;

    for (int outer = 0; outer < config.maxIterations; ++outer) {
        int num_converged = 0;

        for (int k = 0; k < N_opt; ++k) {
            int ai = opt_indices[k];
            Eigen::Vector3d saved_pos = anchors[ai]->local_positions[0];

            double params[3] = {
                saved_pos.x(), saved_pos.y(), saved_pos.z()
            };

            // 3-param Ceres problem for this single anchor
            ceres::Problem problem;
            problem.AddParameterBlock(params, 3);

            // Bounds: initial_positions[k] ± maxDisp (fixed across all outer iters)
            bool is_origin_insertion = (ai == 0 || ai == static_cast<int>(anchors.size()) - 1);
            double max_disp = is_origin_insertion
                ? config.maxDisplacementOriginInsertion
                : config.maxDisplacement;
            for (int dim = 0; dim < 3; ++dim) {
                problem.SetParameterLowerBound(params, dim, initial_positions[k][dim] - max_disp);
                problem.SetParameterUpperBound(params, dim, initial_positions[k][dim] + max_disp);
            }

            // Add combined residual (shape + length in one cost function)
            problem.AddResidualBlock(
                new PerAnchorCombinedCost(contexts[k], ai,
                    config.lambdaShape, config.lambdaLengthCurve),
                nullptr, params);

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            // Apply immediately (next anchor sees updated neighbor)
            Eigen::Vector3d new_pos(params[0], params[1], params[2]);
            anchors[ai]->local_positions[0] = new_pos;
            subject_muscle->UpdateGeometry();

            // Per-anchor convergence: check if parameters barely moved
            double displacement = (new_pos - saved_pos).norm();
            if (displacement < config.parameterTolerance) {
                ++num_converged;
            }
        }

        ++total_outer_iters;

        if (config.verbose && (total_outer_iters % 50 == 0 || num_converged == N_opt)) {
            LOG_INFO("[WaypointOpt] outer " << total_outer_iters
                     << ": converged " << num_converged << "/" << N_opt);
        }

        if (num_converged == N_opt) break;
    }

    result.num_iterations = total_outer_iters;

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

    // Force direction change diagnostic
    if (config.verbose) {
        LOG_INFO("[WaypointOpt] Force direction change (ref_pose):");
        int n = static_cast<int>(anchors.size());
        for (int i = 0; i < n; ++i) {
            Eigen::Vector3d after_pt = anchors[i]->GetPoint();

            // Incoming segment direction (from previous anchor)
            if (i > 0) {
                Eigen::Vector3d prev_before = initial_world_points[i] - initial_world_points[i-1];
                Eigen::Vector3d prev_after  = after_pt - anchors[i-1]->GetPoint();
                double nb = prev_before.norm(), na = prev_after.norm();
                if (nb > kEpsilon && na > kEpsilon) {
                    Eigen::Vector3d ub = prev_before / nb;
                    Eigen::Vector3d ua = prev_after / na;
                    double dot = std::clamp(ub.dot(ua), -1.0, 1.0);
                    double angle_deg = std::acos(dot) * 180.0 / M_PI;
                    LOG_INFO("  [" << i << "] seg_in:  before=" << ub.transpose()
                             << "  after=" << ua.transpose()
                             << "  angle=" << std::fixed << std::setprecision(2) << angle_deg << "°");
                }
            }

            // Outgoing segment direction (to next anchor)
            if (i < n - 1) {
                Eigen::Vector3d next_before = initial_world_points[i+1] - initial_world_points[i];
                Eigen::Vector3d next_after  = anchors[i+1]->GetPoint() - after_pt;
                double nb = next_before.norm(), na = next_after.norm();
                if (nb > kEpsilon && na > kEpsilon) {
                    Eigen::Vector3d ub = next_before / nb;
                    Eigen::Vector3d ua = next_after / na;
                    double dot = std::clamp(ub.dot(ua), -1.0, 1.0);
                    double angle_deg = std::acos(dot) * 180.0 / M_PI;
                    LOG_INFO("  [" << i << "] seg_out: before=" << ub.transpose()
                             << "  after=" << ua.transpose()
                             << "  angle=" << std::fixed << std::setprecision(2) << angle_deg << "°");
                }
            }
        }
    }

    // Detect bound hits
    constexpr double kBoundTolerance = 1e-4;
    result.num_bound_hits = 0;
    result.num_anchors = static_cast<int>(anchors.size());
    result.bound_hit_indices.clear();
    for (int k = 0; k < N_opt; ++k) {
        int ai = opt_indices[k];
        bool is_origin_insertion = (ai == 0 || ai == static_cast<int>(anchors.size()) - 1);
        double max_disp = is_origin_insertion
            ? config.maxDisplacementOriginInsertion
            : config.maxDisplacement;

        Eigen::Vector3d current_pos = anchors[ai]->local_positions[0];
        for (int dim = 0; dim < 3; ++dim) {
            double disp = std::abs(current_pos[dim] - initial_positions[k][dim]);
            if (disp >= max_disp - kBoundTolerance) {
                result.num_bound_hits++;
                result.bound_hit_indices.push_back(ai);
                if (config.verbose) {
                    LOG_WARN("[WaypointOpt] Anchor " << ai << " dim " << dim
                             << " hit bound: disp=" << disp << " max=" << max_disp);
                }
                break;
            }
        }
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
    bool total_energy_increased = result.final_total_cost > result.initial_total_cost;

    if (total_energy_increased) {
        LOG_WARN("[WaypointOpt] " << subject_muscle->name << " FAILED:"
                 << " Shape(" << result.initial_shape_energy << "->" << result.final_shape_energy << ")"
                 << " Length(" << result.initial_length_energy << "->" << result.final_length_energy << ")"
                 << " — reverting");

        for (int k = 0; k < N_opt; ++k) {
            int ai = opt_indices[k];
            anchors[ai]->local_positions[0] = initial_positions[k];
        }

        // Re-initialize muscle state with restored positions
        subject_skeleton->setPositions(Eigen::VectorXd::Zero(subject_skeleton->getNumDofs()));
        subject_muscle->SetMuscle();
        subject_skeleton->setPositions(ref_pose);
    }

    result.success = !total_energy_increased;
    restorePoses();
    return result;
}

} // namespace PMuscle
