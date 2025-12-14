#ifndef ROLLOUT_RECORD_H
#define ROLLOUT_RECORD_H

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <unordered_map>

// Configuration for what fields to record
struct RecordConfig {
    // Basic fields (always recorded)
    bool step = true;
    bool time = true;
    bool phase = true;
    bool cycle = true;

    // Contact/GRF fields
    struct FootConfig {
        bool enabled = false;
        bool contact_left = false;
        bool contact_right = false;
        bool grf_left = false;
        bool grf_right = false;
    } foot;

    // Kinematics
    struct KinematicsConfig {
        bool enabled = false;

        // All joint positions (getSkeleton()->getPositions())
        bool all = false;

        // Root position
        bool root = false;  // Records root_x, root_y, root_z

        // Specific angle fields
        struct AngleConfig {
            bool enabled = false;
            bool hip = false;      // HipR: FemurR joint position(0)
            bool hip_ir = false;   // HipIRR: FemurR joint position(1)
            bool hip_ab = false;   // HipAbR: FemurR joint position(2)
            bool knee = false;     // KneeR: TibiaR joint position(0)
            bool ankle = false;    // AnkleR: TalusR joint position(0)
            bool pelvic_tilt = false;     // Pelvis joint position(0)
            bool pelvic_rotation = false; // Pelvis joint position(1)
            bool pelvic_obliquity = false; // Pelvis joint position(2)
        } angle;

        // Angular velocity fields
        struct AnvelConfig {
            bool enabled = false;
            bool hip = false;
            bool knee = false;
            bool ankle = false;
        } anvel;

        // Sway and foot progression
        struct SwayConfig {
            bool enabled = false;
            bool foot = false;        // sway/FootR, sway/FootL (X position relative to root)
            bool toe = false;         // sway/ToeR, sway/ToeL (Y position)
            bool fpa = false;         // sway/FPA_R, sway/FPA_L (Foot Progression Angle)
            bool anteversion = false; // sway/AnteversionR, sway/AnteversionL
            bool torso = false;       // sway/Torso (X position relative to root)
        } sway;
    } kinematics;

    // Metabolic energy
    struct MetabolicConfig {
        bool enabled = false;
        std::string type = "LEGACY";  // LEGACY, A, A2, MA, MA2
        bool step_energy = false;      // Record per-step energy (array)
        bool cumulative = false;       // Record cumulative energy per cycle (computed in Python filter)
    } metabolic;

    // Muscle recording (per-muscle vectors stored in matrix_data)
    struct MuscleConfig {
        bool enabled = false;
        bool activation = false;   // Per-muscle activation vector
        bool passive = false;      // Per-muscle passive force vector
        bool force = false;        // Per-muscle total force vector
        bool lm_norm = false;      // Per-muscle normalized length (lm_rel / lm_opt)
    } muscle;

    static RecordConfig LoadFromYAML(const std::string& yaml_path);
};

class RolloutRecord {
public:
    explicit RolloutRecord(const std::vector<std::string>& field_names);
    explicit RolloutRecord(const RecordConfig& config);
    virtual ~RolloutRecord() = default;

    // Add data for a simulation step
    virtual void add(unsigned int sim_step, const std::unordered_map<std::string, float>& data) = 0;

    // Add vector data for a specific step (for matrix datasets)
    virtual void addVector(const std::string& key, int step, const Eigen::VectorXf& data) = 0;

    // Getters
    unsigned int get_nrow() const { return mNrow; }
    unsigned int get_ncol() const { return mNcol; }
    const Eigen::MatrixXf& get_data() const { return mData; }
    const std::vector<std::string>& get_fields() const { return mFieldNames; }
    const std::unordered_map<std::string, int>& get_field_to_idx() const { return mFieldToIdx; }
    
    // Reset for reuse
    void reset();
    
    // Build field list from config
    static std::vector<std::string> FieldsFromConfig(const RecordConfig& config, int skeleton_dof = 0);
    static std::vector<std::string> VectorFieldsFromConfig(const RecordConfig& config);

private:
    void resize_if_needed(unsigned int requested_size);

    Eigen::MatrixXf mData;
    std::unordered_map<std::string, int> mFieldToIdx;
    std::vector<std::string> mFieldNames;
    unsigned int mNcol;
    unsigned int mNrow;
    
    static constexpr unsigned int DATA_CHUNK_SIZE = 1000;
};

#endif // ROLLOUT_RECORD_H

