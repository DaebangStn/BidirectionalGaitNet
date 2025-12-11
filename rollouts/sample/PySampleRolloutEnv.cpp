#include "RolloutSampleEnv.h"
#include "RolloutRecord.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

PYBIND11_MODULE(pysamplerollout, m) {
    m.doc() = "Single-core sample rollout module (pysamplerollout)";

    // RecordConfig binding (same as pyrollout for compatibility)
    py::class_<RecordConfig>(m, "RecordConfig")
        .def(py::init<>())
        .def_static("load_from_yaml", &RecordConfig::LoadFromYAML)
        .def_readwrite("metabolic", &RecordConfig::metabolic);

    py::class_<RecordConfig::MetabolicConfig>(m, "MetabolicConfig")
        .def(py::init<>())
        .def_readwrite("enabled", &RecordConfig::MetabolicConfig::enabled)
        .def_readwrite("type", &RecordConfig::MetabolicConfig::type)
        .def_readwrite("step_energy", &RecordConfig::MetabolicConfig::step_energy)
        .def_readwrite("cumulative", &RecordConfig::MetabolicConfig::cumulative);

    // RolloutSampleEnv binding
    py::class_<RolloutSampleEnv>(m, "RolloutSampleEnv")
        .def(py::init<const std::string&>(), py::arg("metadata_xml"),
             "Create RolloutSampleEnv from metadata XML string")

        // Configuration
        .def("load_config", &RolloutSampleEnv::LoadRecordConfig, py::arg("yaml_path"),
             "Load record configuration from YAML file")
        .def("set_target_cycles", &RolloutSampleEnv::SetTargetCycles, py::arg("cycles"),
             "Set target number of gait cycles for rollout")

        // Weight loading
        .def("load_policy_weights", &RolloutSampleEnv::LoadPolicyWeights, py::arg("state_dict"),
             "Load policy network weights from PyTorch state_dict")
        .def("load_muscle_weights", &RolloutSampleEnv::LoadMuscleWeights, py::arg("weights"),
             "Load muscle network weights (for hierarchical control)")

        // Main rollout function
        .def("collect_rollout", &RolloutSampleEnv::CollectRollout,
             py::arg("param_dict") = py::none(),
             "Run complete rollout and return results dict.\n"
             "If param_dict is None, random parameters will be sampled.\n"
             "Returns dict with: data, matrix_data, fields, param_state, cycle_attributes, success, metrics")

        // Queries
        .def("get_state_dim", &RolloutSampleEnv::GetStateDim,
             "Get observation/state dimension")
        .def("get_action_dim", &RolloutSampleEnv::GetActionDim,
             "Get action dimension")
        .def("get_skeleton_dof", &RolloutSampleEnv::GetSkeletonDOF,
             "Get skeleton degrees of freedom")
        .def("get_mass", &RolloutSampleEnv::GetMass,
             "Get character mass")
        .def("get_parameter_names", &RolloutSampleEnv::GetParameterNames,
             "Get list of parameter names")
        .def("get_record_fields", &RolloutSampleEnv::GetRecordFields,
             "Get list of record field names")
        .def("is_hierarchical", &RolloutSampleEnv::IsHierarchical,
             "Check if using hierarchical (muscle) control")

        // Properties
        .def_property_readonly("target_cycles", &RolloutSampleEnv::GetTargetCycles,
             "Target number of gait cycles");
}
