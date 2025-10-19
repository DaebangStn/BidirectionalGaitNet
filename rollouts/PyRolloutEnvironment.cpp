#include "RolloutEnvironment.h"
#include "PyRolloutRecord.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_RolloutEnvironment(py::module& m) {
    py::class_<RolloutEnvironment>(m, "RolloutEnvironment")
        .def(py::init<const std::string&>())
        .def("load_config", &RolloutEnvironment::LoadRecordConfig)
        .def("reset", &RolloutEnvironment::Reset)
        .def("get_state", &RolloutEnvironment::GetState)
        .def("set_action", &RolloutEnvironment::SetAction)
        .def("step", [](RolloutEnvironment& env, PyRolloutRecord* record) {
            // PyRolloutRecord inherits from RolloutRecord, so this is safe
            env.Step(static_cast<RolloutRecord*>(record));
        }, py::arg("record") = nullptr)
        .def("get_cycle_count", &RolloutEnvironment::GetCycleCount)
        .def("is_eoe", &RolloutEnvironment::IsEndOfEpisode)
        .def("get_record_fields", &RolloutEnvironment::GetRecordFields)
        .def("get_simulation_hz", &RolloutEnvironment::GetSimulationHz)
        .def("get_control_hz", &RolloutEnvironment::GetControlHz)
        .def("set_mcn_weights", &RolloutEnvironment::SetMuscleNetworkWeight)
        .def("set_parameters", &RolloutEnvironment::SetParameters,
             py::arg("params"),
             "Set simulation parameters from a dictionary {param_name: value}")
        .def("get_parameter_names", &RolloutEnvironment::GetParameterNames,
             "Get list of available parameter names")
        .def("get_param_state", &RolloutEnvironment::GetParamState,
             py::arg("is_mirror") = false,
             "Get current parameter state as vector")
        .def("get_param_default", &RolloutEnvironment::GetParamDefault,
             "Get default parameter values as vector");
}

PYBIND11_MODULE(pyrollout, m) {
    bind_PyRolloutRecord(m);
    bind_RolloutEnvironment(m);
}

