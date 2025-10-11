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
        .def("step", &RolloutEnvironment::Step, py::arg("record") = nullptr)
        .def("get_cycle_count", &RolloutEnvironment::GetCycleCount)
        .def("is_eoe", &RolloutEnvironment::IsEndOfEpisode)
        .def("get_record_fields", &RolloutEnvironment::GetRecordFields)
        .def("get_simulation_hz", &RolloutEnvironment::GetSimulationHz)
        .def("get_control_hz", &RolloutEnvironment::GetControlHz);
}

PYBIND11_MODULE(pyrollout, m) {
    bind_PyRolloutRecord(m);
    bind_RolloutEnvironment(m);
}

