#include "PyRolloutRecord.h"
#include <pybind11/stl.h>

py::array_t<double> PyRolloutRecord::get_data_array() const {
    const auto& data = get_data();
    
    // Return numpy array with shape (nrow, ncol)
    py::array_t<double> arr({get_nrow(), get_ncol()});
    auto buf = arr.mutable_unchecked<2>();
    
    for (size_t i = 0; i < get_nrow(); ++i) {
        for (size_t j = 0; j < get_ncol(); ++j) {
            buf(i, j) = data(i, j);
        }
    }
    
    return arr;
}

void bind_PyRolloutRecord(py::module& m) {
    py::class_<RecordConfig>(m, "RecordConfig")
        .def(py::init<>())
        .def_static("load_from_yaml", &RecordConfig::LoadFromYAML);
    
    py::class_<PyRolloutRecord>(m, "RolloutRecord")
        .def(py::init<const std::vector<std::string>&>())
        .def(py::init<const RecordConfig&>())
        .def_property_readonly("data", &PyRolloutRecord::get_data_array)
        .def_property_readonly("fields", &PyRolloutRecord::get_fields)
        .def_property_readonly("nrow", &PyRolloutRecord::get_nrow)
        .def_property_readonly("ncol", &PyRolloutRecord::get_ncol)
        .def("reset", &PyRolloutRecord::reset);
}

