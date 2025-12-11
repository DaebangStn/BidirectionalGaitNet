#include "PyRolloutRecord.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

void PyRolloutRecord::resize_matrix_if_needed(const std::string& key, unsigned int requested_size, unsigned int cols) {
    if (mMatrixData.find(key) == mMatrixData.end() || requested_size >= mMatrixData.at(key).rows()) {
        unsigned int new_rows = (requested_size / MATRIX_DATA_CHUNK_SIZE + 1) * MATRIX_DATA_CHUNK_SIZE;
        Eigen::MatrixXf new_data(new_rows, cols);
        if (mMatrixData.find(key) != mMatrixData.end() && mMatrixRows.at(key) > 0) {
            new_data.topRows(mMatrixRows.at(key)) = mMatrixData.at(key).topRows(mMatrixRows.at(key));
        }
        mMatrixData[key] = std::move(new_data);
    }
}

void PyRolloutRecord::addVector(const std::string& key, int step, const Eigen::VectorXf& data) {
    if (mMatrixData.find(key) == mMatrixData.end()) {
        mMatrixRows[key] = 0;
    }
    resize_matrix_if_needed(key, step + 1, data.size());
    mMatrixData.at(key).row(step) = data;
    if (step >= mMatrixRows.at(key)) {
        mMatrixRows[key] = step + 1;
    }
}

void PyRolloutRecord::reset() {
    RolloutRecord::reset();
    mMatrixData.clear();
    mMatrixRows.clear();
    mCycleAttributes.clear();
}

void PyRolloutRecord::addCycleAttribute(int cycle_idx, const std::string& key, float value) {
    mCycleAttributes[cycle_idx][key] = value;
}

py::array_t<float> PyRolloutRecord::get_data_array() const {
    const auto& data = get_data();

    // Return numpy array with shape (nrow, ncol)
    py::array_t<float> arr({get_nrow(), get_ncol()});
    auto buf = arr.mutable_unchecked<2>();
    
    for (size_t i = 0; i < get_nrow(); ++i) {
        for (size_t j = 0; j < get_ncol(); ++j) {
            buf(i, j) = data(i, j);
        }
    }
    
    return arr;
}

py::dict PyRolloutRecord::get_matrix_data() const {
    py::dict dict;
    for (const auto& [key, matrix] : mMatrixData) {
        unsigned int num_rows = mMatrixRows.at(key);
        unsigned int num_cols = matrix.cols();

        // Create numpy float32 array explicitly to avoid pybind11 auto-conversion to float64
        py::array_t<float> arr({num_rows, num_cols});
        auto buf = arr.mutable_unchecked<2>();

        for (size_t i = 0; i < num_rows; ++i) {
            for (size_t j = 0; j < num_cols; ++j) {
                buf(i, j) = matrix(i, j);
            }
        }

        dict[py::str(key)] = arr;
    }
    return dict;
}

py::dict PyRolloutRecord::get_cycle_attributes() const {
    py::dict outer_dict;
    for (const auto& [cycle_idx, attrs] : mCycleAttributes) {
        py::dict inner_dict;
        for (const auto& [key, value] : attrs) {
            inner_dict[py::str(key)] = value;
        }
        outer_dict[py::int_(cycle_idx)] = inner_dict;
    }
    return outer_dict;
}

void bind_PyRolloutRecord(py::module& m) {
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

    py::class_<PyRolloutRecord>(m, "RolloutRecord")
        .def(py::init<const std::vector<std::string>&>())
        .def(py::init<const RecordConfig&>())
        .def_property_readonly("data", &PyRolloutRecord::get_data_array)
        .def_property_readonly("matrix_data", &PyRolloutRecord::get_matrix_data)
        .def_property_readonly("cycle_attributes", &PyRolloutRecord::get_cycle_attributes)
        .def_property_readonly("fields", &PyRolloutRecord::get_fields)
        .def_property_readonly("nrow", &PyRolloutRecord::get_nrow)
        .def_property_readonly("ncol", &PyRolloutRecord::get_ncol)
        .def("reset", &PyRolloutRecord::reset)
        .def_static("FieldsFromConfig", &RolloutRecord::FieldsFromConfig,
                    py::arg("config"), py::arg("skeleton_dof") = 0);
}

