#ifndef PY_ROLLOUT_RECORD_H
#define PY_ROLLOUT_RECORD_H

#include "RolloutRecord.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <unordered_map>
#include <string>

namespace py = pybind11;

class PyRolloutRecord : public RolloutRecord {
public:
    using RolloutRecord::RolloutRecord;
    
    void add(unsigned int sim_step, const std::unordered_map<std::string, float>& data) override {
        RolloutRecord::add(sim_step, data);
    }

    void addVector(const std::string& key, int step, const Eigen::VectorXf& data) override;

    void reset();

    py::array_t<float> get_data_array() const;
    py::dict get_matrix_data() const;
    py::dict get_cycle_attributes() const;

    void addCycleAttribute(int cycle_idx, const std::string& key, float value);

private:
    void resize_matrix_if_needed(const std::string& key, unsigned int requested_size, unsigned int cols);

    // Store matrix data separately from scalar data
    std::unordered_map<std::string, Eigen::MatrixXf> mMatrixData;
    std::unordered_map<std::string, unsigned int> mMatrixRows;

    // Store cycle-level scalar attributes (e.g., metabolic/cumulative)
    // Map: cycle_idx -> (attribute_name -> value)
    std::unordered_map<int, std::unordered_map<std::string, float>> mCycleAttributes;

    static constexpr unsigned int MATRIX_DATA_CHUNK_SIZE = 1000;
};

void bind_PyRolloutRecord(py::module& m);

#endif // PY_ROLLOUT_RECORD_H
