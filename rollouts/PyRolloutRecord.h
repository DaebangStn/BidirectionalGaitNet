#ifndef PY_ROLLOUT_RECORD_H
#define PY_ROLLOUT_RECORD_H

#include "RolloutRecord.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class PyRolloutRecord : public RolloutRecord {
public:
    using RolloutRecord::RolloutRecord;
    
    void add(unsigned int sim_step, const std::unordered_map<std::string, double>& data) override {
        RolloutRecord::add(sim_step, data);
    }

    void addVector(const std::string& key, int step, const Eigen::VectorXd& data) override {
        RolloutRecord::addVector(key, step, data);
    }

    py::array_t<double> get_data_array() const;
};

void bind_PyRolloutRecord(py::module& m);

#endif // PY_ROLLOUT_RECORD_H
