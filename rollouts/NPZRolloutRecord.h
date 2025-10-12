#ifndef NPZ_ROLLOUT_RECORD_H
#define NPZ_ROLLOUT_RECORD_H

#include "RolloutRecord.h"
#include <unordered_map>
#include <string>
#include <pybind11/numpy.h>

namespace py = pybind11;

class NPZRolloutRecord : public RolloutRecord {
public:
    NPZRolloutRecord(const std::vector<std::string>& field_names);
    explicit NPZRolloutRecord(const RecordConfig& config);
    ~NPZRolloutRecord() override = default;

    void add(unsigned int sim_step, const std::unordered_map<std::string, double>& data) override;
    void addVector(const std::string& key, int step, const Eigen::VectorXd& data) override;

    // Save the data to an NPZ file
    void saveToFile(const std::string& filename) const;

private:
    // Store matrix data separately from scalar data
    std::unordered_map<std::string, Eigen::MatrixXd> mMatrixData;
    std::unordered_map<std::string, unsigned int> mMatrixRows;
};

#endif // NPZ_ROLLOUT_RECORD_H
