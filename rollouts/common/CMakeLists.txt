# Shared rollout components (RolloutRecord, RecordConfig, PyRolloutRecord)
add_library(rollout_common
    RolloutRecord.cpp
    PyRolloutRecord.cpp
)

target_include_directories(rollout_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/sim
)

# Find Eigen
find_package(Eigen3 REQUIRED)

target_link_libraries(rollout_common PUBLIC
    yaml-cpp
    Eigen3::Eigen
    pybind11::embed
)
