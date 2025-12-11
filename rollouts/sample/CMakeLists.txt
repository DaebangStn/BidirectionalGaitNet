# Sample rollout library (single-core, libtorch-based)

# Sample rollout library (includes PolicyNet.cpp from ppo/)
add_library(sample_rollout
    RolloutSampleEnv.cpp
    ${CMAKE_SOURCE_DIR}/ppo/PolicyNet.cpp
)

target_include_directories(sample_rollout PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/rollouts/common
    ${CMAKE_SOURCE_DIR}/sim
    ${CMAKE_SOURCE_DIR}/ppo
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(sample_rollout PUBLIC
    rollout_common
    sim
    ${TORCH_LIBRARIES}
)

# Python bindings module
pybind11_add_module(pysamplerollout
    PySampleRolloutEnv.cpp
)

target_link_libraries(pysamplerollout PRIVATE
    sample_rollout
    rollout_common
    sim
)

target_include_directories(pysamplerollout PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/rollouts/common
    ${CMAKE_SOURCE_DIR}/sim
    ${CMAKE_SOURCE_DIR}/ppo
    ${TORCH_INCLUDE_DIRS}
)

set_target_properties(pysamplerollout PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/rollouts/sample
)
