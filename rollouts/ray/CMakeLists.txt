# Ray-based rollout (RolloutEnvironment + pyrollout bindings)

# Rollout library
add_library(rollout
    RolloutEnvironment.cpp
)

target_include_directories(rollout PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/rollouts/common
    ${CMAKE_SOURCE_DIR}/sim
)

target_link_libraries(rollout PUBLIC
    rollout_common
    sim
)

# Python bindings module (PyRolloutRecord is now in common/)
pybind11_add_module(pyrollout
    PyRolloutEnvironment.cpp
)

target_link_libraries(pyrollout PRIVATE
    rollout
    rollout_common
    sim
)

target_include_directories(pyrollout PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/rollouts/common
    ${CMAKE_SOURCE_DIR}/sim
)

set_target_properties(pyrollout PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/rollout
)
