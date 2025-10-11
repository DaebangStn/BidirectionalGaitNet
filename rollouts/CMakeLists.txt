# Add sim directory to include path
include_directories(../sim/)
link_directories(../sim/)

# Rollout library sources (exclude Python binding files)
set(ROLLOUT_SOURCES
    RolloutRecord.cpp
    RolloutEnvironment.cpp
)

# Create rollout library
add_library(rollout ${ROLLOUT_SOURCES})
target_include_directories(rollout PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../sim)
target_link_libraries(rollout PUBLIC sim)

# Python bindings module
set(PYROLLOUT_SOURCES
    PyRolloutRecord.cpp
    PyRolloutEnvironment.cpp
)

pybind11_add_module(pyrollout ${PYROLLOUT_SOURCES})
target_link_libraries(pyrollout PRIVATE rollout sim)
target_include_directories(pyrollout PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../sim)
set_target_properties(pyrollout PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../python
)
