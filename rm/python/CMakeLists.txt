# Python bindings for Resource Manager
pybind11_add_module(pyrm bindings.cpp)

target_link_libraries(pyrm PRIVATE rm)

# Output to rm/ directory for package structure (like python/pysim.so)
set_target_properties(pyrm PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)
