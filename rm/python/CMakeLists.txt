# Python bindings for Resource Manager
pybind11_add_module(pyrm bindings.cpp)

target_link_libraries(pyrm PRIVATE rm)

# Output to rm/python/ directory (source tree) for easy import
set_target_properties(pyrm PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
