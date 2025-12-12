# Resource Manager Core
cmake_minimum_required(VERSION 3.14)

# Find dependencies
find_package(CURL REQUIRED)
find_package(yaml-cpp REQUIRED)

# Collect sources
set(RM_SOURCES
    src/uri.cpp
    src/handle.cpp
    src/manager.cpp
    src/backends/local.cpp
    src/backends/ftp.cpp
    src/backends/pid.cpp
)

set(RM_HEADERS
    include/rm/rm.hpp
    include/rm/error.hpp
    include/rm/uri.hpp
    include/rm/handle.hpp
    include/rm/backend.hpp
    include/rm/manager.hpp
    include/rm/backends/local.hpp
    include/rm/backends/ftp.hpp
    include/rm/backends/pid.hpp
)

# Create library
add_library(rm SHARED ${RM_SOURCES} ${RM_HEADERS})

target_include_directories(rm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(rm PUBLIC
    CURL::libcurl
    yaml-cpp
    stdc++fs
)

# Python bindings
add_subdirectory(python)

# Tests
add_executable(test_rm tests/test_rm.cpp)
target_link_libraries(test_rm rm)
