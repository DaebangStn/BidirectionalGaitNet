# Use ncurses from conda environment
set(CURSES_LIBRARY "$ENV{CONDA_PREFIX}/lib/libncursesw.so")
set(CURSES_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")

add_executable(extract_cycle extract_cycle.cpp)

target_link_libraries(extract_cycle
    ${HDF5_CXX_LIBRARIES}
    ${CURSES_LIBRARY}
)

target_include_directories(extract_cycle PRIVATE
    ${HDF5_INCLUDE_DIRS}
    ${CURSES_INCLUDE_DIR}
)

# HDF DOF Remapper - Convert 56-DOF to 50-DOF motion files
add_executable(hdf_dof_remapper hdf_dof_remapper.cpp)

target_link_libraries(hdf_dof_remapper
    ${HDF5_CXX_LIBRARIES}
)

target_include_directories(hdf_dof_remapper PRIVATE
    ${HDF5_INCLUDE_DIRS}
)

# C3D Merge Tool - Merge Trimmed_*.c3d files into unified C3D
find_package(ezc3d CONFIG REQUIRED)

add_executable(c3d_merge c3d_merge.cpp)

target_link_libraries(c3d_merge
    rm
    ezc3d
)

target_include_directories(c3d_merge PRIVATE
    ${CMAKE_SOURCE_DIR}/rm/include
)

# C3D Inspect Tool - Inspect C3D file metadata (frames, labels)
add_executable(c3d_inspect c3d_inspect.cpp)

target_link_libraries(c3d_inspect
    rm
    ezc3d
    ${CURSES_LIBRARY}
)

target_include_directories(c3d_inspect PRIVATE
    ${CMAKE_SOURCE_DIR}/rm/include
    ${CURSES_INCLUDE_DIR}
)
