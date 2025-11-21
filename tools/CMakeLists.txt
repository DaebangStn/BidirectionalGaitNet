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
