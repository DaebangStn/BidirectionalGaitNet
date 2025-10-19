find_package(Curses REQUIRED)

add_executable(extract_cycle extract_cycle.cpp)

target_link_libraries(extract_cycle
    ${HDF5_CXX_LIBRARIES}
    ${CURSES_LIBRARIES}
)

target_include_directories(extract_cycle PRIVATE
    ${HDF5_INCLUDE_DIRS}
    ${CURSES_INCLUDE_DIR}
)
