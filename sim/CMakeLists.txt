find_package(TinyXML2 REQUIRED)
find_package(DART REQUIRED COMPONENTS collision-bullet)
find_package(CURL REQUIRED)
find_package(yaml-cpp REQUIRED)

file(GLOB srcs "*.h" "*.cpp")

add_library(sim ${srcs})
target_include_directories(sim PUBLIC
        ${CMAKE_SOURCE_DIR}/viewer
)
target_link_libraries(sim PUBLIC
        tinyxml2::tinyxml2
        dart
        dart-collision-bullet
        CURL::libcurl
        yaml-cpp
        ${HDF5_CXX_LIBRARIES}
)
