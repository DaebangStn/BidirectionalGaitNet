cmake_minimum_required(VERSION 3.12)
project(sim)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

find_package(TinyXML2 REQUIRED)
find_package(DART REQUIRED COMPONENTS dart gui collision-bullet)

file(GLOB srcs "*.h" "*.cpp")

add_library(sim ${srcs})
target_link_libraries(sim ${DART_LIBRARIES} tinyxml2::tinyxml2)
