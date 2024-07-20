cmake_minimum_required(VERSION 3.12)
project(sim)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

find_package(TinyXML2 REQUIRED)

file(GLOB srcs "*.h" "*.cpp")

add_library(sim ${srcs})
target_link_libraries(sim ${DART_LIBRARIES} ${TinyXML_LIBRARIES})
