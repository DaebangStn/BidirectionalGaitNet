cmake_minimum_required(VERSION 3.12)
project(mysim)

add_compile_options(-fvisibility=hidden)

link_directories(../sim/)
include_directories(../sim/)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(TinyXML2 REQUIRED)

file(GLOB srcs "*.h" "*.cpp")

add_executable(viewer "main.cpp" ${srcs})
target_link_libraries(viewer ${Python_LIBRARIES} tinyxml2::tinyxml2 GL GLU glut glad pybind11::embed glfw imgui sim)