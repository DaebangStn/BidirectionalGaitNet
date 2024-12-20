cmake_minimum_required(VERSION 3.28)
project(bidirectional)

add_compile_options(-fPIC)
add_compile_options(-Wdeprecated-declarations)
add_compile_options(-DHAVE_CSTDDEF)

set(CMAKE_AR gcc-ar)
set(CMAKE_RANLIB gcc-ranlib)

set(OpenGL_GL_PREFERENCE LEGACY)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_PREFIX_PATH $ENV{ENVDIR})
set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake")
set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX} ${CMAKE_PREFIX_PATH})
set(CMAKE_PREFIX_PATH $ENV{ENVDIR} ${CMAKE_PREFIX_PATH})
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=gold")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, 
      options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -march=native")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops -flto=30")
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -funroll-loops -flto=30")

# Common packages
find_package(Python3 EXACT 3.8 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
include_directories(${Python3_INCLUDE_DIRS})
link_libraries(${Python3_LIBRARIES})
find_package(pybind11 REQUIRED)
link_libraries(stdc++fs)

add_subdirectory( sim )
add_subdirectory( python )
if(NOT SERVER_BUILD)
  add_subdirectory( libs )
  add_subdirectory( viewer )
endif()
