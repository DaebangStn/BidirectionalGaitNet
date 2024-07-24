cmake_minimum_required(VERSION 3.12)
project(mysim)

add_compile_options(-fPIC)
add_compile_options(-Wdeprecated-declarations)
add_compile_options(-DHAVE_CSTDDEF)

set(CMAKE_AR gcc-ar)
set(CMAKE_RANLIB gcc-ranlib)

set(OpenGL_GL_PREFERENCE LEGACY)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_PREFIX_PATH $ENV{ENVDIR})
set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, 
      options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
  set(CMAKE_C_FLAGS_DEBUG "-g -O0")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops -flto")
  set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -funroll-loops -flto")
endif()

include_directories($ENV{ENVDIR}/include)
include_directories(${PYTHON_INCLUDE_DIR})

link_libraries(stdc++fs)
link_directories($ENV{ENVDIR}/lib)

# Common packages
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

add_subdirectory( libs )
add_subdirectory( sim )
add_subdirectory( python )
if(NOT SERVER_BUILD)
add_subdirectory( viewer )
endif()
