cmake_minimum_required(VERSION 3.28)

set(CMAKE_C_COMPILER $ENV{CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER $ENV{CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++)

project(bidirectional)

add_compile_options(-fPIC)
add_compile_options(-Wdeprecated-declarations)
add_compile_options(-DHAVE_CSTDDEF)

# Define PROJECT_ROOT for URI resolution
add_compile_definitions(PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

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
set(CMAKE_PREFIX_PATH $ENV{HOME}/pkgenv ${CMAKE_PREFIX_PATH})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, 
      options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -march=native")
# set(CMAKE_C_FLAGS_DEBUG "-O0 -g -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops -flto=30")
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -funroll-loops -flto=30")

# Common packages
set(Python3_EXECUTABLE $ENV{CONDA_PREFIX}/bin/python)
set(Python3_INCLUDE_DIRS $ENV{CONDA_PREFIX}/include/python3.10)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")

include_directories(${Python3_INCLUDE_DIRS})
link_libraries(${Python3_LIBRARIES})

find_package(pybind11 REQUIRED)
link_libraries(stdc++fs)

# Find libtorch (PyTorch C++ API)
set(CAFFE2_USE_CUDNN ON)

# CUDA paths are set via CMakePresets.json (CUDAToolkit_ROOT, CUDA_TOOLKIT_ROOT_DIR, CMAKE_CUDA_COMPILER)
# Default values if not set by preset (for backward compatibility)
if(NOT DEFINED CUDAToolkit_ROOT)
    set(CUDAToolkit_ROOT "/usr/local/cuda")
endif()
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
endif()
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

set(CMAKE_CUDA_ARCHITECTURES "native")  # Auto-detect GPU architecture

find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "Torch found: ${TORCH_LIBRARIES}")
message(STATUS "Torch CXX flags: ${TORCH_CXX_FLAGS}")

# Find native HDF5 C++ library from conda environment
find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold -Wl,-O1 -Wl,--as-needed -Wl,--hash-style=gnu")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=gold -Wl,-O1 -Wl,--as-needed -Wl,--hash-style=gnu")

add_subdirectory( rm )
add_subdirectory( sim )
add_subdirectory( rollouts )
add_subdirectory( python )
add_subdirectory( ppo )
if(NOT SERVER_BUILD)
  add_subdirectory( surgery )
  add_subdirectory( libs )
  add_subdirectory( viewer )
  add_subdirectory( tools )
endif()
