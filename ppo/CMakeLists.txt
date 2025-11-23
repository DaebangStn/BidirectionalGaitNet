set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()



link_directories(../sim/)
include_directories(../sim/)
include_directories($ENV{CONDA_PREFIX}/include/eigen3)

# GymEnvManager module (existing single-env wrapper)
set(gymenv_srcs
    GymEnvManager.cpp
)
add_library(gymenv SHARED ${gymenv_srcs})
target_link_libraries(gymenv
        ${Python3_LIBRARIES}
        sim
        pybind11::embed
        ${TORCH_LIBRARIES}
)
set_target_properties(gymenv PROPERTIES PREFIX "" )

# BatchEnv module (new high-performance batched wrapper)
set(batchenv_srcs
    BatchEnv.cpp
)
add_library(batchenv SHARED ${batchenv_srcs})
target_link_libraries(batchenv
        ${Python3_LIBRARIES}
        sim
        pybind11::embed
        ${TORCH_LIBRARIES}
)
set_target_properties(batchenv PROPERTIES PREFIX "" )

# BatchRolloutEnv module (autonomous C++ rollout with libtorch inference)
# Includes PolicyNet, TrajectoryBuffer, and NUMAThreadPool compiled directly
set(batchrolloutenv_srcs
    BatchRolloutEnv.cpp
    TrajectoryBuffer.cpp
    PolicyNet.cpp
    NUMAThreadPool.cpp
)
add_library(batchrolloutenv SHARED ${batchrolloutenv_srcs})
target_link_libraries(batchrolloutenv
        ${Python3_LIBRARIES}
        sim
        pybind11::embed
        ${TORCH_LIBRARIES}
        numa  # libnuma for NUMA-aware thread affinity
)
set_target_properties(batchrolloutenv PROPERTIES PREFIX "" )
