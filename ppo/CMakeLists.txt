set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

link_directories(../sim/)
include_directories(../sim/)

file(GLOB srcs "*.h" "*.cpp")

add_library(gymenv SHARED ${srcs})

target_link_libraries(gymenv
        ${Python3_LIBRARIES}
        sim
        pybind11::embed
)
set_target_properties(gymenv PROPERTIES PREFIX "" )
