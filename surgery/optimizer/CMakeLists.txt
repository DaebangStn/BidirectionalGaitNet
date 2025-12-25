# Ceres-based optimizers for surgery module

file(GLOB OPTIMIZER_SRCS
    "*.h"
    "*.cpp"
)
find_package(Ceres REQUIRED)

add_library(surgery_optimizer STATIC ${OPTIMIZER_SRCS})

# Link required libraries
# Note: sim already links DART and Eigen, no need to repeat
target_link_libraries(surgery_optimizer
    PUBLIC
    sim                    # For Character, Muscle, HDF classes (includes DART, Eigen)
    Ceres::ceres          # Ceres solver (automatic differentiation, optimization)
)

# Include directories
target_include_directories(surgery_optimizer
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/sim     # For sim/ headers (Character.h, HDF.h, etc.)
    ${CMAKE_SOURCE_DIR}/viewer  # For viewer/ headers (C3D_Reader.h, RenderCharacter.h)
)

# Inherit compile definitions from sim
target_compile_definitions(surgery_optimizer
    PRIVATE
    $<TARGET_PROPERTY:sim,COMPILE_DEFINITIONS>
)

# C++17 required
set_target_properties(surgery_optimizer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
