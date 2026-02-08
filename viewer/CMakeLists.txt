add_compile_options(-fvisibility=hidden)

link_directories(../sim/)
include_directories(../sim/)
include_directories(../surgery/)
include_directories(../libs/imgui)
include_directories(../libs/ascii)

find_package(GLUT REQUIRED)
find_package(TinyXML2 REQUIRED)
find_package(DART REQUIRED COMPONENTS gui)
find_package(ezc3d CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

# Common viewer library for shared components
add_library(viewer_common STATIC
    common/PIDImGui.h
    common/PIDImGui.cpp
    common/PIDFileFilter.h
    common/imgui_common.h
    common/imgui_common.cpp
    common/ViewerAppBase.h
    common/ViewerAppBase.cpp
    common/timeline.h
    common/timeline.cpp
    # Shared rendering components
    common/ShapeRenderer.h
    common/ShapeRenderer.cpp
    common/GLfunctions.h
    common/GLfunctions.cpp
)
target_link_libraries(viewer_common PUBLIC
    rm
    dart
    dart-gui
    imgui
    glad
    glfw
    GL
    GLU
    glut
)
target_include_directories(viewer_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/tinycolormap/include
)

# Viewer sources
set(VIEWER_SOURCES
    "render_ckpt_main.cpp"
    "RenderCkpt.h"
    "RenderCkpt.cpp"
    "RenderEnvironment.h"
    "RenderEnvironment.cpp"
    "RenderCharacter.h"
    "RenderCharacter.cpp"
    "C3D_Reader.h"
    "C3D_Reader.cpp"
    # Motion processor infrastructure
    "motion/PlaybackContext.h"
    "motion/PlaybackController.h"
    "motion/PlaybackController.cpp"
    "motion/MotionProcessorFactory.h"
    "motion/MotionProcessorFactory.cpp"
    # Unified motion processor (replaces HDF/C3D processors)
    "motion/MotionProcessor.h"
    "motion/MotionProcessor.cpp"
    # PolicyNet for C++ TorchScript loading (Phase 1: backward compatible)
    "${CMAKE_CURRENT_SOURCE_DIR}/../ppo/PolicyNet.cpp"
)

add_executable(render_ckpt ${VIEWER_SOURCES})

# Add ppo include for PolicyNet
target_include_directories(render_ckpt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../ppo
)

target_link_libraries(render_ckpt
        tinyxml2::tinyxml2
        GL GLU glut glad glfw imgui
        sim
        dart-gui
        ${HDF5_CXX_LIBRARIES}
        ezc3d
        Boost::program_options
        rm
        viewer_common
        surgery_optimizer
)

# Marker Editor - Standalone application for editing marker positions
add_executable(marker_editor
    "marker_editor_main.cpp"
    "MarkerEditorApp.h"
    "MarkerEditorApp.cpp"
    "RenderCharacter.h"
    "RenderCharacter.cpp"
)
target_link_libraries(marker_editor
    tinyxml2::tinyxml2
    sim
    dart-gui
    viewer_common
)

# C3D Processor - Standalone C3D processing application
# NOTE: No HDF5 linking - C3D processor does NOT support HDF
set(C3D_PROCESSOR_SOURCES
    "c3d_processor_main.cpp"
    "C3DProcessorApp.h"
    "C3DProcessorApp.cpp"
    "RenderCharacter.h"
    "RenderCharacter.cpp"
    "C3D_Reader.h"
    "C3D_Reader.cpp"
    "common/PlotUtils.h"
    "common/PlotUtils.cpp"
)

add_executable(c3d_processor ${C3D_PROCESSOR_SOURCES})

target_link_libraries(c3d_processor
    tinyxml2::tinyxml2
    GL GLU glut glad glfw imgui
    sim
    dart-gui
    ezc3d
    Boost::program_options
    rm
    viewer_common
    surgery_optimizer
)

# Motion Editor - Standalone H5 motion editing/trimming application
# Inherits from ViewerAppBase for common GLFW/ImGui/camera handling
add_executable(motion_editor
    "motion_editor_main.cpp"
    "MotionEditorApp.h"
    "MotionEditorApp.cpp"
    "RenderCharacter.h"
    "RenderCharacter.cpp"
)
target_link_libraries(motion_editor
    tinyxml2::tinyxml2
    sim
    dart-gui
    ${HDF5_CXX_LIBRARIES}
    Boost::program_options
    viewer_common
)

# Muscle Personalizer - Interactive muscle configuration tool
# Inherits from ViewerAppBase for common GLFW/ImGui/camera handling
add_executable(muscle_personalizer
    "muscle_personalizer_main.cpp"
    "MusclePersonalizerApp.h"
    "MusclePersonalizerApp.cpp"
    "RenderCharacter.h"
    "RenderCharacter.cpp"
)

target_include_directories(muscle_personalizer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/tinycolormap/include
)

target_link_libraries(muscle_personalizer
    tinyxml2::tinyxml2
    sim
    surgery              # Surgery library
    surgery_optimizer    # For ContractureOptimizer
    dart-gui
    ${HDF5_CXX_LIBRARIES}
    Boost::program_options
    viewer_common        # Provides ViewerAppBase, ShapeRenderer, GLfunctions, ImGuiCommon
)

# C3D Calibration CLI - Headless skeleton calibration from C3D files
# This tool automates the calibration pipeline from C3DProcessorApp
set(C3D_CALIBRATION_CLI_SOURCES
    "../tools/c3d_calibration_cli_main.cpp"
    "RenderCharacter.h"
    "RenderCharacter.cpp"
    "C3D_Reader.h"
    "C3D_Reader.cpp"
)

add_executable(c3d_calibration_cli ${C3D_CALIBRATION_CLI_SOURCES})

target_include_directories(c3d_calibration_cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../sim
    ${CMAKE_CURRENT_SOURCE_DIR}/../surgery
    ${CMAKE_CURRENT_SOURCE_DIR}/../surgery/optimizer
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/tinycolormap/include
)

target_link_libraries(c3d_calibration_cli
    tinyxml2::tinyxml2
    sim
    dart
    ezc3d
    Boost::program_options
    yaml-cpp
    rm
    ${HDF5_CXX_LIBRARIES}
)

# Skeleton Global Export - Export global transforms of skeleton body nodes
# Used for visualizing skeleton calibration variability across motion trials
add_executable(skeleton_global_export
    "../tools/skeleton_global_export.cpp"
)

target_include_directories(skeleton_global_export PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../sim
)

target_link_libraries(skeleton_global_export
    sim
    dart
    Boost::program_options
    yaml-cpp
    rm
)