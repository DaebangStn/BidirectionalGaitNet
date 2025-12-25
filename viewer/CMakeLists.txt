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
    common/PIDNavigator.h
    common/PIDNavigator.cpp
    common/PIDFileFilter.h
    common/imgui_common.h
    common/imgui_common.cpp
)
target_link_libraries(viewer_common PUBLIC rm dart imgui)
target_include_directories(viewer_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Viewer sources
set(VIEWER_SOURCES
    "main.cpp"
    "GLFWApp.h"
    "GLFWApp.cpp"
    "RenderEnvironment.h"
    "RenderEnvironment.cpp"
    "RenderCharacter.h"
    "RenderCharacter.cpp"
    "ShapeRenderer.h"
    "ShapeRenderer.cpp"
    "GLfunctions.h"
    "GLfunctions.cpp"
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
)

add_executable(viewer ${VIEWER_SOURCES})

target_link_libraries(viewer
        ${Python3_LIBRARIES}
        tinyxml2::tinyxml2
        GL GLU glut glad pybind11::embed glfw imgui
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
    "ShapeRenderer.h"
    "ShapeRenderer.cpp"
    "GLfunctions.h"
    "GLfunctions.cpp"
)
target_link_libraries(marker_editor
    tinyxml2::tinyxml2
    GL GLU glut glad glfw imgui
    sim
    dart-gui
)

# C3D Processor - Standalone C3D processing application
# NOTE: No HDF5 linking - C3D processor does NOT support HDF
set(C3D_PROCESSOR_SOURCES
    "c3d_processor_main.cpp"
    "C3DProcessorApp.h"
    "C3DProcessorApp.cpp"
    "RenderCharacter.h"
    "RenderCharacter.cpp"
    "ShapeRenderer.h"
    "ShapeRenderer.cpp"
    "GLfunctions.h"
    "GLfunctions.cpp"
    "C3D_Reader.h"
    "C3D_Reader.cpp"
    "PlotUtils.h"
    "PlotUtils.cpp"
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
add_executable(motion_editor
    "motion_editor_main.cpp"
    "MotionEditorApp.h"
    "MotionEditorApp.cpp"
    "RenderCharacter.h"
    "RenderCharacter.cpp"
    "ShapeRenderer.h"
    "ShapeRenderer.cpp"
    "GLfunctions.h"
    "GLfunctions.cpp"
)
target_link_libraries(motion_editor
    tinyxml2::tinyxml2
    GL GLU glut glad glfw imgui
    sim
    dart-gui
    ${HDF5_CXX_LIBRARIES}
    Boost::program_options
    rm
    viewer_common
)

# Muscle Personalizer - Interactive muscle configuration tool
set(MUSCLE_PERSONALIZER_SOURCES
    "muscle_personalizer_main.cpp"
    "MusclePersonalizerApp.h"
    "MusclePersonalizerApp.cpp"
    "RenderCharacter.h"
    "RenderCharacter.cpp"
    "ShapeRenderer.h"
    "ShapeRenderer.cpp"
    "GLfunctions.h"
    "GLfunctions.cpp"
)

add_executable(muscle_personalizer ${MUSCLE_PERSONALIZER_SOURCES})

target_link_libraries(muscle_personalizer
    tinyxml2::tinyxml2
    GL GLU glut glad glfw imgui
    sim
    surgery              # Surgery library
    surgery_optimizer    # For ContractureOptimizer
    dart-gui
    ${HDF5_CXX_LIBRARIES}
    Boost::program_options
    rm
)