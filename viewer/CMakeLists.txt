add_compile_options(-fvisibility=hidden)

link_directories(../sim/)
include_directories(../sim/)
include_directories(../libs/imgui)
include_directories(../libs/ascii)

find_package(GLUT REQUIRED)
find_package(TinyXML2 REQUIRED)
find_package(DART REQUIRED COMPONENTS gui)
find_package(ezc3d CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(viewer
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
target_link_libraries(viewer
        ${Python3_LIBRARIES}
        tinyxml2::tinyxml2
        GL GLU glut glad pybind11::embed glfw imgui
        sim
        dart-gui
        ${HDF5_CXX_LIBRARIES}
        ezc3d
        Boost::program_options
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
add_executable(c3d_processor
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
)
target_link_libraries(c3d_processor
    tinyxml2::tinyxml2
    GL GLU glut glad glfw imgui
    sim
    dart-gui
    ezc3d
    Boost::program_options
)

# Test for marker save cyclic consistency
add_executable(test_marker_save
    "test_marker_save.cpp"
    "RenderCharacter.h"
    "RenderCharacter.cpp"
)
target_link_libraries(test_marker_save
    tinyxml2::tinyxml2
    sim
    dart-gui
)
