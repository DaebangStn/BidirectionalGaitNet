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
