add_compile_options(-fvisibility=hidden)

link_directories(../sim/)
include_directories(../sim/)

find_package(GLUT REQUIRED)
find_package(TinyXML2 REQUIRED)
find_package(DART REQUIRED COMPONENTS gui)

add_executable(viewer
    "main.cpp"
    "GLFWApp.h"
    "GLFWApp.cpp"
    "ShapeRenderer.h"
    "ShapeRenderer.cpp"
    "GLfunctions.h"
    "GLfunctions.cpp"
    "C3D_Reader.h"
    "C3D_Reader.cpp"
)
target_link_libraries(viewer
        ${Python3_LIBRARIES}
        tinyxml2::tinyxml2
        GL GLU glut glad pybind11::embed glfw imgui
        sim
        dart-gui
)