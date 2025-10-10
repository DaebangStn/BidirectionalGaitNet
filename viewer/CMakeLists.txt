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

# Physical Examination executable (standalone, no GLFWApp dependency)
add_executable(physical_exam
    "PhysicalExam.h"
    "PhysicalExam.cpp"
    "physical_exam_main.cpp"
    "ShapeRenderer.h"
    "ShapeRenderer.cpp"
    "GLfunctions.h"
    "GLfunctions.cpp"
    "SurgeryOperation.h"
    "SurgeryOperation.cpp"
    "SurgeryScript.h"
    "SurgeryScript.cpp"
)

target_link_libraries(physical_exam
    ${Python3_LIBRARIES}
    tinyxml2::tinyxml2
    GL GLU glut glad pybind11::embed glfw imgui
    sim
    dart-gui
    yaml-cpp
)

# Surgery Tool executable (CLI-only, no GUI dependencies)
add_executable(surgery_tool
    "surgery_main.cpp"
    "SurgeryOperation.h"
    "SurgeryOperation.cpp"
    "SurgeryScript.h"
    "SurgeryScript.cpp"
)

target_link_libraries(surgery_tool
    tinyxml2::tinyxml2
    sim
    dart
    yaml-cpp
)