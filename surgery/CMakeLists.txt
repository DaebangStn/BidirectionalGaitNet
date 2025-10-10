add_compile_options(-fvisibility=hidden)

find_package(TinyXML2 REQUIRED)
find_package(DART REQUIRED COMPONENTS gui)

# Surgery library - shared by physical_exam and surgery_tool
add_library(surgery STATIC
    "SurgeryExecutor.h"
    "SurgeryExecutor.cpp"
    "SurgeryOperation.h"
    "SurgeryOperation.cpp"
    "SurgeryScript.h"
    "SurgeryScript.cpp"
    "SurgeryPanel.h"
    "SurgeryPanel.cpp"
)

target_include_directories(surgery PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../sim
    ${CMAKE_CURRENT_SOURCE_DIR}/../viewer
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/implot
)

target_link_libraries(surgery
    tinyxml2::tinyxml2
    sim
    dart
    dart-gui
    yaml-cpp
    imgui
)

# Surgery Tool executable (CLI-only)
add_executable(surgery_tool
    "surgery_main.cpp"
)

target_link_libraries(surgery_tool
    surgery
    tinyxml2::tinyxml2
    sim
    dart
    yaml-cpp
)

# Physical Examination executable (standalone, no GLFWApp dependency)
add_executable(physical_exam
    "PhysicalExam.h"
    "PhysicalExam.cpp"
    "physical_exam_main.cpp"
    "../viewer/ShapeRenderer.h"
    "../viewer/ShapeRenderer.cpp"
    "../viewer/GLfunctions.h"
    "../viewer/GLfunctions.cpp"
    "SurgeryExecutor.h"
    "SurgeryExecutor.cpp"
    "SurgeryOperation.h"
    "SurgeryOperation.cpp"
    "SurgeryScript.h"
    "SurgeryScript.cpp"
    "SurgeryPanel.h"
    "SurgeryPanel.cpp"
)

target_include_directories(physical_exam PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../sim
    ${CMAKE_CURRENT_SOURCE_DIR}/../viewer
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/implot
)

target_link_libraries(physical_exam
    ${Python3_LIBRARIES}
    tinyxml2::tinyxml2
    GL GLU glut glad pybind11::embed glfw imgui
    sim
    dart-gui
    yaml-cpp
)
