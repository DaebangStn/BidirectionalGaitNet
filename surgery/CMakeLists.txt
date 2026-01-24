add_compile_options(-fvisibility=hidden)

find_package(TinyXML2 REQUIRED)
find_package(DART REQUIRED COMPONENTS gui)
find_package(Boost REQUIRED COMPONENTS program_options)

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

# Ceres-based optimizers (always available via pre-built Ceres)
add_subdirectory(optimizer)

target_link_libraries(surgery
    PUBLIC
    tinyxml2::tinyxml2
    sim
    dart
    dart-gui
    yaml-cpp
    imgui
    PRIVATE
    surgery_optimizer
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
    Boost::program_options
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
    tinyxml2::tinyxml2
    GL GLU glut glad glfw imgui
    sim
    surgery_optimizer
    dart-gui
    yaml-cpp
    Boost::program_options
    viewer_common
)

# Contracture Optimizer CLI (standalone, no GUI)
add_executable(contracture_cli
    "contracture_cli_main.cpp"
    "SurgeryExecutor.h"
    "SurgeryExecutor.cpp"
)

target_include_directories(contracture_cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../sim
    ${CMAKE_CURRENT_SOURCE_DIR}/../viewer
)

target_link_libraries(contracture_cli
    sim
    surgery_optimizer
    tinyxml2::tinyxml2
    dart
    yaml-cpp
    Boost::program_options
)
