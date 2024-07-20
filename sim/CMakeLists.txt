find_package(TinyXML2 REQUIRED)
find_package(DART REQUIRED COMPONENTS collision-bullet)

file(GLOB srcs "*.h" "*.cpp")

add_library(sim ${srcs})
target_link_libraries(sim PUBLIC
        tinyxml2::tinyxml2
        dart
        dart-collision-bullet
)
