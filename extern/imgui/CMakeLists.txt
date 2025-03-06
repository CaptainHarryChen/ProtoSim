project(imgui LANGUAGES CXX)

file(GLOB_RECURSE IMGUI_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_library(imgui STATIC ${IMGUI_SOURCES})
target_include_directories(imgui SYSTEM PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

set(glfw3_DIR ${Extern_Dir}/glfw/lib/cmake/glfw3)
find_package(glfw3 CONFIG REQUIRED)

target_link_libraries(imgui PUBLIC glad glfw)
