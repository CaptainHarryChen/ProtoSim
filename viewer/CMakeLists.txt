cmake_minimum_required(VERSION 3.10)

file(GLOB_RECURSE SRC "src/*.cpp")
file(GLOB_RECURSE GLSL "shader/*.glsl")
add_library(Viewer STATIC ${SRC})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/viewer_config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/src/viewer_config.h)

target_include_directories(Viewer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(Viewer PUBLIC glm imgui stb_image tiny_obj_loader)
