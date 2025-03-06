cmake_minimum_required(VERSION 3.10)

project(Viewer)

file(GLOB_RECURSE SRC "src/*.cpp")
file(GLOB_RECURSE GLSL "shader/*.glsl")
add_library(Viewer STATIC ${SRC})

foreach(SRC IN ITEMS ${GLSL}) 
    get_filename_component(SRC_PATH "${SRC}" PATH)
    file(RELATIVE_PATH SRC_PATH_REL "${CMAKE_CURRENT_SOURCE_DIR}" "${SRC_PATH}")
    string(REPLACE "/" "\\" GROUP_PATH "${SRC_PATH_REL}")
    source_group("${GROUP_PATH}" FILES "${SRC}")
endforeach()

set(Viewer_Dir ${CMAKE_CURRENT_SOURCE_DIR})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/viewer_config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/viewer_config.h)

target_include_directories(Viewer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(Viewer PUBLIC glm imgui stb_image tiny_obj_loader)
