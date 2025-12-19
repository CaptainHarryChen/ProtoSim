cmake_minimum_required(VERSION 3.10)

if(BUILD_WITH_CUDA)
    project(viewer C CXX CUDA)
else()
    project(viewer C CXX)
endif()

file(GLOB_RECURSE SRC "src/*.c*")
file(GLOB_RECURSE GLSL "shader/*.glsl")
add_library(viewer STATIC ${SRC})
set_target_properties(viewer PROPERTIES FOLDER viewer)
set(VIEWER_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

if(BUILD_WITH_CUDA)
    set_target_properties(viewer PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}"
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON) ##all-major
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/viewer/viewer_config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/src/viewer/viewer_config.h)

target_include_directories(viewer PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty # for glm
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glad/include
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/stb_image
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/tinyobjloader
)

file(GLOB_RECURSE IMGUI_SRC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/*.cpp)
target_sources(viewer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glad/src/glad.c
    ${IMGUI_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/stb_image/stb_image.c
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/tinyobjloader/tiny_obj_loader.cc
)

set(glfw3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glfw/lib/cmake/glfw3)
find_package(glfw3 CONFIG REQUIRED)
find_package(OpenGL REQUIRED)
target_link_libraries(viewer PUBLIC glfw OpenGL::GL)

add_subdirectory(example)
