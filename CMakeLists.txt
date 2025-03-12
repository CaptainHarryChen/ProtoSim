cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(ProtoSim LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 89)

set(Proj_Dir ${CMAKE_CURRENT_SOURCE_DIR})
set(Asset_Dir ${CMAKE_CURRENT_SOURCE_DIR}/asset)
set(Extern_Dir ${CMAKE_CURRENT_SOURCE_DIR}/extern)
set(Viewer_Dir ${CMAKE_CURRENT_SOURCE_DIR}/viewer)
set(Physics_Dir ${CMAKE_CURRENT_SOURCE_DIR}/physics)
set(App_Dir ${CMAKE_CURRENT_SOURCE_DIR}/app)

add_compile_definitions($<$<CONFIG:Debug>:DEBUG>)
add_compile_definitions($<$<CONFIG:Release>:NDEBUG>)

add_subdirectory(extern)
add_subdirectory(viewer)
add_subdirectory(Physics)
add_subdirectory(app)
add_subdirectory(examples)
