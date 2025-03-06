cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(ENABLE_CUDA "ENABLE CUDA" ON)

if(${ENABLE_CUDA} STREQUAL "ON" AND NOT APPLE)
    project(ProtoSim LANGUAGES CXX CUDA)
    message("ENABLE CUDA")
else()
    project(ProtoSim LANGUAGES CXX)
    message("DISABLE CUDA")
endif()

set(Proj_Dir ${CMAKE_CURRENT_SOURCE_DIR})
set(Asset_Dir ${CMAKE_CURRENT_SOURCE_DIR}/asset)
set(Extern_Dir ${CMAKE_CURRENT_SOURCE_DIR}/extern)
set(Viewer_Dir ${CMAKE_CURRENT_SOURCE_DIR}/viewer)
set(Physics_Dir ${CMAKE_CURRENT_SOURCE_DIR}/physics)
set(App_Dir ${CMAKE_CURRENT_SOURCE_DIR}/app)

add_subdirectory(extern)
add_subdirectory(viewer)
add_subdirectory(physics)
add_subdirectory(app)
add_subdirectory(examples)
