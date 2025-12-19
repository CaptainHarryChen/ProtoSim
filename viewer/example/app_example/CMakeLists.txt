cmake_minimum_required(VERSION 3.11.4)

project(viewer_app_example)

FILE(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

set(TARGET_NAME viewer_app_example)
add_executable(${TARGET_NAME} ${SRC_FILES})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER viewer/example)

target_link_libraries(${TARGET_NAME} PRIVATE viewer)
