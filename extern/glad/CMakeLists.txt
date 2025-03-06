project(glad C)

add_library(glad STATIC src/glad.c)
target_include_directories(glad SYSTEM PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")
