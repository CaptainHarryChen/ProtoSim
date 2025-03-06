project(stb_image CXX)

add_library(stb_image STATIC stb_image.cpp)
target_include_directories(stb_image SYSTEM PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
