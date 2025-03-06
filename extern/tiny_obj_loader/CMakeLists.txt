project(tiny_obj_loader CXX)

add_library(tiny_obj_loader STATIC tiny_obj_loader.cc)
target_include_directories(tiny_obj_loader SYSTEM PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
