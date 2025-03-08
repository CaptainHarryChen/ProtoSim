file(GLOB_RECURSE SRC ./*.cpp)
add_library(App STATIC ${SRC})
target_include_directories(App PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(App PUBLIC Viewer Physics)
