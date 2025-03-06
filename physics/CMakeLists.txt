configure_file(proj_config.h.in proj_config.h)

file(GLOB_RECURSE SRC ./*.cpp)
add_library(Physics STATIC ${SRC})
target_include_directories(Physics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Eigen3 REQUIRED)
target_link_libraries(Physics PUBLIC glad Eigen3::Eigen)
