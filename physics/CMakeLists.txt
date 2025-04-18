file(GLOB_RECURSE SRC ./*.cpp ./*.cu)
add_library(Physics STATIC ${SRC})
target_include_directories(Physics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set(Eigen3_DIR ${Extern_Dir}/eigen/share/eigen3/cmake)
find_package(Eigen3 REQUIRED)
target_link_libraries(Physics PUBLIC Eigen3::Eigen)
