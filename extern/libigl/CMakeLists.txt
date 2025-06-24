project(libigl CXX)

# file(GLOB_RECURSE LIBIGL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cpp")
# add_library(libigl STATIC ${LIBIGL_SOURCES})
# target_compile_definitions(libigl PUBLIC IGL_STATIC_LIBRARY)

add_library(libigl STATIC placeholder.cpp)

set(Eigen3_DIR ${Extern_Dir}/eigen/share/eigen3/cmake)
find_package(Eigen3 REQUIRED)
target_link_libraries(libigl PUBLIC Eigen3::Eigen)

target_include_directories(libigl SYSTEM PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")
