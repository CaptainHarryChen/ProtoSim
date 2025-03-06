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

add_subdirectory(extern)
add_subdirectory(viewer)
add_subdirectory(examples)

# file(GLOB_RECURSE SRC src/*)
# add_library(Physics STATIC ${SRC})
# target_include_directories(
#     Physics
#     PUBLIC
#     src
#     extern/eigen
# )

# target_link_libraries(Physics PUBLIC Viewer)

# foreach(SRC IN ITEMS ${SRC}) 
#     get_filename_component(SRC_PATH "${SRC}" PATH)
#     file(RELATIVE_PATH SRC_PATH_REL "${CMAKE_CURRENT_SOURCE_DIR}" "${SRC_PATH}")
#     string(REPLACE "/" "\\" GROUP_PATH "${SRC_PATH_REL}")
#     source_group("${GROUP_PATH}" FILES "${SRC}")
# endforeach()

# if(APPLE)
#     set(CMAKE_C_COMPILER clang)
#     set(CMAKE_CXX_COMPILER clang++)
#     if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
#         set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
#         set(OpenMP_C_LIB_NAMES "omp")
#         set(OpenMP_omp_LIBRARY omp)
#     endif()
#     if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
#         set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
#         set(OpenMP_CXX_LIB_NAMES "omp")
#         set(OpenMP_omp_LIBRARY omp)
#     endif()
#     find_package(OpenMP REQUIRED)
# else()
#     find_package(OpenMP REQUIRED)
#     if(OPENMP_FOUND)
#         message("OPENMP FOUND")
#         set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#         set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#     endif()
# endif()

# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/proj_config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/src/proj_config.h)

# include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/add_physics_examples.cmake)
# add_subdirectory(examples)

# if(${ENABLE_CUDA} STREQUAL "ON" AND NOT APPLE)
#     add_subdirectory(CUDA)
#     add_subdirectory(examplesCUDA)
# endif()

# export CUDA_HOME=/usr/local/cuda
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
# export PATH=$PATH:$CUDA_HOME/bin
