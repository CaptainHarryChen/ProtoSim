file(GLOB_RECURSE SRC ./*.cpp ./*.cu)
add_library(Physics STATIC ${SRC})
target_include_directories(Physics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
