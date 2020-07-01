cmake_minimum_required(VERSION 3.5)
project(renderer_prototype)

set(CMAKE_CXX_STANDARD 20)

find_package(SDL2 REQUIRED)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/sdl2-cmake-scripts")
find_package(SDL2_image REQUIRED)
find_package(Assimp REQUIRED)
find_package(spdlog REQUIRED)
find_package(Microsoft.GSL CONFIG)

include_directories(renderer_prototype ${SDL2_INCLUDE_DIRS} ${SDL2_IMAGE_INCLUDE_DIRS} ${Assimp_SOURCE_DIR}/include)


add_executable(renderer_prototype src/main.cpp src/ApplicationWindow.cpp src/ApplicationWindow.hpp src/Buffer.hpp src/texture.cpp src/texture.hpp src/util.cpp src/util.hpp src/Model.hpp src/Model.cpp src/config.h)

target_link_libraries(renderer_prototype ${SDL2_LIBRARIES} ${SDL2_IMAGE_LIBRARIES} assimp spdlog::spdlog tbb)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Wextra")