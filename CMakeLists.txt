cmake_minimum_required(VERSION 3.5)
project(renderer_prototype)

set(CMAKE_CXX_STANDARD 20)

find_package(SDL2 REQUIRED)
include_directories(renderer_prototype ${SDL2_INCLUDE_DIRS})

add_executable(renderer_prototype src/main.cpp src/ApplicationWindow.cpp src/ApplicationWindow.hpp src/Buffer.hpp)

target_link_libraries(renderer_prototype ${SDL2_LIBRARIES})