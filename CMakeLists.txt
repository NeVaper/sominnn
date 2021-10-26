cmake_minimum_required(VERSION 3.15)

set (PROJECT_NAME sominnn)

set(CMAKE_CXX_STANDARD 17)

project(${PROJECT_NAME})

file(GLOB SRC_CPP src/*.cpp)
file(GLOB INC_CPP src/*.h)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(${PROJECT_NAME} ${SRC_CPP} ${INC_CPP})