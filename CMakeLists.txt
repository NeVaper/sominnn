cmake_minimum_required(VERSION 3.15)

set (PROJECT_NAME sominnn)

set(CMAKE_CXX_STANDARD 17)

project(${PROJECT_NAME})

file(GLOB SRC_CPP src/*.cpp)
file(GLOB INC_CPP src/*.h src/*.inl)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(LIB ${PROJECT_NAME})

add_library(${LIB} ${SRC_CPP} ${INC_CPP})

file(GLOB TEST_SRC test/*.cpp test/*.h)

set(TEST ${LIB}_test)

add_executable(${TEST} ${TEST_SRC})

target_link_libraries(${TEST} ${LIB})
