cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)

set(PROJECT_NAME sominnn)
set(LIB ${PROJECT_NAME})
set(TEST ${LIB}_test)

project(${PROJECT_NAME})

file(GLOB SRC_CPP src/*.cpp)
file(GLOB INC_CPP src/*.h src/*.inl)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})


add_library(${LIB} ${SRC_CPP} ${INC_CPP})

file(GLOB TEST_SRC test/*.cpp test/*.h)


add_executable(${TEST} ${TEST_SRC})

target_link_libraries(${TEST} ${LIB})

add_compile_options(
    -Werror

    -Wall
    -Wextra
    -Wpedantic

    -Wcast-align
    -Wcast-qual
    -Wconversion
    -Wctor-dtor-privacy
    -Wenum-compare
    -Wfloat-equal
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Woverloaded-virtual
    -Wredundant-decls
    -Wsign-conversion
    -Wsign-promo
)

add_custom_command(TARGET ${TEST}
    POST_BUILD
    COMMAND ./${TEST}
)
