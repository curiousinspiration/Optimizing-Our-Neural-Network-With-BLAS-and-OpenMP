cmake_minimum_required(VERSION 3.1)
project(neural_cpp)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "-Wall -std=c++0x -O0 -g3")

# Project Headers
include_directories(include)

# Project Sources
file(GLOB_RECURSE SOURCES "src/*.cpp")
add_library(neural_cpp STATIC ${SOURCES})

# Third Party
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(${GLOG_INCLUDE_DIR})
include_directories(${GTEST_INCLUDE_DIR})
include_directories(${BLAS_INCLUDE_DIR})

link_directories(${GLOG_LIB_DIR})
link_directories(${GTEST_LIB_DIR})
link_directories(${BLAS_LIB_DIR})

set(LIBS
    ${LIBS}
    glog
    gtest
    pthread
    blas
    omp
    neural_cpp
)

link_directories(build)

# tests
file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")
add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests ${LIBS})

# tools
add_executable(feedforward_neural_net tools/feedforward_neural_net/main.cpp)
target_link_libraries(feedforward_neural_net ${LIBS})
