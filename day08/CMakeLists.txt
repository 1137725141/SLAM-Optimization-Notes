cmake_minimum_required(VERSION 3.0)
project(day08_optical_flow)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(day08_lk day08_optical_flow.cpp)
target_link_libraries(day08_lk ${OpenCV_LIBS})