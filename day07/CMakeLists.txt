cmake_minimum_required(VERSION 3.0)
project(day07_icp)
set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CERES_INCLUDE_DIRS})

add_executable(day07_icp day07_icp.cpp)
target_link_libraries(day07_icp ${CERES_LIBRARIES})