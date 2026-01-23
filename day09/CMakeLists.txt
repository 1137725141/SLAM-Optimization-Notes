cmake_minimum_required(VERSION 3.0)
project(day09_direct)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)

# --- 【添加这一行】寻找 fmt 库 ---
find_package(fmt REQUIRED) 

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Sophus_INCLUDE_DIRS})

add_executable(day09_direct day09_direct_method.cpp)

# --- 【修改这一行】把 fmt::fmt 加进去 ---
target_link_libraries(day09_direct ${OpenCV_LIBS} ${Sophus_LIBRARIES} fmt::fmt)