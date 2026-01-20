cmake_minimum_required(VERSION 3.0)
project(day06_ba)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
# 确保你的 Ceres 路径正确
# set(Ceres_DIR "/usr/lib/cmake/Ceres") 
find_package(Ceres REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})

add_executable(day06_ba day06_pose_optimization.cpp)
target_link_libraries(day06_ba ${OpenCV_LIBS} ${CERES_LIBRARIES})