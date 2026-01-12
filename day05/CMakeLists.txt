cmake_minimum_required(VERSION 3.0)
project(day05_optimization)

set(CMAKE_CXX_STANDARD 14)

# 寻找 OpenCV (为了产生随机数，当然用 C++ 自带的也行，但习惯用 CV 了)
find_package(OpenCV REQUIRED)
# 寻找 Ceres
find_package(Ceres REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})

add_executable(curve_fitting day05_ceres_fitting.cpp)
target_link_libraries(curve_fitting ${OpenCV_LIBS} ${CERES_LIBRARIES})