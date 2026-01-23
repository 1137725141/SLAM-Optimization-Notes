cmake_minimum_required(VERSION 3.0)
project(day12_mapping)
set(CMAKE_CXX_STANDARD 14) # PCL 通常需要 C++14

# 找 PCL
find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(day12_mapping day12_mapping.cpp)
target_link_libraries(day12_mapping ${PCL_LIBRARIES})