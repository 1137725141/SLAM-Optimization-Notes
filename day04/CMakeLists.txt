cmake_minimum_required(VERSION 3.0)
project(SLAM_Learning)

# 1. 设置 C++ 标准
# SLAM 相关的库（如 Sophus, g2o）通常需要 C++11 或 C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. 寻找 Eigen3 库
# 推荐方式：使用 find_package (最规范)
find_package(Eigen3 REQUIRED)

# 如果 find_package 找不到，或者你想手动指定路径
# 注意：通常路径是 /usr/include/eigen3，不是 engine3
# 只有当 find_package 失败时，才需要手动解开下面这行的注释
# include_directories("/usr/include/eigen3") 

# 3. 输出一些调试信息（可选，方便确认路径对不对）
message(STATUS "Eigen3 found: ${EIGEN3_FOUND}")
message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")

# 4. 声明可执行文件
# 语法: add_executable(生成的可执行文件名 源代码文件)
add_executable(day04 day04.cpp)

# 5. 链接库
# 虽然 Eigen 是纯头文件库，但在 CMake 中最好显式链接
# 这样 CMake 会自动帮你处理 include 路径
# 5. 链接库
# 这里不仅要连 Eigen，还要连 Sophus
find_package(Sophus REQUIRED)
target_link_libraries(day04 
    Eigen3::Eigen
    Sophus::Sophus
)

find_package(OpenCV REQUIRED)
target_link_libraries(day04 
    ${OpenCV_LIBS}
)

