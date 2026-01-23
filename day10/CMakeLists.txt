cmake_minimum_required(VERSION 3.0)
project(day10_g2o)

# --- 【修复1】必须升级到 C++17，因为新版 g2o 需要 ---
set(CMAKE_CXX_STANDARD 17)

# 找包
find_package(OpenCV REQUIRED) # --- 【修复2】加上 OpenCV ---
find_package(Eigen3 REQUIRED)

# 包含路径
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories("/usr/local/include") # 找 g2o 头文件

# g2o 库列表
set(G2O_LIBS 
    g2o_core 
    g2o_stuff 
    g2o_types_sba 
    g2o_solver_csparse 
    g2o_solver_dense
    g2o_csparse_extension
)

add_executable(day10_g2o day10_g2o_curve_fitting.cpp)

# --- 【修复2】记得链接 OpenCV 库 ---
target_link_libraries(day10_g2o ${G2O_LIBS} ${OpenCV_LIBS} cxsparse)