cmake_minimum_required(VERSION 3.0)
project(day11_pose_graph)
set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED) # 最好加上这个，虽然这节课可能没直接用

# --- 【关键修改】添加 suitesparse 的路径 ---
include_directories("/usr/include/suitesparse") 
# ----------------------------------------

include_directories("/usr/local/include")
include_directories(${EIGEN3_INCLUDE_DIR})

# 设置 g2o 库
set(G2O_LIBS 
    g2o_core 
    g2o_stuff 
    g2o_types_slam3d 
    g2o_solver_cholmod  # 现在的 g2o 有这个模块了
    g2o_solver_dense
    g2o_csparse_extension
)

add_executable(day11_pose_graph day11_pose_graph.cpp)

# 链接库 (为了保险，显示链接 cholmod 库，防止链接报错)
target_link_libraries(day11_pose_graph 
    ${G2O_LIBS} 
    cxsparse 
    cholmod 
)