#include <iostream>
#include <fstream>
#include <string>

// --- g2o 的核心头文件 ---
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h> // 稀疏求解器

// --- g2o 预定义好的类型 ---
#include <g2o/types/slam3d/types_slam3d.h> // 包含 VertexSE3, EdgeSE3

using namespace std;

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: ./day11_pose_graph my_circle.g2o" << endl;
        return 1;
    }

    ifstream fin(argv[1]);
    if (!fin) {
        cout << "File " << argv[1] << " not found!" << endl;
        return 1;
    }

    // 1. 配置优化器 (标准套路，但这次用稀疏求解器)
    //    Pose Graph 的矩阵非常稀疏，必须用 Cholmod 或 CSparse
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
    typedef g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType> LinearSolverType;

    //    创建求解器: LM 算法 (Levenberg-Marquardt)
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // 2. 加载数据 (g2o 有自带的 load 函数，但为了教学我们稍微看一眼结构)
    //    其实更简单的写法是: optimizer.load(argv[1]); 
    //    但为了演示，我们还是用 load，它会自动帮我们创建 VertexSE3 和 EdgeSE3
    if (!optimizer.load(argv[1])) {
        cout << "Error loading graph" << endl;
        return 1;
    }

    cout << "Graph loaded! Vertices: " << optimizer.vertices().size() 
         << ", Edges: " << optimizer.edges().size() << endl;

    // 3. 必须固定一个点 (Gauge Freedom)
    //    如果所有点都能动，整个球就会在空间中乱飘。
    //    我们把第 0 个点钉死，作为世界坐标系的原点。
    //    注意：g2o 的 vertices() 返回的是一个 map，key 是 ID
    auto first_vertex = dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().begin()->second);
    if (first_vertex) {
        first_vertex->setFixed(true); 
    }

    // 4. 执行优化
    cout << "Optimizing..." << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30); // 迭代 30 次

    // 5. 保存结果
    optimizer.save("sphere_after.g2o");
    cout << "Optimization done. Result saved to 'sphere_after.g2o'" << endl;

    return 0;
}