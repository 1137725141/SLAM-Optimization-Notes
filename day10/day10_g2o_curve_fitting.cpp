#include <iostream>
// g2o 的核心头文件
#include <g2o/core/base_vertex.h>      // 顶点基类
#include <g2o/core/base_unary_edge.h>  // 一元边基类 (因为边只连这一个顶点)
#include <g2o/core/block_solver.h>     // 块求解器 (求解矩阵的核心)
// 优化算法 (三种常用的迭代策略)
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
// 线性求解器 (解 Hx = -b)
#include <g2o/solvers/dense/linear_solver_dense.h> 

#include <Eigen/Core>
#include <cmath>
#include <chrono>
#include <opencv2/core/core.hpp> 

using namespace std;

// ==========================================
// 1. 定义顶点 (Vertex)
//    继承自 g2o::BaseVertex
//    模板参数 <优化变量维度, 数据类型>
//    这里优化变量是 a,b,c，所以维度是 3，类型是 Eigen::Vector3d
// ==========================================
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    // Eigen 内存对齐宏，凡是类成员里有 Eigen 变量，必须加这个，否则段错误
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 

    // 重置函数：把估计值归零 (g2o 内部可能会调用)
    virtual void setToOriginImpl() override {
        _estimate << 0, 0, 0;
    }

    // 更新函数：核心逻辑！
    // 优化器计算出了增量 v (也就是 delta_x)，我们需要告诉它如何把 v 加到当前估计值上。
    // 对于欧氏空间 (Vector3d)，直接加法：x_new = x_old + v
    // (如果是李代数/旋转矩阵，这里就得是左乘或右乘)
    virtual void oplusImpl(const double *update) override {
        _estimate += Eigen::Vector3d(update);
    }

    // 读写函数，通常用于保存/读取图文件，这里不需要，直接返回 true
    virtual bool read(istream &in) { return true; }
    virtual bool write(ostream &out) const { return true; }
};

// ==========================================
// 2. 定义边 (Edge)
//    继承自 g2o::BaseUnaryEdge
//    模板参数 <观测值维度, 观测值类型, 连接的顶点类型>
//    观测值 y 是标量，维度 1，类型 double。连接的是上面定义的 CurveFittingVertex
// ==========================================
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // 构造函数：存入已知量 x
    // 注意：y 是观测值 (measurement)，通过 setMeasurement 传入，不在这里传
    CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}

    // 【重要】计算误差函数
    // 作用：计算 "观测值" 与 "根据当前参数算出的理论值" 之间的差距
    virtual void computeError() override {
        // 1. 获取连接的顶点 (因为是一元边，所以取第0个)
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
        // 2.以此获取当前的估计值 [a, b, c]
        const Eigen::Vector3d abc = v->estimate();
        
        // 3. 计算理论值 prediction = exp(ax^2 + bx + c)
        double y_pred = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
        
        // 4. 计算误差 = 观测值 - 理论值
        // _measurement 是基类里存的 y_data[i]
        _error(0, 0) = _measurement - y_pred; 
    }

    // 【重要】计算雅可比矩阵 (导数)
    // 作用：告诉优化器，当 a,b,c 变化一点点时，误差 error 会变化多少？
    // 也就是 error 对 [a, b, c] 的偏导数。
    // 链式法则：error = y_meas - exp(ax^2+bx+c)
    // d(error)/da = -exp(...) * x^2
    // d(error)/db = -exp(...) * x
    // d(error)/dc = -exp(...) * 1
    virtual void linearizeOplus() override {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]); // 这就是 exp(...) 部分

        // _jacobianOplusXi 是 g2o 定义好的变量，用来存导数
        _jacobianOplusXi[0] = -_x * _x * y; // 对 a 的导数
        _jacobianOplusXi[1] = -_x * y;      // 对 b 的导数
        _jacobianOplusXi[2] = -y;           // 对 c 的导数
    }

    virtual bool read(istream &in) { return true; }
    virtual bool write(ostream &out) const { return true; }
public:
    double _x; // 这条边对应的 x 值 (已知数据)
};

int main(int argc, char **argv) {
    // --- 1. 生成模拟数据 ---
    double ar = 1.0, br = 2.0, cr = 1.0;         // 真实参数 (Ground Truth)
    double ae = 2.0, be = -1.0, ce = 5.0;        // 初始估计值 (随便瞎猜一个，作为起跑线)
    int N = 100;                                 // 数据点数量
    double w_sigma = 1.0;                        // 噪声方差
    cv::RNG rng;                                 // OpenCV 随机数生成器
    
    vector<double> x_data, y_data;
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        // y = 真实方程 + 高斯噪声
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    // --- 2. 配置 g2o 优化器 (这是最繁琐的一步，但套路固定) ---
    
    // Step A: 定义块求解器类型
    // BlockSolverTraits<3, 1> 表示：优化变量维度为 3 (a,b,c)，误差维度为 1 (y)
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;
    
    // Step B: 定义线性求解器类型
    // 我们使用的是稠密线性求解器 (LinearSolverDense)，因为问题规模小
    // 对于 SLAM 大规模问题，通常用 LinearSolverSparse (CSparse/Cholmod)
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    // Step C: 创建总求解器 (Solver)
    // 逻辑层级：OptimizationAlgorithm -> BlockSolver -> LinearSolver
    // 也就是：最上层是“高斯牛顿法”，中间层负责“把图变成矩阵块”，最底层负责“解 Ax=b 线性方程”
    // 注意：std::make_unique 是 C++14/17 标准，用于管理内存，防止内存泄漏
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));

    // Step D: 创建稀疏优化器 (也就是图模型本身)
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver); // 设置算法
    optimizer.setVerbose(true);     // 打开调试输出，让它打印每一步的残差

    // --- 3. 向图中添加顶点 ---
    CurveFittingVertex *v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(ae, be, ce)); // 设定初始值 (2, -1, 5)
    v->setId(0); // 顶点的 ID，图中唯一
    optimizer.addVertex(v);

    // --- 4. 向图中添加边 (观测数据) ---
    for (int i = 0; i < N; i++) {
        // 创建一条边，告诉它对应的 x 是多少
        CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i); // 边的 ID
        edge->setVertex(0, v); // 这条边连接到哪个顶点？连接到顶点 v (ID=0)
        edge->setMeasurement(y_data[i]); // 告诉它观测到的 y 是多少
        
        // 设置信息矩阵 (Information Matrix)
        // 信息矩阵 = 协方差矩阵的逆 = 权重的意思
        // 误差越大，权重越小。这里假设所有点权重一样，设为 1/sigma^2
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1/(w_sigma*w_sigma));
        
        optimizer.addEdge(edge); // 把边加入图
    }

    // --- 5. 执行优化 ---
    cout << "Start optimization..." << endl;
    optimizer.initializeOptimization(); // 初始化 (检查图结构是否正确)
    optimizer.optimize(10);             // 迭代 10 次

    // --- 6. 输出结果 ---
    Eigen::Vector3d abc_est = v->estimate(); // 取出优化后的结果
    cout << "Estimated model: " << abc_est.transpose() << endl;

    return 0;
}