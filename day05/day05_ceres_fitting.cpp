#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;

// 1. 定义代价函数 (Cost Function) 的模型
// 类似于 EKF 里的 h(x) - z
struct CURVE_FITTING_COST {
    CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

    // 残差计算 (template 是 Ceres 的要求，为了自动求导)
    template<typename T>
    bool operator()(const T *const abc, T *residual) const {
        // abc[0]=a, abc[1]=b, abc[2]=c
        // 预测值 y_pred = exp(ax^2 + bx + c)
        // 残差 = y_true - y_pred
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
        return true;
    }

    const double _x, _y;
};

int main(int argc, char **argv) {
    // 真实参数
    double a = 1.0, b = 2.0, c = 1.0; 
    int N = 100;                          // 数据点数量
    double w_sigma = 1.0;                 // 噪声标准差
    cv::RNG rng;                          // OpenCV 随机数生成器

    vector<double> x_data, y_data;        // 数据容器

    cout << "Generating data..." << endl;
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        double y = exp(a * x * x + b * x + c) + rng.gaussian(w_sigma * w_sigma);
        x_data.push_back(x);
        y_data.push_back(y);
    }

    // 2. 构建最小二乘问题
    double abc[3] = {0, 0, 0}; // 参数初值 (随便猜一个，全0)

    ceres::Problem problem;
    for (int i = 0; i < N; i++) {
        // 向问题中添加误差项
        // AutoDiffCostFunction: Ceres 最强大的功能，自动求导！不需要你手算雅可比
        // <CostFunctor, ResidualDim, ParamDim> -> <类名, 残差维度1, 参数维度3>
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
                new CURVE_FITTING_COST(x_data[i], y_data[i])
            ),
            nullptr, // 核函数 (Huber等)，这里为空
            abc      // 待优化参数
        );
    }

    // 3. 配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR; // 增量方程如何求解
    options.minimizer_progress_to_stdout = true;  // 输出过程

    ceres::Solver::Summary summary;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    
    // 4. 开始优化！
    ceres::Solve(options, &problem, &summary);
    
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);

    // 5. 输出结果
    cout << summary.BriefReport() << endl;
    cout << "Optimization Cost time: " << time_used.count() << " seconds." << endl;
    cout << "Estimated a,b,c = ";
    for (auto a : abc) cout << a << " ";
    cout << endl;
    cout << "Truth: " << a << " " << b << " " << c << endl;

    return 0;
}