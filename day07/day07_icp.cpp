#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <chrono>

using namespace std;
using namespace Eigen;

typedef Eigen::Vector3d Point3d; 

// =========================================
// 1. Ceres 的残差定义 (Cost Functor)
//    误差 = p_target - (R * p_source + t)
//    注意：这里是 3D 距离，不是像素距离
// =========================================
struct ICPCost {
    ICPCost(Point3d p_source, Point3d p_target) 
        : _p_source(p_source), _p_target(p_target) {}

    template <typename T>
    bool operator()(const T* const camera_pose, T* residuals) const {
        // camera_pose[0,1,2] = angle_axis (rotation)
        // camera_pose[3,4,5] = translation

        // 1. 取出点 (转为 T 类型)
        T p_s[3] = {T(_p_source.x()), T(_p_source.y()), T(_p_source.z())};
        T p_t[3] = {T(_p_target.x()), T(_p_target.y()), T(_p_target.z())};

        // 2. 变换: p_transformed = R * p_s
        T p_trans[3];
        ceres::AngleAxisRotatePoint(camera_pose, p_s, p_trans);

        // 3. 加上平移 t
        p_trans[0] += camera_pose[3];
        p_trans[1] += camera_pose[4];
        p_trans[2] += camera_pose[5];

        // 4. 计算残差 (XYZ 三个维度)
        residuals[0] = p_t[0] - p_trans[0];
        residuals[1] = p_t[1] - p_trans[1];
        residuals[2] = p_t[2] - p_trans[2];

        return true;
    }

    static ceres::CostFunction* Create(const Point3d& p_source, const Point3d& p_target) {
        // <CostFunctor, 残差维度3, 参数维度6>
        return (new ceres::AutoDiffCostFunction<ICPCost, 3, 6>(
            new ICPCost(p_source, p_target)));
    }

    Point3d _p_source, _p_target;
};

// =========================================
// 2. SVD 方法实现 (手工推导)
// =========================================
void pose_estimation_3d3d(
    const vector<Point3d>& pts1,
    const vector<Point3d>& pts2,
    Matrix3d& R, Vector3d& t
) {
    // A. 计算质心
    Point3d p1 = Point3d::Zero(); // 必须初始化为 0，否则是随机数！
    Point3d p2 = Point3d::Zero();
    int N = pts1.size();
    for (int i = 0; i < N; i++) {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = p1 / N;
    p2 = p2 / N;

    // B. 去质心坐标
    vector<Point3d> q1(N), q2(N);
    for (int i = 0; i < N; i++) {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // C. 计算 W = sum( q1 * q2^T )
    Matrix3d W = Matrix3d::Zero();
    for (int i = 0; i < N; i++) {
        W += q1[i] * q2[i].transpose();
    }

    // D. SVD 分解 W = U * Sigma * V^T
    // ComputeFullU | ComputeFullV 表示我们需要完整的 U 和 V 矩阵
    JacobiSVD<Matrix3d> svd(W, ComputeFullU | ComputeFullV);
    Matrix3d U = svd.matrixU();
    Matrix3d V = svd.matrixV();

    // E. 计算 R = U * V^T (注意这里有的书是 VU^T，取决于 W 定义是 q1*q2^T 还是 q2*q1^T)
    // 按照 SLAM 14讲定义 W = sum(q1 * q2^T)，则 R = U * V^T
    R = U * (V.transpose());

    // 检查行列式，如果为 -1，说明是反射矩阵，需要取反
    if (R.determinant() < 0) {
        R = -R;
    }

    // F. 计算 t = p2 - R * p1
    t = Vector3d(p2.x(), p2.y(), p2.z()) - R * Vector3d(p1.x(), p1.y(), p1.z());
}

int main() {
    // ------------------------------------
    // 1. 生成数据
    // ------------------------------------
    vector<Point3d> pts1, pts2;
    int N = 20; // 点的数量
    
    // 真值位姿
    Matrix3d R_gt = AngleAxisd(M_PI / 4, Vector3d(0, 0, 1)).toRotationMatrix(); // 绕Z轴旋转45度
    Vector3d t_gt(1, 2, 3); // 平移 (1, 2, 3)

    srand((unsigned int)time(0));
    for (int i = 0; i < N; i++) {
        Point3d p1;
        p1.x() = (rand() % 200) / 10.0 - 10.0; // [-10, 10]
        p1.y() = (rand() % 200) / 10.0 - 10.0;
        p1.z() = (rand() % 200) / 10.0 - 10.0;
        pts1.push_back(p1);

        // p2 = R * p1 + t
        Point3d p2 = R_gt * p1 + t_gt;
        // 加一点噪声
        // p2 += Vector3d(0.01, -0.01, 0.01); 
        if (i==0) {
            p2.x() += 1000;
            p2.y() += 2000;
        }
        pts2.push_back(p2);
    }

    cout << "Ground Truth:\n R=\n" << R_gt << "\n t=\n" << t_gt.transpose() << endl << endl;

    // ------------------------------------
    // 2. 方法一：SVD 求解
    // ------------------------------------
    Matrix3d R_svd;
    Vector3d t_svd;
    
    auto t1 = chrono::steady_clock::now();
    pose_estimation_3d3d(pts1, pts2, R_svd, t_svd);
    auto t2 = chrono::steady_clock::now();
    
    cout << "--- Method 1: SVD ---" << endl;
    cout << "Time cost: " << chrono::duration_cast<chrono::duration<double>>(t2 - t1).count() << "s" << endl;
    cout << "R_svd=\n" << R_svd << "\n t_svd=\n" << t_svd.transpose() << endl;


    // ------------------------------------
    // 3. 方法二：Ceres 求解
    // ------------------------------------
    double pose_ceres[6] = {0, 0, 0, 0, 0, 0}; // 初值全0

    ceres::Problem problem;
    for (int i = 0; i < N; i++) {
        problem.AddResidualBlock(
            ICPCost::Create(pts1[i], pts2[i]),
            new ceres::HuberLoss(0.5), // 这里可以加 new ceres::HuberLoss(0.5)
            pose_ceres
        );
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    
    ceres::Solver::Summary summary;
    t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    t2 = chrono::steady_clock::now();

    cout << "\n--- Method 2: Ceres ---" << endl;
    cout << "Time cost: " << chrono::duration_cast<chrono::duration<double>>(t2 - t1).count() << "s" << endl;
    
    // 转换结果以便打印
    // Ceres 用的旋转向量，我们需要转回旋转矩阵对比
    Vector3d r_vec(pose_ceres[0], pose_ceres[1], pose_ceres[2]);
    Matrix3d R_ceres;
    // Eigen 的 AngleAxis 和 Rodrigues 公式一样
    // 若 r_vec 模长为0，则是单位阵；否则构造 AngleAxis
    if (r_vec.norm() > 1e-6) {
        R_ceres = AngleAxisd(r_vec.norm(), r_vec.normalized()).toRotationMatrix();
    } else {
        R_ceres = Matrix3d::Identity();
    }
    Vector3d t_ceres(pose_ceres[3], pose_ceres[4], pose_ceres[5]);

    cout << "R_ceres=\n" << R_ceres << "\n t_ceres=\n" << t_ceres.transpose() << endl;

    // ------------------------------------
    // 4. 误差对比
    // ------------------------------------
    cout << "\n--- Error Analysis ---" << endl;
    cout << "R_error (SVD vs GT): " << (R_svd - R_gt).norm() << endl;
    cout << "t_error (SVD vs GT): " << (t_svd - t_gt).norm() << endl;
    cout << "R_error (Ceres vs GT): " << (R_ceres - R_gt).norm() << endl;
    cout << "t_error (Ceres vs GT): " << (t_ceres - t_gt).norm() << endl;

    return 0;
}
