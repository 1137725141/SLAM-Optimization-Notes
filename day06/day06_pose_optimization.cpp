#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp> // 这一行包含了 core, calib3d, highgui 等所有常用模块
#include <ceres/ceres.h>
#include <ceres/rotation.h> // 必须引用，用于处理旋转向量
#include <chrono>

using namespace std;
using namespace cv;

// 相机内参 (为了简化，直接硬编码在 Functor 里，实际工程应通过构造函数传入)
const double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;

// =========================================
// 1. 定义代价函数 (Cost Function)
//    计算：观测像素 - 预测像素
// =========================================
struct ReprojectionError {
    ReprojectionError(double observed_u, double observed_v, const Point3f& point_3d)
        : _observed_u(observed_u), _observed_v(observed_v), _point_3d(point_3d) {}

    // template <typename T> 是 Ceres 自动求导的灵魂
    // camera_pose[0,1,2] 是旋转向量 (angle-axis)
    // camera_pose[3,4,5] 是平移向量 (t)
    template <typename T>
    bool operator()(const T* const camera_pose, T* residuals) const {
        // A. 取出 3D 点 (转为 T 类型)
        T p_world[3];
        p_world[0] = T(_point_3d.x);
        p_world[1] = T(_point_3d.y);
        p_world[2] = T(_point_3d.z);

        // B. 坐标变换：World -> Camera
        // P_c = R * P_w + t
        T p_camera[3];
        
        // ceres::AngleAxisRotatePoint 实现了 R * P 的功能
        // 它接收旋转向量(3维)和点(3维)，输出旋转后的点
        ceres::AngleAxisRotatePoint(camera_pose, p_world, p_camera);

        // 加上平移 t
        p_camera[0] += camera_pose[3];
        p_camera[1] += camera_pose[4];
        p_camera[2] += camera_pose[5];

        // C. 投影：Camera -> Pixel
        // u = fx * X/Z + cx
        // v = fy * Y/Z + cy
        T xp = p_camera[0] / p_camera[2]; // x/z
        T yp = p_camera[1] / p_camera[2]; // y/z

        T predicted_u = T(fx) * xp + T(cx);
        T predicted_v = T(fy) * yp + T(cy);

        // D. 计算残差 (Residual)
        residuals[0] = predicted_u - T(_observed_u);
        residuals[1] = predicted_v - T(_observed_v);

        return true;
    }

    // 工厂模式：方便创建 AutoDiffCostFunction
    static ceres::CostFunction* Create(const double observed_u,
                                       const double observed_v,
                                       const Point3f& point_3d) {
        // <ReprojectionError, 残差维度=2, 参数维度=6>
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6>(
            new ReprojectionError(observed_u, observed_v, point_3d)));
    }

    double _observed_u, _observed_v;
    Point3f _point_3d;
};

// =========================================
// 辅助：数据保存函数 (用于 Python 画图)
// =========================================
void saveData(const string& filename, 
              const vector<Point3f>& points, 
              const double* pose_gt,    // 真值 [rx, ry, rz, tx, ty, tz]
              const double* pose_init,  // 初始猜测 (有噪声)
              const double* pose_est) { // 优化结果
    ofstream f(filename);
    if (!f.is_open()) return;

    // 1. 保存 3D 点
    f << points.size() << endl;
    for (const auto& p : points) f << p.x << " " << p.y << " " << p.z << endl;

    // 2. 保存 GT Pose
    for(int i=0; i<6; i++) f << pose_gt[i] << " "; f << endl;
    // 3. 保存 Init Pose
    for(int i=0; i<6; i++) f << pose_init[i] << " "; f << endl;
    // 4. 保存 Est Pose
    for(int i=0; i<6; i++) f << pose_est[i] << " "; f << endl;

    f.close();
    cout << "Data saved to " << filename << endl;
}

int main(int argc, char** argv) {
    // 1. 生成模拟数据
    vector<Point3f> points_3d;
    vector<Point2f> points_2d;
    
    // 真值位姿：旋转向量 (0, 0, 0) -> 无旋转, 平移 (0, 0, 0) -> 原点
    // 为了让效果明显，我们设一个稍微复杂的真值
    // 绕 Y 轴转 20 度，平移 (1, 0.5, 2)
    double pose_gt[6] = {0, 20 * CV_PI / 180.0, 0, 1.0, 0.5, 2.0};
    
    // 将旋转向量转为矩阵方便生成数据
    Mat R_gt_mat;
    Mat r_vec = (Mat_<double>(3,1) << pose_gt[0], pose_gt[1], pose_gt[2]);
    Rodrigues(r_vec, R_gt_mat);
    
    // 生成 3D 点并投影
    cv::RNG rng;
    for (int i = 0; i < 50; i++) {
        double x = rng.uniform(-2.0, 2.0);
        double y = rng.uniform(-2.0, 2.0);
        double z = rng.uniform( 3.0, 8.0); // 在前方 3-8 米处
        Point3f p_w(x, y, z);
        points_3d.push_back(p_w);

        // 投影 P_c = R*P_w + t
        Mat p_w_mat = (Mat_<double>(3, 1) << x, y, z);
        Mat t_vec = (Mat_<double>(3, 1) << pose_gt[3], pose_gt[4], pose_gt[5]);
        Mat p_c_mat = R_gt_mat * p_w_mat + t_vec;
        
        Point3f p_c(p_c_mat.at<double>(0), p_c_mat.at<double>(1), p_c_mat.at<double>(2));
        
        // 投影到像素 (加一点点观测噪声)
        double u = fx * p_c.x / p_c.z + cx + rng.gaussian(0.5); // 0.5 像素噪声
        double v = fy * p_c.y / p_c.z + cy + rng.gaussian(0.5);
        points_2d.push_back(Point2f(u, v));
    }

    // 2. 制造一个“糟糕的初值”
    // 假设 PnP 算出来的或者是上一帧的位姿，有点偏差
    double camera_pose[6];
    for(int i=0; i<6; i++) camera_pose[i] = pose_gt[i];
    
    // 给初值加噪声：旋转偏差 0.1弧度，平移偏差 0.5米
    camera_pose[0] += 0.1; 
    camera_pose[1] -= 0.1;
    camera_pose[3] -= 0.5;
    camera_pose[5] += 0.5;

    // 备份一份初始值用于画图
    double pose_init[6];
    for(int i=0; i<6; i++) pose_init[i] = camera_pose[i];

    cout << "Initial Pose (with noise): " << endl;
    for(int i=0; i<6; i++) cout << camera_pose[i] << " ";
    cout << endl;

    // =========================================
    // 3. 构建优化问题 (BA)
    // =========================================
    ceres::Problem problem;
    for (int i = 0; i < points_3d.size(); ++i) {
        // 向问题中添加残差块
        // 故意引入噪声很大的点
        if (i==0) {
            points_2d[i].x += 100;
            points_2d[i].y += 200;
        }
        ceres::CostFunction* cost_function = 
            ReprojectionError::Create(points_2d[i].x, points_2d[i].y, points_3d[i]);
        
        // 核函数 (Loss Function)：这里用 Huber 核，防止外点干扰
        // 如果是 nullptr，就是标准的最小二乘
        ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

        problem.AddResidualBlock(cost_function, loss_function, camera_pose);
    }

    // 4. 配置并求解
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR; // BA 问题常用 Schur 消元
    options.minimizer_progress_to_stdout = true;
    
    ceres::Solver::Summary summary;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Solve time: " << time_used.count() << " seconds." << endl;
    cout << summary.BriefReport() << endl;

    cout << "Estimated Pose: " << endl;
    for(int i=0; i<6; i++) cout << camera_pose[i] << " ";
    cout << endl;
    
    cout << "Ground Truth Pose: " << endl;
    for(int i=0; i<6; i++) cout << pose_gt[i] << " ";
    cout << endl;

    saveData("ba_data.txt", points_3d, pose_gt, pose_init, camera_pose);

    return 0;
}