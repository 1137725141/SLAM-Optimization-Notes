#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.hpp" // 使用SE3来表达位姿

using namespace std;
using namespace Eigen;

// 全局定义一个相机内参 (假设是 640x480 的相机)
// fx, fy, cx, cy
double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;

int main(int argc, char **argv) {
    
    // 1. 定义一个世界坐标系下的点 P_w (比如在前方 3 米，右边 0.5 米，高 0.5 米)
    // 注意：世界坐标系定义随意，我们假设 Z 前 X 右 Y 下
    Vector3d P_w(0.5, 0.5, 3.0);
    cout << "World Point: " << P_w.transpose() << endl;

    // 2. 定义相机位姿 T_cw (World -> Camera)
    // 假设相机稍微旋转了一点点，并且平移了一点点
    // 旋转：绕 Y 轴转 5 度 (模拟车头偏了一点)
    // 平移：x 轴移动 -0.1 (模拟相机安装在左边一点)
    Matrix3d R_cw;
    R_cw = AngleAxisd(5 * M_PI / 180.0, Vector3d::UnitY());
    Vector3d t_cw(-0.1, 0, 0);
    
    Sophus::SE3d T_cw(R_cw, t_cw);
    cout << "Camera Pose T_cw:\n" << T_cw.matrix() << endl;

    // ==========================================
    // TODO 1: 正向投影 (World -> Pixel)
    // ==========================================
    // Step A: 把 P_w 转换到相机坐标系 P_c
    Vector3d P_c = T_cw * P_w; // 利用 Sophus 重载的乘法
    cout << "Camera Point: " << P_c.transpose() << endl;

    // Step B: 归一化 (除以 Z) -> 得到归一化平面坐标
    // 提示：P_c[0] 是 X, P_c[2] 是 Z
    Vector2d P_norm;
    // P_norm = ... (请补全)
    P_norm << P_c[0] / P_c[2], P_c[1] / P_c[2]; // 参考答案

    // Step C: 投影到像素坐标 (Pixel)
    // u = fx * x + cx
    Vector2d P_uv;
    // P_uv = ... (请补全)
    P_uv << fx * P_norm[0] + cx, fy * P_norm[1] + cy; // 参考答案

    cout << "Pixel (u, v): " << P_uv.transpose() << endl;


    // ==========================================
    // TODO 2: 逆向投影 (Pixel + Depth -> Camera Point)
    // ==========================================
    // 假设我们要把刚刚算出来的像素点 P_uv 和已知的深度 (P_c[2]) 还原回去
    // 这在构建地图时非常重要
    double depth = P_c[2];
    double u = P_uv[0];
    double v = P_uv[1];

    Vector3d P_c_reconstructed;
    // P_c_reconstructed[0] = ... (X = (u - cx) * depth / fx)
    // P_c_reconstructed[1] = ...
    // P_c_reconstructed[2] = depth;
    
    // 参考答案
    P_c_reconstructed[0] = (u - cx) * depth / fx;
    P_c_reconstructed[1] = (v - cy) * depth / fy;
    P_c_reconstructed[2] = depth;

    cout << "Reconstructed Camera Point: " << P_c_reconstructed.transpose() << endl;

    cout << "Reconstructed World Point: " << (T_cw.inverse() *  P_c_reconstructed).transpose() << endl;
    
    // 验证一下误差
    cout << "Error: " << (P_c - P_c_reconstructed).norm() << endl;

    return 0;
}