#include <iostream>
#include <cmath>
#include <iomanip> // 用于控制输出精度
#include <Eigen/Core>
#include <Eigen/Geometry>

// 【新增】引入 Sophus 头文件
// 注意：根据安装版本不同，有时是 "sophus/so3.hpp" 或 "sophus/so3.h"
#include "sophus/so3.hpp" 

using namespace std;
using namespace Eigen;

int main(int argc, char **argv)
{
    // 设置输出精度
    cout << fixed << setprecision(3);

    // ==========================================
    // Part 1: Eigen 基础几何变换 (回顾 Day 01 上半场)
    // ==========================================
    cout << "*****************************************" << endl;
    cout << "      Part 1: Eigen Geometry Basics      " << endl;
    cout << "*****************************************" << endl;

    // 0. 准备数据：定义一个 绕Z轴旋转 45度 的旋转
    Eigen::AngleAxisd rotation_vector(M_PI / 4, Eigen::Vector3d::UnitZ());
    
    cout << "--- [0] 原始定义 (绕Z轴转45度) ---" << endl;
    cout << "Angle: " << rotation_vector.angle() << endl;
    cout << "Axis:  " << rotation_vector.axis().transpose() << endl << endl;

    // 7. 旋转向量 -> 旋转矩阵
    Eigen::Matrix3d R = rotation_vector.toRotationMatrix();
    cout << "--- [7] 旋转向量 -> 旋转矩阵 ---" << endl;
    cout << "R =\n" << R << endl << endl;

    // 1. 旋转矩阵 -> 四元数
    Eigen::Quaterniond q(R);
    cout << "--- [1] 旋转矩阵 -> 四元数 ---" << endl;
    cout << "q (x,y,z,w) = " << q.coeffs().transpose() << endl << endl;

    // 2. 四元数 -> 旋转矩阵
    Eigen::Matrix3d R_from_q = q.toRotationMatrix(); 
    cout << "--- [2] 四元数 -> 旋转矩阵 ---" << endl;
    cout << "R_from_q =\n" << R_from_q << endl << endl;

    // 3. 欧拉角 -> 旋转矩阵
    Eigen::Vector3d euler_angles(M_PI/4, 0, 0); // Z, Y, X
    Eigen::Matrix3d R_from_euler;
    R_from_euler = Eigen::AngleAxisd(euler_angles[0], Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(euler_angles[1], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(euler_angles[2], Eigen::Vector3d::UnitX());
    cout << "--- [3] 欧拉角(Z-Y-X) -> 旋转矩阵 ---" << endl;
    cout << "R_from_euler =\n" << R_from_euler << endl << endl;

    // 4. 旋转矩阵 -> 欧拉角
    Eigen::Vector3d euler_extracted = R.eulerAngles(2, 1, 0);
    cout << "--- [4] 旋转矩阵 -> 欧拉角(Z-Y-X) ---" << endl;
    cout << "Euler (Yaw, Pitch, Roll) = " << euler_extracted.transpose() << endl << endl;

    // 5. 四元数 -> 欧拉角
    Eigen::Vector3d euler_from_q = q.toRotationMatrix().eulerAngles(2, 1, 0);
    cout << "--- [5] 四元数 -> 欧拉角 ---" << endl;
    cout << "Euler from q = " << euler_from_q.transpose() << endl << endl;

    // 6. 欧拉角 -> 四元数
    Eigen::Quaterniond q_from_euler;
    q_from_euler = Eigen::AngleAxisd(euler_angles[0], Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(euler_angles[1], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(euler_angles[2], Eigen::Vector3d::UnitX());
    cout << "--- [6] 欧拉角 -> 四元数 ---" << endl;
    cout << "q_from_euler (x,y,z,w) = " << q_from_euler.coeffs().transpose() << endl << endl;


    // ==========================================
    // Part 2: Sophus 李群与李代数 (Day 01 下半场核心)
    // ==========================================
    cout << "\n\n*****************************************" << endl;
    cout << "      Part 2: Sophus Lie Group & Algebra " << endl;
    cout << "*****************************************" << endl;

    // 1. 重新定义一个旋转：沿Z轴旋转90度
    // 为了防止混淆，变量名加上 _90
    Matrix3d R_90 = AngleAxisd(M_PI/2, Vector3d(0,0,1)).toRotationMatrix();
    Quaterniond q_90(R_90);
    
    cout << "--- [Sophus 1] 构建 SO(3) 群 ---" << endl;
    // Sophus::SO3d 可以直接从旋转矩阵或四元数构造
    Sophus::SO3d SO3_R(R_90); 
    // Sophus::SO3d SO3_q(q_90); // 也是一样的
    
    cout << "SO(3) matrix from Sophus:\n" << SO3_R.matrix() << endl;
    
    // 2. 对数映射 (Logarithmic Map): SO(3) -> so(3)
    // 作用：将“旋转矩阵”转换成“旋转向量”(李代数)
    // 物理意义：获得旋转轴和旋转角
    cout << "\n--- [Sophus 2] 对数映射 (Log Map) ---" << endl;
    Vector3d so3 = SO3_R.log();
    cout << "so3 (Lie Algebra) = " << so3.transpose() << endl;
    cout << "Expected: [0, 0, 1.571] (because PI/2 approx 1.57)" << endl;
    
    // 3. hat 操作 (向量 -> 反对称矩阵)
    // 这是李代数的重要性质：a^ 
    cout << "\n--- [Sophus 3] Hat (向量变反对称矩阵) ---" << endl;
    cout << "so3 hat=\n" << Sophus::SO3d::hat(so3) << endl;

    // 4. 扰动模型 (Perturbation Model) - 核心中的核心
    // 场景：在优化中，我们想对旋转进行微调。
    // 我们不在 SO(3) 上做加法，而是在 so(3) 上定义一个小量 update_w，然后指数映射回去。
    cout << "\n--- [Sophus 4] 扰动更新 (Update) ---" << endl;
    
    // 假设更新量：绕 X 轴旋转 0.01 弧度 (很小的扰动)
    Vector3d update_w(0.01, 0, 0); 
    
    // 左乘更新模型：R_new = exp(update_w^) * R_old
    // 注意：Sophus 重载了 * 运算符，SO3 * SO3 就是矩阵乘法
    // Sophus::SO3d::exp(update_w) 将李代数向量转换回群(旋转矩阵)
    Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_w) * SO3_R;
    
    cout << "Original R (Z-90):\n" << SO3_R.matrix() << endl;
    cout << "Updated  R (Z-90 + Small X-rot):\n" << SO3_updated.matrix() << endl;
    
    // 验证：看看新的旋转是不是稍微偏了一点点
    cout << "\nDiff:\n" << (SO3_updated.matrix() - SO3_R.matrix()) << endl;

    return 0;
}