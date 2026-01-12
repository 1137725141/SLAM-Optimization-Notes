#include <iostream>
#include <vector>
#include <fstream> 
#include <iomanip> // 新增：用于控制输出格式 (setprecision)
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp> 
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace cv;

double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
Mat K = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

// 将三维点p投影到二维平面上
Point2f project(const Point3f& p) {
    // 根据相机内参矩阵将三维点投影到二维平面上
    return Point2f(fx * p.x / p.z + cx, fy * p.y / p.z + cy);
}

void saveData(const string& filename, 
              const vector<Point3f>& points, 
              const Mat& R_gt, const Mat& t_gt, 
              const Mat& R_est, const Mat& t_est) {
    // 打开文件
    ofstream f(filename);
    // 如果文件打开失败，则返回
    if (!f.is_open()) return;

    // 保存点的数量
    f << points.size() << endl;
    // 保存每个点的坐标
    for (const auto& p : points) f << p.x << " " << p.y << " " << p.z << endl;

    // 保存真实旋转矩阵
    for(int i=0; i<3; i++) for(int j=0; j<3; j++) f << R_gt.at<double>(i,j) << " ";
    f << endl;
    // 保存真实平移向量
    f << t_gt.at<double>(0) << " " << t_gt.at<double>(1) << " " << t_gt.at<double>(2) << endl;

    // 保存估计旋转矩阵
    for(int i=0; i<3; i++) for(int j=0; j<3; j++) f << R_est.at<double>(i,j) << " ";
    f << endl;
    // 保存估计平移向量
    f << t_est.at<double>(0) << " " << t_est.at<double>(1) << " " << t_est.at<double>(2) << endl;

    // 关闭文件
    f.close();
    cout << "\n[IO] Data saved to " << filename << endl;
}

int main(int argc, char **argv) {
    // 设置控制台输出精度，避免看到一堆科学计数法
    cout << fixed << setprecision(3); 
    
    cout << "========== 1. Data Generation ==========" << endl;
    
    // 1. 准备数据
    vector<Point3f> points_3d;
    for (int i = 0; i < 30; i++) { 
        float z = rand() % 50 / 10.0 + 3.0; 
        float x = (rand() % 40 - 20) / 10.0;
        float y = (rand() % 40 - 20) / 10.0;
        points_3d.push_back(Point3f(x, y, z));
    }
    // [调试] 打印前两个 3D 点看看长什么样
    cout << "Generated " << points_3d.size() << " 3D points." << endl;
    cout << "Example P_w[0]: " << points_3d[0] << endl;
    cout << "Example P_w[1]: " << points_3d[1] << endl;


    cout << "\n========== 2. Ground Truth Setup ==========" << endl;
    
    // 2. 设定真实位姿
    Mat R1 = Mat::eye(3, 3, CV_64F);
    Mat t1 = Mat::zeros(3, 1, CV_64F);

    Mat R2_gt;
    Rodrigues(Mat(vector<double>{0, 10 * CV_PI / 180.0, 0}), R2_gt); 
    Mat t2_gt = (Mat_<double>(3, 1) << -3.0, 0, 0); 
    
    // [调试] 打印真值
    cout << "True Rotation (R2_gt):\n" << R2_gt << endl;
    cout << "True Translation (t2_gt): " << t2_gt.t() << endl; // 转置打印成一行方便看


    cout << "\n========== 3. Projection & Matching ==========" << endl;

    // 3. 产生观测
    vector<Point2f> pts1, pts2;
    for (const auto& p : points_3d) {
        pts1.push_back(project(p));
        Mat p_mat = (Mat_<double>(3, 1) << p.x, p.y, p.z);
        Mat p_c2_mat = R2_gt * p_mat + t2_gt;
        Point3f p_c2(p_c2_mat.at<double>(0,0), p_c2_mat.at<double>(1,0), p_c2_mat.at<double>(2,0));
        
        if (p_c2.z > 0) {
            pts2.push_back(project(p_c2));
        }
    }
    // [调试] 打印匹配点对情况
    cout << "Valid Matches: " << pts1.size() << " pairs." << endl;
    cout << "Example Match[0]: Frame1 " << pts1[0] << " <--> Frame2 " << pts2[0] << endl;
    
 // ==========================================
    // 核心任务：仅根据 pts1 和 pts2，恢复 R 和 t
    // ==========================================

    cout << "\n========== 4. Essential Matrix ==========" << endl;

    // 4. 计算位姿
    // [知识点] 本质矩阵 E = t^R，包含了位姿信息
    Mat E = findEssentialMat(pts1, pts2, fx, Point2d(cx, cy), RANSAC);
    
    // [调试] 打印本质矩阵
    cout << "Essential Matrix E:\n" << E << endl;
    cout << "Note: E is singular (det approx 0) and has scale ambiguity." << endl;


    cout << "\n========== 5. Pose Recovery ==========" << endl;

    Mat R_est, t_est;
    // [知识点] recoverPose 会自动把 t 归一化
    recoverPose(E, pts1, pts2, R_est, t_est, fx, Point2d(cx, cy));

    // [调试] 打印解算结果
    cout << "Estimated Rotation (R_est):\n" << R_est << endl;
    cout << "Estimated Translation (t_est): " << t_est.t() << endl;

    // 验证旋转矩阵误差
    Mat R_diff = R_est * R2_gt.t(); // 应该是单位矩阵
    // trace(R) = 1+2cos(theta)，这里简单看对角线
    cout << "Rotation Error Check (Trace of R_est * R_gt^T): " << trace(R_diff) << " (Should be 3.0)" << endl;


    cout << "\n========== 6. Scale Analysis ==========" << endl;
    
    // 5. 输出结果并保存
    double n_gt = norm(t2_gt);
    double n_est = norm(t_est);
    
    cout << "Magnitude of True t: " << n_gt << endl;
    cout << "Magnitude of Est  t: " << n_est << " (Look! It is always 1.0)" << endl;
    cout << ">>> Scale Ratio (GT / Est): " << n_gt / n_est << " <<<" << endl;
    
    saveData("../pose_data.txt", points_3d, R2_gt, t2_gt, R_est, t_est);

    return 0;
}