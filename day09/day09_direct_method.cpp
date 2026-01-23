#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
#include <ctime>
#include <climits>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

// Sophus 库用于处理李代数 SE3
#include <sophus/se3.hpp> 

using namespace std;
using namespace cv;

// 类型定义
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// 相机内参 (TUM数据集典型参数)
double fx = 517.3, fy = 516.5, cx = 325.1, cy = 249.7;
double d_scale = 1000.0; // 深度图数值单位，通常是毫米

// 双线性插值获取灰度 (复用 Day 08)
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    if (x < 0) x = 0; if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;
    int x_floor = floor(x); int y_floor = floor(y);
    float dx = x - x_floor; float dy = y - y_floor;
    return (1 - dx) * (1 - dy) * img.ptr<uchar>(y_floor)[x_floor] +
           dx * (1 - dy) * img.ptr<uchar>(y_floor)[x_floor + 1] +
           (1 - dx) * dy * img.ptr<uchar>(y_floor + 1)[x_floor] +
           dx * dy * img.ptr<uchar>(y_floor + 1)[x_floor + 1];
}

// 稀疏直接法类
class SparseDirectTracker {
public:
    SparseDirectTracker(const Mat &img1, const Mat &img2, const VecVector2d &pixels_ref, const vector<double> &depth_ref)
        : img1_(img1), img2_(img2), pixels_ref_(pixels_ref), depth_ref_(depth_ref) {}

    // 计算
    void calculate(Sophus::SE3d &T21) {
        // Gauss-Newton 迭代
        int iterations = 10;
        double cost = 0, lastCost = 0;

        for (int iter = 0; iter < iterations; iter++) {
            Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
            cost = 0;
            int effective_pixels = 0;

            // 遍历 Frame 1 中的每一个参考点
            for (size_t i = 0; i < pixels_ref_.size(); i++) {
                Eigen::Vector2d u = pixels_ref_[i];
                double depth = depth_ref_[i];
                if (depth < 0.1 || depth > 5.0) continue; // 忽略太近或太远的点

                // 3D 点坐标 (在 Camera 1 系下)
                Eigen::Vector3d P_ref;
                P_ref[0] = (u[0] - cx) / fx * depth;
                P_ref[1] = (u[1] - cy) / fy * depth;
                P_ref[2] = depth;

                // 投影到 Camera 2 系下: P_cur = T * P_ref
                Eigen::Vector3d P_cur = T21 * P_ref;
                if (P_cur[2] < 0) continue; // 深度必须为正

                double u_cur = fx * P_cur[0] / P_cur[2] + cx;
                double v_cur = fy * P_cur[1] / P_cur[2] + cy;

                // 边界检查
                if (u_cur < 2 || u_cur > img2_.cols - 2 || v_cur < 2 || v_cur > img2_.rows - 2) continue;

                // --- 核心：以投影点为中心，计算 4x4 Patch 的光度误差 ---
                for (int x = -2; x < 2; x++) {
                    for (int y = -2; y < 2; y++) {
                        // 1. 残差 e = I1 - I2
                        double error = GetPixelValue(img1_, u[0] + x, u[1] + y) -
                                       GetPixelValue(img2_, u_cur + x, v_cur + y);

                        // 2. 图像梯度 (在 img2 当前投影点附近算)
                        Eigen::Vector2d J_img;
                        J_img[0] = 0.5 * (GetPixelValue(img2_, u_cur + x + 1, v_cur + y) - GetPixelValue(img2_, u_cur + x - 1, v_cur + y));
                        J_img[1] = 0.5 * (GetPixelValue(img2_, u_cur + x, v_cur + y + 1) - GetPixelValue(img2_, u_cur + x, v_cur + y - 1));

                        // 3. 几何导数 (投影导数) 2x6
                        // Day 04 那个矩阵
                        double X = P_cur[0], Y = P_cur[1], Z = P_cur[2];
                        double Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = 1.0 / Z2;
                        Eigen::Matrix<double, 2, 6> J_pixel_xi;
                        J_pixel_xi(0, 0) = fx * Z_inv;
                        J_pixel_xi(0, 1) = 0;
                        J_pixel_xi(0, 2) = -fx * X * Z2_inv;
                        J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
                        J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
                        J_pixel_xi(0, 5) = -fx * Y * Z_inv;

                        J_pixel_xi(1, 0) = 0;
                        J_pixel_xi(1, 1) = fy * Z_inv;
                        J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
                        J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                        J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
                        J_pixel_xi(1, 5) = fy * X * Z_inv;

                        // 4. 总 Jacobian = - J_img * J_pixel_xi (1x6)
                        Eigen::Matrix<double, 1, 6> J = -J_img.transpose() * J_pixel_xi;

                        // 5. 累加 H, b
                        H += J.transpose() * J;
                        b += -J.transpose() * error;
                        cost += error * error;
                        effective_pixels++;
                    }
                }
            }

            if (effective_pixels < 20) {
                cout << "Not enough pixels!" << endl;
                break;
            }

            // 6. 求解 H * dx = b
            Eigen::Matrix<double, 6, 1> update = H.ldlt().solve(b);

            if (iter > 0 && cost > lastCost) {
                // cout << "Cost increased: " << cost << ", stop." << endl;
                // break;
            }
            
            // 更新位姿 (左乘扰动)
            T21 = Sophus::SE3d::exp(update) * T21;
            lastCost = cost;
            cout << "Iter " << iter << " cost=" << cost / effective_pixels << endl;
            
            if (update.norm() < 1e-3) break;
        }
    }

private:
    const Mat &img1_, &img2_;
    const VecVector2d &pixels_ref_;
    const vector<double> &depth_ref_;
};

int main(int argc, char **argv) {
    // 1. 读取图像 (请确保目录下有 1.png 和 2.png)
    // 这里的 1.png 和 2.png 最好有 3-5 个像素的平移，不要太大
    Mat img1 = imread("1.png", IMREAD_GRAYSCALE);
    Mat img2 = imread("2.png", IMREAD_GRAYSCALE);
    
    // 假设这是从 depth.png 读出来的深度，或者我们用随机特征点+固定深度来模拟
    VecVector2d pixels_ref;
    vector<double> depth_ref;

    // 提取 FAST 角点作为稀疏点
    vector<KeyPoint> keypoints;
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
    detector->detect(img1, keypoints);

    for (auto kp : keypoints) {
        // 简单模拟：假设所有特征点都在前方 2 米处 (平面场景)
        // 在真实 SLAM 中，这里应该读取 depth 图
        if (kp.pt.x < 20 || kp.pt.y < 20 || kp.pt.x > img1.cols-20 || kp.pt.y > img1.rows-20) continue;
        pixels_ref.push_back(Eigen::Vector2d(kp.pt.x, kp.pt.y));
        depth_ref.push_back(2.0 + (rand()%10)/10.0); // 2.0 ~ 3.0m 的随机深度
    }

    cout << "Points number: " << pixels_ref.size() << endl;

    // 初始位姿猜测 (单位阵)
    Sophus::SE3d T_cur_ref; 

    // 开始直接法优化
    SparseDirectTracker tracker(img1, img2, pixels_ref, depth_ref);
    tracker.calculate(T_cur_ref);

    cout << "Estimated Pose T21: \n" << T_cur_ref.matrix() << endl;

    // 画图验证
    // 把 img1 的点投影到 img2 上看看对不对
    Mat img2_show = cv::imread("2.png");
    for (size_t i = 0; i < pixels_ref.size(); i++) {
        Eigen::Vector2d u = pixels_ref[i];
        double depth = depth_ref[i];
        Eigen::Vector3d P_ref((u[0] - cx) / fx * depth, (u[1] - cy) / fy * depth, depth);
        Eigen::Vector3d P_cur = T_cur_ref * P_ref;
        
        if (P_cur[2] > 0) {
            double u_cur = fx * P_cur[0] / P_cur[2] + cx;
            double v_cur = fy * P_cur[1] / P_cur[2] + cy;
            cv::circle(img2_show, Point2f(u_cur, v_cur), 2, Scalar(0, 250, 0), 2); // 绿色是追踪结果
            cv::circle(img2_show, Point2f(u[0], u[1]), 2, Scalar(0, 0, 250), 1);   // 红色是原位置
            cv::line(img2_show, Point2f(u[0], u[1]), Point2f(u_cur, v_cur), Scalar(250, 0, 0)); // 连线
        }
    }

    imshow("Direct Method", img2_show);
    waitKey(0);

    return 0;
}