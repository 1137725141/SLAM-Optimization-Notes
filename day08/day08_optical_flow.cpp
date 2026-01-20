#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

// ==========================================
// 辅助函数：双线性插值获取浮点坐标的灰度值
// ==========================================
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // 边界检查
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;

    int x_floor = floor(x);
    int y_floor = floor(y);
    float dx = x - x_floor;
    float dy = y - y_floor;

    float p1 = (float)img.ptr<uchar>(y_floor)[x_floor];
    float p2 = (float)img.ptr<uchar>(y_floor)[x_floor + 1];
    float p3 = (float)img.ptr<uchar>(y_floor + 1)[x_floor];
    float p4 = (float)img.ptr<uchar>(y_floor + 1)[x_floor + 1];

    // 双线性插值公式
    return (1 - dx) * (1 - dy) * p1 + dx * (1 - dy) * p2 +
           (1 - dx) * dy * p3 + dx * dy * p4;
}

// ==========================================
// 手写光流追踪 (Gauss-Newton)
// ==========================================
void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false // 是否使用反向法(进阶技巧，这里默认false用正向法)
) {
    kp2.resize(kp1.size());
    success.resize(kp1.size());

    // 我们可以并行处理每个点，这里为了清晰用单线程循环
    for (size_t i = 0; i < kp1.size(); i++) {
        KeyPoint kp = kp1[i];
        double dx = 0, dy = 0; // 待估计的运动量 (u, v)
        
        // 这里的迭代优化实际上是在解最小二乘问题
        // Cost = sum( (I2(x+dx) - I1(x))^2 )
        bool has_converged = false;
        for (int iter = 0; iter < 10; iter++) {
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero(); // Hessian = J^T * J
            Eigen::Vector2d b = Eigen::Vector2d::Zero(); // bias = -J^T * error
            double cost = 0;

            // 在 8x8 的窗口内累加
            for (int x = -4; x < 4; x++) {
                for (int y = -4; y < 4; y++) {
                    double u = kp.pt.x + x;
                    double v = kp.pt.y + y;
                    
                    // 1. 计算误差 error = I2(x+dx, y+dy) - I1(x, y)
                    double val1 = GetPixelValue(img1, u, v);
                    double val2 = GetPixelValue(img2, u + dx, v + dy);
                    double error = val2 - val1;

                    // 2. 计算雅可比 J (图像梯度)
                    // 我们在 img2 的当前估计位置求梯度
                    // J = [dI2/dx, dI2/dy]
                    double Jx = (GetPixelValue(img2, u + dx + 1, v + dy) - GetPixelValue(img2, u + dx - 1, v + dy)) / 2.0;
                    double Jy = (GetPixelValue(img2, u + dx, v + dy + 1) - GetPixelValue(img2, u + dx, v + dy - 1)) / 2.0;

                    Eigen::Vector2d J(Jx, Jy);

                    // 3. 累加 H 和 b
                    H += J * J.transpose();
                    b += -J * error;
                    cost += error * error;
                }
            }

            // 4. 求解 H * delta = b
            Eigen::Vector2d update = H.ldlt().solve(b);

            if (std::isnan(update[0])) {
                // 解挂了（比如 H 不可逆，通常是图像这一块没纹理）
                has_converged = false; 
                break;
            }

            if (iter > 0 && cost > 0) {
                // 这里可以加 cost 判断
            }

            // 更新估计值
            dx += update[0];
            dy += update[1];
            
            // 收敛判定
            if (update.norm() < 1e-2) {
                has_converged = true;
                break;
            }
        }

        success[i] = has_converged;
        kp2[i].pt = kp.pt + Point2f(dx, dy);
    }
}

int main(int argc, char **argv) {
    // 读取两张图像 (灰度)
    // 请确保目录下有 LK1.png 和 LK2.png (你可以用 Day04 的 1.png, 2.png 改名)
    Mat img1 = imread("1.png", IMREAD_GRAYSCALE);
    Mat img2 = imread("2.png", IMREAD_GRAYSCALE);
    
    if (img1.empty() || img2.empty()) {
        cerr << "Cannot find images!" << endl;
        return -1;
    }

    // 1. 提取第一帧的特征点 (GFTT角点适合光流)
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20);
    detector->detect(img1, kp1);

    // 2. 使用 OpenCV 自带的光流 (作为标准答案对比)
    vector<Point2f> pt1, pt2_opencv;
    for (auto &kp : kp1) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    
    auto t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2_opencv, status, error);
    auto t2 = chrono::steady_clock::now();
    cout << "OpenCV LK cost: " << chrono::duration_cast<chrono::duration<double>>(t2 - t1).count() << "s" << endl;

    // 3. 使用我们要手写的光流
    vector<KeyPoint> kp2_my;
    vector<bool> success_my;
    t1 = chrono::steady_clock::now();
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_my, success_my);
    t2 = chrono::steady_clock::now();
    cout << "My LK cost: " << chrono::duration_cast<chrono::duration<double>>(t2 - t1).count() << "s" << endl;

    // 4. 画图对比
    Mat img2_CV = imread("2.png"); // 彩色用于画图
    Mat img2_My = img2_CV.clone();

    for (int i = 0; i < pt2_opencv.size(); i++) {
        if (status[i]) {
            cv::circle(img2_CV, pt2_opencv[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2_opencv[i], cv::Scalar(0, 250, 0));
        }
    }

    for (int i = 0; i < kp2_my.size(); i++) {
        if (success_my[i]) {
            cv::circle(img2_My, kp2_my[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_My, kp1[i].pt, kp2_my[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("OpenCV LK", img2_CV);
    cv::imshow("My LK", img2_My);
    cv::waitKey(0);

    return 0;
}