#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    // 1. 读取图像 (请替换为您电脑上任意一张图片的路径)
    // 建议用一张只有黑白纹理或者普通室内场景图
    string image_path = "./test_image.jpg"; 
    Mat img = imread(image_path, IMREAD_COLOR);

    if (img.empty()) {
        cerr << "Error: Image not found at " << image_path << endl;
        // 如果没图，创建一个灰色的图凑合一下
        img = Mat::zeros(480, 640, CV_8UC3);
        randn(img, Scalar(128,128,128), Scalar(20,20,20)); 
    }

    // 2. 初始化 ORB 检测器
    // 参数含义:
    // nfeatures: 目标提取特征点数量 (ORB-SLAM2 默认 1000-2000)
    // scaleFactor: 金字塔缩放比例 (ORB-SLAM2 默认 1.2)
    // nlevels: 金字塔层数 (ORB-SLAM2 默认 8)
    int nFeatures = 500;
    float scaleFactor = 1.2f;
    int nLevels = 8;
    
    Ptr<ORB> orb = ORB::create(nFeatures, scaleFactor, nLevels);

    // 3. 检测关键点 (Keypoints) 和 计算描述子 (Descriptors)
    vector<KeyPoint> keypoints;
    Mat descriptors;
    
    // ORB 内部会自动构建金字塔进行检测，这一步封装了复杂的数学
    orb->detectAndCompute(img, Mat(), keypoints, descriptors);

    cout << "Detected " << keypoints.size() << " keypoints." << endl;
    
    // 让我们看看第一个点的具体信息
    if (!keypoints.empty()) {
        KeyPoint kp = keypoints[0];
        cout << "--- Sample Keypoint ---" << endl;
        cout << "Position: (" << kp.pt.x << ", " << kp.pt.y << ")" << endl;
        cout << "Orientation: " << kp.angle << " degrees" << endl; // 方向
        cout << "Size (Diameter): " << kp.size << endl; 
        cout << "Pyramid Octave (Level): " << kp.octave << endl; // 属于金字塔哪一层
    }

    // 4. 可视化
    Mat img_show;
    // drawKeypoints 会把关键点的位置和方向(如果是DRAW_RICH_KEYPOINTS)画出来
    drawKeypoints(img, keypoints, img_show, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // 显示
    imshow("ORB Features", img_show);
    
    // 5. 模拟手动金字塔 (为了理解原理，我们手动缩放一层看看)
    Mat img_level1;
    resize(img, img_level1, Size(), 1.0/scaleFactor, 1.0/scaleFactor);
    imshow("Pyramid Level 1", img_level1);

    cout << "Press any key to exit..." << endl;
    waitKey(0);

    return 0;
}