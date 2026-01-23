#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>

// --- PCL 头文件 ---
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

using namespace std;

// 定义位姿结构体
struct Pose {
    Eigen::Vector3d t;
    Eigen::Quaterniond q;
};

// 读取 g2o 文件里的位姿
vector<Pose> readPoses(const string& filename) {
    vector<Pose> poses;
    ifstream fin(filename);
    if (!fin) {
        cerr << "Cannot open " << filename << endl;
        return poses;
    }

    while (!fin.eof()) {
        string line;
        getline(fin, line);
        if (line.empty()) continue;

        stringstream ss(line);
        string tag;
        ss >> tag;
        // 我们只关心顶点 VERTEX_SE3:QUAT
        if (tag == "VERTEX_SE3:QUAT") {
            int id;
            double x, y, z, qx, qy, qz, qw;
            ss >> id >> x >> y >> z >> qx >> qy >> qz >> qw;
            Pose p;
            p.t << x, y, z;
            p.q = Eigen::Quaterniond(qw, qx, qy, qz); // 注意Eigen构造顺序是 (w, x, y, z)
            poses.push_back(p);
        }
    }
    return poses;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        // 提示用法
        cout << "Usage: ./day12_mapping <g2o_filename>" << endl;
        return 1;
    }

    // --- 【修改处 1】解析文件名 ---
    string input_filename = argv[1];
    string output_filename = "map.pcd"; // 默认兜底名称

    // 查找最后一个点的位置
    size_t last_dot = input_filename.find_last_of(".");
    if (last_dot != string::npos) {
        // 截取点之前的部分 (例如 my_circle) 并加上后缀
        string base_name = input_filename.substr(0, last_dot);
        output_filename = base_name + "_map.pcd";
    } else {
        // 如果没有后缀，直接加
        output_filename = input_filename + "_map.pcd";
    }
    // ----------------------------

    // 1. 读取优化后的轨迹
    cout << "Reading poses from " << input_filename << " ..." << endl;
    vector<Pose> poses = readPoses(input_filename);
    cout << "Read " << poses.size() << " poses." << endl;

    if (poses.empty()) {
        cerr << "Error: No poses found!" << endl;
        return 1;
    }

    // 定义全局地图
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr global_map(new pcl::PointCloud<pcl::PointXYZRGB>);

    // 2. 模拟建图过程
    for (size_t i = 0; i < poses.size(); i++) {
        // --- 模拟传感器数据 (假设机器人左右两边各有一道墙) ---
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr local_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        
        // 在局部坐标系下生成一些点
        // 假设机器人前方是 X 轴，左边是 Y 轴 (常见 ROS 坐标系: 前X左Y上Z)
        // 或者 g2o 的设定可能不同，我们假设生成两条“轨道”
        for (double z = 0; z < 2.0; z += 0.1) { // 高度
             // 左边的墙 (y = 2.0)
             pcl::PointXYZRGB p_left;
             p_left.x = 0; p_left.y = 2.0; p_left.z = z;
             p_left.r = 255; p_left.g = 0; p_left.b = 0; // 红色墙
             local_cloud->push_back(p_left);

             // 右边的墙 (y = -2.0)
             pcl::PointXYZRGB p_right;
             p_right.x = 0; p_right.y = -2.0; p_right.z = z;
             p_right.r = 0; p_right.g = 255; p_right.b = 0; // 绿色墙
             local_cloud->push_back(p_right);
        }

        // --- 核心步骤：将局部点云转到世界坐标系 ---
        // P_world = T_world_camera * P_local
        
        // 构造变换矩阵 T
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.pretranslate(poses[i].t);
        T.rotate(poses[i].q);

        // 使用 PCL 的 transformPointCloud 函数
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::transformPointCloud(*local_cloud, *cloud_transformed, T.matrix());

        // 拼接到全局地图
        *global_map += *cloud_transformed;
    }

    // 3. 保存结果
    // --- 【修改处 2】使用生成的输出文件名 ---
    cout << "Mapping done. Total points: " << global_map->size() << endl;
    pcl::io::savePCDFileBinary(output_filename, *global_map);
    cout << "Map saved to " << output_filename << endl;

    return 0;
}