import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_camera(ax, R, t, color, label, scale=0.5):
    """
    在 3D 图里画一个相机模型（金字塔形状）
    R, t: 相机在世界坐标系的位姿 (R_wc, t_wc)
    注意：OpenCV 算出的是 R_cw, t_cw，所以传入前要取逆
    """
    # 相机模型的 5 个顶点 (金字塔: 1个光心 + 4个底角)
    # 在相机坐标系下
    w = scale
    h = scale * 0.75
    z = scale * 0.6
    points_cam = np.array([
        [0, 0, 0],      # 光心
        [w, h, z],      # 右下
        [w, -h, z],     # 右上
        [-w, -h, z],    # 左上
        [-w, h, z]      # 左下
    ]).T # 3x5

    # 转换到世界坐标系: P_w = R_wc * P_c + t_wc
    points_world = R @ points_cam + t.reshape(3, 1)

    # 画线连接顶点
    # 光心到四个角
    for i in range(1, 5):
        ax.plot([points_world[0, 0], points_world[0, i]],
                [points_world[1, 0], points_world[1, i]],
                [points_world[2, 0], points_world[2, i]], color=color)
    
    # 连接底座四边
    order = [1, 2, 3, 4, 1]
    ax.plot(points_world[0, order], points_world[1, order], points_world[2, order], color=color)
    
    # 标注重心位置
    ax.text(points_world[0, 0], points_world[1, 0], points_world[2, 0], label, color=color)

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 1. 读取 3D 点
    n_points = int(lines[0])
    points = []
    current_line = 1
    for i in range(n_points):
        vals = list(map(float, lines[current_line].strip().split()))
        points.append(vals)
        current_line += 1
    points = np.array(points).T # 3xN

    # 2. 读取 GT Pose
    vals = list(map(float, lines[current_line].strip().split()))
    R_gt = np.array(vals).reshape(3, 3)
    current_line += 1
    vals = list(map(float, lines[current_line].strip().split()))
    t_gt = np.array(vals).reshape(3, 1)
    current_line += 1

    # 3. 读取 Est Pose
    vals = list(map(float, lines[current_line].strip().split()))
    R_est = np.array(vals).reshape(3, 3)
    current_line += 1
    vals = list(map(float, lines[current_line].strip().split()))
    t_est = np.array(vals).reshape(3, 1)

    return points, R_gt, t_gt, R_est, t_est

def main():
    try:
        points, R_gt, t_gt, R_est, t_est = read_data("pose_data.txt")
    except FileNotFoundError:
        print("错误：找不到 pose_data.txt，请先运行 C++ 程序生成数据！")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 画 3D 点云
    ax.scatter(points[0, :], points[1, :], points[2, :], s=20, c='k', label='3D Points')

    # 2. 画 Camera 1 (原点，黑色)
    # Pose: R=I, t=0
    draw_camera(ax, np.eye(3), np.zeros((3,1)), 'black', 'Cam1')

    # 3. 画 Camera 2 GT (红色)
    # 注意：文件存的是 T_cw (World -> Camera)，画图需要 T_wc (Camera -> World)
    # T_wc = T_cw_inv
    # R_wc = R_cw.T
    # t_wc = -R_cw.T * t_cw
    R_wc_gt = R_gt.T
    t_wc_gt = -R_wc_gt @ t_gt
    draw_camera(ax, R_wc_gt, t_wc_gt, 'red', 'Cam2 GT')

    # 4. 画 Camera 2 Est (蓝色)
    R_wc_est = R_est.T
    t_wc_est = -R_wc_est @ t_est
    draw_camera(ax, R_wc_est, t_wc_est, 'blue', 'Cam2 Est')

    # 设置图例和轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('SLAM Initialization: Scale Ambiguity Demo\nRed: Ground Truth, Blue: Estimated (Unit Scale)')
    ax.legend()
    
    # 调整视角以便观察
    ax.view_init(elev=-80, azim=-90) # 俯视视角
    plt.show()

if __name__ == "__main__":
    main()