import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 # 使用 OpenCV 的 Rodrigues 转换

def to_matrix(pose_6d):
    """把 6维向量 [rx,ry,rz, tx,ty,tz] 转为 4x4 变换矩阵"""
    r_vec = np.array(pose_6d[0:3], dtype=np.float32)
    t_vec = np.array(pose_6d[3:6], dtype=np.float32).reshape(3, 1)
    R, _ = cv2.Rodrigues(r_vec) # 旋转向量 -> 旋转矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t_vec.flatten()
    return T

def draw_camera(ax, T_wc, color, label, scale=0.5):
    """ 
    画相机视锥
    注意：C++ 里的位姿是 T_cw (World->Camera)，也就是 P_c = R*P_w + t
    我们要画相机的姿态，需要的是 T_wc (Camera->World)，也就是 T_cw 的逆
    """
    # 构造 T_cw
    R_cw = T_wc[:3, :3]
    t_cw = T_wc[:3, 3].reshape(3,1)
    
    # 求逆得到 T_wc (相机的实际位姿)
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw
    
    # 相机模型
    w = scale; h = scale * 0.75; z = scale * 0.6
    points_cam = np.array([
        [0, 0, 0], [w, h, z], [w, -h, z], [-w, -h, z], [-w, h, z]
    ]).T
    
    # 变换到世界坐标
    points_world = R_wc @ points_cam + t_wc

    # 画线
    for i in range(1, 5):
        ax.plot([points_world[0, 0], points_world[0, i]],
                [points_world[1, 0], points_world[1, i]],
                [points_world[2, 0], points_world[2, i]], color=color)
    order = [1, 2, 3, 4, 1]
    ax.plot(points_world[0, order], points_world[1, order], points_world[2, order], color=color)
    ax.text(points_world[0, 0], points_world[1, 0], points_world[2, 0], label, color=color)

def main():
    try:
        with open("ba_data.txt", 'r') as f:
            lines = f.readlines()
    except:
        print("Run C++ first!")
        return

    # 1. 读取 3D 点
    n_points = int(lines[0])
    points = []
    line_idx = 1
    for i in range(n_points):
        points.append(list(map(float, lines[line_idx].split())))
        line_idx += 1
    points = np.array(points).T

    # 2. 读取 Pose (6d vector)
    pose_gt = list(map(float, lines[line_idx].split())); line_idx+=1
    pose_init = list(map(float, lines[line_idx].split())); line_idx+=1
    pose_est = list(map(float, lines[line_idx].split())); line_idx+=1

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 画点云
    ax.scatter(points[0], points[1], points[2], s=2, c='k', alpha=0.3)

    # 画相机 (注意：输入的是 vector，需要转矩阵)
    draw_camera(ax, to_matrix(pose_gt), 'red', 'GT')
    draw_camera(ax, to_matrix(pose_init), 'green', 'Init (Noisy)')
    draw_camera(ax, to_matrix(pose_est), 'blue', 'Optimized')

    ax.set_title(f'Bundle Adjustment Demo\nGreen(Start) -> Blue(End) == Red(GT)')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlim(0, 10)
    ax.view_init(elev=-20, azim=-90) 
    plt.show()

if __name__ == "__main__":
    main()