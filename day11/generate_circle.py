import numpy as np
import math

# 配置：生成一个圆形轨迹
NODE_COUNT = 100
RADIUS = 10.0
NOISE_SIGMA = 0.05  # 噪声大小

def to_quat(r, p, y):
    # 简单的欧拉角转四元数 (这里只涉及平面旋转 yaw)
    cr = math.cos(r * 0.5); sr = math.sin(r * 0.5)
    cp = math.cos(p * 0.5); sp = math.sin(p * 0.5)
    cy = math.cos(y * 0.5); sy = math.sin(y * 0.5)
    return [sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy] # x, y, z, w

print("正在生成仿真数据 my_circle.g2o ...")

with open("my_circle.g2o", "w") as f:
    # 1. 生成顶点 (VERTEX_SE3:QUAT)
    true_poses = []
    for i in range(NODE_COUNT):
        theta = 2.0 * math.pi * i / NODE_COUNT
        # 真实位置
        x = RADIUS * math.cos(theta)
        y = RADIUS * math.sin(theta)
        q = to_quat(0, 0, theta + math.pi/2) # 朝向切线方向
        
        # 加上一点噪声作为初值 (模拟里程计漂移)
        noisy_x = x + np.random.normal(0, 0.2)
        noisy_y = y + np.random.normal(0, 0.2)
        
        # ID x y z qx qy qz qw
        line = f"VERTEX_SE3:QUAT {i} {noisy_x:.6f} {noisy_y:.6f} 0.0 {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n"
        f.write(line)
        true_poses.append((x, y, theta))

    # 2. 生成边 (EDGE_SE3:QUAT)
    # 信息矩阵 (权重)
    info = "100 0 0 0 0 0 100 0 0 0 0 100 0 0 0 100 0 0 100 0 100"
    
    # 2.1 里程计约束 (相邻节点)
    for i in range(NODE_COUNT):
        next_id = (i + 1) % NODE_COUNT
        # 简单的相对运动：沿 x 轴走一段，转一点弯
        # 这里为了简化，直接用真实相对位姿加噪声生成观测
        dx = true_poses[next_id][0] - true_poses[i][0]
        dy = true_poses[next_id][1] - true_poses[i][1]
        # ... (简化生成，直接写入带噪声的观测)
        # 实际为了教学，我们只生成一个简单的环形约束
        
        # 模拟：每一步都有一点相对位移的噪声
        f.write(f"EDGE_SE3:QUAT {i} {next_id} 0.62 0.0 0.0 0.0 0.0 0.03 0.999 {info}\n")

print("生成完成！请运行: ./day11_pose_graph my_circle.g2o")