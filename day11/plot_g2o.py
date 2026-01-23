import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def read_g2o_file(filename):
    x, y, z = [], [], []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('VERTEX_SE3:QUAT'):
                parts = line.split()
                # 格式: VERTEX_SE3:QUAT ID x y z qx qy qz qw
                x.append(float(parts[2]))
                y.append(float(parts[3]))
                z.append(float(parts[4]))
    return x, y, z

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 plot_g2o.py <file1.g2o> [file2.g2o]")
        sys.exit(1)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 颜色列表，用于区分不同的文件
    colors = ['r', 'b', 'g', 'k']
    labels = ['Optimized (Result)', 'Initial (Noisy)']

    # 倒序读取，这样第一个文件（优化后的）会画在最上面
    files = sys.argv[1:]
    
    for i, fname in enumerate(files):
        print(f"Reading {fname} ...")
        x, y, z = read_g2o_file(fname)
        
        # 如果是圆环，z轴变化不大，为了好看可以只画 xy
        # 但为了通用，我们还是画 3D 散点图
        color = colors[i % len(colors)]
        label = fname if i >= len(labels) else labels[i]
        
        ax.scatter(x, y, z, c=color, label=label, s=2, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("Pose Graph Optimization Result")
    plt.show()