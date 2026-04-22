import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # 允许重复加载库，解决 OMP Error #15
import torch
# ... 其他导入 ...

# ==========================================
# 0. 路径自动切换 (解决 FileNotFoundError)
# ==========================================
# 自动获取当前运行的 .py 脚本所在的绝对路径
base_path = os.path.dirname(os.path.abspath(__file__))
# 锁定数据文件夹的绝对路径
DATA_DIR = os.path.join(base_path, "data")

print(f"当前数据搜索路径: {DATA_DIR}")

# ==========================================
# 1. 基础配置与超参数
# ==========================================
NUM_VIEWS = 50
NUM_POINTS = 20000
IMAGE_SIZE = 1024
CX = IMAGE_SIZE / 2
CY = IMAGE_SIZE / 2
EPOCHS = 2000

# 检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. 辅助函数：欧拉角转旋转矩阵 (纯 PyTorch 实现)
# ==========================================
def euler_angles_to_matrix(euler_angles, convention="XYZ"):
    """手动实现欧拉角转旋转矩阵，不需要安装 pytorch3d"""
    x = euler_angles[:, 0]
    y = euler_angles[:, 1]
    z = euler_angles[:, 2]

    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)

    Rx = torch.zeros((len(x), 3, 3), device=euler_angles.device)
    Rx[:, 0, 0] = 1; Rx[:, 1, 1] = cx; Rx[:, 1, 2] = -sx; Rx[:, 2, 1] = sx; Rx[:, 2, 2] = cx
    
    Ry = torch.zeros((len(y), 3, 3), device=euler_angles.device)
    Ry[:, 0, 0] = cy; Ry[:, 0, 2] = sy; Ry[:, 1, 1] = 1; Ry[:, 2, 0] = -sy; Ry[:, 2, 2] = cy

    Rz = torch.zeros((len(z), 3, 3), device=euler_angles.device)
    Rz[:, 0, 0] = cz; Rz[:, 0, 1] = -sz; Rz[:, 1, 0] = sz; Rz[:, 1, 1] = cz; Rz[:, 2, 2] = 1

    return Rx @ Ry @ Rz

# ==========================================
# 3. 数据加载
# ==========================================
def load_data():
    # 使用之前定义好的 DATA_DIR 绝对路径加载
    points2d_path = os.path.join(DATA_DIR, "points2d.npz")
    colors_path = os.path.join(DATA_DIR, "points3d_colors.npy")
    
    points2d_data = np.load(points2d_path)
    colors = np.load(colors_path)
    
    obs_pts = []
    obs_vis = []
    
    for i in range(NUM_VIEWS):
        key = f"view_{i:03d}"
        obs = points2d_data[key]
        obs_pts.append(obs[:, :2])  # (20000, 2) [x, y]
        obs_vis.append(obs[:, 2])   # (20000,)   [visibility]
        
    obs_pts = torch.tensor(np.array(obs_pts), dtype=torch.float32, device=device) 
    obs_vis = torch.tensor(np.array(obs_vis), dtype=torch.bool, device=device)    
    
    return obs_pts, obs_vis, colors

# ==========================================
# 4. 定义 Bundle Adjustment 模型
# ==========================================
class BundleAdjustmentModel(nn.Module):
    def __init__(self, num_views, num_points):
        super().__init__()
        self.points3d = nn.Parameter(torch.randn(num_points, 3) * 0.1)
        self.euler_angles = nn.Parameter(torch.zeros(num_views, 3))
        
        init_T = torch.zeros(num_views, 3)
        init_T[:, 2] = -2.5 # 初始化相机距离物体 2.5 单位
        self.T = nn.Parameter(init_T)
        
        self.f = nn.Parameter(torch.tensor(886.0))

    def forward(self):
        R = euler_angles_to_matrix(self.euler_angles)
        P = self.points3d.unsqueeze(0).expand(NUM_VIEWS, -1, -1)
        
        # Pc = P @ R^T + T
        Pc = torch.matmul(P, R.transpose(1, 2)) + self.T.unsqueeze(1)
        
        Xc, Yc, Zc = Pc[..., 0], Pc[..., 1], Pc[..., 2]
        
        # 投影公式 (根据作业要求)
        u = -self.f * (Xc / Zc) + CX
        v =  self.f * (Yc / Zc) + CY
        
        return torch.stack([u, v], dim=-1)

def save_colored_obj(filename, points, colors):
    # 保存到脚本同级目录
    out_path = os.path.join(base_path, filename)
    print(f"Saving 3D points to {out_path}...")
    with open(out_path, 'w') as f:
        for p, c in zip(points, colors):
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")

# ==========================================
# 5. 主程序
# ==========================================
def main():
    obs_pts, obs_vis, colors_np = load_data()
    model = BundleAdjustmentModel(NUM_VIEWS, NUM_POINTS).to(device)
    
    optimizer = optim.Adam([
        {'params': [model.points3d], 'lr': 1e-2},
        {'params': [model.euler_angles, model.T], 'lr': 1e-3},
        {'params': [model.f], 'lr': 1e-1}
    ])
    
    loss_history = []
    
    print("Starting Bundle Adjustment Optimization...")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        preds2d = model()
        
        # 只计算可见点的重投影误差
        diff = preds2d[obs_vis] - obs_pts[obs_vis]
        loss = torch.mean(diff ** 2)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1:04d}/{EPOCHS}] | Loss: {loss.item():.4f} | f: {model.f.item():.2f}")

    # 保存曲线图
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.title("Bundle Adjustment Loss")
    plt.savefig(os.path.join(base_path, "loss_curve.png"))
    
    # 保存结果
    final_points = model.points3d.detach().cpu().numpy()
    save_colored_obj("reconstructed_model.obj", final_points, colors_np)
    print("Task 1 Finished!")

if __name__ == "__main__":
    main()