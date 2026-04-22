

-----

# 实验报告：Bundle Adjustment 与 COLMAP 三维重建

本实验是计算机视觉课程的 Assignment 3，主要包含两部分内容：(1) 使用 PyTorch 从零实现 Bundle Adjustment (BA) 优化；(2) 使用 COLMAP 工具链完成从多视图图像到稠密三维模型的重建全过程。

## 实验环境 (Requirements)

为了运行本实验的代码并完成重建，需要配置以下环境：

```bash
# 基础 Python 环境
pip install torch numpy matplotlib opencv-python

# 如果需要使用 PyTorch3D 的变换函数（可选，本代码已实现内置版本）
# pip install pytorch3d
```

  - **硬件/软件要求**：
      - 支持 CUDA 的 GPU（可选，代码支持 CPU 运行，但 GPU 更快）。
      - 已安装 [COLMAP](https://colmap.github.io/) 命令行工具。

## 实验内容一：基于 PyTorch 的 Bundle Adjustment (Task 1)

### 1\. 实现原理

Bundle Adjustment 的目标是通过优化相机参数和 3D 点坐标，最小化 2D 观测点与 3D 点投影结果之间的**重投影误差**。

  - **3D 点云**：初始化为 20,000 个随机 3D 坐标。
  - **相机模型**：优化 50 个视角的旋转（使用 **Euler 角** 参数化以避免正交矩阵约束）和平移向量 $T$，以及一个全局共享的焦距 $f$。
  - **损失函数**：均方误差 (MSE)，仅针对每个视角下可见的（Visibility Mask 为 1）点进行计算。

### 2\. 训练/优化过程 (Training)

在 `1.py` 中，我们使用了分层学习率的 Adam 优化器。运行以下命令开始优化：

```bash
python 1.py
```

**关键超参数设置：**

  - **Epochs**: 2000
  - **学习率 (LR)**:
      - 3D 点云: `1e-2`
      - 相机位姿 (R, T): `1e-3`
      - 焦距 $f$: `1e-1`
  - **初始化策略**: 相机平移初始化为 `[0, 0, -3.0]`，旋转初始化为 0。

### 3\. 实验结果 (Results)

#### 3.1 损失下降曲线

优化过程中，Loss 随着迭代次数的增加稳步下降，证明了自动微分在非线性最小二乘问题中的有效性。
<img width="800" height="500" alt="loss_curve" src="https://github.com/user-attachments/assets/2a2787cd-7a60-4c35-84ce-ae1f13f7521d" />

#### 3.2 3D 重建模型

优化完成后，代码生成了 `reconstructed_model.obj` 文件。该模型成功恢复了人脸/头部的几何轮廓。

  - **重建点数**: 20,000
  - **颜色恢复**: 结合 `points3d_colors.npy` 成功导出了带有 RGB 信息的点云。
<img width="1344" height="879" alt="task1_result" src="https://github.com/user-attachments/assets/a62ace0c-114c-4e04-ab22-6637ceeb51b0" />

-----

## 实验内容二：使用 COLMAP 进行三维重建 (Task 2)

本部分利用工业级工具 COLMAP 对 `data/images/` 下的 50 张图像进行重建。

### 1\. 评估流程 (Evaluation)

执行以下标准重建步骤：

1.  **特征提取与匹配**：
    ```bash
    colmap feature_extractor --database_path ./db.db --image_path ./data/images
    colmap exhaustive_matcher --database_path ./db.db
    ```
2.  **稀疏重建 (Sparse Reconstruction)**：
    ```bash
    mkdir sparse
    colmap mapper --database_path ./db.db --image_path ./data/images --output_path ./sparse
    ```
3.  **稠密重建 (Dense Reconstruction)**：
    通过 `image_undistorter`、`patch_match_stereo` 和 `stereo_fusion` 步骤生成高密度点云。

### 2\. 结果对比

| 维度 | 自研 BA (Task 1) | COLMAP 重建 (Task 2) |
| :--- | :--- | :--- |
| **输入** | 已知的 2D-3D 对应关系 | 原始 2D 图像 |
| **精度** | 取决于优化迭代次数 | 极高，包含去畸变处理 |
| **密度** | 稀疏 (20,000 点) | 稠密 (百万级点云) |
| **计算量** | 较小，仅优化已知点 | 较大，需进行特征提取与匹配 |
<img width="1537" height="889" alt="A2155AFC292A04F498F208955EA51B54" src="https://github.com/user-attachments/assets/0f69fbd4-c93f-4bac-868c-b9131aa0fc5e" />

-----

## 实验总结与讨论 (Summary)

1.  **参数化的重要性**：在优化旋转时，直接优化 $3\times3$ 矩阵会导致其失去正交性。使用欧拉角或四元数是更好的选择，本实验通过欧拉角成功简化了约束。
2.  **局部最优问题**：BA 是一个高度非线性的问题。实验中发现，如果焦距 $f$ 的初始值偏差过大，模型可能无法收敛。将 $f$ 初始化在合理范围内（根据 FoV 估算）至关重要。
3.  **工具链协同**：通过对比发现，自研 BA 帮助深入理解了底层数学原理，而 COLMAP 则展示了在处理真实场景图像时，特征匹配和鲁棒性估计（RANSAC）的重要性。

-----

