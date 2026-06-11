# Assignment 4 Report - Simplified 3D Gaussian Splatting

## Task 1: Structure-from-Motion with COLMAP

本项目选择 `data/lego` 作为实验场景。已在 `dip` 环境中安装并调用 COLMAP CPU 版本完成 SfM，相机参数和稀疏点云输出如下。为了便于提交，关键结果已统一放到项目根目录的 `results/lego_hq/`：

- `data/lego/sparse/0_text/cameras.txt`
- `data/lego/sparse/0_text/images.txt`
- `data/lego/sparse/0_text/points3D.txt`
- `results/lego_hq/colmap/cameras.txt`
- `results/lego_hq/colmap/images.txt`
- `results/lego_hq/colmap/points3D.txt`

运行命令：

```powershell
D:\software\miniconda3\envs\dip\python.exe mvs_with_colmap.py --data_dir data/lego
D:\software\miniconda3\envs\dip\python.exe debug_mvs_by_projecting_pts.py --data_dir data/lego
```

`debug_mvs_by_projecting_pts.py` 已生成 100 张重投影验证图，位于 `results/lego_hq/projections/`。这些图片将 COLMAP 恢复出的 3D 稀疏点投影回原始视角，用于检查相机内外参和点云坐标是否一致。

## Task 2: Simplified 3DGS Implementation

已完成 README 中要求的核心 TODO：

- `gaussian_model.py`: 由四元数构造旋转矩阵 `R`，由尺度参数构造对角缩放矩阵 `S`，并用 `Sigma = (R S)(R S)^T` 得到每个 3D Gaussian 的协方差矩阵。
- `gaussian_renderer.py`: 将 3D Gaussian 投影到相机平面，使用透视投影 Jacobian 计算二维协方差 `Sigma' = J W Sigma W^T J^T`。
- `gaussian_renderer.py`: 按二维高斯公式计算每个 Gaussian 在每个像素处的贡献。
- `gaussian_renderer.py`: 按深度排序后进行 alpha blending，累积透射率并得到最终 RGB 图像。

为了保证当前 Windows + CPU 环境能稳定跑通，还补充了以下工程处理：

- 用 `torch.cdist + topk` 替代 `pytorch3d.ops.knn_points`，减少额外依赖。
- 为 `natsort` 增加 fallback，自然排序依赖缺失时仍可读取数据。
- 给训练和渲染脚本加入 `--max_views`、`--maximum_pts_num`、`--downsample_factor`、`--num_workers` 等参数，便于在 CPU 上完成可验证实验。
- 给训练脚本加入 `--scale_multiplier`，可以用更小初始高斯半径减少模糊。
- 将 Gaussian 初始化从 50 近邻大半径改为 8 近邻局部半径，并降低初始 opacity，避免粗大 splat 过早遮挡细节。
- 渲染 alpha 使用未归一化 Gaussian 响应，更接近 3DGS splatting 的实际形式，减少由大协方差归一化带来的发灰和模糊。
- 对 2D covariance、determinant、Gaussian exponent、alpha 和最终图像做数值稳定处理，避免训练中出现 NaN。
- 在训练结束时额外保存最终 epoch checkpoint。

最终训练命令：

```powershell
D:\software\miniconda3\envs\dip\python.exe train.py `
  --colmap_dir data/lego `
  --checkpoint_dir results/lego_hq `
  --num_epochs 30 `
  --device cpu `
  --max_views 30 `
  --maximum_pts_num 3000 `
  --debug_samples 4 `
  --debug_every 15 `
  --save_every 15 `
  --num_workers 0 `
  --downsample_factor 4 `
  --scale_multiplier 0.25
```

训练使用 30 个视角、3000 个 COLMAP 稀疏点，并将图像下采样 4 倍。相比最初的快速版本，这版分辨率从 50x50 提高到 100x100，Gaussian 数量翻倍，最终 loss 约为 `0.0262`，画面明显更清晰。最终 checkpoint 中所有张量均为 finite，没有 NaN 或 Inf。

主要输出文件：

- `results/lego_hq/checkpoint_000030.pt`
- `results/lego_hq/debug_images/epoch_0030.png`
- `results/lego_hq/debug_images/epoch_0030_200px.png`
- `results/lego_hq/debug_rendering.mp4`

训练完成后，使用最终 checkpoint 渲染了 120 帧环绕视频：

```powershell
D:\software\miniconda3\envs\dip\python.exe render_3dgs_mv.py `
  --colmap_dir data/lego `
  --checkpoint results/lego_hq/checkpoint_000030.pt `
  --output results/lego_hq/render_mv_final.mp4 `
  --num_frames 120 `
  --fps 30 `
  --device cpu `
  --max_views 30 `
  --maximum_pts_num 3000 `
  --downsample_factor 4 `
  --scale_multiplier 0.25
```

最终视频输出：

- `results/lego_hq/render_mv_final.mp4`
- `results/lego_hq/render_mv_final_200px.mp4`

另外，为了避免播放器放大 100x100 视频时显得发糊，还额外用同一 checkpoint 按 200x200 分辨率渲染了 `render_mv_final_200px.mp4` 和 `epoch_0030_200px.png`，作为更适合检查的展示结果。

### Quantitative Evaluation and Ablation

为了更清楚地展示训练改进效果，额外编写了 `evaluate_3dgs.py`，对 `checkpoint_000000.pt`、`checkpoint_000015.pt`、`checkpoint_000030.pt` 在固定视角 `[0, 8, 16, 24]` 上进行评估，计算 L1、PSNR 和 SSIM。

运行命令：

```powershell
D:\software\miniconda3\envs\dip\python.exe evaluate_3dgs.py `
  --colmap_dir data/lego `
  --checkpoint_dir results/lego_hq `
  --output_dir results/lego_hq/analysis `
  --downsample_factor 4 `
  --maximum_pts_num 3000 `
  --max_views 30 `
  --scale_multiplier 0.25 `
  --eval_indices 0,8,16,24 `
  --device cpu
```

评估结果如下：

| Epoch | L1 lower better | PSNR higher better | SSIM higher better |
| --- | ---: | ---: | ---: |
| 0 | 0.0725 | 16.34 | 0.661 |
| 15 | 0.0254 | 24.06 | 0.869 |
| 30 | 0.0218 | 25.15 | 0.892 |

可以看到，训练从 epoch 0 到 epoch 30 后，L1 明显下降，PSNR 和 SSIM 持续上升，说明优化确实提升了重建结果。相关评估产物如下：

- `results/lego_hq/analysis/metrics.csv`
- `results/lego_hq/analysis/per_view_metrics.csv`
- `results/lego_hq/analysis/metrics_summary.md`
- `results/lego_hq/analysis/metric_curves.png`
- `results/lego_hq/analysis/final_comparison.png`

## Task 3: Comparison with Official 3DGS

| 方面 | 本项目 PyTorch 简化版 | 官方 3DGS |
| --- | --- | --- |
| 渲染质量 | 直接使用 COLMAP 稀疏点初始化，没有 adaptive densification，细节覆盖不足，边缘和薄结构容易缺失。 | 会在训练中动态 densification / pruning，能增加有效 Gaussian 数量，重建细节更完整。 |
| 颜色表达 | 每个 Gaussian 优化固定 RGB，不能建模明显的视角相关颜色变化。 | 通常使用 spherical harmonics 表达视角相关颜色，高光和反射效果更自然。 |
| 训练速度 | 纯 PyTorch 直接构造 `N x H x W` 张量，计算量和内存开销较大，CPU 训练较慢。 | 使用 CUDA tile-based rasterizer，只处理局部覆盖区域，训练和渲染速度明显更快。 |
| 显存/内存占用 | 中间张量随 Gaussian 数量和图像分辨率快速增长。 | 自定义 rasterization 和 backward 更节省显存。 |
| 工程完整度 | 适合教学展示投影、协方差传播和 alpha blending 的可微链路。 | 面向真实重建任务，包含完整训练策略、稠密化、裁剪和高性能 rasterizer。 |

差异主要来自三个方面。第一，官方实现有自适应 Gaussian 增密，可以弥补 SfM 稀疏点云覆盖不足的问题；本项目只从初始点云优化固定数量 Gaussian，因此细节更少。第二，官方 CUDA rasterizer 会按 tile 组织计算，避免每个 Gaussian 对每个像素都做全量计算；本项目的纯 PyTorch 实现更直观，但速度和内存效率更低。第三，官方实现的颜色模型、训练调度和 pruning 机制更完整，因此在质量、速度和资源利用上都优于本教学版。
