# StreamMapNet vs. MapTR 对比概要

## 概览
- **StreamMapNet (WACV 2024)**：专注多相机流式在线矢量 HD 图构建，内置时序 BEV 记忆与流式更新，支持一键可视化/视频生成。
- **MapTR / MapTRv2 (ICLR Spotlight / IJCV 2024)**：端到端矢量 HD 图构建，提出 permutation-equivalent 建模与层次查询；MapTRv2 强化性能与收敛。

## 任务与场景
| 维度 | StreamMapNet | MapTR / MapTRv2 |
| --- | --- | --- |
| 任务类型 | 流式在线矢量 HD 图 | 在线矢量 HD 图（含 v2 增强） |
| 传感器 | 纯视觉 (6 摄像头) | 视觉为主，可扩展 LiDAR/融合 |
| 时序建模 | Streaming BEV + GRU 记忆 | 可选时序/多模态设置（v2 支持中心线） |
| 输出 | 矢量多类别车道/路沿/隔离带 | 矢量车道/路沿/中心线等 |

## 输入与输出详解 (Inputs & Outputs)

### StreamMapNet
- **输入 (Inputs)**：
  - **时序多视角图像流**：连续帧的 6-Cam 图像输入（通常需要按时间戳严格排序）。
  - **历史 BEV 记忆 (Temporal Memory)**：上一帧传递的 BEV 隐藏状态（Hidden State/Recurrent State），这是流式处理的核心。
  - **自车运动变换 (Ego-Motion)**：上一帧到当前帧的车辆位姿变化矩阵，用于将历史 BEV 特征对齐到当前坐标系。
  - **相机参数**：内参、外参及当前帧的自车位姿。
- **输出 (Outputs)**：
  - **当前帧矢量结果**：实时生成的矢量化地图元素（折线点集、类别、分数）。
  - **更新后的记忆 (Updated Memory)**：融合了当前信息的 BEV 特征，传递给下一帧，无需重复计算历史帧图像。
- **使用提示**：它是“有状态”(Stateful) 的模型，推理必须严格按时间顺序进行；首帧需初始化记忆；内存开销相对平稳（不需要像传统时序模型那样一次并行提取多帧 Backbone）。

### MapTR / MapTRv2
- **输入 (Inputs)**：
  - 多视角摄像头图像（典型为 6-Cam），按相机顺序输入（image tensors，已归一化与 resize）。
  - 相机内参 / 外参（intrinsics / extrinsics），用于将像素特征变换到 BEV。
  - 可选：历史帧 / 时间信息（用于时序设置）、LiDAR 或视觉-雷达融合特征（在多模态配置中）。
  - 常见预处理：resize、normalize、按模型配置的裁剪与尺度（例如 NuScenes 的 BEV 范围配置）。
- **输出 (Outputs)**：
  - 矢量化地图元素（多段折线 / 控制点形式），每个元素包含：类别标签（车道、路沿、中心线等）、有序点序列（x,y，BEV 或世界坐标系）、置信度分数与实例 id。
  - 输出坐标系：以车辆为中心的 BEV 坐标（范围由配置决定，如 60×30 m 等）。
  - 表示与匹配：模型以查询为单位预测多段线，采用 permutation-invariant 的匹配策略对齐预测与标注（训练时常用 L1/Chamfer 损失）。
- **使用提示**：确保使用与训练一致的相机标定与 BEV 范围配置以获得可复现结果，输出可直接用于下游 HD Map 构建或可视化。

## 性能（NuScenes 示例）
- **StreamMapNet**（newsplit，60×30m）：AP≈34.1（ped/div/bound 综合），单卡 480×800×6-cam 下约 7.3 FPS（本地样例，未量化)。
- **MapTR/MapTRv2**（官方报告，batch=1，6视角，RTX3090）：
  - MapTR-tiny R50 24e：mAP ≈50.0，FPS ≈15.1。
  - MapTRv2 R50 24e：mAP ≈61.4，FPS ≈14.1。
  - 更大模型（V2-99）可达更高 mAP（≈73.4）但 FPS 较低（≈9.9）。


### Benchmark 表（补充 CPU / GPU / 内存 / 显存）（本地评测）
| 模型 | 硬件 | 分辨率 / 视角 | Batch | 单帧推理 (ms) | GPU 显存 (used)  | CPU / 内存 |
| --- | --- | --- | --- | --- | --- | --- |
| StreamMapNet (nusc 60×30) | 单卡 | 480×800 ×6 cam | 1 | 136.3 | ~812 MB | 200% |
| MapTR-tiny R50 110e* | 单卡（模型同上采样日志） | cam or lidar| 1 | 160 | ~2588 MB | 200.8%|

## 体系架构差异
- **StreamMapNet**：
  - BEVFormer-style 感知 + Streaming BEV 记忆（GRU）
  - 强调在线/流式与低显存占用
- **MapTR / MapTRv2**：
  - 矢量查询与层次匹配，支持多模态与中心线（v2）
  - 训练/收敛速度与精度平衡良好
  - 生态成熟（多配置、多数据集、社区模型）

## 可视化视频比对
<div align="center">

| MapTR 演示 | StreamMapNet 演示 |
| :---: | :---: |
| <video src="./MapTr.mp4" width="100%" controls autoplay loop muted></video> | <video src="./StreamMapNet.mp4" width="100%" controls autoplay loop muted></video> |
| [如果无法播放请点击这里下载 MapTr.mp4](./MapTr.mp4) | [如果无法播放请点击这里下载 StreamMapNet.mp4](./StreamMapNet.mp4) |

</div>

## 场景建议
- **追求更高精度/成熟生态/多模态与中心线支持**：建议优先尝试 MapTR
