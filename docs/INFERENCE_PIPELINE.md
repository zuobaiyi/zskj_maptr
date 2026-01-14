# MapTR 推理与可视化完整流程（针对离线数据、只看预测）

本文档描述如何从自有数据开始，生成用于 `tools/maptr/vis_pred.py` 的最小化 nuScenes-style infos `.pkl`，运行模型推理并可视化预测（PRED maps），以及把可视化结果合成视频。

适用场景：没有完整 nuScenes JSON 表（v1.0-mini）但有 camera images、calib（或可从 manifest 提取）的情况。

目录结构（示例）

- `data/nuscenses/manifest.json`            # 源数据清单（你当前仓库中的 manifest）
- `data/nuscenses/samples/`                 # 多相机图片，按 manifest 的 data_path 存放
- `data/nuscenses/calib_ori/`               # 可选：相机内参/外参文本文件
- `data/nuscenes/`                          # 为兼容性创建的软链接（见下）
- `tools/generate_minimal_infos.py`         # 生成 infos 的脚本（仓库已添加）
- `tools/maptr/vis_pred.py`                 # 仓库自带的可视化推理脚本（已做兼容性补丁）
- `inference_only/generate_video_custom.py` # 合成视频的简化脚本（已做兼容性补丁）

推荐目录树（可复制）

```
MapTR
├── mmdetection3d/
├── projects/
├── tools/
├── configs/
├── ckpts/
│   ├── r101_dcn_fcos3d_pretrain.pth
│   ├── maptr_tiny_r50_110e.pth
├── data/
│   ├── can_bus/                   # 可选：CAN 总线/定位日志
│   ├── nuscenes/                  # 或者 'nuscenses'，两者可用软链互通
│   │   ├── maps/                   # 地图文件（仅有时需要）
│   │   ├── samples/                # 相机图片，可按相机子目录组织
│   │   │   ├── CAM_FRONT/
│   │   │   ├── CAM_FRONT_LEFT/
│   │   │   └── ...
│   │   ├── sweeps/                 # 可选：lidar sweeps
│   │   ├── v1.0-mini/              # 可选：nuScenes JSON 表（完整数据）
│   │   ├── nuscenes_infos_temporal_train.pkl
│   │   └── nuscenes_infos_temporal_val.pkl
│   └── nuscenses/                 
```

## 数据目录详解

以下说明细化了仓库中三个关键目录的预期结构和命名规范，方便你准备数据或检查现有数据是否符合要求。

- `data/nuscenses/samples/`（必须）
  - 功能：存放多相机采样图片。`manifest.json` 中 `frames[].images` 的路径应指向该目录下的图片文件。
  - 组织建议：按相机子目录存放（推荐）：
    - `data/nuscenses/samples/CAM_FRONT/`、`CAM_FRONT_LEFT/`、`CAM_FRONT_RIGHT/`、`CAM_BACK/`、`CAM_BACK_LEFT/`、`CAM_BACK_RIGHT/`
    - 也可以按 scene 或时间戳扁平放置，但 `build_manifest_from_synced_folders.py` 假设每个相机为一个文件夹。
  - 文件名：任意可含时间戳的格式，但若使用 `build_manifest_from_synced_folders.py`，图片文件名需包含可解析的数字时间戳（例如 `img_1672567890123456789.jpg` 或 `1672567890123456789.jpg`）。脚本会从文件名中提取最后一个数字串并根据 `--timestamp-unit` 转换为秒。
  - 支持的扩展名：`.jpg`, `.jpeg`, `.png`, `.bmp`。
  - 路径在 `manifest.json` 中可以是绝对路径或相对于仓库的相对路径；`tools/generate_minimal_infos.py` 会尝试解析和修正路径，但最好直接使用仓库内路径（例如 `data/nuscenses/samples/CAM_FRONT/00001.jpg`）。

- `data/nuscenses/calib/`（必须）
  - 功能：包含原始相机标定文本（或其它格式）以便提取内参 K 和外参（相机到 LiDAR 或 lidar2cam 矩阵）。`tools/generate_minimal_infos.py` 和 `build_manifest_from_synced_folders.py` 将尝试解析这些文件来计算 `lidar2img`。
  - 常见布局：
    - `data/nuscenses/calib_ori/CAM_FRONT/mid-lidar-mid-camera.txt`（或其它命名），文件中包含类似 `Intrinsic:`、`Extrinsic:` 字样的行，脚本会解析矩阵。
    - 一些导出格式把每个相机的内参/外参放到单个 `.txt` 或 `.json` 文件中，脚本有多种解析分支（支持 yaml/json 的 `lidar2img` 字段、以及从 txt 中读 Intrinsic/Extrinsic）。
  - 内容示例（txt 片段，脚本会用正则提取）：
    - Intrinsic: [ [fx, 0, cx], [0, fy, cy], [0,0,1] ]
    - Extrinsic: [ [r00, r01, r02, t0], ..., [0,0,0,1] ]
  - 注意：如果只提供内参且没有直接的 lidar2img，脚本会尝试用内参与外参组合计算 `lidar2img`（视具体文件而定）。若没有任何标定信息，你仍可手工在 `manifest.json` 中填入 `calib.lidar2img`。



小结与建议：
- 推荐流程：先把图片导出到 `data/nuscenses/samples/<CAM>/`，把原始标定放到 `data/nuscenses/calib/`，再运行 `build_manifest_from_synced_folders.py` 生成 `manifest.json`；随后运行 `tools/generate_minimal_infos.py` 生成 `nuscenes_infos_temporal_val.pkl`，最后运行 `tools/maptr/vis_pred.py` 做推理与可视化。
- 对于快速试验，可手工写一个最小 `manifest.json`（参见文档前面示例），但要确保每帧包含 `images`（与 `camera_order` 一致）及 `calib.lidar2img`（或在 `calib' 中能推断）。



关于 `manifest.json` 的生成（补充）

如果你没有现成的 `manifest.json`（仓库中为 `data/nuscenses/manifest.json`），仓库提供了一个用来从按相机组织的图片文件夹生成 `manifest.json` 的工具：

`build_manifest_from_synced_folders.py`

主要用途：把每个相机导出的图片（文件名包含时间戳）按时间同步，生成每帧的多相机条目和对应的 `lidar2img` 标定矩阵。

示例命令：

```bash
python build_manifest_from_synced_folders.py \
  --scene-dir /path/to/scene_dir \
  --calib /path/to/calib \
  --out data/nuscenses/manifest.json \
  --timestamp-unit ns \
  --ref-cam CAM_FRONT \
  --max-delta-ms 20.0
```

要点说明：
- `--scene-dir`：包含每个相机子文件夹（例如 `CAM_FRONT/`, `CAM_BACK/` 等）的目录。脚本会做若干容错匹配（大小写、后缀 `_undistorted` 等）。
- `--calib`：可以是一个包含相机/传感器条目的 JSON/YAML 文件，或一个目录（含每个摄像头的 yaml/json 或 `calib_ori` 文本），脚本会尝试解析 `lidar2img` 矩阵或根据原始 txt 计算它。
- `--timestamp-unit`：图片文件名中时间戳的单位，常见为 `ns`（纳秒）或 `ms`（毫秒）。脚本会从文件名中解析数字并转换为秒。
- `--max-delta-ms`：不同相机之间同步时间容忍阈值（毫秒），超过该差值的帧会被丢弃。

输出的 `manifest.json` 格式要点（脚本生成的字段）：
- 顶层 `meta`：包含 `camera_order`（按脚本使用的相机顺序）、`ref_cam`、`timestamp_unit` 等信息。
- `frames`：列表，每个条目包含：
  - `sample_idx`：帧 id（字符串）。
  - `scene_token`：场景 id（可选）。
  - `timestamp`：以秒为单位的浮点时间戳。
  - `images`：字典，键为相机名（与 `camera_order` 一致），值为图片文件绝对或相对路径。
  - `calib`：包含 `lidar2img` 字典，按相机名给出 4x4 矩阵（list-of-lists），示例：
    "calib": { "lidar2img": { "CAM_FRONT": [[...],[...],[...],[...]], ... } }
  - `can_bus`：可选，车辆状态数组（脚本会尝试读取 `localization__global_fusion__Location/messages.jsonl` 来填充）。

示例片段（简化）：

```json
{
  "meta": {"camera_order": ["CAM_FRONT","CAM_FRONT_RIGHT",...], "timestamp_unit": "ns"},
  "frames": [
    {
      "sample_idx": "000000",
      "scene_token": "custom_scene_000",
      "timestamp": 1672567890.123456,
      "images": {"CAM_FRONT": ".../CAM_FRONT/1672567890123456789.jpg", ...},
      "calib": {"lidar2img": {"CAM_FRONT": [[...],[...],[...],[...]], ...}},
      "can_bus": [0.0, 0.0, ...]
    }
  ]
}
```


步骤二：生成 minimal infos .pkl

仓库中有脚本 `tools/generate_minimal_infos.py`，会从 `data/nuscenses/manifest.json` 和 `data/nuscenses/calib/`（如果有）提取必要字段，生成 nuScenes-style infos `.pkl`，路径为 `data/nuscenes/nuscenes_infos_temporal_val.pkl`（以及 `_train.pkl` 备份）。

运行：

```bash
python tools/generate_minimal_infos.py \
  --manifest data/nuscenses/manifest.json \
  --out-dir data/nuscenes
```

脚本要点：
- 填充每帧 `cams` 字段：包含 `data_path`（图片路径，相对于仓库）、`cam_intrinsic`、`sensor2lidar_rotation`、`sensor2lidar_translation`。
- 优先使用 manifest 中的 `calib.lidar2img` 矩阵；若没有，则尝试解析 `data/nuscenses/calib_ori/` 的 txt 文件来提取 K/R/t。
- 输出为 `{ 'infos': [...], 'metadata': {'version':'v1.0-mini'} }` 的 pkl，兼容 `projects` 中的 `CustomNuScenesDataset` 最小需求。

如果你的 manifest 字段形式不同，请先检查并适当修改脚本以匹配你的键名。

步骤三：运行可视化推理（生成 PRED maps）

我们对 `tools/maptr/vis_pred.py` 做了兼容性增强，能在没有完整 nuScenes DB 的情况下，用刚生成的 `nuscenes_infos_temporal_val.pkl` 做推理与绘图。

基本命令：

```bash
python tools/maptr/vis_pred.py \
  configs/maptr/maptr_tiny_r50_110e.py \
  ckpts/maptr_tiny_r50_110e.pth \
  --data-root data/nuscenes \
  --work-dir work_dirs/maptr_tiny_r50_110e \
  --max-samples 200
```

关键说明：
- `--max-samples` 可选，用于快速测试，避免生成过多文件导致内存/时间耗尽。脚本已加入对 `np.int/np.float` 等兼容别名的处理。
- 如果脚本找不到 `ann_file` 或 `img` 路径，会尝试自动在 `data/` 下搜索并修正路径；但仍推荐确保 `data_path` 在 infos 中指向正确的图片文件。
- 输出位置：`work_dirs/<exp>/vis_pred/_<idx>/`（每帧是一个子目录），包含：
  - `CAM_<NAME>.jpg`（6 个相机视角）
  - `surround_view.jpg` 或 `surroud_view.jpg`（脚本里存在小写/错拼变体）
  - `PRED_MAP_plot.png` 或 `pred_map_plot.png`
  - `SAMPLE_VIS.jpg`（合成的样例图）

注意：如果你没有 GT annotations（map labels），GT 图像不会生成；vis_pred 仅会写 PRED maps（这是目前你的数据情况）。

步骤四：生成视频

仓库提供两个选项：
- `tools/maptr/generate_video.py`：生成带 6-camera 网格 + PRED + GT 的样例视频（如果 GT 缺失，脚本会用占位黑图代替），并能写 `SAMPLE_VIS.jpg`。
- `inference_only/generate_video_custom.py`：我们保留了一个简化脚本，只把 `surround_view` 与 `pred_map` 横向拼接并生成视频，适合快速查看预测结果。该脚本已被修补以兼容 `surroud_view.jpg`（拼写变体）和 `PRED_MAP_plot.png`（大小写差异）。

示例（用自定义脚本）：

```bash
python inference_only/generate_video_custom.py \
  work_dirs/maptr_tiny_r50_110e/vis_pred/ \
  --fps 25 \
  --video-name inference_demo \
  --width 1600 --height 600
```

示例（用 repo 的完整脚本）：

```bash
python tools/maptr/generate_video.py work_dirs/maptr_tiny_r50_110e/vis_pred/ \
  --fps 10 --video-name maptr_demo --max-samples 1000
```

生成的视频位置：一般写在 `work_dirs/maptr_tiny_r50_110e/inference_demo.mp4`（或 `maptr_demo.mp4`），请以脚本输出为准。

常见问题与排查

- 找不到 `projects` 包：请确保仓库根加入 `PYTHONPATH`（参见先决条件）。
- NumPy 的 `np.int/np.float` 导致导入链失败：如果你遇到 ImportError/AttributeError，升级/修复 mmcv 或在脚本顶部加入向后兼容别名（我们已在 `vis_pred.py` 做了这项改动）。
- `vis_pred` 找不到图片路径：检查 pkl 中每个 `cams[i]['data_path']` 是否存在，路径应指向真实文件，例如 `data/nuscenses/samples/XXXXX.jpg`。
- 视频中缺少 GT：因为没有 map annotations；要显示 GT，需要提供 `ann_file`（map annotations）或完整 nuScenes JSON+annotations，或自行实现 GT 渲染（超出当前文档范围）。
- 文件名大小写或错拼：脚本现在支持 `surroud_view.jpg` 与 `surround_view.jpg`，以及 `PRED_MAP_plot.png` / `pred_map_plot.png`。

附：示例快速命令集合

```bash
# 1) （可选）软链接
ln -s $(pwd)/data/nuscenses $(pwd)/data/nuscenes

# 2) 生成 infos
python tools/generate_minimal_infos.py --manifest data/nuscenses/manifest.json --out-dir data/nuscenes

# 3) 小批量可视化推理（快速测试）
python tools/maptr/vis_pred.py configs/maptr/maptr_tiny_r50_110e.py ckpts/maptr_tiny_r50_110e.pth \
  --data-root data/nuscenes --work-dir work_dirs/maptr_tiny_r50_110e --max-samples 100

# 4) 合成视频
python inference_only/generate_video_custom.py work_dirs/maptr_tiny_r50_110e/vis_pred/ --fps 25 --video-name inference_demo
```

后续可选项（进阶）

- 如果你能提供 nuScenes 的 `v1.0-mini` JSON 表（`category.json`, `sample.json` 等）并放到 `data/nuscenes/v1.0-mini/`，建议使用官方 `tools/create_data.py nuscenes ...` 生成完整的 infos `.pkl`，以获得完整的 GT 支持。
- 如果想把 GT 加回视频里，需要提供或生成地图注释（map annotations/labels），并确保 `vis_pred.py` 能读取到 `GT_fixednum_pts_MAP.png`（或让 vis_pred 渲染 GT 并保存）。

文件与输出位置回顾

- 生成的 infos：`data/nuscenes/nuscenes_infos_temporal_val.pkl`
- vis 输出（每帧目录）：`work_dirs/<exp>/vis_pred/_000000/`（含 `CAM_*.jpg`, `PRED_MAP_plot.png`, `surround_view.jpg/surroud_view.jpg` 等）
- 合成视频：`work_dirs/<exp>/inference_demo.mp4`

如果你愿意，我可以：
- 把这份文档再写成中文 README 并提交到 `docs/`（已完成），或把关键命令写入一个 shell 脚本方便一键运行；
- 尝试用 10 个样本完整跑一遍并把生成的视频（小样本）提供给你；
- 指导你如何准备 GT annotations 并把 GT 显示在 vis 中。

---
文档已保存：`docs/INFERENCE_PIPELINE.md`。后续我会把 TODO 列表更新为已完成项。
