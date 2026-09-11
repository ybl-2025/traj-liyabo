# ETH/UCY 行人轨迹分析说明

本目录按照课程图片中的 A1–A5 要求处理 5 个序列：`seq_eth`、`seq_hotel`、`zara01`、`zara02`、`students03`。

## 结果目录

每个序列目录都包含：

- `A1_trajectories.png`：每位行人一条线的世界坐标轨迹图。
- `A2_kinematics.png`：速度、运动方向和加速度模长的分布。
- `A3_distance_TTC.png`：最近邻距离和碰撞时间 TTC 的分布。
- `A4_q_k_v.png`：密度–速度（k–v）、流率–密度（q–k）、流率–速度（q–v）图。
- `A5_behavior.png`：成行、避让、震荡和瓶颈候选现象图。
- `trajectory_metrics.csv`：逐行人逐帧数据及运动学、最近邻、TTC 指标。
- `pair_events.csv`：同一时刻相距 5 m 内的行人对，以及成行/TTC 判定。
- `frame_qkv.csv`：逐帧人数、平均速度、密度和流率。

根目录的 `summary.csv` 汇总 5 个序列的行人数、观测数、时长、平均速度、最近邻距离等指标。

## 计算口径

原始 `obsmat.txt` 的有效列为：`frame, pedestrian_ID, x, z, y, vx, vz, vy`，分析使用地面平面的 `x, y, vx, vy`。五个序列的轨迹标注频率均按 2.5 Hz 处理；ETH 原视频帧率按 15 FPS，UCY 按 25 FPS 将原始帧号换算为秒。

- 速度：`sqrt(vx² + vy²)`。
- 方向：`atan2(vy, vx)`，换算到 0–360°。
- 加速度：同一行人相邻有效观测的速度差除以时间差。
- 最近邻距离：同一帧内到其他行人的最小欧氏距离。
- TTC：假设两人在短时段内保持当前速度，求相对距离首次达到 0.6 m 的时间，只保留 0–10 s 的解。没有有限碰撞解时留空。
- 场景面积：所有坐标的 1%–99% 分位矩形面积，用于减少极端坐标影响。
- 密度 `k`：当帧人数除以上述场景面积。
- 平均速度 `v`：该帧所有行人的速度均值。
- 流率 `q`：`k × v`。

## A5 判定说明

`A5_behavior.png` 是从数据中自动筛出的代表性候选，不等同于人工标注结论：

- 成行：间距 0.5–2.5 m、航向差不超过 20°，且相对位置以横向为主。
- 避让：恒速外推下 5 s 内可能进入 0.6 m 距离的行人对。
- 震荡：按运动方向变化符号次数筛选代表轨迹。
- 瓶颈：空间网格内的平均速度分布，持续低速区域可作为瓶颈线索。

课程报告中建议使用“候选”或“现象线索”等表述，并结合轨迹或原视频人工复核后再下结论。

## 复现

```powershell
python analyze_eth_ucy.py `
  --data-root "E:\traffic_data_lyb\context-group-detection-main\datasets" `
  --output "E:\traffic_data_lyb\analysis_results"
```

依赖：Python 3.10+、NumPy、pandas、Matplotlib。
