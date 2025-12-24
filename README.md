# MPSCAP 项目说明

本项目提供肌电/IMU/手套数据的采集、实时可视化与基础算法示例（含 Kafka 发布与空域手部姿态显示）。下文给出快速上手与新增的 EMGNet 训练示例使用方法。

## 主要功能
- 数据采集：通过 UI 进行设备配置、波形显示与数据保存/发布（Kafka、文件、TSDB）。
- 眼动信号：内置 Tobii 4C 启动器，添加眼动仪设备后实时采集并绘制 X/Y 双通道轨迹，可 Kafka 发布。
- 空域可视化：IMU 手部姿态模型、坐标系映射、静止检测与互补滤波。
- 数据解码与共享内存：驱动层解码 EMG/ACC/Glove 并写入共享内存，供 UI 实时读取。
- Kafka 发布：可配置 Kafka 服务器，实时发布采集数据。
- 模型训练示例：`scripts/train_emgnet.py` 演示基于 4 个数据文件的分类训练与实时可视化。

## 环境准备
```bash
pip install -r requirements.txt
# 训练示例所需：PyTorch + matplotlib + scikit-learn（需额外安装）
pip install torch torchvision torchaudio matplotlib scikit-learn
```

## EMGNet 训练示例（使用 4 个文件：x_train, y_train, x_test, y_test）
- 数据要求：
  - `x_train.npy`, `x_test.npy` 形状推荐为 `(样本数, 通道数, 序列长度)`，浮点型。
  - `y_train.npy`, `y_test.npy` 为整型类别标签，取值 `0..(num_classes-1)`。
- 运行示例：
```bash
python scripts/train_emgnet.py ^
  --x-train path/to/x_train.npy --y-train path/to/y_train.npy ^
  --x-test path/to/x_test.npy   --y-test path/to/y_test.npy ^
  --epochs 20 --batch-size 64 --lr 1e-3
```
- 实时可视化：
  - 左图：训练/验证 loss 曲线实时更新。
  - 右图：每个 epoch 更新一次的混淆矩阵。
  - 训练完成后输出整体精度与保存的最佳模型（`best_emgnet.pt`，可通过 `--save-path` 自定义）。

## 数据保存（统一 EDF）
- 点击 UI 的“开始保存数据”选择目录，采集中仅在内存缓存增量；点击“停止保存”后统一导出 `MPSCAP_data.edf`（包含可用的 EMG/EEG/眼动信号）。
- EMG/EEG 采样率取共享内存的设备采样率（当前肌电 500 Hz），眼动采样率固定 30 Hz（当前设备）。
- 需先安装 EDF 依赖：`pip install pyedflib`

## 目录速览
- `mpscap/ui/pages/data_acquisition.py`：采集界面，含眼动实时绘制、Kafka 配置与波形显示。
- `mpscap/ui/widgets/spatial_visualizer.py`：手部姿态可视化组件，含互补滤波与坐标映射。
- `mpscap/core/drivers/data_decoder.py`：数据解码，含 Kafka 发布与通道数维护。
- `mpscap/core/drivers/eye_tracker.py`：Tobii 4C 眼动采集封装，启动本地服务并输出 gaze 坐标。
- `mpscap/core/utils/`：共享内存、数据管理、滤波工具。
- `scripts/train_emgnet.py`：EMGNet 训练与实时可视化示例。

## 常用命令
- 启动采集应用：`python -m mpscap.app`
- 训练示例（见上）：`python scripts/train_emgnet.py ...`

## 备注
- 若使用 Kafka，请先根据 `docs/Kafka安装和启动指南.md` 启动服务，并在 UI 中配置服务器地址。
- 训练脚本默认使用 CPU；如有 GPU 且已安装 CUDA 版 PyTorch，会自动切换到 GPU。


