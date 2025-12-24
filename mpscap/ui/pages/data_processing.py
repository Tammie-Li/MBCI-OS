"""
数据处理页面：提供数据分段、特征提取、模型训练与性能评估流程。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets
from scipy import signal
from sklearn.metrics import confusion_matrix


@dataclass
class DummyModel:
    """简单的最近均值分类器，用于占位示例。"""

    class_means: Dict[int, np.ndarray]

    def predict(self, features: np.ndarray) -> np.ndarray:
        preds = []
        for feat in features:
            distances = {
                label: np.linalg.norm(feat - mean_vec) for label, mean_vec in self.class_means.items()
            }
            preds.append(min(distances, key=distances.get))
        return np.asarray(preds, dtype=int)


class AlgorithmConfigDialog(QtWidgets.QDialog):
    """算法配置对话框（参考添加设备的交互风格）。"""

    def __init__(
        self,
        algo: str,
        epochs: int,
        batch: int,
        lr: float,
        workers: int,
        time_point: int,
        N_t: int,
        N_s: int,
        dropout: float,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("配置算法")
        layout = QtWidgets.QFormLayout(self)

        self._algo_combo = QtWidgets.QComboBox()
        self._algo_combo.addItems(["EMGNet (深度学习)", "最近均值", "SVM", "随机森林"])
        self._algo_combo.setCurrentText(algo)
        layout.addRow("算法", self._algo_combo)

        # 仅 EMGNet 使用的参数
        self._epochs_spin = QtWidgets.QSpinBox()
        self._epochs_spin.setRange(1, 500)
        self._epochs_spin.setValue(epochs)
        layout.addRow("Epochs", self._epochs_spin)

        self._batch_spin = QtWidgets.QSpinBox()
        self._batch_spin.setRange(1, 2048)
        self._batch_spin.setValue(batch)
        layout.addRow("Batch size", self._batch_spin)

        self._lr_spin = QtWidgets.QDoubleSpinBox()
        self._lr_spin.setDecimals(6)
        self._lr_spin.setRange(1e-6, 1.0)
        self._lr_spin.setSingleStep(1e-4)
        self._lr_spin.setValue(lr)
        layout.addRow("Learning rate", self._lr_spin)

        self._workers_spin = QtWidgets.QSpinBox()
        self._workers_spin.setRange(0, 16)
        self._workers_spin.setValue(workers)
        layout.addRow("DataLoader 线程", self._workers_spin)

        self._time_point_spin = QtWidgets.QSpinBox()
        self._time_point_spin.setRange(3, 199)
        self._time_point_spin.setSingleStep(2)
        self._time_point_spin.setValue(time_point)
        layout.addRow("time_point(奇数)", self._time_point_spin)

        self._nt_spin = QtWidgets.QSpinBox()
        self._nt_spin.setRange(1, 256)
        self._nt_spin.setValue(N_t)
        layout.addRow("N_t", self._nt_spin)

        self._ns_spin = QtWidgets.QSpinBox()
        self._ns_spin.setRange(1, 512)
        self._ns_spin.setValue(N_s)
        layout.addRow("N_s(可被N_t整除)", self._ns_spin)

        self._dropout_spin = QtWidgets.QDoubleSpinBox()
        self._dropout_spin.setRange(0.0, 0.95)
        self._dropout_spin.setSingleStep(0.05)
        self._dropout_spin.setValue(dropout)
        layout.addRow("Dropout", self._dropout_spin)

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addRow(btn_box)

        self._update_visibility()
        self._algo_combo.currentTextChanged.connect(self._update_visibility)

    def _update_visibility(self) -> None:
        is_emgnet = self._algo_combo.currentText().startswith("EMGNet")
        for w in [
            self._epochs_spin,
            self._batch_spin,
            self._lr_spin,
            self._workers_spin,
            self._time_point_spin,
            self._nt_spin,
            self._ns_spin,
            self._dropout_spin,
        ]:
            w.setEnabled(is_emgnet)

    def get_config(self) -> Dict[str, object]:
        return {
            "algo": self._algo_combo.currentText(),
            "epochs": self._epochs_spin.value(),
            "batch": self._batch_spin.value(),
            "lr": self._lr_spin.value(),
            "workers": self._workers_spin.value(),
            "time_point": self._time_point_spin.value(),
            "N_t": self._nt_spin.value(),
            "N_s": self._ns_spin.value(),
            "dropout": self._dropout_spin.value(),
        }

class EMGNetTrainWorker(QtCore.QThread):
    """在后台线程中执行 EMGNet 训练，避免阻塞 UI。"""

    progress = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, paths: Dict[str, str], params: Dict[str, float], parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.paths = paths
        self.params = params

    def run(self) -> None:  # pragma: no cover - 运行期线程
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
        except Exception:
            self.error.emit("未安装 PyTorch，请先安装：pip install torch torchvision torchaudio")
            return

        # 读取数据
        try:
            x_train = np.load(self.paths["x_train"])
            y_train = np.load(self.paths["y_train"])
            x_test = np.load(self.paths["x_test"])
            y_test = np.load(self.paths["y_test"])
        except Exception as exc:
            self.error.emit(f"数据加载失败：{exc}")
            return

        # 校验形状
        if x_train.ndim != 3 or x_test.ndim != 3:
            self.error.emit("x_train/x_test 需为三维张量 (样本, 通道, 序列长度)")
            return
        if x_train.shape[1] != x_test.shape[1]:
            self.error.emit("x_train 与 x_test 的通道数不一致")
            return
        if y_train.ndim != 1 or y_test.ndim != 1:
            self.error.emit("y_train/y_test 需为一维标签数组")
            return

        # 不再对标签减1；按数据动态确定类别数
        num_classes = int(np.max([y_train.max(), y_test.max()])) + 1
        class_names = [str(i) for i in range(num_classes)]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 数据集与加载器
        train_ds = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        test_ds = TensorDataset(
            torch.tensor(x_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(self.params["batch_size"]),
            shuffle=True,
            num_workers=int(self.params["num_workers"]),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=int(self.params["batch_size"]),
            shuffle=False,
            num_workers=int(self.params["num_workers"]),
        )

        # 预处理（可选）：参考给定 band_pass_filter，默认带通 20-150Hz，fs=500
        def apply_preprocess(x: np.ndarray) -> np.ndarray:
            x_proc = x
            ftype = self.params.get("filter_type", "无")
            fs = 500.0  # 参考示例
            try:
                if ftype == "高通":
                    b, a = signal.butter(3, self.params["f_low"] / (fs / 2), btype="highpass")
                    x_proc = signal.filtfilt(b, a, x_proc, axis=-1)
                elif ftype == "低通":
                    b, a = signal.butter(3, self.params["f_high"] / (fs / 2), btype="lowpass")
                    x_proc = signal.filtfilt(b, a, x_proc, axis=-1)
                elif ftype == "带通":
                    low = self.params["f_low"] / (fs / 2)
                    high = self.params["f_high"] / (fs / 2)
                    if 0 < low < high < 1:
                        b, a = signal.butter(3, [low, high], btype="bandpass")
                        x_proc = signal.filtfilt(b, a, x_proc, axis=-1)
            except Exception:
                pass
            if self.params.get("do_zscore", True):
                mean = x_proc.mean(axis=-1, keepdims=True)
                std = x_proc.std(axis=-1, keepdims=True) + 1e-8
                x_proc = (x_proc - mean) / std
            return x_proc

        # 应用预处理
        x_train = apply_preprocess(x_train)
        x_test = apply_preprocess(x_test)

        # 标签与分布记录（用于比对）
        label_info = {
            "train_unique": np.unique(y_train).tolist(),
            "test_unique": np.unique(y_test).tolist(),
            "train_counts": np.bincount(y_train.astype(int), minlength=int(np.max(y_train)) + 1).tolist(),
            "test_counts": np.bincount(y_test.astype(int), minlength=int(np.max(y_test)) + 1).tolist(),
            "x_train_shape": x_train.shape,
            "x_test_shape": x_test.shape,
            "y_train_shape": y_train.shape,
            "y_test_shape": y_test.shape,
        }

        # 模型定义（参考根目录 EMGNet.py，实现动态计算输出尺寸）
        class EMGNet(nn.Module):
            def __init__(
                self,
                num_classes: int,
                drop_out: float,
                time_point: int,
                channel: int,
                N_t: int,
                N_s: int,
                seq_len: int,
            ):
                super().__init__()
                if time_point % 2 == 0:
                    raise ValueError("time_point 需为奇数")
                if N_s % N_t != 0:
                    raise ValueError("N_s 必须能被 N_t 整除（depthwise groups 约束）")

                self.block_1 = nn.Sequential(
                    nn.ZeroPad2d((time_point // 2, time_point // 2 + 1, 0, 0)),
                    nn.Conv2d(1, N_t, (1, time_point), bias=False),
                    nn.BatchNorm2d(N_t),
                )
                self.block_2 = nn.Sequential(
                    nn.Conv2d(N_t, N_s, (channel, 1), groups=N_t, bias=False),
                    nn.BatchNorm2d(N_s),
                    nn.ELU(),
                    nn.AvgPool2d((1, 4)),
                    nn.Dropout(drop_out),
                )
                self.block_3 = nn.Sequential(
                    nn.ZeroPad2d((N_s // 2 - 1, N_s // 2, 0, 0)),
                    nn.Conv2d(N_s, N_s, (1, N_s), groups=N_s, bias=False),
                    nn.Conv2d(N_s, N_s, (1, 1), bias=False),
                    nn.BatchNorm2d(N_s),
                    nn.ELU(),
                    nn.AvgPool2d((1, 8)),
                    nn.Dropout(drop_out),
                )
                # 动态计算全连接层尺寸，避免固定 256 假设
                with torch.no_grad():
                    dummy = torch.zeros(1, 1, channel, seq_len)
                    out = self.block_3(self.block_2(self.block_1(dummy)))
                    flat_dim = out.view(1, -1).shape[1]
                self.fc1 = nn.Linear(flat_dim, num_classes)

            def forward(self, x):
                # 输入 x: [B, C, T]，转换为 [B, 1, C, T] 以匹配 2D 卷积
                x = x.unsqueeze(1)
                x = self.block_1(x)
                x = self.block_2(x)
                x = self.block_3(x)
                x = x.view(x.size(0), -1)
                logits = self.fc1(x)
                return logits

        # 使用默认超参（可后续做为 UI 参数暴露）
        model = EMGNet(
            num_classes=num_classes,
            drop_out=0.5,
            time_point=9,
            channel=x_train.shape[1],
            N_t=8,
            N_s=16,  # 必须能被 N_t 整除
            seq_len=x_train.shape[2],
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=float(self.params["lr"]), weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        def evaluate(loader):
            model.eval()
            total_loss, correct, total = 0.0, 0, 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for bx, by in loader:
                    bx = bx.to(device)
                    by = by.to(device)
                    logits = model(bx)
                    loss = criterion(logits, by)
                    total_loss += loss.item() * bx.size(0)
                    preds = logits.argmax(dim=1)
                    correct += (preds == by).sum().item()
                    total += bx.size(0)
                    all_preds.append(preds.cpu())
                    all_labels.append(by.cpu())
            avg_loss = total_loss / max(total, 1)
            acc = correct / max(total, 1)
            preds_np = torch.cat(all_preds).numpy() if all_preds else np.array([])
            labels_np = torch.cat(all_labels).numpy() if all_labels else np.array([])
            if labels_np.size > 0:
                class_count = np.bincount(labels_np, minlength=num_classes)
                correct_count = np.bincount(
                    labels_np, minlength=num_classes, weights=(preds_np == labels_np)
                )
                per_class_acc = np.divide(
                    correct_count,
                    class_count,
                    out=np.zeros_like(correct_count, dtype=float),
                    where=class_count > 0,
                )
                cm = confusion_matrix(labels_np, preds_np, labels=np.arange(num_classes))
            else:
                per_class_acc = np.zeros(num_classes, dtype=float)
                cm = np.zeros((num_classes, num_classes), dtype=int)
            return avg_loss, acc, per_class_acc, cm

        train_losses: List[float] = []
        test_losses: List[float] = []
        best_acc = 0.0
        best_path = self.params["save_path"]
        epochs = int(self.params["epochs"])

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            total = 0
            for bx, by in train_loader:
                bx = bx.to(device)
                by = by.to(device)
                optimizer.zero_grad()
                logits = model(bx)
                print(logits)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * bx.size(0)
                total += bx.size(0)

            avg_train_loss = running_loss / max(total, 1)
            train_losses.append(avg_train_loss)

            # 仅在最后一次评估并汇报（混淆矩阵/每类准确率基于训练集）
            if epoch == epochs:
                train_eval_loss, train_eval_acc, train_per_class_acc, train_cm_mat = evaluate(train_loader)
                test_loss, test_acc, _, _ = evaluate(test_loader)
                test_losses.append(test_loss)
                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save({"model_state": model.state_dict(), "acc": best_acc}, best_path)
                self.progress.emit(
                    {
                        "epoch": epoch,
                        "train_loss": avg_train_loss,
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                        "train_losses": train_losses.copy(),
                        "test_losses": test_losses.copy(),
                        "per_class_acc": train_per_class_acc,
                        "cm": train_cm_mat,
                        "class_names": class_names,
                        "device": str(device),
                        "label_info": label_info,
                        "train_eval_acc": train_eval_acc,
                        "train_eval_loss": train_eval_loss,
                    }
                )

        self.finished.emit()


class DataProcessingPage(QtWidgets.QWidget):
    """数据处理流程界面。"""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._data: Optional[np.ndarray] = None  # C×T
        self._sample_rate: float = 1000.0
        self._segments: Optional[np.ndarray] = None  # N×C×L
        self._segment_labels: Optional[np.ndarray] = None
        self._features: Optional[np.ndarray] = None  # N×F
        self._feature_labels: Optional[np.ndarray] = None
        self._model: Optional[DummyModel] = None
        self._train_split: float = 0.8
        # EMGNet 训练相关
        self._emg_worker: Optional[QtCore.QThread] = None
        self._emg_train_losses: List[float] = []
        self._emg_test_losses: List[float] = []
        self._emg_class_names: List[str] = []
        self._init_ui()

    # region UI init
    def _init_ui(self) -> None:
        main_layout = QtWidgets.QHBoxLayout(self)

        # 左侧参数区
        controls_panel = QtWidgets.QWidget()
        controls_panel.setMaximumWidth(380)
        controls_layout = QtWidgets.QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(10)
        controls_layout.addWidget(self._build_data_section())
        controls_layout.addWidget(self._build_preprocess_section())
        controls_layout.addWidget(self._build_feature_section())
        controls_layout.addWidget(self._build_algorithm_section())
        self._config_algo_btn = QtWidgets.QPushButton("配置算法...")
        self._config_algo_btn.clicked.connect(self._on_config_algorithm)
        controls_layout.addWidget(self._config_algo_btn)
        # 通用“开始训练”按钮，始终可见（内部会校验当前算法）
        self._start_train_btn = QtWidgets.QPushButton("开始训练")
        self._start_train_btn.clicked.connect(self._on_start_emg_training)
        controls_layout.addWidget(self._start_train_btn)
        # 隐藏的 EMGNet 参数组，仅用于持有参数，不在左侧显示
        self._emgnet_hidden_group = self._build_emgnet_section()
        self._emgnet_hidden_group.hide()
        controls_layout.addStretch()
        main_layout.addWidget(controls_panel, stretch=0)

        # 右侧显示区（状态 + 图表 + 控制台）
        display_panel = QtWidgets.QWidget()
        display_layout = QtWidgets.QVBoxLayout(display_panel)
        display_layout.setContentsMargins(8, 8, 8, 8)
        display_layout.setSpacing(8)

        self._status_label = QtWidgets.QLabel("数据未加载")
        display_layout.addWidget(self._status_label)

        # 实时可视化（train loss + 混淆矩阵 + per-class accuracy）
        self._emgnet_fig = Figure(figsize=(11.5, 4.2), tight_layout=True)
        gs = self._emgnet_fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1])
        self._emgnet_ax_loss = self._emgnet_fig.add_subplot(gs[0, 0])
        self._emgnet_ax_cm = self._emgnet_fig.add_subplot(gs[0, 1])
        self._emgnet_ax_bar = self._emgnet_fig.add_subplot(gs[0, 2])
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # 延迟导入以加快启动
        self._emgnet_canvas = FigureCanvas(self._emgnet_fig)
        # 兼容旧变量，避免属性错误
        self._emgnet_cm_cbar = None
        self._emgnet_cm_im = None
        self._emgnet_cm_texts: List = []
        display_layout.addWidget(self._emgnet_canvas, stretch=2)
        self._reset_emgnet_plots()

        self._log_view = QtWidgets.QTextEdit()
        self._log_view.setReadOnly(True)
        display_layout.addWidget(self._log_view, stretch=1)

        main_layout.addWidget(display_panel, stretch=1)

    def _build_data_section(self) -> QtWidgets.QGroupBox:
        """数据加载（四个文件）。"""
        group = QtWidgets.QGroupBox("数据加载（4 文件）")
        layout = QtWidgets.QFormLayout(group)

        self._x_train_edit = QtWidgets.QLineEdit()
        self._y_train_edit = QtWidgets.QLineEdit()
        self._x_test_edit = QtWidgets.QLineEdit()
        self._y_test_edit = QtWidgets.QLineEdit()
        for edit in [self._x_train_edit, self._y_train_edit, self._x_test_edit, self._y_test_edit]:
            edit.setReadOnly(True)

        def add_row(label: str, edit: QtWidgets.QLineEdit):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(edit)
            btn = QtWidgets.QPushButton("选择")
            btn.clicked.connect(lambda: self._on_browse_emg_file(edit))
            row.addWidget(btn)
            layout.addRow(label, row)

        add_row("x_train.npy", self._x_train_edit)
        add_row("y_train.npy", self._y_train_edit)
        add_row("x_test.npy", self._x_test_edit)
        add_row("y_test.npy", self._y_test_edit)

        return group

    def _build_preprocess_section(self) -> QtWidgets.QGroupBox:
        """预处理参数（勾选即应用）。"""
        group = QtWidgets.QGroupBox("预处理参数")
        layout = QtWidgets.QVBoxLayout(group)
        self._preprocess_filter_combo = QtWidgets.QComboBox()
        self._preprocess_filter_combo.addItems(["无", "高通", "低通", "带通"])
        layout.addWidget(self._preprocess_filter_combo)

        freq_form = QtWidgets.QFormLayout()
        self._preprocess_f_low = QtWidgets.QDoubleSpinBox()
        self._preprocess_f_low.setRange(0.1, 1000.0)
        self._preprocess_f_low.setValue(20.0)
        self._preprocess_f_low.setSuffix(" Hz")
        freq_form.addRow("F_low", self._preprocess_f_low)

        self._preprocess_f_high = QtWidgets.QDoubleSpinBox()
        self._preprocess_f_high.setRange(0.1, 2000.0)
        self._preprocess_f_high.setValue(150.0)
        self._preprocess_f_high.setSuffix(" Hz")
        freq_form.addRow("F_high", self._preprocess_f_high)

        layout.addLayout(freq_form)
        self._preprocess_norm_cb = QtWidgets.QCheckBox("标准化 (z-score)")
        self._preprocess_norm_cb.setChecked(True)
        layout.addWidget(self._preprocess_norm_cb)
        layout.addStretch()
        return group

    def _build_feature_section(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("机器学习特征")
        layout = QtWidgets.QVBoxLayout(group)

        self._feature_mean_cb = QtWidgets.QCheckBox("通道均值")
        self._feature_mean_cb.setChecked(True)
        self._feature_std_cb = QtWidgets.QCheckBox("通道标准差")
        self._feature_std_cb.setChecked(True)
        self._feature_energy_cb = QtWidgets.QCheckBox("能量")
        self._feature_energy_cb.setChecked(False)

        layout.addWidget(self._feature_mean_cb)
        layout.addWidget(self._feature_std_cb)
        layout.addWidget(self._feature_energy_cb)
        layout.addStretch()
        return group

    def _build_algorithm_section(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("模型选择")
        layout = QtWidgets.QFormLayout(group)

        self._algo_combo = QtWidgets.QComboBox()
        self._algo_combo.addItems(["EMGNet (深度学习)", "最近均值", "SVM", "随机森林"])
        self._algo_combo.currentTextChanged.connect(self._on_algo_changed)
        layout.addRow("算法", self._algo_combo)

        self._label_shift_cb = QtWidgets.QCheckBox("标签最小值归零（防止缺少0类）")
        self._label_shift_cb.setChecked(True)
        layout.addRow(self._label_shift_cb)
        return group

    def _on_algo_changed(self, text: str) -> None:
        """切换算法时，仅记录当前算法（参数在对话框中设置）。"""

    def _on_config_algorithm(self) -> None:
        """弹出算法配置对话框（参考设备添加的交互样式）。"""
        dlg = AlgorithmConfigDialog(
            algo=self._algo_combo.currentText(),
            epochs=self._emg_epochs_spin.value(),
            batch=self._emg_batch_spin.value(),
            lr=self._emg_lr_spin.value(),
            workers=self._emg_workers_spin.value(),
            time_point=self._emg_time_point_spin.value(),
            N_t=self._emg_nt_spin.value(),
            N_s=self._emg_ns_spin.value(),
            dropout=self._emg_dropout_spin.value(),
            parent=self,
        )
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            cfg = dlg.get_config()
            self._algo_combo.setCurrentText(cfg["algo"])
            self._emg_epochs_spin.setValue(cfg["epochs"])
            self._emg_batch_spin.setValue(cfg["batch"])
            self._emg_lr_spin.setValue(cfg["lr"])
            self._emg_workers_spin.setValue(cfg["workers"])
            self._emg_time_point_spin.setValue(cfg["time_point"])
            self._emg_nt_spin.setValue(cfg["N_t"])
            self._emg_ns_spin.setValue(cfg["N_s"])
            self._emg_dropout_spin.setValue(cfg["dropout"])

    def _build_emgnet_section(self) -> QtWidgets.QGroupBox:
        """深度学习参数（EMGNet）。"""
        group = QtWidgets.QGroupBox("深度学习参数 (EMGNet)")
        group.setVisible(False)  # 默认仅在选中 EMGNet 时显示
        self._emgnet_params_group = group
        layout = QtWidgets.QFormLayout(group)

        self._emg_epochs_spin = QtWidgets.QSpinBox()
        self._emg_epochs_spin.setRange(1, 500)
        self._emg_epochs_spin.setValue(20)
        layout.addRow("Epochs", self._emg_epochs_spin)

        self._emg_batch_spin = QtWidgets.QSpinBox()
        self._emg_batch_spin.setRange(1, 1024)
        self._emg_batch_spin.setValue(64)
        layout.addRow("Batch size", self._emg_batch_spin)

        self._emg_lr_spin = QtWidgets.QDoubleSpinBox()
        self._emg_lr_spin.setDecimals(6)
        self._emg_lr_spin.setRange(1e-6, 1.0)
        self._emg_lr_spin.setSingleStep(1e-4)
        self._emg_lr_spin.setValue(1e-3)
        layout.addRow("Learning rate", self._emg_lr_spin)

        self._emg_workers_spin = QtWidgets.QSpinBox()
        self._emg_workers_spin.setRange(0, 8)
        self._emg_workers_spin.setValue(0)  # Windows 建议 0
        layout.addRow("DataLoader 线程", self._emg_workers_spin)

        self._emg_time_point_spin = QtWidgets.QSpinBox()
        self._emg_time_point_spin.setRange(3, 99)
        self._emg_time_point_spin.setSingleStep(2)
        self._emg_time_point_spin.setValue(15)
        layout.addRow("time_point(奇数)", self._emg_time_point_spin)

        self._emg_nt_spin = QtWidgets.QSpinBox()
        self._emg_nt_spin.setRange(1, 128)
        self._emg_nt_spin.setValue(8)
        layout.addRow("N_t", self._emg_nt_spin)

        self._emg_ns_spin = QtWidgets.QSpinBox()
        self._emg_ns_spin.setRange(1, 256)
        self._emg_ns_spin.setValue(16)
        layout.addRow("N_s(可被N_t整除)", self._emg_ns_spin)

        self._emg_dropout_spin = QtWidgets.QDoubleSpinBox()
        self._emg_dropout_spin.setRange(0.0, 0.9)
        self._emg_dropout_spin.setSingleStep(0.05)
        self._emg_dropout_spin.setValue(0.5)
        layout.addRow("Dropout", self._emg_dropout_spin)

        self._emg_save_edit = QtWidgets.QLineEdit("best_emgnet.pt")
        layout.addRow("保存路径", self._emg_save_edit)

        self._emg_start_btn = QtWidgets.QPushButton("开始训练")
        self._emg_start_btn.clicked.connect(self._on_start_emg_training)
        layout.addRow(self._emg_start_btn)

        # 初始化可见性（根据当前算法）
        self._emgnet_params_group.setVisible(self._algo_combo.currentText().startswith("EMGNet"))
        return group

    # endregion

    # region EMGNet helpers
    def _reset_emgnet_plots(self) -> None:
        # 重置状态
        self._emgnet_cm_cbar = None
        self._emgnet_cm_im = None
        self._emgnet_cm_texts = []
        self._emgnet_ax_loss.cla()
        self._emgnet_ax_loss.set_title("Loss")
        self._emgnet_ax_loss.set_xlabel("Epoch")
        self._emgnet_ax_loss.set_ylabel("Loss")
        self._emgnet_ax_loss.grid(True, ls="--", alpha=0.5)
        self._emgnet_ax_cm.cla()
        self._emgnet_ax_cm.set_title("Confusion Matrix (up to 5 classes)")
        self._emgnet_ax_cm.set_xticks([])
        self._emgnet_ax_cm.set_yticks([])
        self._emgnet_ax_bar.cla()
        self._emgnet_ax_bar.set_title("Per-class Test Accuracy")

        self._emgnet_canvas.draw_idle()

    def _on_browse_emg_file(self, target_edit: QtWidgets.QLineEdit) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择 .npy 文件", "", "Numpy 数据 (*.npy)")
        if path:
            target_edit.setText(path)
            # 读取形状并输出到控制台/日志
            try:
                arr = np.load(path, mmap_mode="r")
                shape_info = arr.shape
                self._append_log(f"加载 {Path(path).name}，形状={shape_info}")
            except Exception as exc:
                self._append_log(f"加载 {Path(path).name} 失败：{exc}")

    def _on_start_emg_training(self) -> None:
        if self._emg_worker is not None and self._emg_worker.isRunning():
            QtWidgets.QMessageBox.information(self, "提示", "训练已在进行中")
            return

        if not self._algo_combo.currentText().startswith("EMGNet"):
            QtWidgets.QMessageBox.information(self, "提示", "当前仅支持 EMGNet 训练，传统算法后续补充。")
            return

        paths = {
            "x_train": self._x_train_edit.text(),
            "y_train": self._y_train_edit.text(),
            "x_test": self._x_test_edit.text(),
            "y_test": self._y_test_edit.text(),
        }
        missing = [k for k, v in paths.items() if not v]
        if missing:
            QtWidgets.QMessageBox.warning(self, "缺少文件", "请完整选择四个文件：x_train, y_train, x_test, y_test")
            return

        params = {
            "epochs": self._emg_epochs_spin.value(),
            "batch_size": self._emg_batch_spin.value(),
            "lr": self._emg_lr_spin.value(),
            "num_workers": self._emg_workers_spin.value(),
            "save_path": self._emg_save_edit.text() or "best_emgnet.pt",
            "time_point": self._emg_time_point_spin.value(),
            "N_t": self._emg_nt_spin.value(),
            "N_s": self._emg_ns_spin.value(),
            "dropout": self._emg_dropout_spin.value(),
            "filter_type": self._preprocess_filter_combo.currentText(),
            "f_low": self._preprocess_f_low.value(),
            "f_high": self._preprocess_f_high.value(),
            "do_zscore": self._preprocess_norm_cb.isChecked(),
            "shift_labels": self._label_shift_cb.isChecked(),
        }
        if params["time_point"] % 2 == 0:
            QtWidgets.QMessageBox.warning(self, "参数错误", "time_point 需为奇数")
            return
        if params["N_s"] % params["N_t"] != 0:
            QtWidgets.QMessageBox.warning(self, "参数错误", "N_s 必须能被 N_t 整除")
            return

        self._emg_train_losses.clear()
        self._emg_test_losses.clear()
        self._emg_class_names = []
        self._reset_emgnet_plots()
        self._status_label.setText("EMGNet 训练中...")
        self._append_log(f"开始 EMGNet 训练，算法={self._algo_combo.currentText()}")

        self._emg_worker = EMGNetTrainWorker(paths, params)
        self._emg_worker.progress.connect(self._on_emg_progress)
        self._emg_worker.error.connect(self._on_emg_error)
        self._emg_worker.finished.connect(self._on_emg_finished)
        self._emg_start_btn.setEnabled(False)
        self._emg_worker.start()

    def _on_emg_progress(self, info: dict) -> None:
        # 更新曲线
        self._emg_train_losses = info.get("train_losses", self._emg_train_losses)
        class_names = info.get("class_names", self._emg_class_names)
        if class_names:
            self._emg_class_names = class_names
        per_class_acc = info.get("per_class_acc")
        cm_mat = info.get("cm")
        label_info = info.get("label_info")

        # loss 图（仅 train，线加粗）
        self._emgnet_ax_loss.cla()
        self._emgnet_ax_loss.set_title("Loss")
        self._emgnet_ax_loss.set_xlabel("Epoch")
        self._emgnet_ax_loss.set_ylabel("Loss")
        self._emgnet_ax_loss.grid(True, ls="--", alpha=0.5)
        epochs = np.arange(1, len(self._emg_train_losses) + 1)
        if len(self._emg_train_losses) > 0:
            self._emgnet_ax_loss.plot(
                epochs,
                self._emg_train_losses,
                label="train loss",
                color="tab:blue",
                linewidth=2.5,
            )
            self._emgnet_ax_loss.legend()

        # 混淆矩阵（最多展示前 5 类）
        self._emgnet_ax_cm.cla()
        self._emgnet_ax_cm.set_title("Confusion Matrix (up to 5 classes)")
        if cm_mat is not None and len(self._emg_class_names) > 0:
            show_len = min(5, len(self._emg_class_names))
            cm_show = cm_mat[:show_len, :show_len]
            class_names_show = self._emg_class_names[:show_len]
            im = self._emgnet_ax_cm.imshow(cm_show, cmap="Blues")
            self._emgnet_ax_cm.set_xticks(range(len(class_names_show)))
            self._emgnet_ax_cm.set_yticks(range(len(class_names_show)))
            self._emgnet_ax_cm.set_xticklabels(class_names_show)
            self._emgnet_ax_cm.set_yticklabels(class_names_show)
            # 文本标注
            for i in range(cm_show.shape[0]):
                for j in range(cm_show.shape[1]):
                    self._emgnet_ax_cm.text(j, i, str(int(cm_show[i, j])), ha="center", va="center", color="black")
            if self._emgnet_cm_cbar is None:
                self._emgnet_cm_cbar = self._emgnet_fig.colorbar(im, ax=self._emgnet_ax_cm, fraction=0.046, pad=0.04)
            else:
                try:
                    self._emgnet_cm_cbar.update_normal(im)
                except Exception:
                    pass
        else:
            self._emgnet_ax_cm.set_xticks([])
            self._emgnet_ax_cm.set_yticks([])

        # 每类测试准确率柱状图
        self._emgnet_ax_bar.cla()
        self._emgnet_ax_bar.set_title("Per-class Test Accuracy")
        if per_class_acc is not None and len(self._emg_class_names) > 0:
            show_len = min(5, len(per_class_acc))
            x_pos = np.arange(show_len)
            acc_vals = np.asarray(per_class_acc)[:show_len]
            class_names_show = self._emg_class_names[:show_len]
            self._emgnet_ax_bar.bar(x_pos, acc_vals, color="tab:green", alpha=0.8)
            self._emgnet_ax_bar.set_xticks(x_pos)
            self._emgnet_ax_bar.set_xticklabels(class_names_show, rotation=0)
            self._emgnet_ax_bar.set_ylim(0.0, 1.0)
            for i, v in enumerate(acc_vals):
                self._emgnet_ax_bar.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        else:
            self._emgnet_ax_bar.text(0.5, 0.5, "无测试数据", ha="center", va="center", transform=self._emgnet_ax_bar.transAxes)

        self._emgnet_canvas.draw_idle()

        # 状态日志
        epoch = info.get("epoch")
        if epoch is not None:
            msg = (
                f"[Epoch {epoch}] train_loss={info.get('train_loss', 0):.4f}, "
                f"test_loss={info.get('test_loss', 0):.4f}, "
                f"test_acc={info.get('test_acc', 0):.4f}"
            )
            self._status_label.setText(msg)
            self._append_log(msg)
            if label_info:
                self._append_log(
                    f"标签分布: train_unique={label_info.get('train_unique')}, "
                    f"test_unique={label_info.get('test_unique')}, "
                    f"train_counts={label_info.get('train_counts')}, "
                    f"test_counts={label_info.get('test_counts')}, "
                    f"x_train_shape={label_info.get('x_train_shape')}, "
                    f"x_test_shape={label_info.get('x_test_shape')}, "
                    f"y_train_shape={label_info.get('y_train_shape')}, "
                    f"y_test_shape={label_info.get('y_test_shape')}"
                )

    def _on_emg_error(self, message: str) -> None:
        self._status_label.setText("训练失败")
        self._emg_start_btn.setEnabled(True)
        self._emg_worker = None
        QtWidgets.QMessageBox.critical(self, "EMGNet 训练失败", message)
        self._append_log(f"训练失败：{message}")

    def _on_emg_finished(self) -> None:
        self._status_label.setText("EMGNet 训练完成")
        self._emg_start_btn.setEnabled(True)
        self._emg_worker = None
        self._append_log("训练完成")
    # endregion
    # region callbacks
    def _on_browse_data(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择数据文件", "", "Numpy 数据 (*.npy)"
        )
        if not file_path:
            return
        path = Path(file_path)
        try:
            data = np.load(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "加载失败", str(exc))
            return

        if data.ndim != 2:
            QtWidgets.QMessageBox.warning(self, "格式错误", "数据必须是二维数组 (C×T)")
            return

        self._data = data.astype(np.float32, copy=True)
        self._sample_rate = self._sample_rate_spin.value()
        self._segments = None
        self._segment_labels = None
        self._features = None
        self._feature_labels = None
        self._model = None
        self._data_path_edit.setText(str(path))
        self._summary_label.setText(f"通道: {data.shape[0]} | 样本: {data.shape[1]}")
        self._status_label.setText("数据加载完成")
        self._append_log(f"加载数据：{path.name}，形状={data.shape}")

    def _on_run_segmentation(self) -> None:
        if self._data is None:
            QtWidgets.QMessageBox.warning(self, "提示", "请先加载数据")
            return

        window_seconds = self._window_length_spin.value()
        overlap = self._overlap_spin.value()
        sample_rate = self._sample_rate_spin.value()
        window_samples = int(window_seconds * sample_rate)
        if window_samples <= 0:
            QtWidgets.QMessageBox.warning(self, "参数错误", "窗口长度过小")
            return

        step = int(window_samples * (1 - overlap))
        if step <= 0:
            QtWidgets.QMessageBox.warning(self, "参数错误", "重叠比例过大")
            return

        segments, labels = self._segment_data(self._data, window_samples, step)
        if segments.size == 0:
            QtWidgets.QMessageBox.warning(self, "失败", "无法生成任何分段，请调整参数")
            return

        self._segments = segments
        self._segment_labels = labels
        self._features = None
        self._feature_labels = None
        self._model = None
        self._status_label.setText(f"分段完成：{segments.shape[0]} 段")
        self._append_log(f"分段完成：segments={segments.shape}")

    def _on_run_feature_extraction(self) -> None:
        if self._segments is None:
            QtWidgets.QMessageBox.warning(self, "提示", "请先执行数据分段")
            return
        selected = {
            "mean": self._feature_mean_cb.isChecked(),
            "std": self._feature_std_cb.isChecked(),
            "energy": self._feature_energy_cb.isChecked(),
        }
        if not any(selected.values()):
            QtWidgets.QMessageBox.warning(self, "提示", "请至少选择一种特征")
            return
        features = self._extract_features(self._segments, selected)
        self._features = features
        self._feature_labels = self._segment_labels.copy()
        self._model = None
        self._status_label.setText(f"特征提取完成：形状 {features.shape}")
        self._append_log(f"特征矩阵 Shape={features.shape}")

    def _on_run_training(self) -> None:
        if self._features is None or self._feature_labels is None:
            QtWidgets.QMessageBox.warning(self, "提示", "请先提取特征")
            return
        split_ratio = self._split_spin.value()
        train_data, train_labels, test_data, test_labels = self._train_test_split(
            self._features, self._feature_labels, split_ratio
        )
        if train_data.size == 0 or test_data.size == 0:
            QtWidgets.QMessageBox.warning(self, "失败", "训练/测试样本不足")
            return

        model_name = self._model_combo.currentText()
        params = self._collect_model_params()
        self._append_log(f"训练 {model_name}，超参数: {params}")
        if model_name == "最近均值":
            model = self._train_dummy_model(train_data, train_labels)
        else:
            model = self._train_placeholder_model(model_name, train_data, train_labels, params)
        self._model = model
        self._cached_eval_data = (test_data, test_labels)
        self._status_label.setText(
            f"模型训练完成（训练 {train_data.shape[0]}，测试 {test_data.shape[0]}）"
        )
        self._append_log("模型训练完成：使用最近均值分类器实现占位")

    def _on_run_evaluation(self) -> None:
        if not self._model:
            QtWidgets.QMessageBox.warning(self, "提示", "请先训练模型")
            return
        if not hasattr(self, "_cached_eval_data"):
            QtWidgets.QMessageBox.warning(self, "提示", "未找到评估数据，请重新训练")
            return
        test_data, test_labels = getattr(self, "_cached_eval_data")
        preds = self._model.predict(test_data)
        metrics = self._compute_metrics(test_labels, preds)
        self._update_metrics_table(metrics)
        self._status_label.setText("评估完成")
        self._append_log(f"性能：{metrics}")

    # endregion

    # region processing helpers
    @staticmethod
    def _segment_data(
        data: np.ndarray,
        window_samples: int,
        step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        channels, total = data.shape
        segments: List[np.ndarray] = []
        labels: List[int] = []
        start = 0
        label = 0
        while start + window_samples <= total:
            window = data[:, start : start + window_samples]
            segments.append(window)
            labels.append(label)
            label = 1 - label
            start += step
        if not segments:
            return np.empty((0, channels, window_samples), dtype=np.float32), np.empty((0,), dtype=int)
        seg_arr = np.stack(segments).astype(np.float32)
        label_arr = np.asarray(labels, dtype=int)
        return seg_arr, label_arr

    @staticmethod
    def _extract_features(
        segments: np.ndarray,
        options: Dict[str, bool],
    ) -> np.ndarray:
        num_segments, channels, samples = segments.shape
        feats: List[np.ndarray] = []
        if options.get("mean"):
            feats.append(segments.mean(axis=2))
        if options.get("std"):
            feats.append(segments.std(axis=2))
        if options.get("energy"):
            feats.append(np.sqrt((segments ** 2).mean(axis=2)))
        if not feats:
            return np.empty((num_segments, 0), dtype=np.float32)
        feature_matrix = np.concatenate(feats, axis=1)
        return feature_matrix.astype(np.float32)

    @staticmethod
    def _train_test_split(
        features: np.ndarray,
        labels: np.ndarray,
        ratio: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        count = features.shape[0]
        split_idx = max(1, min(count - 1, int(count * ratio)))
        indices = np.arange(count)
        np.random.shuffle(indices)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        return (
            features[train_idx],
            labels[train_idx],
            features[test_idx],
            labels[test_idx],
        )

    @staticmethod
    def _train_dummy_model(features: np.ndarray, labels: np.ndarray) -> DummyModel:
        class_means: Dict[int, np.ndarray] = {}
        for label in np.unique(labels):
            class_feats = features[labels == label]
            class_means[int(label)] = class_feats.mean(axis=0)
        return DummyModel(class_means=class_means)

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0
        precision = DataProcessingPage._safe_metric(y_true, y_pred, positive=1, metric="precision")
        recall = DataProcessingPage._safe_metric(y_true, y_pred, positive=1, metric="recall")
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
        }

    @staticmethod
    def _safe_metric(y_true: np.ndarray, y_pred: np.ndarray, positive: int, metric: str) -> float:
        tp = np.sum((y_true == positive) & (y_pred == positive))
        fp = np.sum((y_true != positive) & (y_pred == positive))
        fn = np.sum((y_true == positive) & (y_pred != positive))
        if metric == "precision":
            denom = tp + fp
            return float(tp / denom) if denom > 0 else 0.0
        if metric == "recall":
            denom = tp + fn
            return float(tp / denom) if denom > 0 else 0.0
        return 0.0

    # endregion

    # region UI helpers
    def _update_metrics_table(self, metrics: Dict[str, float]) -> None:
        self._result_table.setRowCount(len(metrics))
        for row, (name, value) in enumerate(metrics.items()):
            self._result_table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self._result_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{value:.4f}"))

    def _append_log(self, message: str) -> None:
        self._log_view.append(f"[{QtCore.QDateTime.currentDateTime().toString('hh:mm:ss')}] {message}")

    # endregion

    # region model config helpers
    def _build_model_configs(self) -> Dict[str, List[Dict]]:
        return {
            "最近均值": [],
            "轻量 SVM": [
                {"name": "C", "label": "C 值", "type": "float", "min": 0.01, "max": 100.0, "step": 0.1, "default": 1.0},
                {"name": "gamma", "label": "Gamma", "type": "float", "min": 0.0001, "max": 10.0, "step": 0.0001, "default": 0.01},
            ],
            "随机森林": [
                {"name": "estimators", "label": "树数量", "type": "int", "min": 10, "max": 500, "step": 10, "default": 100},
                {"name": "max_depth", "label": "最大深度", "type": "int", "min": 1, "max": 50, "step": 1, "default": 10},
            ],
            "EEGNet": [
                {"name": "epochs", "label": "Epochs", "type": "int", "min": 5, "max": 200, "step": 5, "default": 50},
                {"name": "learning_rate", "label": "学习率", "type": "float", "min": 0.0001, "max": 0.01, "step": 0.0001, "default": 0.001},
                {"name": "dropout", "label": "Dropout", "type": "float", "min": 0.0, "max": 0.8, "step": 0.05, "default": 0.25},
            ],
            "DeepConvNet": [
                {"name": "epochs", "label": "Epochs", "type": "int", "min": 5, "max": 200, "step": 5, "default": 60},
                {"name": "learning_rate", "label": "学习率", "type": "float", "min": 0.0001, "max": 0.01, "step": 0.0001, "default": 0.0005},
                {"name": "filters", "label": "卷积核数量", "type": "int", "min": 4, "max": 128, "step": 4, "default": 32},
            ],
        }

    def _on_model_changed(self, model_name: str) -> None:
        # 清空旧的表单
        while self._param_form.count():
            item = self._param_form.takeAt(0)
            if widget := item.widget():
                widget.deleteLater()
        self._param_inputs.clear()

        config = self._model_configs.get(model_name, [])
        if not config:
            notice = QtWidgets.QLabel("该模型无需额外参数")
            self._param_form.addRow(notice)
            return

        for field in config:
            if field["type"] == "float":
                widget = QtWidgets.QDoubleSpinBox()
                widget.setRange(field["min"], field["max"])
                widget.setSingleStep(field["step"])
                widget.setValue(field["default"])
            else:
                widget = QtWidgets.QSpinBox()
                widget.setRange(int(field["min"]), int(field["max"]))
                widget.setSingleStep(int(field["step"]))
                widget.setValue(int(field["default"]))
            self._param_inputs[field["name"]] = widget
            self._param_form.addRow(field["label"], widget)

    def _collect_model_params(self) -> Dict[str, float]:
        params: Dict[str, float] = {}
        for name, widget in self._param_inputs.items():
            if isinstance(widget, QtWidgets.QDoubleSpinBox) or isinstance(widget, QtWidgets.QSpinBox):
                params[name] = float(widget.value())
        return params

    def _train_placeholder_model(
        self,
        model_name: str,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        params: Dict[str, float],
    ) -> DummyModel:
        # 真实项目应在此接入深度学习训练流程，这里做占位实现
        self._append_log(f"[模拟] {model_name} 训练耗时约 {int(params.get('epochs', 50))} epochs")
        # 仍然使用最近均值模型作为占位，使后续评估逻辑可复用
        return self._train_dummy_model(train_data, train_labels)

    # endregion


