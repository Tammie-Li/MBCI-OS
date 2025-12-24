"""
离线数据转换页面：支持导入各种格式并可视化。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui
from scipy.signal import butter, filtfilt

from ...core.data_pipeline.converters.base import BaseConverter
from ...core.data_pipeline.converters.emg_dat import EmgDatConverter
from ...core.data_pipeline.converters.edf import EdfConverter
from ...core.data_pipeline.protocol import MetaInfo

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")


class ChannelControlDialog(QtWidgets.QDialog):
    """通道缩放和平移控制弹窗。"""

    def __init__(
        self,
        channel_count: int,
        channel_gains: Dict[int, float],
        channel_offsets: Dict[int, float],
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("通道缩放/平移控制")
        self.resize(400, 500)
        self._channel_count = channel_count
        self._channel_gains = channel_gains.copy()
        self._channel_offsets = channel_offsets.copy()
        self._gain_spins: Dict[int, QtWidgets.QDoubleSpinBox] = {}
        self._offset_spins: Dict[int, QtWidgets.QDoubleSpinBox] = {}

        layout = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        controls_widget = QtWidgets.QWidget()
        self._controls_layout = QtWidgets.QVBoxLayout(controls_widget)

        for i in range(channel_count):
            if i not in self._channel_gains:
                self._channel_gains[i] = 1.0
            if i not in self._channel_offsets:
                self._channel_offsets[i] = 0.0

            group = QtWidgets.QGroupBox(f"通道 {i+1}")
            group_layout = QtWidgets.QHBoxLayout()

            group_layout.addWidget(QtWidgets.QLabel("缩放:"))
            gain_spin = QtWidgets.QDoubleSpinBox()
            gain_spin.setRange(0.01, 100.0)
            gain_spin.setSingleStep(0.1)
            gain_spin.setValue(self._channel_gains[i])
            gain_spin.valueChanged.connect(lambda v, ch=i: self._on_gain_changed(ch, v))
            self._gain_spins[i] = gain_spin
            group_layout.addWidget(gain_spin)

            group_layout.addWidget(QtWidgets.QLabel("平移:"))
            offset_spin = QtWidgets.QDoubleSpinBox()
            offset_spin.setRange(-10000.0, 10000.0)
            offset_spin.setSingleStep(10.0)
            offset_spin.setValue(self._channel_offsets[i])
            offset_spin.valueChanged.connect(lambda v, ch=i: self._on_offset_changed(ch, v))
            self._offset_spins[i] = offset_spin
            group_layout.addWidget(offset_spin)

            group.setLayout(group_layout)
            self._controls_layout.addWidget(group)

        self._controls_layout.addStretch()
        scroll.setWidget(controls_widget)
        layout.addWidget(scroll)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_gain_changed(self, channel: int, value: float) -> None:
        self._channel_gains[channel] = value

    def _on_offset_changed(self, channel: int, value: float) -> None:
        self._channel_offsets[channel] = value

    def get_gains(self) -> Dict[int, float]:
        return self._channel_gains.copy()

    def get_offsets(self) -> Dict[int, float]:
        return self._channel_offsets.copy()


class OfflineConverterPage(QtWidgets.QWidget):
    """离线数据转换页面。"""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._converters: Dict[str, BaseConverter] = {
            "emg_dat": EmgDatConverter(),
            "edf": EdfConverter(),
        }
        self._selected_paths: List[Path] = []
        self._current_data_raw: Optional[np.ndarray] = None
        self._current_data_processed: Optional[np.ndarray] = None
        self._channel_names: List[str] = []
        self._current_meta: Optional[MetaInfo] = None
        self._channel_gains: Dict[int, float] = {}
        self._channel_offsets: Dict[int, float] = {}
        self._marker_times: List[float] = []
        self._marker_lines: List[pg.InfiniteLine] = []
        self._marker_label_items: List[pg.TextItem] = []
        self._channel_label_items: List[pg.TextItem] = []
        self._selected_marker_index: Optional[int] = None
        self._channel_name_view: Optional[QtWidgets.QListWidget] = None
        self._marker_list_view: Optional[QtWidgets.QListWidget] = None
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QtWidgets.QHBoxLayout(self)

        # 左侧：文件选择和转换控制（紧凑布局）
        left_panel = QtWidgets.QWidget()
        left_panel.setMaximumWidth(300)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)

        # 文件选择（紧凑）
        file_group = QtWidgets.QGroupBox("文件导入")
        file_layout = QtWidgets.QVBoxLayout()
        self._file_path_edit = QtWidgets.QLineEdit()
        self._file_path_edit.setReadOnly(True)
        file_layout.addWidget(self._file_path_edit)
        browse_btn = QtWidgets.QPushButton("浏览（可多选）...")
        browse_btn.clicked.connect(self._on_browse_clicked)
        file_layout.addWidget(browse_btn)
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        # 转换选项移除，直接预览
        self._output_dir_edit = QtWidgets.QLineEdit()

        # 数据信息（紧凑）
        info_group = QtWidgets.QGroupBox("数据信息")
        self._info_text = QtWidgets.QTextEdit()
        self._info_text.setReadOnly(True)
        self._info_text.setMaximumHeight(100)
        info_group_layout = QtWidgets.QVBoxLayout()
        info_group_layout.addWidget(self._info_text)
        info_group.setLayout(info_group_layout)
        left_layout.addWidget(info_group)

        # 预处理设置
        preprocess_group = QtWidgets.QGroupBox("预处理")
        preprocess_layout = QtWidgets.QVBoxLayout()
        self._filter_combo = QtWidgets.QComboBox()
        self._filter_combo.addItem("无滤波", "none")
        self._filter_combo.addItem("高通滤波", "highpass")
        self._filter_combo.addItem("带通滤波", "bandpass")
        preprocess_layout.addWidget(QtWidgets.QLabel("滤波:"))
        preprocess_layout.addWidget(self._filter_combo)

        freq_form = QtWidgets.QFormLayout()
        self._low_cut_spin = QtWidgets.QDoubleSpinBox()
        self._low_cut_spin.setRange(0.1, 1000.0)
        self._low_cut_spin.setSingleStep(0.5)
        self._low_cut_spin.setValue(1.0)
        freq_form.addRow("低截止(Hz)", self._low_cut_spin)

        self._high_cut_spin = QtWidgets.QDoubleSpinBox()
        self._high_cut_spin.setRange(0.1, 2000.0)
        self._high_cut_spin.setSingleStep(0.5)
        self._high_cut_spin.setValue(40.0)
        freq_form.addRow("高截止(Hz)", self._high_cut_spin)
        preprocess_layout.addLayout(freq_form)

        self._norm_combo = QtWidgets.QComboBox()
        self._norm_combo.addItem("无归一化", "none")
        self._norm_combo.addItem("Z-Score", "zscore")
        self._norm_combo.addItem("Min-Max [0,1]", "minmax")
        preprocess_layout.addWidget(QtWidgets.QLabel("归一化:"))
        preprocess_layout.addWidget(self._norm_combo)

        apply_pre_btn = QtWidgets.QPushButton("应用预处理")
        apply_pre_btn.clicked.connect(self._on_apply_preprocessing_clicked)
        preprocess_layout.addWidget(apply_pre_btn)
        preprocess_group.setLayout(preprocess_layout)
        left_layout.addWidget(preprocess_group)
        self._preprocess_group = preprocess_group

        # 数据集导出
        export_group = QtWidgets.QGroupBox("数据集导出")
        export_layout = QtWidgets.QFormLayout()
        self._slice_len_spin = QtWidgets.QDoubleSpinBox()
        self._slice_len_spin.setRange(0.1, 600.0)
        self._slice_len_spin.setSingleStep(0.1)
        self._slice_len_spin.setValue(2.0)
        self._slice_step_spin = QtWidgets.QDoubleSpinBox()
        self._slice_step_spin.setRange(0.05, 600.0)
        self._slice_step_spin.setSingleStep(0.05)
        self._slice_step_spin.setValue(1.0)
        export_layout.addRow("切片长度 (s)", self._slice_len_spin)
        export_layout.addRow("切片步长 (s)", self._slice_step_spin)
        export_btn = QtWidgets.QPushButton("生成数据集 (npy)")
        export_btn.clicked.connect(self._on_export_dataset_clicked)
        export_layout.addRow(export_btn)
        export_group.setLayout(export_layout)
        left_layout.addWidget(export_group)

        # 通道与标记信息
        info2_group = QtWidgets.QGroupBox("通道与标记信息")
        info2_layout = QtWidgets.QVBoxLayout()
        info2_layout.addWidget(QtWidgets.QLabel("通道名称:"))
        self._channel_name_view = QtWidgets.QListWidget()
        self._channel_name_view.setMaximumHeight(100)
        info2_layout.addWidget(self._channel_name_view)
        info2_layout.addWidget(QtWidgets.QLabel("标记时间 (s):"))
        self._marker_list_view = QtWidgets.QListWidget()
        self._marker_list_view.setMaximumHeight(80)
        info2_layout.addWidget(self._marker_list_view)
        info2_group.setLayout(info2_layout)
        left_layout.addWidget(info2_group)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # 右侧：可视化区域
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)

        # 控制工具栏
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.addWidget(QtWidgets.QLabel("时间窗起始 (s):"))
        self._start_spin = QtWidgets.QDoubleSpinBox()
        self._start_spin.setRange(0.0, 10000.0)
        self._start_spin.setSingleStep(0.1)
        self._start_spin.setValue(0.0)
        self._start_spin.valueChanged.connect(self._update_plot)
        toolbar.addWidget(self._start_spin)

        toolbar.addWidget(QtWidgets.QLabel("时间窗长度 (s):"))
        self._length_spin = QtWidgets.QDoubleSpinBox()
        self._length_spin.setRange(0.1, 10000.0)
        self._length_spin.setSingleStep(0.1)
        self._length_spin.setValue(5.0)
        self._length_spin.valueChanged.connect(self._update_plot)
        toolbar.addWidget(self._length_spin)

        toolbar.addWidget(QtWidgets.QLabel("Y轴范围:"))
        self._y_min_spin = QtWidgets.QDoubleSpinBox()
        self._y_min_spin.setRange(-100000.0, 100000.0)
        self._y_min_spin.setSingleStep(100.0)
        self._y_min_spin.setValue(-500.0)
        self._y_min_spin.valueChanged.connect(self._update_plot)
        toolbar.addWidget(QtWidgets.QLabel("最小:"))
        toolbar.addWidget(self._y_min_spin)

        self._y_max_spin = QtWidgets.QDoubleSpinBox()
        self._y_max_spin.setRange(-100000.0, 100000.0)
        self._y_max_spin.setSingleStep(100.0)
        self._y_max_spin.setValue(500.0)
        self._y_max_spin.valueChanged.connect(self._update_plot)
        toolbar.addWidget(QtWidgets.QLabel("最大:"))
        toolbar.addWidget(self._y_max_spin)

        toolbar.addStretch()

        toolbar.addWidget(QtWidgets.QLabel("通道选择:"))
        self._channel_list = QtWidgets.QListWidget()
        self._channel_list.setMaximumWidth(120)
        self._channel_list.setMaximumHeight(100)
        self._channel_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._channel_list.itemSelectionChanged.connect(self._update_plot)
        toolbar.addWidget(self._channel_list)

        channel_control_btn = QtWidgets.QPushButton("通道控制")
        channel_control_btn.clicked.connect(self._on_channel_control_clicked)
        toolbar.addWidget(channel_control_btn)

        toolbar.addWidget(QtWidgets.QLabel("标记时间 (s):"))
        self._marker_time_spin = QtWidgets.QDoubleSpinBox()
        self._marker_time_spin.setRange(0.0, 10000.0)
        self._marker_time_spin.setSingleStep(0.1)
        toolbar.addWidget(self._marker_time_spin)

        add_marker_btn = QtWidgets.QPushButton("添加标记")
        add_marker_btn.clicked.connect(self._on_add_marker_clicked)
        toolbar.addWidget(add_marker_btn)

        remove_marker_btn = QtWidgets.QPushButton("删除标记")
        remove_marker_btn.clicked.connect(self._on_remove_marker_clicked)
        toolbar.addWidget(remove_marker_btn)

        right_layout.addLayout(toolbar)

        # 绘图区域
        self._plot = pg.PlotWidget()
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.setLabel("bottom", "时间", units="s")
        self._plot.setLabel("left", "幅值")
        self._plot.setMouseEnabled(x=True, y=False)  # 只允许X轴滑动，Y轴固定
        # 支持鼠标点击添加/选择标记
        self._plot.scene().sigMouseClicked.connect(self._on_plot_clicked)
        right_layout.addWidget(self._plot)

        main_layout.addWidget(right_panel, stretch=1)

    def _on_browse_clicked(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "选择数据文件（可多选，支持数据+meta json）",
            "",
            "数据文件 (*.dat *.edf *.bdf *.vhdr *.tsv *.npy *.json);;所有文件 (*.*)",
        )
        if files:
            self._file_path_edit.setText("; ".join(files))
            paths = [Path(f) for f in files]
            self._try_load_preview_paths(paths)

    def _try_load_preview_paths(self, paths: List[Path]) -> None:
        """尝试加载文件预览信息（支持多文件：数据+meta）。"""
        if not paths:
            return
        self._selected_paths = paths
        # 优先直接加载 .npy（若有）并配对 .json 元信息
        npy_paths = [p for p in paths if p.suffix.lower() == ".npy"]
        if npy_paths:
            try:
                self._load_npy_with_meta(npy_paths[0], paths)
                return
            except Exception as exc:
                self._info_text.setText(f"加载 npy 失败: {exc}")
                return

        path = paths[0]
        ext = path.suffix.lower()

        # 根据扩展名选择转换器
        converter = None
        for conv in self._converters.values():
            if ext in conv.source_extensions:
                converter = conv
                break

        if converter is None:
            self._info_text.setText(f"未找到支持 {ext} 格式的转换器")
            return

        try:
            params = None
            # 对 .dat 提示用户输入采样率和通道数
            if isinstance(converter, EmgDatConverter) and ext == ".dat":
                params = self._prompt_dat_params(path)
                if params is None:
                    return
            # 执行转换到临时目录
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                meta = converter.convert(paths, Path(tmpdir), params=params)  # type: ignore[arg-type]
                tmp_data_file = Path(tmpdir) / "data.npy"
                if tmp_data_file.exists():
                    data = np.load(tmp_data_file)
                    self._current_meta = meta
                    self._update_preprocess_controls_for_modality(meta.modality)
                    info_text = f"""模态: {meta.modality}
通道数: {meta.channels}
采样率: {meta.sample_rate} Hz
时长: {meta.duration_seconds:.2f} s
总样本数: {data.shape[1]}
"""
                    self._info_text.setText(info_text)

                    # 初始化通道列表和参数
                    self._channel_list.clear()
                    channel_count = data.shape[0]
                    channel_names: Optional[List[str]] = None
                    if isinstance(meta.extra, dict):
                        raw_names = meta.extra.get("channel_names")
                        if isinstance(raw_names, (list, tuple)) and len(raw_names) == channel_count:
                            channel_names = [str(n) for n in raw_names]
                    if channel_names is None:
                        channel_names = [f"Ch{i+1}" for i in range(channel_count)]
                    self._channel_names = channel_names

                    for i in range(channel_count):
                        name = self._channel_names[i]
                        item = QtWidgets.QListWidgetItem(f"{name}")
                        item.setData(QtCore.Qt.UserRole, i)
                        item.setSelected(i < min(8, channel_count))
                        self._channel_list.addItem(item)

                    # 左侧通道名称视图
                    if self._channel_name_view is not None:
                        self._channel_name_view.clear()
                        for i, name in enumerate(self._channel_names):
                            self._channel_name_view.addItem(f"{i+1}: {name}")

                    # 初始化数据、通道增益和偏移
                    self._current_data_raw = data.astype(np.float32, copy=True)
                    self._current_data_processed = self._current_data_raw.copy()
                    self._channel_gains = {i: 1.0 for i in range(channel_count)}
                    self._channel_offsets = {i: 0.0 for i in range(channel_count)}
                    self._marker_times.clear()
                    self._clear_marker_lines()

                    # 更新时间窗范围
                    self._start_spin.setMaximum(max(0.0, meta.duration_seconds - 0.1))
                    self._length_spin.setMaximum(meta.duration_seconds)
                    self._length_spin.setValue(min(5.0, meta.duration_seconds))
                    self._start_spin.setValue(0.0)
                    self._marker_time_spin.setMaximum(meta.duration_seconds)
                    self._marker_time_spin.setValue(0.0)

                    self._apply_preprocessing(update_plot=False)
                    self._update_plot()
                else:
                    self._info_text.setText("转换成功，但未找到数据文件")
        except Exception as exc:
            self._info_text.setText(f"预览失败: {str(exc)}")

    def _prompt_dat_params(self, path: Path) -> Optional[Dict[str, int]]:
        """针对 .dat 文件，提示用户输入采样率和通道数。"""
        # 先尝试读取头部作为默认值
        is_eye_guess = path.stem.upper().startswith("ET_")
        default = {"srate": 500, "emg_chs": 8, "acc_chs": 12, "glove_chs": 14, "dtype": "float64", "is_eye": is_eye_guess}
        try:
            with open(path, "rb") as f:
                header = np.frombuffer(f.read(7 * 4), dtype=np.int32, count=7)
                if header.size == 7:
                    default["srate"] = int(header[3])
                    default["emg_chs"] = int(header[4])
                    default["acc_chs"] = int(header[5])
                    default["glove_chs"] = int(header[6])
                    default["dtype"] = "float32" if int(header[2]) == 2 else "float64"
        except Exception:
            pass

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("DAT 参数确认")
        form = QtWidgets.QFormLayout(dialog)

        srate_spin = QtWidgets.QSpinBox()
        srate_spin.setRange(1, 5000)
        srate_spin.setValue(default["srate"])
        form.addRow("采样率(Hz)", srate_spin)

        emg_spin = QtWidgets.QSpinBox()
        emg_spin.setRange(0, 128)
        emg_spin.setValue(default["emg_chs"])
        form.addRow("肌电通道数", emg_spin)

        acc_spin = QtWidgets.QSpinBox()
        acc_spin.setRange(0, 128)
        acc_spin.setValue(default["acc_chs"])
        form.addRow("IMU通道数", acc_spin)

        glove_spin = QtWidgets.QSpinBox()
        glove_spin.setRange(0, 128)
        glove_spin.setValue(default["glove_chs"])
        form.addRow("手套通道数", glove_spin)

        dtype_combo = QtWidgets.QComboBox()
        dtype_combo.addItems(["float32", "float64"])
        dtype_combo.setCurrentText(default["dtype"])
        form.addRow("数据类型", dtype_combo)

        eye_chk = QtWidgets.QCheckBox("眼动数据（仅X,Y，无时间戳）")
        eye_chk.setChecked(default["is_eye"])
        form.addRow(eye_chk)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        form.addRow(btns)
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return None

        return {
            "srate": int(srate_spin.value()),
            "emg_chs": int(emg_spin.value()),
            "acc_chs": int(acc_spin.value()),
            "glove_chs": int(glove_spin.value()),
            "dtype": dtype_combo.currentText(),
            "is_eye": eye_chk.isChecked(),
        }

    def _on_export_dataset_clicked(self) -> None:
        """基于当前预处理结果，按切片导出 npy 数据集。"""
        if self._current_data_processed is None or self._current_meta is None:
            QtWidgets.QMessageBox.warning(self, "错误", "请先导入并预览数据")
            return

        output_dir = self._output_dir_edit.text().strip()
        if not output_dir:
            QtWidgets.QMessageBox.warning(self, "错误", "请先选择输出目录")
            return

        sr = float(self._current_meta.sample_rate)
        if sr <= 0:
            QtWidgets.QMessageBox.warning(self, "错误", "采样率无效，无法切片")
            return

        slice_len_s = float(self._slice_len_spin.value())
        slice_step_s = float(self._slice_step_spin.value())
        slice_len = int(slice_len_s * sr)
        slice_step = int(slice_step_s * sr)
        if slice_len <= 0 or slice_step <= 0:
            QtWidgets.QMessageBox.warning(self, "错误", "切片长度/步长必须大于0")
            return

        data = self._current_data_processed
        total_samples = data.shape[1]
        slices = []
        for start in range(0, total_samples - slice_len + 1, slice_step):
            end = start + slice_len
            slices.append(data[:, start:end])

        if not slices:
            QtWidgets.QMessageBox.warning(self, "提示", "数据长度不足，无法按当前切片参数生成数据集")
            return

        dataset = np.stack(slices, axis=0)  # (N, C, T)
        ts = time.strftime("%Y%m%d_%H%M%S")
        modality = self._current_meta.modality or "EEG"
        save_dir = Path(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        data_file = save_dir / f"dataset_{modality}_{ts}.npy"
        meta_file = save_dir / f"dataset_meta_{modality}_{ts}.json"

        np.save(data_file, dataset.astype(np.float32, copy=False))
        meta_dict = {
            "modality": modality,
            "channels": self._current_meta.channels,
            "sample_rate": sr,
            "slice_len_s": slice_len_s,
            "slice_step_s": slice_step_s,
            "slice_len_samples": slice_len,
            "slice_step_samples": slice_step,
            "num_slices": int(dataset.shape[0]),
            "source_file": self._file_path_edit.text().strip(),
            "channel_names": self._channel_names,
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2, ensure_ascii=False)

        QtWidgets.QMessageBox.information(
            self,
            "导出成功",
            f"已生成数据集：\n{data_file.name}\n切片数: {dataset.shape[0]}，形状: {dataset.shape}",
        )

    def _on_channel_control_clicked(self) -> None:
        if self._current_data_raw is None:
            QtWidgets.QMessageBox.warning(self, "错误", "请先导入数据")
            return

        channel_count = self._current_data_raw.shape[0]
        dialog = ChannelControlDialog(
            channel_count, self._channel_gains, self._channel_offsets, parent=self
        )
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self._channel_gains = dialog.get_gains()
            self._channel_offsets = dialog.get_offsets()
            self._update_plot()

    def _update_plot(self) -> None:
        if self._current_data_processed is None or self._current_meta is None:
            return

        self._plot.clear()
        self._marker_lines.clear()
        # 清理旧的通道与标记文字
        for item in self._channel_label_items:
            try:
                self._plot.removeItem(item)
            except Exception:
                pass
        self._channel_label_items.clear()
        for item in self._marker_label_items:
            try:
                self._plot.removeItem(item)
            except Exception:
                pass
        self._marker_label_items.clear()

        # 获取选中的通道
        selected_items = self._channel_list.selectedItems()
        if not selected_items:
            return

        visible_channels = [item.data(QtCore.Qt.UserRole) for item in selected_items]
        channel_count, total_samples = self._current_data_processed.shape
        sample_rate = self._current_meta.sample_rate
        duration = total_samples / sample_rate

        # 计算时间窗范围
        window_start = self._start_spin.value()
        window_length = self._length_spin.value()
        max_start = max(0.0, duration - window_length)
        window_start = min(window_start, max_start)

        start_sample = int(window_start * sample_rate)
        end_sample = int((window_start + window_length) * sample_rate)
        end_sample = min(end_sample, total_samples)
        start_sample = max(0, start_sample)

        if start_sample >= end_sample:
            return

        time_axis = np.linspace(window_start, window_start + window_length, end_sample - start_sample)

        # 绘制选中的通道（使用 MATLAB 默认颜色）
        palette = [
            (0, 114, 189),      # 蓝色
            (217, 83, 25),      # 红色
            (237, 177, 32),     # 黄色
            (126, 47, 142),     # 紫色
            (119, 172, 48),     # 绿色
            (77, 190, 238),     # 青色
            (162, 20, 47),      # 深红色
        ]

        channel_spacing = 100.0
        for idx, ch in enumerate(visible_channels):
            if ch >= channel_count:
                continue
            gain = self._channel_gains.get(ch, 1.0)
            offset = self._channel_offsets.get(ch, 0.0)
            y_offset = idx * channel_spacing

            channel_data = self._current_data_processed[ch, start_sample:end_sample]
            scaled_data = channel_data * gain + offset + y_offset

            color = palette[idx % len(palette)]
            curve = pg.PlotCurveItem(
                time_axis, scaled_data, pen=pg.mkPen(color=color, width=2), name=f"通道 {ch+1}"
            )
            self._plot.addItem(curve)

            # 在每根线左侧标注通道名称
            if self._channel_names and ch < len(self._channel_names):
                label_text = self._channel_names[ch]
            else:
                label_text = f"Ch{ch+1}"
            text_item = pg.TextItem(
                text=label_text,
                color=(0, 0, 0),
                anchor=(1.0, 0.5),
            )
            text_item.setPos(window_start, y_offset)
            self._plot.addItem(text_item)
            self._channel_label_items.append(text_item)

        # 设置 Y 轴范围（固定，不可滑动）
        y_min = self._y_min_spin.value()
        y_max = self._y_max_spin.value()
        self._plot.setYRange(y_min, y_max, padding=0)

        # 设置 X 轴范围
        self._plot.setXRange(window_start, window_start + window_length, padding=0)
        self._render_marker_lines()

    def _on_add_marker_clicked(self) -> None:
        if self._current_data_raw is None or self._current_meta is None:
            QtWidgets.QMessageBox.warning(self, "错误", "请先导入数据")
            return
        time_value = self._marker_time_spin.value()
        duration = self._current_data_raw.shape[1] / self._current_meta.sample_rate
        time_value = max(0.0, min(time_value, duration))
        self._marker_times.append(time_value)
        self._render_marker_lines()

    def _on_remove_marker_clicked(self) -> None:
        if not self._marker_times:
            QtWidgets.QMessageBox.information(self, "提示", "暂无标记")
            return
        target = self._marker_time_spin.value()
        closest_idx = min(
            range(len(self._marker_times)), key=lambda idx: abs(self._marker_times[idx] - target)
        )
        self._marker_times.pop(closest_idx)
        self._render_marker_lines()

    def _render_marker_lines(self) -> None:
        self._clear_marker_lines()
        y_max = self._y_max_spin.value()
        for idx, t in enumerate(self._marker_times):
            # 被选中的标记更粗更亮
            if idx == self._selected_marker_index:
                pen = pg.mkPen((217, 83, 25), width=4, style=QtCore.Qt.SolidLine)
            else:
                pen = pg.mkPen((217, 83, 25), width=3, style=QtCore.Qt.DashLine)
            line = pg.InfiniteLine(pos=t, angle=90, pen=pen, movable=True)
            line.setZValue(50)
            # 拖动时更新对应时间戳
            line.sigPositionChanged.connect(self._make_marker_move_handler(idx))
            self._plot.addItem(line)
            self._marker_lines.append(line)
        # 更新左侧标记时间列表
        if self._marker_list_view is not None:
            self._marker_list_view.clear()
            for idx, t in enumerate(self._marker_times):
                self._marker_list_view.addItem(f"{idx+1}: {t:.3f} s")
        # 在每个标记线上方标注时间戳
        for idx, t in enumerate(self._marker_times):
            label_color = (217, 83, 25) if idx != self._selected_marker_index else (128, 0, 0)
            label = pg.TextItem(text=f"{t:.3f}s", color=label_color, anchor=(0.5, 1.0))
            label.setPos(t, y_max)
            label.setZValue(51)
            self._plot.addItem(label)
            self._marker_label_items.append(label)

    def _clear_marker_lines(self) -> None:
        for line in self._marker_lines:
            try:
                self._plot.removeItem(line)
            except Exception:
                pass
        self._marker_lines.clear()
        for item in self._marker_label_items:
            try:
                self._plot.removeItem(item)
            except Exception:
                pass
        self._marker_label_items.clear()

    def _make_marker_move_handler(self, index: int):
        def _handler(line: pg.InfiniteLine) -> None:
            # 根据拖动结果更新时间戳，再整体重绘标记
            new_time = float(line.value())
            if 0 <= index < len(self._marker_times):
                self._marker_times[index] = new_time
                # 同步当前选中标记时间
                if self._selected_marker_index == index:
                    self._marker_time_spin.setValue(new_time)
                self._render_marker_lines()
        return _handler

    def _on_plot_clicked(self, event) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            return
        if self._current_data_raw is None or self._current_meta is None:
            return
        mouse_point = self._plot.getPlotItem().vb.mapSceneToView(event.scenePos())
        x = float(mouse_point.x())
        if x < 0:
            x = 0.0
        duration = self._current_data_raw.shape[1] / self._current_meta.sample_rate
        if x > duration:
            x = duration

        # 若已有标记，优先选择离得最近的一根
        if self._marker_times:
            diffs = [abs(t - x) for t in self._marker_times]
            idx_min = int(np.argmin(diffs))
            # 阈值：当前时间窗口长度的 5% 或 0.2s 中的较大者
            window_len = max(self._length_spin.value(), 0.001)
            threshold = max(0.2, window_len * 0.05)
            if diffs[idx_min] <= threshold:
                self._selected_marker_index = idx_min
                self._marker_time_spin.setValue(self._marker_times[idx_min])
                self._render_marker_lines()
                return

        # 否则添加新标记
        self._marker_times.append(x)
        self._selected_marker_index = len(self._marker_times) - 1
        self._marker_time_spin.setValue(x)
        self._render_marker_lines()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # type: ignore[override]
        if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            if self._selected_marker_index is not None and 0 <= self._selected_marker_index < len(
                self._marker_times
            ):
                self._marker_times.pop(self._selected_marker_index)
                self._selected_marker_index = None
                self._render_marker_lines()
                event.accept()
                return
        super().keyPressEvent(event)

    def _on_apply_preprocessing_clicked(self) -> None:
        if self._current_data_raw is None:
            QtWidgets.QMessageBox.warning(self, "错误", "请先导入数据")
            return
        self._apply_preprocessing()

    def _apply_preprocessing(self, update_plot: bool = True) -> None:
        if self._current_data_raw is None or self._current_meta is None:
            return
        data = self._current_data_raw.copy()

        filter_mode = self._filter_combo.currentData()
        # 眼动/EOG 数据不做滤波，强制无滤波
        modality_upper = str(self._current_meta.modality).upper() if self._current_meta else ""
        if modality_upper in {"EYE", "EOG", "EYE_TRACK", "EYETRACK", "EYE-TRACK"}:
            filter_mode = "none"
        if filter_mode != "none":
            data = self._apply_filter(
                data,
                filter_mode,
                self._current_meta.sample_rate,
                self._low_cut_spin.value(),
                self._high_cut_spin.value(),
            )

        norm_mode = self._norm_combo.currentData()
        if norm_mode != "none":
            data = self._apply_normalization(data, norm_mode)

        self._current_data_processed = data.astype(np.float32, copy=False)
        if update_plot:
            self._update_plot()

    @staticmethod
    def _apply_filter(
        data: np.ndarray,
        mode: str,
        sample_rate: float,
        low_cut: float,
        high_cut: float,
    ) -> np.ndarray:
        nyquist = sample_rate / 2.0
        order = 4
        if mode == "highpass":
            cutoff = max(0.1, min(low_cut, nyquist - 0.1))
            normalized = cutoff / nyquist
            b, a = butter(order, normalized, btype="highpass")
            return filtfilt(b, a, data, axis=1)
        if mode == "bandpass":
            low = max(0.1, min(low_cut, nyquist - 0.1))
            high = max(low + 0.1, min(high_cut, nyquist - 0.1))
            normalized = [low / nyquist, high / nyquist]
            b, a = butter(order, normalized, btype="bandpass")
            return filtfilt(b, a, data, axis=1)
        return data

    def _update_preprocess_controls_for_modality(self, modality: str) -> None:
        """根据模态控制预处理可用性：眼动类禁用滤波。"""
        mod = str(modality).upper()
        is_eye = mod in {"EYE", "EOG", "EYE_TRACK", "EYETRACK", "EYE-TRACK"}
        if is_eye:
            self._filter_combo.setCurrentIndex(0)  # 无滤波
            self._filter_combo.setEnabled(False)
            self._low_cut_spin.setEnabled(False)
            self._high_cut_spin.setEnabled(False)
        else:
            self._filter_combo.setEnabled(True)
            self._low_cut_spin.setEnabled(True)
            self._high_cut_spin.setEnabled(True)

    # 直接加载已有 npy + json 元信息
    def _load_npy_with_meta(self, npy_path: Path, all_paths: List[Path]) -> None:
        if not npy_path.exists():
            raise FileNotFoundError(f"找不到数据文件: {npy_path}")
        data = np.load(npy_path)
        channel_count, total_samples = data.shape[:2] if data.ndim == 2 else (data.shape[0], data.shape[-1])
        # 查找配套 json
        meta_path = None
        for p in all_paths:
            if p.suffix.lower() == ".json":
                meta_path = p
                break
        sample_rate = 1.0
        modality = "EEG"
        channel_names: Optional[List[str]] = None
        if meta_path and meta_path.exists():
            try:
                meta_json = json.loads(meta_path.read_text(encoding="utf-8"))
                sample_rate = float(meta_json.get("sample_rate", sample_rate))
                modality = str(meta_json.get("modality", modality))
                names = meta_json.get("channel_names")
                if isinstance(names, (list, tuple)) and len(names) == channel_count:
                    channel_names = [str(n) for n in names]
            except Exception:
                pass
        duration = total_samples / sample_rate if sample_rate > 0 else 0.0

        meta = MetaInfo(
            subject_id=npy_path.stem,
            modality=modality,  # type: ignore[arg-type]
            channels=channel_count,
            sample_rate=sample_rate,
            duration_seconds=duration,
            extra={"source_file": str(npy_path), "meta_file": str(meta_path) if meta_path else None},
        )

        self._current_meta = meta
        self._current_data_raw = data.astype(np.float32, copy=True)
        self._current_data_processed = self._current_data_raw.copy()
        self._update_preprocess_controls_for_modality(meta.modality)
        self._channel_names = channel_names or [f"Ch{i+1}" for i in range(channel_count)]
        self._channel_list.clear()
        for i, name in enumerate(self._channel_names):
            item = QtWidgets.QListWidgetItem(f"{name}")
            item.setData(QtCore.Qt.UserRole, i)
            item.setSelected(i < min(8, channel_count))
            self._channel_list.addItem(item)
        if self._channel_name_view is not None:
            self._channel_name_view.clear()
            for i, name in enumerate(self._channel_names):
                self._channel_name_view.addItem(f"{i+1}: {name}")

        self._channel_gains = {i: 1.0 for i in range(channel_count)}
        self._channel_offsets = {i: 0.0 for i in range(channel_count)}
        self._marker_times.clear()
        self._clear_marker_lines()

        self._start_spin.setMaximum(max(0.0, duration - 0.1))
        self._length_spin.setMaximum(max(duration, 0.1))
        self._length_spin.setValue(min(5.0, duration if duration > 0 else 5.0))
        self._start_spin.setValue(0.0)
        self._marker_time_spin.setMaximum(max(duration, 0.0))
        self._marker_time_spin.setValue(0.0)

        self._apply_preprocessing(update_plot=True)
        info_text = f"""模态: {meta.modality}
通道数: {meta.channels}
采样率: {meta.sample_rate} Hz
时长: {meta.duration_seconds:.2f} s
总样本数: {self._current_data_processed.shape[1]}
源数据: {npy_path.name}
"""
        if meta_path:
            info_text += f"元信息: {meta_path.name}"
        self._info_text.setText(info_text)

    @staticmethod
    def _apply_normalization(data: np.ndarray, mode: str) -> np.ndarray:
        if mode == "zscore":
            mean = data.mean(axis=1, keepdims=True)
            std = data.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            return (data - mean) / std
        if mode == "minmax":
            min_v = data.min(axis=1, keepdims=True)
            max_v = data.max(axis=1, keepdims=True)
            span = max_v - min_v
            span[span == 0] = 1.0
            return (data - min_v) / span
        return data

