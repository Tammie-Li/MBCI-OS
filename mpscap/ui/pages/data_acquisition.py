"""
数据采集页面：支持实时数据采集和可视化。
"""

from __future__ import annotations

import colorsys
import os
import socket
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from scipy.signal import butter, filtfilt

from ...core.utils.shm import CreateShm, EEGTYPE
from ...core.utils.data_manager import DataManager
from ...core.utils.butter_filter import ButterFilter
from ...core.drivers.rda1299 import RDA1299
from ...core.drivers.eye_tracker import TobiiEyeTracker
from ...core.data_pipeline.tsdb_storage import TSDBStorage, TSDBConfig
from ...core.data_pipeline.kafka_producer import KafkaConfig, KafkaDataProducer

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")


class DeviceConfigDialog(QtWidgets.QDialog):
    """设备配置对话框，先选类型再设置对应参数。"""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("添加设备")
        self.resize(520, 360)
        self._result: Optional[dict] = None

        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()

        # 设备类型
        self._type_combo = QtWidgets.QComboBox()
        self._type_combo.addItems(["自研肌电腕带", "眼动仪"])
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        form.addRow("设备类型:", self._type_combo)

        # 设备别名
        self._alias_edit = QtWidgets.QLineEdit()
        self._alias_edit.setPlaceholderText("设备显示名称（可选）")
        form.addRow("设备别名:", self._alias_edit)

        # COM口选择（肌电专用）
        self._port_combo = QtWidgets.QComboBox()
        self._port_combo.setEditable(True)
        self._refresh_ports()
        refresh_btn = QtWidgets.QPushButton("刷新")
        refresh_btn.clicked.connect(self._refresh_ports)
        port_layout = QtWidgets.QHBoxLayout()
        port_layout.setContentsMargins(0, 0, 0, 0)
        port_layout.setSpacing(6)
        port_layout.addWidget(self._port_combo)
        port_layout.addWidget(refresh_btn)
        self._port_widget = QtWidgets.QWidget()
        self._port_widget.setLayout(port_layout)
        self._port_label = QtWidgets.QLabel("COM口:")
        form.addRow(self._port_label, self._port_widget)

        # 比特率设置（肌电专用）
        self._baudrate_combo = QtWidgets.QComboBox()
        self._baudrate_combo.addItems(["115200", "230400", "460800", "921600"])
        self._baudrate_combo.setCurrentText("460800")
        self._baud_label = QtWidgets.QLabel("比特率:")
        form.addRow(self._baud_label, self._baudrate_combo)

        layout.addLayout(form)
        layout.addStretch()

        # 按钮
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._center_on_parent()

        # 初始状态
        self._on_type_changed(self._type_combo.currentIndex())
        self._center_on_parent()

    def _on_type_changed(self, idx: int) -> None:
        """根据设备类型显隐串口参数。"""
        is_emg = idx == 0
        self._port_widget.setVisible(is_emg)
        self._port_label.setVisible(is_emg)
        self._baudrate_combo.setVisible(is_emg)
        self._baud_label.setVisible(is_emg)

    def _refresh_ports(self) -> None:
        """刷新串口列表。"""
        self._port_combo.clear()
        try:
            ports = RDA1299.getallserial()
            if ports:
                self._port_combo.addItems(ports)
            else:
                self._port_combo.addItem("未找到设备")
        except Exception as e:
            print(f"[ERROR] COM口刷新失败: {e}")
            import traceback
            traceback.print_exc()
            self._port_combo.addItem("刷新失败")

    def accept(self) -> None:
        """确认配置。"""
        device_type = "emg_wristband" if self._type_combo.currentIndex() == 0 else "eye_tracker"
        alias = self._alias_edit.text().strip() or ("自研肌电腕带" if device_type == "emg_wristband" else "眼动仪")

        if device_type == "emg_wristband":
            port = self._port_combo.currentText().strip()
            if not port or port == "未找到设备" or port == "刷新失败":
                QtWidgets.QMessageBox.warning(self, "错误", "请选择有效的COM口")
                return
            try:
                baudrate = int(self._baudrate_combo.currentText())
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "错误", "无效的比特率")
                return
            self._result = {
                'type': device_type,
                'port': port,
                'baudrate': baudrate,
                'alias': alias,
            }
        else:
            self._result = {
                'type': device_type,
                'alias': alias,
            }
        super().accept()

    def get_config(self) -> Optional[dict]:
        """获取配置结果。"""
        return self._result

    def _center_on_parent(self) -> None:
        """弹窗居中显示（相对父窗口或屏幕）。"""
        try:
            parent = self.parentWidget()
            if parent is not None:
                geo = parent.frameGeometry()
                center = geo.center()
            else:
                screen = QtWidgets.QApplication.primaryScreen()
                center = screen.availableGeometry().center() if screen else QtCore.QPoint(0, 0)
            fg = self.frameGeometry()
            fg.moveCenter(center)
            self.move(fg.topLeft())
        except Exception:
            pass


class EyeTrackerPanel(QtWidgets.QGroupBox):
    """眼动信号实时获取与绘制。"""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("眼动信号")
        self._tracker: Optional[TobiiEyeTracker] = None
        # 保存最近一段时间的 (t, x, y)，默认约 8 秒窗口
        self._history = deque(maxlen=260)
        self._start_ts: float = 0.0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        ctrl = QtWidgets.QHBoxLayout()
        self._start_btn = QtWidgets.QPushButton("开始采集")
        self._start_btn.setCheckable(True)
        self._start_btn.toggled.connect(self._on_toggled)
        ctrl.addWidget(self._start_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # 双通道曲线：X、Y 随时间
        self._plot = pg.PlotWidget()
        self._plot.showGrid(True, True)
        self._plot.enableAutoRange(x=False, y=True)
        self._plot.setLabel("left", "")
        self._plot.setLabel("bottom", "")
        self._curve_x = pg.PlotCurveItem(pen=pg.mkPen(color=(250, 100, 80), width=2), name="X")
        self._curve_y = pg.PlotCurveItem(pen=pg.mkPen(color=(80, 140, 255), width=2), name="Y")
        self._plot.addItem(self._curve_x)
        self._plot.addItem(self._curve_y)
        self._plot.addLegend()
        layout.addWidget(self._plot, stretch=1)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)

    def _on_toggled(self, checked: bool) -> None:
        if checked:
            self._start_btn.setText("停止采集")
            self._start()
        else:
            self._start_btn.setText("开始采集")
            self._stop()

    def _start(self) -> None:
        # 启动眼动仪数据流
        try:
            self._tracker = TobiiEyeTracker()
            self._history.clear()
            self._start_ts = time.time()
            self._timer.start(30)  # ~30ms 刷新
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"眼动采集启动失败: {e}")
            self._start_btn.setChecked(False)
            self._tracker = None

    def _stop(self) -> None:
        self._timer.stop()
        if self._tracker is not None:
            try:
                self._tracker.stop()
            except Exception:
                pass
        self._tracker = None
        self._history.clear()
        self._curve_x.setData([], [])
        self._curve_y.setData([], [])

    def _on_tick(self) -> None:
        if self._tracker is None:
            return
        x, y = self._tracker.latest_xy()
        if x < 0 or y < 0:
            return
        t = time.time() - self._start_ts
        self._history.append((t, x, y))

        ts = [p[0] for p in self._history]
        xs = [p[1] for p in self._history]
        ys = [p[2] for p in self._history]

        self._curve_x.setData(ts, xs)
        self._curve_y.setData(ts, ys)

        # 窗口随时间滚动，保持最近 8 秒
        if ts:
            tmax = ts[-1]
            tmin = max(0.0, tmax - 8.0)
            self._plot.setXRange(tmin, tmax, padding=0.02)

    def shutdown(self) -> None:
        self._stop()


class EyeTrackerDevice(QtWidgets.QGroupBox):
    """眼动仪设备包装，含启动、绘制与Kafka发布。"""

    def __init__(self, title: str, on_remove, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(title)
        self._on_remove = on_remove
        self._panel = EyeTrackerPanel()
        self._kafka_config: Optional[KafkaConfig] = None
        self._kafka_producer: Optional[KafkaDataProducer] = None
        self._publishing = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # 控件统一与肌电样式和顺序
        self._start_btn = self._panel._start_btn  # 复用面板按钮
        self._start_btn.setStyleSheet("""
            QPushButton:checked {
                background-color: rgb(223, 4, 4);
                color: white;
            }
        """)
        self._start_btn.toggled.connect(self._on_start_toggled)

        self._kafka_publish_btn = QtWidgets.QPushButton("Kafka发布")
        self._kafka_publish_btn.setCheckable(True)
        self._kafka_publish_btn.setEnabled(False)
        self._kafka_publish_btn.setStyleSheet("""
            QPushButton:checked {
                background-color: rgb(0, 150, 0);
                color: white;
            }
        """)
        self._kafka_publish_btn.toggled.connect(self._on_kafka_publish_toggled)

        self._kafka_config_btn = QtWidgets.QPushButton("Kafka配置")
        self._kafka_config_btn.clicked.connect(self._on_kafka_config_clicked)

        remove_btn = QtWidgets.QPushButton("移除")
        remove_btn.clicked.connect(self._on_remove_clicked)

        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addStretch()  # 按钮整体右对齐
        ctrl.addWidget(self._kafka_publish_btn)
        ctrl.addWidget(self._kafka_config_btn)
        ctrl.addWidget(self._start_btn)
        ctrl.addWidget(remove_btn)
        layout.addLayout(ctrl)

        layout.addWidget(self._panel)

        # 扩展 panel 的 tick 以便在发布时取最新数据
        self._panel._timer.timeout.disconnect()
        self._panel._timer.timeout.connect(self._on_tick_wrapper)

    def _on_kafka_config_clicked(self) -> None:
        dialog = KafkaConfigDialog(self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            config = dialog.get_config()
            if config:
                # 眼动信号：固定小批次和默认topic
                config.batch_size = 3
                config.topic = "et_data"
                self._kafka_config = config
                self._kafka_publish_btn.setEnabled(True)
                self._kafka_publish_btn.setChecked(True)

    def _on_start_toggled(self, checked: bool) -> None:
        """采集开关：关闭时自动停Kafka发布。"""
        if not checked and self._kafka_publish_btn.isChecked():
            self._kafka_publish_btn.setChecked(False)

    def _on_kafka_publish_toggled(self, checked: bool) -> None:
        if checked:
            if self._kafka_config is None:
                QtWidgets.QMessageBox.warning(self, "警告", "请先配置Kafka服务器")
                self._kafka_publish_btn.setChecked(False)
                return
            try:
                self._kafka_producer = KafkaDataProducer(self._kafka_config)
                if not self._kafka_producer.connect():
                    raise RuntimeError("Kafka连接失败")
                self._publishing = True
                self._kafka_publish_btn.setText("停止发布")
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "警告", f"Kafka发布启用失败: {e}")
                self._kafka_publish_btn.setChecked(False)
                self._publishing = False
        else:
            self._publishing = False
            self._kafka_publish_btn.setText("Kafka发布")
            if self._kafka_producer:
                try:
                    self._kafka_producer.disconnect()
                except Exception:
                    pass
                self._kafka_producer = None

    def _on_tick_wrapper(self) -> None:
        # 原有绘制
        self._panel._on_tick()
        # 发布最新点
        if not self._publishing or self._kafka_producer is None:
            return
        if not self._panel._history:
            return
        t, x, y = self._panel._history[-1]
        try:
            import numpy as np

            data = np.array([[x], [y]], dtype=float)
            self._kafka_producer.publish_data(data, timestamps=[time.time()], device_id="eye_tracker")
        except Exception as e:
            print(f"[WARNING] 眼动Kafka发布失败: {e}")

    def _on_remove_clicked(self) -> None:
        self.shutdown()
        if callable(self._on_remove):
            self._on_remove(self)

    def shutdown(self) -> None:
        self._publishing = False
        if self._kafka_producer:
            try:
                self._kafka_producer.disconnect()
            except Exception:
                pass
            self._kafka_producer = None
        try:
            self._panel.shutdown()
        except Exception:
            pass


class WavePanel(QtWidgets.QWidget):
    """单个设备波形显示面板（参考eegdisplay.py/EEGDisplay）。"""

    def __init__(self, title: str, history_seconds: float = 2.5, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.history_seconds = history_seconds
        
        # 共享内存（参考eegdisplay.py第36行）
        self.shm = CreateShm(master=False)
        
        # 绘图相关（参考eegdisplay.py第42-56行）
        self.flttype = 2  # 0-None, 1-high, 2-band
        self.scale = 100.0  # 通道间距（默认100）
        self.ygain = 1.0
        self.curves: List[pg.PlotCurveItem] = []
        self.period = history_seconds  # 绘图时长
        self.prepare = True  # True：update_one_frame跳过
        
        # 绘图控制（参考eegdisplay.py第69行）
        # 仅保留波形绘制，姿态独立弹窗
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 时域波形绘图区域
        self.pgplot = pg.PlotWidget()
        self.pgplot.showGrid(True, True)
        layout.addWidget(self.pgplot)
        
        # 绘图数据配置（参考eegdisplay.py第61行）
        self.dm = DataManager()  # 绘图数据管理器
        
        # 数据读取和滤波控制（参考eegdisplay.py第83-86行）
        self.index = 0
        self.filter: Optional[ButterFilter] = None
        
        # 定时器（参考eegdisplay.py第88-89行）
        self.pgTimer = QtCore.QTimer()
        self.pgTimer.timeout.connect(self.update_one_frame)
        
        # 初始化标志
        self.virgin = True
        self.emgChsNum = 0
        self.accChsNum = 0
        self.gloveChsNum = 0
        self.localSrate = 0
        self.downsampleScale = 1
        # 可视化开关默认值
        self._show_emg = True
        self._show_imu1 = False
        self._show_imu2 = False
        self._show_pressure = False
        self._show_bend = False
        # 时间轴起点（秒），None 表示使用当前时间
        self.time_origin: Optional[float] = None

    def relayout(self) -> None:
        """重新布局（参考eegdisplay.py/relayout）。"""
        self.prepare = True  # 暂时屏蔽绘图更新
        
        # 从共享内存获取配置信息
        emgChsNum = self.shm.getvalue('emgchs')
        accChsNum = self.shm.getvalue('accchs')
        gloveChsNum = self.shm.getvalue('glovechs')
        rawsrate = self.shm.getvalue('srate')
        
        if emgChsNum == 0 or rawsrate == 0:
            self.prepare = False
            return
        
        self.downsampleScale = rawsrate // 250
        self.localSrate = rawsrate // self.downsampleScale
        self.emgChsNum = emgChsNum
        self.accChsNum = accChsNum
        self.gloveChsNum = gloveChsNum
        
        # 总显示通道数 = EMG + ACC（IMU1+IMU2）+ Glove（压力+弯曲等）
        totalChsNum = self.emgChsNum + self.accChsNum + self.gloveChsNum
        scale = self.scale
        self.pgplot.setYRange(0, scale * totalChsNum)
        
        if self.virgin:
            self.virgin = False
            self.curves = []
            # 参考eegdisplay.py第143-147行：创建曲线（这里为 EMG+IMU 通道）
            for idx in range(totalChsNum):
                curve = pg.PlotCurveItem(pen=pg.mkPen(color=(250, 100, 50), width=2))
                self.pgplot.addItem(curve)
                curve.setPos(0, idx * scale + 0.5 * scale)
                self.curves.append(curve)
            
            # 初始化DataManager（参考eegdisplay.py第149行）
            # 固定使用1500点，确保绘图窗口正好占满横轴
            FIXED_POINTS = 1500
            self.dm.config(self.localSrate, totalChsNum, self.period, EEGTYPE, fixed_points=FIXED_POINTS)
            # X轴范围设置为0到1500，确保1500点正好占满横轴
            self.pgplot.setXRange(0, FIXED_POINTS)
            
            # 初始化滤波器（参考eegdisplay.py第85-86行）
            self.filter = ButterFilter()
            self.filter.reset(
                srate=self.localSrate,
                chs=self.emgChsNum,
                fltparam=[(49, 51), (20, 150), (1, 0), None],
                eegtype=EEGTYPE
            )
        
        if scale != self.scale:
            self.scale = scale
            for idx in range(totalChsNum):
                if idx < len(self.curves):
                    self.curves[idx].setPos(0, idx * scale + 0.5 * scale)
        
        self.prepare = False
    
    def add_trigger_line(self, line: pg.InfiniteLine) -> None:
        """将外部触发竖线添加到波形图。"""
        try:
            self.pgplot.addItem(line)
        except Exception:
            pass
    
    def update_one_frame(self) -> None:
        """更新一帧数据（完全参考eegdisplay.py/update_one_frame）。"""
        # 读数据（参考eegdisplay.py第181-183行）
        ind = self.shm.info[0]
        if ind == 0:
            return  # 设备未启动
        if self.prepare:
            return  # 准备状态不更新
        
        if ind != self.index:  # eeg数据有更新（参考eegdisplay.py第185行）
            self.index = ind
            if self.shm.getvalue('mode') != 1:  # 确保在正确的模式下（参考eegdisplay.py第187行）
                return
            
            self.shm.setvalue('plotting', 1)  # 参考eegdisplay.py第190行
            
            curdataindx = int(self.shm.getvalue('curdataindex'))
            totalChsNum = self.shm.getvalue('emgchs') + self.shm.getvalue('accchs') + self.shm.getvalue('glovechs')
            if totalChsNum == 0:
                self.shm.setvalue('plotting', 0)
                return
            
            pp = int(curdataindx / totalChsNum)  # 参考eegdisplay.py第193行
            if pp == 0:
                self.shm.setvalue('plotting', 0)
                return
            
            # 读取数据并清空共享内存索引（参考eegdisplay.py第195-198行）
            dat = self.shm.eeg[:curdataindx].reshape(pp, totalChsNum).transpose()
            self.shm.setvalue('curbyteindex', 0)
            self.shm.setvalue('curdataindex', 0)
            self.shm.setvalue('plotting', 0)
            
            # 提取EMG、IMU(ACC)、Glove数据并降采样（参考eegdisplay.py第200行）
            if self.emgChsNum == 0:
                self.relayout()  # 重新布局以获取配置
                if self.emgChsNum == 0:
                    return
            
            # EMG 通道
            eeg = dat[:self.emgChsNum, ::self.downsampleScale]
            # ACC/IMU 通道（紧跟在 EMG 后面）
            acc = np.array([], dtype=EEGTYPE).reshape(0, eeg.shape[1])
            if self.accChsNum > 0:
                acc = dat[self.emgChsNum:self.emgChsNum + self.accChsNum, ::self.downsampleScale]
            # Glove 通道（紧跟在 ACC 后面）
            glove = np.array([], dtype=EEGTYPE).reshape(0, eeg.shape[1])
            if self.gloveChsNum > 0:
                start = self.emgChsNum + self.accChsNum
                glove = dat[start:start + self.gloveChsNum, ::self.downsampleScale]
            
            # 仅对 EMG 做滤波（参考eegdisplay.py第202-210行）
            if self.filter is not None:
                self.filter.update(eeg)
                if self.flttype == 0:  # none
                    emg_for_plot = self.filter.rawdata
                elif self.flttype == 1:  # high pass
                    emg_for_plot = self.filter.hdata
                elif self.flttype == 2:  # band pass
                    emg_for_plot = self.filter.bdata
                else:
                    emg_for_plot = self.filter.rawdata
            else:
                emg_for_plot = eeg
            
            # 组合 EMG(已滤波) + ACC/IMU + Glove(均不滤波) 一起送入 DataManager
            parts = [emg_for_plot]
            if acc.size > 0:
                parts.append(acc)
            if glove.size > 0:
                parts.append(glove)
            combined = np.vstack(parts)
            self.dm.update(combined)
            
            # 更新显示（参考eegdisplay.py第212-214行），并根据可视化开关决定是否绘制
            if self.dm.data is None:
                return

            x_vals = None
            if self.localSrate > 0:
                window_sec = self.dm.data.shape[1] / float(self.localSrate)
            else:
                window_sec = float(self.dm.data.shape[1])
            now_t = time.time()
            if self.time_origin is not None:
                t_end = now_t - self.time_origin
            else:
                t_end = now_t
            t_start = t_end - window_sec
            try:
                x_vals = np.linspace(t_start, t_end, self.dm.data.shape[1])
                self.pgplot.setXRange(t_start, t_end, padding=0.0)
            except Exception:
                x_vals = None

            # 前 emgChsNum 条为 EMG，后面依次为 IMU/ACC 和 Glove
            totalChsNum = self.dm.data.shape[0]
            for id in range(totalChsNum):
                if id >= len(self.curves):
                    break

                # 判定当前通道是否需要显示
                visible = True
                if id < self.emgChsNum:
                    # EMG 通道
                    visible = self._show_emg
                elif id < self.emgChsNum + self.accChsNum:
                    # IMU 通道（ACC），前6个为IMU1，后6个为IMU2
                    imu_idx = id - self.emgChsNum
                    if imu_idx < 6:
                        visible = self._show_imu1
                    else:
                        visible = self._show_imu2
                else:
                    # Glove 通道：0-4压力，6-10弯曲（相对于Glove起始）
                    glove_idx = id - (self.emgChsNum + self.accChsNum)
                    if 0 <= glove_idx <= 4:
                        visible = self._show_pressure
                    elif 6 <= glove_idx <= 10:
                        visible = self._show_bend
                    else:
                        visible = False

                if visible:
                    if x_vals is not None:
                        self.curves[id].setData(x_vals, self.dm.data[id, :] * self.ygain)
                    else:
                        self.curves[id].setData(self.dm.data[id, :] * self.ygain)
                else:
                    # 隐藏该通道（清空数据）
                    self.curves[id].setData([])
    
    def startPloting(self, flg: bool) -> None:
        """开始/停止绘图（参考eegdisplay.py/startPloting）。"""
        if flg:
            # 参考采集程序：5ms刷新率（200Hz），这是合理的平衡点
            self.pgTimer.start(5)
        else:
            self.pgTimer.stop()
    
    def set_collecting(self, collecting: bool) -> None:
        """设置采集状态。"""
        if collecting:
            self.startPloting(True)
            # 重新布局以获取最新配置
            self.relayout()
        else:
            self.startPloting(False)
    
    def set_gain(self, gain: float) -> None:
        """设置增益。"""
        self.ygain = gain
    
    def set_filter_type(self, flttype: int) -> None:
        """设置滤波类型：0-None, 1-high, 2-band。"""
        self.flttype = flttype
    
    def set_channel_spacing(self, spacing: float) -> None:
        """设置通道间距。"""
        self.scale = spacing
        self.relayout()

    def set_visibility(
        self,
        emg: bool = True,
        imu1: bool = False,
        imu2: bool = False,
        pressure: bool = False,
        bend: bool = False,
    ) -> None:
        """设置各功能块的可视化开关。

        说明：当前绘图主要显示EMG通道（前8导），IMU/压力/弯曲后续可扩展，
        这里先保存可视化状态，避免频繁修改核心绘图逻辑。
        """
        self._show_emg = emg
        self._show_imu1 = imu1
        self._show_imu2 = imu2
        self._show_pressure = pressure
        self._show_bend = bend

    def set_time_origin(self, t0: Optional[float]) -> None:
        """设置时间轴起点（秒），用于将 X 轴对齐到保存开始的相对时间。"""
        self.time_origin = t0

    def release(self) -> None:
        """释放资源（参考eegdisplay.py/release）。"""
        self.startPloting(False)
        self.shm.release()


class DataAcquisitionPage(QtWidgets.QWidget):
    """数据采集页面。"""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._devices: Dict[str, dict] = {}  # device_key -> {rda, panel, config, ...}
        # 数据保存相关状态（支持时序数据库和文件保存）
        self._saving: bool = False
        self._auto_save_dir: Optional[Path] = None  # 若外部请求自动保存则使用此目录，跳过对话框
        self._trigger_log: List[tuple] = []  # 记录触发值与相对时间（秒）
        self._trigger_dat_path: Optional[Path] = None
        self._save_ts_tag: Optional[str] = None
        self._tsdb_storage: Optional[TSDBStorage] = None
        self._session_id: Optional[str] = None
        self._storage_mode: str = "auto"  # "auto", "tsdb", "file"
        self._save_dir: Optional[str] = None
        self._emg_last_idx: int = 0
        self._emg_srate: float = 0.0
        self._et_srate: float = 33.0  # 眼动默认采样率（若可估计则动态更新）
        self._save_start_ts: float = 0.0
        self._current_paradigm_name: Optional[str] = None
        # 通道数量快照（保存开始时确定，避免采集中途变化导致错位）
        self._emg_chs: int = 0
        self._acc_chs: int = 0
        self._glove_chs: int = 0
        self._emg_dat_path: Optional[Path] = None
        # 眼动保存
        self._et_dat_path: Optional[Path] = None
        self._et_file_handle = None
        self._et_last_t_map: Dict[object, float] = {}
        self._et_pending: Dict[object, deque] = {}
        self._et_buf_x: list[float] = []
        self._et_buf_y: list[float] = []
        # trigger 提示
        self._trigger_label: Optional[pg.TextItem] = None
        # trigger UDP 监听（作为共享内存的备选，默认 127.0.0.1:15000，可用环境变量 MPSCAP_TRIGGER_UDP_PORT 配置）
        self._trigger_udp_port: int = int(os.environ.get("MPSCAP_TRIGGER_UDP_PORT", "15000") or 15000)
        self._trigger_udp_queue: deque = deque(maxlen=256)
        self._trigger_udp_stop: threading.Event = threading.Event()
        self._trigger_udp_thread: Optional[threading.Thread] = None
        self._trigger_udp_socket: Optional[socket.socket] = None
        self._init_ui()
        self._start_trigger_udp_listener()

    def _init_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # 工具栏
        toolbar = QtWidgets.QHBoxLayout()
        self._add_device_btn = QtWidgets.QPushButton("添加设备")
        self._add_device_btn.clicked.connect(self._on_add_device_clicked)
        toolbar.addWidget(self._add_device_btn)
        
        toolbar.addStretch()
        # 触发信息显示：仅显示 trigger + 时间（秒）
        self._trigger_display = QtWidgets.QLabel("Trigger: - | Time(s): -")
        toolbar.addWidget(self._trigger_display)
        self._beijing_time_label = QtWidgets.QLabel("北京时间: --")
        toolbar.addWidget(self._beijing_time_label)
        
        # 全局数据保存按钮
        self._save_data_btn = QtWidgets.QPushButton("开始保存数据")
        self._save_data_btn.setCheckable(True)
        self._save_data_btn.toggled.connect(self._on_save_data_toggled)
        toolbar.addWidget(self._save_data_btn)
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.timeout.connect(self._save_data)
        # trigger 竖线提示
        self._trigger_line = pg.InfiniteLine(pen=pg.mkPen("r", width=2))
        self._trigger_line.setVisible(False)
        self._trigger_label = pg.TextItem(color=(200, 0, 0), anchor=(0.5, 1.1))
        self._trigger_label.setVisible(False)
        
        self._status_label = QtWidgets.QLabel("就绪")
        toolbar.addWidget(self._status_label)
        main_layout.addLayout(toolbar)

        # 滚动区域
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._scroll.setWidget(container)
        self._device_layout = QtWidgets.QVBoxLayout(container)
        self._device_layout.setContentsMargins(8, 8, 8, 8)
        self._device_layout.setSpacing(12)
        main_layout.addWidget(self._scroll)

        # 设备列表
        self._eye_devices: List[EyeTrackerDevice] = []
        # 用于按类型分配纵向空间的权重
        self._device_type_weight = {
            "emg_wristband": 10,
            "eye_tracker": 3,
            "eeg": 60,
        }
        # 触发提示绑定到主绘图（第一个设备波形面板出现后添加）
        self._trigger_attached = False
        self._last_trigger_ts = 0
        self._last_trigger_seen_time = 0.0
        self._trigger_timer = QtCore.QTimer(self)
        self._trigger_timer.timeout.connect(self._poll_trigger)
        self._trigger_timer.start(100)
        self._trigger_reset_timer = QtCore.QTimer(self)
        self._trigger_reset_timer.setSingleShot(True)
        self._trigger_reset_timer.timeout.connect(self._reset_trigger_display)
        self._clock_timer = QtCore.QTimer(self)
        self._clock_timer.timeout.connect(self._update_beijing_time)
        self._clock_timer.start(1000)
        self._update_beijing_time()

    def _on_add_device_clicked(self) -> None:
        """添加设备按钮点击事件。"""
        dialog = DeviceConfigDialog(self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        
        config = dialog.get_config()
        if not config:
            return
        
        device_type = config.get('type', 'emg_wristband')

        # 眼动仪走设备模式：添加后再显示绘图区域
        if device_type == "eye_tracker":
            title = config.get('alias', '眼动仪')
            device = EyeTrackerDevice(title, on_remove=self._remove_eye_device, parent=self)
            device.setProperty("device_type", "eye_tracker")
            self._eye_devices.append(device)
            self._device_layout.insertWidget(0, device)
            self._apply_device_stretch()
            self._status_label.setText(f"已添加设备: {title}")
            return

        # EMG设备
        device_key = f"{device_type}_{len(self._devices)}"
        title = config.get('alias', '自研肌电腕带')
        
        try:
            # 创建RDA1299设备（参考devmanager.py第22行）
            # 第一个设备创建共享内存（master=True），后续设备连接（master=False）
            is_first_device = len(self._devices) == 0
            
            # Kafka配置（从设备配置中获取，如果已配置）
            kafka_config = config.get('kafka_config')
            rda = RDA1299(pysig=None, master=is_first_device, kafka_config=kafka_config)
            
            print(f"[DEBUG UI] 配置设备: port={config['port']}, baudrate={config['baudrate']}")
            rda.configDev(port=config['port'], baudrate=config['baudrate'])
            
            print(f"[DEBUG UI] 打开设备...")
            if not rda.open():
                QtWidgets.QMessageBox.warning(self, "错误", "设备打开失败！")
                return
            
            print(f"[DEBUG UI] 设备连接成功")
            
            # 创建显示参数配置区域（在绘图框上方）
            device_widget = QtWidgets.QWidget()
            device_widget.setProperty("device_type", device_type)
            device_layout = QtWidgets.QVBoxLayout(device_widget)
            device_layout.setContentsMargins(8, 8, 8, 8)
            device_layout.setSpacing(8)
            
            # 参数配置栏（压缩为一行，增加可视化选择）
            param_group = QtWidgets.QGroupBox("显示参数")
            param_layout = QtWidgets.QHBoxLayout()

            # 增益与通道间距
            param_layout.addWidget(QtWidgets.QLabel("增益:"))
            gain_spin = QtWidgets.QDoubleSpinBox()
            gain_spin.setRange(0.01, 10.0)
            gain_spin.setSingleStep(0.1)
            gain_spin.setValue(1.0)
            param_layout.addWidget(gain_spin)

            param_layout.addWidget(QtWidgets.QLabel("通道间距:"))
            spacing_spin = QtWidgets.QDoubleSpinBox()
            spacing_spin.setRange(10.0, 500.0)
            spacing_spin.setSingleStep(10.0)
            spacing_spin.setValue(100.0)
            param_layout.addWidget(spacing_spin)

            # 滤波方式与参数
            param_layout.addWidget(QtWidgets.QLabel("滤波:"))
            filter_combo = QtWidgets.QComboBox()
            filter_combo.addItem("无", "none")
            filter_combo.addItem("高通", "highpass")
            filter_combo.addItem("带通", "bandpass")
            filter_combo.setCurrentIndex(2)  # 默认带通
            param_layout.addWidget(filter_combo)

            param_layout.addWidget(QtWidgets.QLabel("低截止(Hz):"))
            low_cut_spin = QtWidgets.QDoubleSpinBox()
            low_cut_spin.setRange(0.1, 1000.0)
            low_cut_spin.setSingleStep(0.5)
            low_cut_spin.setValue(20.0)
            param_layout.addWidget(low_cut_spin)

            param_layout.addWidget(QtWidgets.QLabel("高截止(Hz):"))
            high_cut_spin = QtWidgets.QDoubleSpinBox()
            high_cut_spin.setRange(0.1, 2000.0)
            high_cut_spin.setSingleStep(0.5)
            high_cut_spin.setValue(150.0)
            param_layout.addWidget(high_cut_spin)

            # 可视化选择（34导联中不同功能块）
            emg_chk = QtWidgets.QCheckBox("肌电")
            emg_chk.setChecked(True)  # 默认只勾选肌电
            imu1_chk = QtWidgets.QCheckBox("IMU1")
            imu2_chk = QtWidgets.QCheckBox("IMU2")
            press_chk = QtWidgets.QCheckBox("压力")
            bend_chk = QtWidgets.QCheckBox("弯曲")
            param_layout.addWidget(emg_chk)
            param_layout.addWidget(imu1_chk)
            param_layout.addWidget(imu2_chk)
            param_layout.addWidget(press_chk)
            param_layout.addWidget(bend_chk)


            param_layout.addStretch()

            # Kafka发布按钮
            kafka_publish_btn = QtWidgets.QPushButton("Kafka发布")
            kafka_publish_btn.setCheckable(True)
            kafka_publish_btn.setStyleSheet("""
                QPushButton:checked {
                    background-color: rgb(0, 150, 0);
                    color: white;
                }
            """)
            param_layout.addWidget(kafka_publish_btn)
            
            # Kafka配置按钮
            kafka_config_btn = QtWidgets.QPushButton("Kafka配置")
            param_layout.addWidget(kafka_config_btn)

            # 开始/停止采集按钮与移除按钮
            start_stop_btn = QtWidgets.QPushButton("开始采集")
            start_stop_btn.setCheckable(True)
            start_stop_btn.setStyleSheet("""
                QPushButton:checked {
                    background-color: rgb(223, 4, 4);
                    color: white;
                }
            """)
            param_layout.addWidget(start_stop_btn)
            
            remove_btn = QtWidgets.QPushButton("移除设备")
            remove_btn.clicked.connect(lambda: self._remove_device(device_key))
            param_layout.addWidget(remove_btn)

            param_group.setLayout(param_layout)
            device_layout.addWidget(param_group)
            
            # 绘图框（2.5秒数据窗口，参考eegdisplay.py）
            panel = WavePanel(title, history_seconds=2.5)
            device_layout.addWidget(panel)
            if not self._trigger_attached:
                try:
                    panel.add_trigger_line(self._trigger_line)
                    if self._trigger_label is not None:
                        panel.pgplot.addItem(self._trigger_label)
                    self._trigger_attached = True
                except Exception:
                    pass
            
            # 连接增益和通道间距控件（参考eegdisplay.py）
            def on_gain_changed(value: float) -> None:
                panel.set_gain(value)
            
            def on_spacing_changed(value: float) -> None:
                panel.set_channel_spacing(value)
            
            gain_spin.valueChanged.connect(on_gain_changed)
            spacing_spin.valueChanged.connect(on_spacing_changed)
            
            # 连接滤波控件（参考eegdisplay.py）
            def on_filter_changed(index: int) -> None:
                flttype = index  # 0-None, 1-high, 2-band
                panel.set_filter_type(flttype)
            
            filter_combo.currentIndexChanged.connect(on_filter_changed)

            # 可视化选择回调（目前绘图主要为肌电，预留IMU/压力/弯曲开关）
            def on_visibility_changed() -> None:
                panel.set_visibility(
                    emg=emg_chk.isChecked(),
                    imu1=imu1_chk.isChecked(),
                    imu2=imu2_chk.isChecked(),
                    pressure=press_chk.isChecked(),
                    bend=bend_chk.isChecked(),
                )

            emg_chk.stateChanged.connect(lambda _: on_visibility_changed())
            imu1_chk.stateChanged.connect(lambda _: on_visibility_changed())
            imu2_chk.stateChanged.connect(lambda _: on_visibility_changed())
            press_chk.stateChanged.connect(lambda _: on_visibility_changed())
            bend_chk.stateChanged.connect(lambda _: on_visibility_changed())
            
            # 开始/停止采集按钮（参考devmanager.py/start_acq和stop_acq）
            def on_start_stop_toggled(checked: bool) -> None:
                if checked:
                    start_stop_btn.setText("停止采集")
                    rda.writeCmd('acquireEEG')  # 开始采集（参考devmanager.py第88行）
                    panel.set_collecting(True)
                    # 更新设备采集状态标志
                    self._devices[device_key]['is_collecting'] = True
                else:
                    start_stop_btn.setText("开始采集")
                    rda.writeCmd('stop')  # 停止采集（参考devmanager.py第97行）
                    panel.set_collecting(False)
                    # 更新设备采集状态标志
                    self._devices[device_key]['is_collecting'] = False
            
            start_stop_btn.toggled.connect(on_start_stop_toggled)
            
            # Kafka配置对话框
            def on_kafka_config_clicked():
                dialog = KafkaConfigDialog(self)
                if dialog.exec_() == QtWidgets.QDialog.Accepted:
                    kafka_config = dialog.get_config()
                    if kafka_config:
                        # 更新设备配置
                        self._devices[device_key]['config']['kafka_config'] = kafka_config
                        # 动态更新DataDecoder的Kafka配置
                        try:
                            rda.dec.set_kafka_config(kafka_config)
                            kafka_publish_btn.setEnabled(True)
                            kafka_publish_btn.setChecked(True)
                            self._status_label.setText(f"Kafka配置已更新: {kafka_config.bootstrap_servers}")
                        except Exception as e:
                            QtWidgets.QMessageBox.warning(self, "警告", f"Kafka配置更新失败: {e}")
                            kafka_publish_btn.setChecked(False)
            
            kafka_config_btn.clicked.connect(on_kafka_config_clicked)
            
            # Kafka发布开关
            def on_kafka_publish_toggled(checked: bool):
                if checked:
                    kafka_config = self._devices[device_key]['config'].get('kafka_config')
                    if kafka_config is None:
                        QtWidgets.QMessageBox.warning(
                            self,
                            "警告",
                            "请先配置Kafka服务器！\n点击\"Kafka配置\"按钮进行配置。"
                        )
                        kafka_publish_btn.setChecked(False)
                        return
                    try:
                        rda.dec.set_kafka_config(kafka_config)
                        kafka_publish_btn.setText("停止发布")
                        self._status_label.setText(f"Kafka发布已启用: {kafka_config.topic}")
                    except Exception as e:
                        QtWidgets.QMessageBox.warning(self, "警告", f"启用Kafka发布失败: {e}")
                        kafka_publish_btn.setChecked(False)
                else:
                    try:
                        rda.dec.set_kafka_config(None)
                        kafka_publish_btn.setText("Kafka发布")
                        self._status_label.setText("Kafka发布已禁用")
                    except Exception as e:
                        print(f"[WARNING] 禁用Kafka发布失败: {e}")
            
            kafka_publish_btn.toggled.connect(on_kafka_publish_toggled)
            
            # 初始化Kafka按钮状态
            if config.get('kafka_config') is not None:
                kafka_publish_btn.setChecked(True)
                kafka_publish_btn.setText("停止发布")
            else:
                kafka_publish_btn.setChecked(False)
                kafka_publish_btn.setText("Kafka发布")
            
            # 保存设备信息（参考devmanager.py）
            self._devices[device_key] = {
                'rda': rda,
                'panel': panel,
                'config': config,
                'widget': device_widget,
                'gain_spin': gain_spin,
                'spacing_spin': spacing_spin,
                'filter_combo': filter_combo,
                'low_cut_spin': low_cut_spin,
                'high_cut_spin': high_cut_spin,
                'start_stop_btn': start_stop_btn,
                'kafka_publish_btn': kafka_publish_btn,
                'kafka_config_btn': kafka_config_btn,
                'is_collecting': False,  # 初始状态为未采集
            }
            
            # 添加到布局
            self._device_layout.insertWidget(0, device_widget)
            self._apply_device_stretch()
            
            self._status_label.setText(f"已添加设备: {title}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"添加设备失败: {e}")
            import traceback
            traceback.print_exc()

    def _on_add_eye_clicked(self) -> None:
        """添加眼动仪设备。"""
        device = EyeTrackerDevice("眼动仪", on_remove=self._remove_eye_device, parent=self)
        self._eye_devices.append(device)
        # 放置在滚动区域顶部
        self._device_layout.insertWidget(0, device)
        self._status_label.setText(f"已添加眼动仪（共 {len(self._eye_devices)} 个）")

    def _remove_eye_device(self, device: EyeTrackerDevice) -> None:
        if device in self._eye_devices:
            try:
                device.shutdown()
            except Exception:
                pass
            self._eye_devices.remove(device)
            self._device_layout.removeWidget(device)
            device.deleteLater()
            self._apply_device_stretch()
        if not self._eye_devices:
            self._status_label.setText("就绪")
        else:
            self._status_label.setText(f"已添加眼动仪（共 {len(self._eye_devices)} 个）")

    def _remove_device(self, device_key: str) -> None:
        """移除设备。"""
        if device_key not in self._devices:
            return
        
        device_info = self._devices[device_key]
        rda = device_info.get('rda')
        
        # 先停止数据流，清理回调（防止回调访问已删除的控件）
        try:
            if rda is not None:
                rda.writeCmd('stop')  # 停止采集
                rda.close()  # 关闭设备
        except Exception as e:
            print(f"[WARNING] 停止设备流失败: {e}")
        
        # 如果正在采集，先停止（在停止数据流之后）
        try:
            if device_info.get('is_collecting', False):
                start_stop_btn = device_info.get('start_stop_btn')
                if start_stop_btn is not None:
                    start_stop_btn.setChecked(False)
        except RuntimeError:
            # 控件可能已被删除，忽略
            pass
        
        # 释放绘图面板资源
        try:
            panel = device_info.get('panel')
            if panel is not None:
                panel.release()
        except Exception as e:
            print(f"[WARNING] 释放绘图面板失败: {e}")
        
        # 移除UI
        widget = device_info['widget']
        self._device_layout.removeWidget(widget)
        widget.deleteLater()
        self._apply_device_stretch()
        
        del self._devices[device_key]
        
        if not self._devices:
            self._status_label.setText("就绪")
        else:
            self._status_label.setText(f"已连接 {len(self._devices)} 个设备")
    
    def _apply_filter(
        self,
        data: np.ndarray,
        mode: str,
        sample_rate: float,
        low_cut: float,
        high_cut: float,
    ) -> np.ndarray:
        """应用滤波（参考butterfilter.py的实现方式）。"""
        if mode == "none" or data.size == 0:
            return data
        
        # 参考butterfilter.py：需要足够的缓存数据才能滤波
        # padlen = max(len(a), len(b)) - 1，对于order=4的滤波器，padlen约为27
        # 为了安全，我们需要至少 padlen * 3 的数据长度
        # 参考代码使用 padL = int(srate)，即1秒的数据作为缓存
        padL = int(sample_rate)  # 1秒的数据作为最小缓存
        r, c = data.shape
        
        # 如果数据长度不足，无法滤波，返回原始数据
        if c < padL:
            return data
        
        nyquist = sample_rate / 2.0
        order = 2  # 参考butterfilter.py使用order=2
        
        try:
            if mode == "highpass":
                cutoff = max(0.1, min(low_cut, nyquist - 0.1))
                normalized = cutoff / nyquist
                b, a = butter(order, normalized, btype="highpass")
                # 对整个数据缓存进行滤波
                filtered = filtfilt(b, a, data, axis=1)
                # 只返回最后c个样本（新数据的长度）
                return filtered[:, -c:]
            
            if mode == "bandpass":
                low = max(0.1, min(low_cut, nyquist - 0.1))
                high = max(low + 0.1, min(high_cut, nyquist - 0.1))
                normalized = [low / nyquist, high / nyquist]
                b, a = butter(order, normalized, btype="bandpass")
                # 对整个数据缓存进行滤波
                filtered = filtfilt(b, a, data, axis=1)
                # 只返回最后c个样本（新数据的长度）
                return filtered[:, -c:]
        except (ValueError, np.linalg.LinAlgError) as e:
            # 滤波失败（数据长度不足、数值不稳定等），返回原始数据
            print(f"[WARNING] 滤波失败: {e}, 返回原始数据")
            return data
        
        return data
    
    def _on_save_data_toggled(self, checked: bool) -> None:
        """数据保存按钮切换事件（保存为本地 .dat 文件）。"""
        if checked:
            # 检查是否有设备正在采集
            shm = None
            collecting_devices = [
                key for key, info in self._devices.items()
                if info.get('is_collecting', False)
            ]
            eye_collecting = any(
                getattr(dev._panel, "_tracker", None) is not None and dev._panel._start_btn.isChecked()
                for dev in getattr(self, "_eye_devices", [])
            )
            if not collecting_devices and not eye_collecting:
                QtWidgets.QMessageBox.warning(
                    self,
                    "警告",
                    "没有设备正在采集数据！\n请先点击设备的\"开始采集\"按钮。"
                )
                self._save_data_btn.setChecked(False)
                return

            # 选择保存文件夹（支持自动模式跳过弹窗）
            if self._auto_save_dir:
                save_dir = str(self._auto_save_dir)
            else:
                save_dir = QtWidgets.QFileDialog.getExistingDirectory(
                    self,
                    "选择数据保存文件夹",
                    str(Path.home() / "mpscap_data")
                )
                if not save_dir:
                    self._save_data_btn.setChecked(False)
                    return
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            
            self._save_dir = save_dir
            self._emg_chs = 0
            self._acc_chs = 0
            self._glove_chs = 0
            self._emg_dat_path = None
            self._et_dat_path = None
            self._et_file_handle = None
            self._et_last_t_map = {dev: -1.0 for dev in self._eye_devices}
            self._et_pending = {dev: deque() for dev in self._eye_devices}
            self._et_buf_x = []
            self._et_buf_y = []
            self._et_srate = 33.0
            self._trigger_log = []
            self._trigger_dat_path = None
            try:
                shm = CreateShm(master=False)
                self._emg_last_idx = shm.getvalue('curdataindex')
                self._emg_srate = float(shm.getvalue('srate') or 0.0)
                self._emg_chs = int(shm.getvalue('emgchs') or 0)
                self._acc_chs = int(shm.getvalue('accchs') or 0)
                self._glove_chs = int(shm.getvalue('glovechs') or 0)
            except Exception:
                shm = None
                self._emg_last_idx = 0
                self._emg_srate = 0.0
                self._emg_chs = self._acc_chs = self._glove_chs = 0

            self._et_srate = 33.0
            self._save_start_ts = time.time()
            self._save_ts_tag = time.strftime("%Y%m%d_%H%M%S")
            self._update_panels_time_origin(self._save_start_ts)
            # 设置 .dat 保存路径并通知采集端开始落盘
            ts = self._save_ts_tag
            emg_dat = Path(save_dir) / f"EMG_{ts}.dat"
            if collecting_devices:
                if shm is None:
                    QtWidgets.QMessageBox.warning(self, "提示", "未连接共享内存，跳过肌电/IMU/手套数据保存。")
                else:
                    try:
                        shm.setPath(str(emg_dat))
                        shm.setvalue('savedata', 1)
                        self._emg_dat_path = emg_dat
                    except Exception as e:
                        QtWidgets.QMessageBox.critical(self, "保存失败", f"设置保存路径失败: {e}")
                        self._save_data_btn.setChecked(False)
                        return

            # 眼动单独以 .dat 方式保存（2 导联 X,Y）
            if eye_collecting and self._eye_devices:
                et_dat = Path(save_dir) / f"ET_{ts}.dat"
                try:
                    self._et_file_handle = open(et_dat, "wb")
                    header = np.array([7, 2, 3, int(self._et_srate), 2, 0, 0], dtype=np.int32)
                    self._et_file_handle.write(header.tobytes())
                    self._et_dat_path = et_dat
                except Exception as e:
                    self._et_file_handle = None
                    QtWidgets.QMessageBox.warning(self, "眼动保存失败", f"打开眼动 .dat 文件失败：{e}")

            self._saving = True
            self._storage_mode = "file"
            self._save_data_btn.setText("停止保存")
            self._save_data_btn.setStyleSheet("""
                QPushButton:checked {
                    background-color: rgb(223, 4, 4);
                    color: white;
                }
            """)
            self._status_label.setText(f"正在保存数据到: {Path(save_dir).name}")
            if eye_collecting and self._eye_devices:
                self._save_timer.start(100)
            # 清除自动保存标记（仅本次）
            self._auto_save_dir = None
        else:
            # 停止保存
            self._saving = False
            try:
                shm = CreateShm(master=False)
                shm.setvalue('savedata', 0)
            except Exception:
                pass
            self._save_timer.stop()
            # 写出残余 trigger
            try:
                self._flush_trigger_log()
            except Exception:
                pass
            # 关闭眼动文件并写出残留
            try:
                if self._et_file_handle and (self._et_buf_x or self._et_buf_y):
                    xs = np.array(self._et_buf_x, dtype=np.float64)
                    ys = np.array(self._et_buf_y, dtype=np.float64)
                    arr = np.vstack([xs, ys]).astype(np.float64).T
                    self._et_file_handle.write(arr.tobytes())
                if self._et_file_handle:
                    self._et_file_handle.close()
            except Exception:
                pass
            self._et_file_handle = None
            self._save_dir = None
            self._save_ts_tag = None
            self._emg_dat_path = None
            self._et_dat_path = None
            self._trigger_dat_path = None
            self._trigger_log = []
            self._et_last_t_map = {}
            self._et_pending = {}
            self._et_buf_x = []
            self._et_buf_y = []
            self._save_start_ts = 0.0
            self._update_panels_time_origin(None)
            self._reset_trigger_display()
            
            self._save_data_btn.setText("开始保存数据")
            self._save_data_btn.setStyleSheet("")
            self._status_label.setText("数据保存已停止")
    
    # EDF 相关逻辑已移除，使用设备端 .dat 写入
    def _save_data(self) -> None:
        """周期写入眼动增量到 ET_*.dat。"""
        if not self._saving or self._et_file_handle is None:
            return
        # 记录眼动最新点（增量方式，保证不丢帧）
        for dev in self._eye_devices:
            hist = dev._panel._history
            last_t = self._et_last_t_map.get(dev, -1.0)
            new_samples = [p for p in hist if p[0] > last_t]
            if new_samples:
                q = self._et_pending.setdefault(dev, deque())
                for _, x, y in new_samples:
                    q.append((x, y))
                self._et_last_t_map[dev] = new_samples[-1][0]

        # 写入所有待写入的样本
        try:
            for dev in self._eye_devices:
                q = self._et_pending.get(dev)
                while q:
                    x, y = q.popleft()
                    self._et_buf_x.append(x)
                    self._et_buf_y.append(y)
            if self._et_buf_x:
                xs = np.array(self._et_buf_x, dtype=np.float64)
                ys = np.array(self._et_buf_y, dtype=np.float64)
                arr = np.vstack([xs, ys]).astype(np.float64).T
                self._et_file_handle.write(arr.tobytes())
                self._et_buf_x = []
                self._et_buf_y = []
        except Exception as e:
            print(f"[WARNING] 写入眼动 .dat 失败: {e}")

    def _poll_trigger(self) -> None:
        """轮询共享内存触发值，显示并记录 trigger + 时间（秒）。"""
        trig_val = 0
        now_s = time.time()
        try:
            shm = CreateShm(master=False)
            trig_val = int(shm.getvalue('includetrigger') or 0)
        except Exception:
            pass

        udp_ts = None
        if trig_val == 0 and self._trigger_udp_queue:
            try:
                while self._trigger_udp_queue:
                    trig_val, udp_ts = self._trigger_udp_queue.popleft()
            except Exception:
                trig_val = 0
                udp_ts = None

        if trig_val == 0:
            return

        if udp_ts is not None:
            now_s = udp_ts

        t_rel = None
        if self._save_start_ts:
            t_rel = now_s - self._save_start_ts

        # 刷新显示（0.5s 后自动恢复为无）
        if t_rel is not None:
            self._trigger_display.setText(f"Trigger: {trig_val} | Time(s): {t_rel:.3f}")
        else:
            self._trigger_display.setText(f"Trigger: {trig_val} | Time(s): -")
        self._last_trigger_ts = trig_val
        self._last_trigger_seen_time = now_s

        try:
            if t_rel is not None:
                self._trigger_line.setValue(t_rel)
            self._trigger_line.setVisible(True)
            if self._trigger_label is not None and t_rel is not None:
                self._trigger_label.setPlainText(str(trig_val))
                self._trigger_label.setPos(self._trigger_line.value(), 0)
                self._trigger_label.setVisible(True)
        except Exception:
            pass

        # 触发文字与线条：2s 后复位；线条持续显示，随时间轴左移直至滑出视窗
        self._trigger_reset_timer.stop()
        self._trigger_reset_timer.start(2000)

        # 记录触发日志（同值也追加，确保不丢）
        if self._saving and t_rel is not None:
            self._trigger_log.append((float(trig_val), float(t_rel)))
            if self._trigger_log and len(self._trigger_log) % 100 == 0:
                self._flush_trigger_log()

    def _flush_trigger_log(self) -> None:
        """将触发日志写盘（仅在保存开启时使用）。"""
        if not (self._saving and self._save_dir and self._trigger_log):
            return
        if self._trigger_dat_path is None and self._save_ts_tag:
            self._trigger_dat_path = Path(self._save_dir) / f"{self._trigger_file_stem()}.dat"
        if self._trigger_dat_path is None:
            return
        try:
            import numpy as np
            arr = np.array(self._trigger_log, dtype=np.float64)
            with open(self._trigger_dat_path, "ab") as f:
                f.write(arr.tobytes())
            self._trigger_log = []
        except Exception as e:
            print(f"[WARNING] 写入 trigger 数据失败: {e}")

    def _reset_trigger_display(self) -> None:
        """恢复触发显示为默认状态并隐藏标记线。"""
        try:
            self._trigger_display.setText("Trigger: - | Time(s): -")
            if self._trigger_line is not None:
                self._trigger_line.setVisible(False)
            if self._trigger_label is not None:
                self._trigger_label.setVisible(False)
        except Exception:
            pass

    def _update_panels_time_origin(self, t0: Optional[float]) -> None:
        """将时间起点同步给所有波形面板。"""
        for info in self._devices.values():
            panel = info.get('panel')
            if panel is not None:
                try:
                    panel.set_time_origin(t0)
                except Exception:
                    pass

    def _trigger_file_stem(self) -> str:
        """返回触发文件名（不含扩展名），包含范式名称前缀。"""
        base = self._current_paradigm_name or "TRIGGER"
        safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in base).strip("_")
        if not safe:
            safe = "TRIGGER"
        return f"{safe}_{self._save_ts_tag}" if self._save_ts_tag else safe

    def _update_beijing_time(self) -> None:
        """更新时间标签为北京时间（UTC+8）。"""
        try:
            now = datetime.now(timezone(timedelta(hours=8)))
            self._beijing_time_label.setText(now.strftime("北京时间: %Y-%m-%d %H:%M:%S"))
        except Exception:
            pass

    def _start_trigger_udp_listener(self) -> None:
        """启动本地 UDP 监听作为触发备选通道（127.0.0.1:port）。"""
        if self._trigger_udp_thread is not None:
            return
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("127.0.0.1", self._trigger_udp_port))
            sock.settimeout(1.0)
            self._trigger_udp_socket = sock
        except Exception as e:
            print(f"[Trigger UDP] 启动失败，继续使用共享内存触发: {e}")
            self._trigger_udp_socket = None
            return

        def _loop() -> None:
            while not self._trigger_udp_stop.is_set():
                try:
                    data, _ = sock.recvfrom(1024)
                except socket.timeout:
                    continue
                except OSError:
                    break
                try:
                    val = int((data or b"0").decode().strip())
                except Exception:
                    continue
                if val == 0:
                    continue
                try:
                    self._trigger_udp_queue.append((val, time.time()))
                except Exception:
                    pass
            try:
                sock.close()
            except Exception:
                pass

        self._trigger_udp_thread = threading.Thread(target=_loop, daemon=True)
        self._trigger_udp_thread.start()

    def _stop_trigger_udp_listener(self) -> None:
        """停止 UDP 触发监听线程。"""
        try:
            self._trigger_udp_stop.set()
        except Exception:
            pass
        try:
            if self._trigger_udp_socket is not None:
                self._trigger_udp_socket.close()
        except Exception:
            pass
        self._trigger_udp_socket = None
        self._trigger_udp_thread = None

    def start_saving_auto(self, save_dir: Optional[str] = None) -> bool:
        """
        外部调用：自动开始保存数据。
        - save_dir 为空则使用 ~/mpscap_data
        - 跳过目录选择对话框
        返回是否成功进入保存状态。
        """
        if self._saving:
            return True
        auto_dir = Path(save_dir) if save_dir else (Path.home() / "mpscap_data")
        auto_dir.mkdir(parents=True, exist_ok=True)
        self._auto_save_dir = auto_dir
        # 触发按钮，进入保存流程（让 toggled 信号正常触发）
        self._save_data_btn.setChecked(True)
        # 如果保存失败，checked 会被复位
        return self._saving

    def _cleanup_all(self) -> None:
        """统一清理资源，供closeEvent/shutdown调用。"""
        # 停止数据保存
        if self._saving:
            self._save_data_btn.setChecked(False)

        # 移除所有设备
        for device_key in list(self._devices.keys()):
            self._remove_device(device_key)

        # 关闭眼动设备
        for dev in list(self._eye_devices):
            self._remove_eye_device(dev)

        self._stop_trigger_udp_listener()

    def set_paradigm_name(self, name: Optional[str]) -> None:
        """供范式页传入当前范式名，用于触发文件命名。"""
        self._current_paradigm_name = name or None

    def closeEvent(self, event) -> None:  # pragma: no cover - UI 行为
        """关闭事件，清理所有设备。"""
        self._cleanup_all()
        super().closeEvent(event)

    def shutdown(self) -> None:
        """提供给外部主动调用的清理接口。"""
        self._cleanup_all()

    def _apply_device_stretch(self) -> None:
        """按设备类型权重分配纵向空间。"""
        count = self._device_layout.count()
        for i in range(count):
            item = self._device_layout.itemAt(i)
            if item is None:
                continue
            w = item.widget()
            if w is None:
                continue
            dtype = w.property("device_type")
            weight = self._device_type_weight.get(dtype, 1)
            w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            self._device_layout.setStretch(i, weight)


class KafkaConfigDialog(QtWidgets.QDialog):
    """Kafka配置对话框。"""
    
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Kafka配置")
        self.resize(500, 300)
        self._result: Optional[KafkaConfig] = None
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # 说明
        info_label = QtWidgets.QLabel("配置Kafka服务器以启用数据发布功能")
        info_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(info_label)
        
        form = QtWidgets.QFormLayout()
        
        # Kafka服务器地址
        self._bootstrap_servers_edit = QtWidgets.QLineEdit()
        self._bootstrap_servers_edit.setText("localhost:9092")
        self._bootstrap_servers_edit.setPlaceholderText("例如: localhost:9092")
        form.addRow("服务器地址:", self._bootstrap_servers_edit)
        
        # Topic名称
        self._topic_edit = QtWidgets.QLineEdit()
        self._topic_edit.setText("emg_data")
        self._topic_edit.setPlaceholderText("例如: emg_data")
        form.addRow("Topic名称:", self._topic_edit)
        
        # 批次大小
        self._batch_size_spin = QtWidgets.QSpinBox()
        self._batch_size_spin.setRange(1, 1000)
        self._batch_size_spin.setValue(50)
        self._batch_size_spin.setToolTip("每个数据包包含的样本数")
        form.addRow("批次大小:", self._batch_size_spin)
        
        # 端口说明
        port_info = QtWidgets.QLabel("注意: Kafka默认端口为9092，Zookeeper端口为2181")
        port_info.setStyleSheet("color: gray; font-size: 10px;")
        form.addRow("", port_info)
        
        layout.addLayout(form)
        layout.addStretch()
        
        # 按钮
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def accept(self) -> None:
        """确认配置。"""
        bootstrap_servers = self._bootstrap_servers_edit.text().strip()
        topic = self._topic_edit.text().strip()
        batch_size = self._batch_size_spin.value()
        
        if not bootstrap_servers:
            QtWidgets.QMessageBox.warning(self, "错误", "请输入Kafka服务器地址")
            return
        
        if not topic:
            QtWidgets.QMessageBox.warning(self, "错误", "请输入Topic名称")
            return
        
        try:
            self._result = KafkaConfig(
                bootstrap_servers=bootstrap_servers,
                topic=topic,
                batch_size=batch_size,
            )
            super().accept()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"创建Kafka配置失败: {e}")
    
    def get_config(self) -> Optional[KafkaConfig]:
        """获取配置结果。"""
        return self._result

    def _center_on_parent(self) -> None:
        """弹窗居中显示（相对父窗口或屏幕）。"""
        try:
            parent = self.parentWidget()
            if parent is not None:
                geo = parent.frameGeometry()
                center = geo.center()
            else:
                screen = QtWidgets.QApplication.primaryScreen()
                center = screen.availableGeometry().center() if screen else QtCore.QPoint(0, 0)
            fg = self.frameGeometry()
            fg.moveCenter(center)
            self.move(fg.topLeft())
        except Exception:
            pass

