"""
空域可视化组件（IMU姿态、手指弯曲和压力）。
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from scipy.spatial.transform import Rotation as R


class IMUPoseWidget(QtWidgets.QWidget):
    """IMU姿态可视化组件（3D坐标系投影）。"""
    
    def __init__(self, title: str = "IMU姿态", parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.title = title
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # 绘图区域 - 充满整个空间
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Y')
        self.plot_widget.setLabel('bottom', 'X')
        # 增大显示范围，使模型更大更清晰
        self.plot_widget.setXRange(-2.0, 2.0)
        self.plot_widget.setYRange(-2.0, 2.0)
        self.plot_widget.setAspectLocked(True)
        layout.addWidget(self.plot_widget, stretch=1)  # 充满剩余空间
        
        # 信息标签（放在底部，占用最小空间）
        self.info_label = QtWidgets.QLabel("等待数据...")
        self.info_label.setStyleSheet("font-size: 9px;")
        layout.addWidget(self.info_label)
        
        # 初始化四元数（单位四元数）
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        
        # 静态检测相关
        self._acc_history = []
        self._gyro_history = []
        self._history_size = 10
        self._static_count = 0
        self._static_threshold_count = 5
        self._is_static = False
        
        # 初始化模型
        self._init_model()
    
    def _init_model(self):
        """
        初始化3D模型（坐标系和手部模型）。
        根据实际设备图片：
        - IMU1位于腕部背面（手腕设备上）
        - IMU2位于手背中心
        """
        # 定义坐标轴（X红色，Y绿色，Z蓝色）- 增大尺寸
        self.axis_length = 1.0  # 从0.6增大到1.0
        self.origin = np.array([0.0, 0.0, 0.0])
        
        # 缩放因子，使整个手部模型更大
        scale_factor = 1.5

        # 手部模型（参考采集程序/imu.py，根据实际手部形状调整）
        # 坐标系：X轴指向手指方向（向前），Y轴指向拇指方向（向右），Z轴垂直向上
        # 原点位于手腕和手背的交界处
        self.hand_model = {
            "palm": [  # 手背（四边形，IMU2位于中心）
                (-0.25 * scale_factor, -0.15 * scale_factor, 0), 
                (0.25 * scale_factor, -0.15 * scale_factor, 0), 
                (0.25 * scale_factor, 0.15 * scale_factor, 0), 
                (-0.25 * scale_factor, 0.15 * scale_factor, 0)
            ],
            "thumb": [  # 拇指（3段，从手背右侧延伸）
                (0.0, 0.15 * scale_factor, 0), 
                (-0.05 * scale_factor, 0.2 * scale_factor, 0), 
                (-0.1 * scale_factor, 0.25 * scale_factor, 0), 
                (-0.15 * scale_factor, 0.3 * scale_factor, 0)
            ],
            "index": [  # 食指（3段，从手背前侧延伸）
                (0.25 * scale_factor, -0.05 * scale_factor, 0), 
                (0.35 * scale_factor, 0.0, 0), 
                (0.45 * scale_factor, 0.05 * scale_factor, 0), 
                (0.55 * scale_factor, 0.1 * scale_factor, 0)
            ],
            "middle": [  # 中指（3段，从手背前侧延伸）
                (0.25 * scale_factor, 0.0, 0), 
                (0.35 * scale_factor, 0.05 * scale_factor, 0), 
                (0.45 * scale_factor, 0.1 * scale_factor, 0), 
                (0.55 * scale_factor, 0.15 * scale_factor, 0)
            ],
            "ring": [  # 无名指（3段，从手背前侧延伸）
                (0.25 * scale_factor, 0.05 * scale_factor, 0), 
                (0.3 * scale_factor, 0.1 * scale_factor, 0), 
                (0.35 * scale_factor, 0.15 * scale_factor, 0), 
                (0.4 * scale_factor, 0.2 * scale_factor, 0)
            ],
            "pinky": [  # 小指（3段，从手背前侧延伸）
                (0.25 * scale_factor, 0.1 * scale_factor, 0), 
                (0.25 * scale_factor, 0.15 * scale_factor, 0), 
                (0.25 * scale_factor, 0.2 * scale_factor, 0), 
                (0.25 * scale_factor, 0.25 * scale_factor, 0)
            ],
            "wrist": [  # 手腕（矩形，IMU1位于此处）
                (-0.5 * scale_factor, -0.12 * scale_factor, 0), 
                (-0.25 * scale_factor, -0.12 * scale_factor, 0), 
                (-0.25 * scale_factor, 0.12 * scale_factor, 0), 
                (-0.5 * scale_factor, 0.12 * scale_factor, 0)
            ]
        }

        # IMU位置标记（根据图片中的实际位置）- 同样放大
        self.imu1_pos = np.array([-0.375 * scale_factor, 0.0, 0.0])  # 腕部背面中心（手腕IMU，在黑色腕带上）
        self.imu2_pos = np.array([0.0, 0.0, 0.0])      # 手背中心（手背IMU）
    
    def _draw_axes(self):
        """绘制坐标系和手部模型。"""
        # 清除之前的绘制
        self.plot_widget.clear()
        self.axes_items = []
        
        # 获取旋转矩阵
        rot_matrix = R.from_quat(self.quaternion).as_matrix()
        
        # 坐标变换：将手部坐标系映射到屏幕坐标系
        # 手部坐标系：X=手指方向（向前），Y=拇指方向（向右），Z=向上
        # 屏幕坐标系：X=向右，Y=向上
        # 映射关系：手部X轴（手指方向）-> 屏幕X轴（向右），手部Z轴（向上）-> 屏幕Y轴（向上）
        # 手部Y轴（拇指方向）作为深度信息，影响显示但不直接映射
        
        # 提取旋转矩阵的列向量（手部坐标系的基向量）
        hand_x = rot_matrix[:, 0]  # 手部X轴（手指方向）
        hand_y = rot_matrix[:, 1]  # 手部Y轴（拇指方向）
        hand_z = rot_matrix[:, 2]  # 手部Z轴（向上）
        
        # 映射到屏幕坐标系：屏幕X = 手部X（手指方向），屏幕Y = 手部Z（向上方向）
        # 手部Y轴（拇指方向）作为深度信息
        screen_x_axis = hand_x * self.axis_length  # 手指方向 -> 屏幕右侧
        screen_y_axis = hand_z * self.axis_length  # 向上方向 -> 屏幕上方
        
        # 绘制坐标轴（在屏幕上的投影）
        # X轴（红色）- 手指方向（屏幕右侧）
        x_line = pg.PlotDataItem(
            [self.origin[0], screen_x_axis[0]], [self.origin[1], screen_x_axis[1]],
            pen=pg.mkPen(color=(255, 0, 0), width=3)
        )
        self.plot_widget.addItem(x_line)
        self.axes_items.append(x_line)
        
        # Y轴（绿色）- 向上方向（屏幕上方）
        y_line = pg.PlotDataItem(
            [self.origin[0], screen_y_axis[0]], [self.origin[1], screen_y_axis[1]],
            pen=pg.mkPen(color=(0, 255, 0), width=3)
        )
        self.plot_widget.addItem(y_line)
        self.axes_items.append(y_line)
        
        # Z轴（蓝色）- 拇指方向（深度方向，用虚线表示）
        # 由于是2D投影，Z轴（拇指方向）用虚线表示，长度根据深度调整
        z_depth = np.dot(hand_y, np.array([0, 0, 1]))  # 拇指方向与屏幕法线的夹角
        z_scale = max(0.3, abs(z_depth))  # 深度缩放因子
        z_axis_screen = hand_y * self.axis_length * z_scale
        z_line = pg.PlotDataItem(
            [self.origin[0], z_axis_screen[0]], [self.origin[1], z_axis_screen[1]],
            pen=pg.mkPen(color=(0, 0, 255), width=2, style=QtCore.Qt.DashLine)
        )
        self.plot_widget.addItem(z_line)
        self.axes_items.append(z_line)
        
        # 绘制手部模型（使用坐标变换）
        self._draw_hand_model(rot_matrix)
    
    def _draw_hand_model(self, rot_matrix: np.ndarray):
        """绘制手部模型（参考采集程序/imu.py）。"""
        # 颜色定义
        colors = {
            "palm": QtGui.QColor(255, 200, 150),  # 手背颜色（浅橙色）
            "thumb": QtGui.QColor(255, 0, 0),     # 拇指颜色（红色）
            "index": QtGui.QColor(0, 255, 0),     # 食指颜色（绿色）
            "middle": QtGui.QColor(0, 0, 255),    # 中指颜色（蓝色）
            "ring": QtGui.QColor(255, 255, 0),    # 无名指颜色（黄色）
            "pinky": QtGui.QColor(255, 0, 255),   # 小指颜色（洋红色）
            "wrist": QtGui.QColor(100, 100, 100), # 手腕颜色（灰色）
        }

        # 坐标变换：手部坐标系 -> 屏幕坐标系
        # 手部坐标系：X=手指方向（向前），Y=拇指方向（向右），Z=向上
        # 屏幕坐标系：X=向右，Y=向上
        # 映射：屏幕X = 手部X（手指方向），屏幕Y = 手部Z（向上方向）
        # 手部Y轴（拇指方向）作为深度信息

        # 绘制手部各部分
        for part_name, points in self.hand_model.items():
            screen_points = []
            for point in points:
                point_3d = np.array(point)
                # 应用旋转矩阵
                point_rotated = rot_matrix @ point_3d
                # 映射到屏幕坐标系：屏幕X = 手部X，屏幕Y = 手部Z
                screen_x = point_rotated[0]  # 手指方向 -> 屏幕右侧
                screen_y = point_rotated[2]  # 向上方向 -> 屏幕上方
                screen_points.append([screen_x, screen_y])

            if part_name in ["palm", "wrist"]:
                # 绘制多边形
                if len(screen_points) >= 3:
                    x_coords = [p[0] for p in screen_points] + [screen_points[0][0]]
                    y_coords = [p[1] for p in screen_points] + [screen_points[0][1]]
                    polygon = pg.PlotDataItem(
                        x_coords, y_coords,
                        pen=pg.mkPen(color=colors[part_name], width=2),
                        brush=pg.mkBrush(color=colors[part_name], alpha=100 if part_name == "palm" else 150)
                    )
                    self.plot_widget.addItem(polygon)
                    self.axes_items.append(polygon)
            else:
                # 手指（线段）
                for i in range(len(points) - 1):
                    p1_3d = np.array(points[i])
                    p2_3d = np.array(points[i + 1])
                    p1_rotated = rot_matrix @ p1_3d
                    p2_rotated = rot_matrix @ p2_3d
                    
                    # 映射到屏幕坐标系：屏幕X = 手部X，屏幕Y = 手部Z
                    screen_x1 = p1_rotated[0]
                    screen_y1 = p1_rotated[2]
                    screen_x2 = p2_rotated[0]
                    screen_y2 = p2_rotated[2]

                    line = pg.PlotDataItem(
                        [screen_x1, screen_x2],
                        [screen_y1, screen_y2],
                        pen=pg.mkPen(color=colors[part_name], width=2)
                    )
                    self.plot_widget.addItem(line)
                    self.axes_items.append(line)

        # 绘制IMU位置标记
        # IMU1（腕部背面）
        imu1_rotated = rot_matrix @ self.imu1_pos
        imu1_screen_x = imu1_rotated[0]  # 手指方向 -> 屏幕X
        imu1_screen_y = imu1_rotated[2]  # 向上方向 -> 屏幕Y
        imu1_marker = pg.ScatterPlotItem(
            [imu1_screen_x], [imu1_screen_y],
            pen=pg.mkPen(color=QtGui.QColor(255, 0, 0), width=2),
            brush=pg.mkBrush(color=QtGui.QColor(255, 0, 0), alpha=200),
            size=8,
            symbol='o'
        )
        self.plot_widget.addItem(imu1_marker)
        self.axes_items.append(imu1_marker)

        # IMU1标签
        imu1_text = pg.TextItem("IMU1", color=QtGui.QColor(255, 0, 0))
        imu1_text.setPos(imu1_screen_x + 0.1, imu1_screen_y)
        self.plot_widget.addItem(imu1_text)
        self.axes_items.append(imu1_text)

        # IMU2（手背中心）
        imu2_rotated = rot_matrix @ self.imu2_pos
        imu2_screen_x = imu2_rotated[0]  # 手指方向 -> 屏幕X
        imu2_screen_y = imu2_rotated[2]  # 向上方向 -> 屏幕Y
        imu2_marker = pg.ScatterPlotItem(
            [imu2_screen_x], [imu2_screen_y],
            pen=pg.mkPen(color=QtGui.QColor(0, 0, 255), width=2),
            brush=pg.mkBrush(color=QtGui.QColor(0, 0, 255), alpha=200),
            size=8,
            symbol='s'
        )
        self.plot_widget.addItem(imu2_marker)
        self.axes_items.append(imu2_marker)

        # IMU2标签
        imu2_text = pg.TextItem("IMU2", color=QtGui.QColor(0, 0, 255))
        imu2_text.setPos(imu2_screen_x + 0.1, imu2_screen_y)
        self.plot_widget.addItem(imu2_text)
        self.axes_items.append(imu2_text)
    
    def update_pose(self, acc: np.ndarray, gyro: np.ndarray, dt: float):
        """
        更新IMU姿态（使用互补滤波，参考采集程序/imu.py）。
        
        参数:
            acc: 加速度计数据 [ax, ay, az] (m/s²)
            gyro: 陀螺仪数据 [gx, gy, gz] (deg/s)
            dt: 时间间隔（秒）
        """
        # 转换陀螺仪数据从度/秒到弧度/秒
        gyro_rad = np.deg2rad(gyro)
        
        # 静态检测（维护历史数据）
        self._acc_history.append(acc.copy())
        self._gyro_history.append(gyro_rad.copy())
        if len(self._acc_history) > self._history_size:
            self._acc_history.pop(0)
            self._gyro_history.pop(0)
        
        # 计算加速度和角速度的标准差来判断是否静止
        if len(self._acc_history) >= self._history_size:
            acc_std = np.std(self._acc_history, axis=0)
            gyro_std = np.std(self._gyro_history, axis=0)
            
            # 如果加速度和角速度的标准差都很小，认为是静止状态
            acc_motion = np.linalg.norm(acc_std) > 0.5  # 加速度变化阈值
            gyro_motion = np.linalg.norm(gyro_std) > 0.05  # 角速度变化阈值（rad/s）
            
            if not acc_motion and not gyro_motion:
                self._static_count = min(self._static_count + 1, self._static_threshold_count * 2)
                self._is_static = self._static_count >= self._static_threshold_count
            else:
                self._is_static = False
                self._static_count = max(0, self._static_count - 1)
        
        # 互补滤波参数（静止时更信任加速度计，运动时更信任陀螺仪）
        if self._is_static:
            alpha = 0.92  # 静止时更信任加速度计，减少陀螺仪漂移
        else:
            alpha = 0.98  # 运动时更信任陀螺仪
        
        # 加速度计姿态估计（俯仰和横滚）
        # 只有在加速度接近重力时才使用加速度计估计姿态
        acc_magnitude = np.linalg.norm(acc)
        use_accel = 8.0 < acc_magnitude < 11.0  # 加速度接近重力
        
        if use_accel:
            acc_norm = acc / acc_magnitude
            roll = np.arctan2(acc_norm[1], acc_norm[2])
            pitch = np.arctan2(-acc_norm[0], np.sqrt(acc_norm[1]**2 + acc_norm[2]**2 + 1e-6))
            # 保持当前的Yaw角（加速度计无法估计Yaw）
            # 使用'xyz'顺序获取当前欧拉角：[roll, pitch, yaw]
            current_euler = R.from_quat(self.quaternion).as_euler('xyz')
            # 使用'xyz'顺序创建四元数：[roll, pitch, yaw]
            acc_quat = R.from_euler('xyz', [roll, pitch, current_euler[2]]).as_quat()
        else:
            # 如果加速度不接近重力，使用当前姿态（不更新）
            acc_quat = self.quaternion
        
        # 陀螺仪姿态估计
        try:
            # 使用陀螺仪积分更新姿态
            current_rot = R.from_quat(self.quaternion)
            # 计算角速度向量（rad/s）
            gyro_quat = R.from_rotvec(gyro_rad * dt)
            # 更新姿态
            gyro_updated_quat = (current_rot * gyro_quat).as_quat()
            
            # 互补滤波融合
            if use_accel and self._is_static:
                # 静止时：更信任加速度计，减少陀螺仪漂移
                self.quaternion = alpha * gyro_updated_quat + (1 - alpha) * acc_quat
            elif use_accel:
                # 运动时：正常融合
                self.quaternion = alpha * gyro_updated_quat + (1 - alpha) * acc_quat
            else:
                # 加速度计不可靠：只使用陀螺仪
                self.quaternion = gyro_updated_quat
            
            # 归一化四元数
            self.quaternion = self.quaternion / (np.linalg.norm(self.quaternion) + 1e-6)
        except Exception as e:
            # 如果计算失败，保持当前姿态或使用加速度计估计
            if use_accel:
                self.quaternion = acc_quat
        
        # 更新显示
        self._draw_axes()
        
        # 更新信息标签（使用'xyz'顺序：Roll, Pitch, Yaw）
        try:
            euler = R.from_quat(self.quaternion).as_euler('xyz', degrees=True)
            static_status = "静止" if self._is_static else "运动"
            self.info_label.setText(
                f"Roll: {euler[0]:.1f}°  Pitch: {euler[1]:.1f}°  Yaw: {euler[2]:.1f}° ({static_status})"
            )
        except:
            self.info_label.setText("姿态计算中...")


class FingerBendWidget(QtWidgets.QWidget):
    """手指弯曲可视化组件。"""
    
    def __init__(self, title: str = "手指弯曲", parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.title = title
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # 标题
        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title_label)
        
        # 条形图
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', '弯曲程度')
        self.plot_widget.setLabel('bottom', '手指')
        self.plot_widget.setYRange(0, 1.0)
        self.plot_widget.setXRange(-0.5, 4.5)
        layout.addWidget(self.plot_widget)
        
        # 手指名称
        self.finger_names = ['小指', '无名指', '中指', '食指', '拇指']
        self.plot_widget.getAxis('bottom').setTicks([
            [(i, name) for i, name in enumerate(self.finger_names)]
        ])
        
        # 条形图项（使用单个BarGraphItem显示所有条形）
        self.bar_item = None
        self.bend_values = np.zeros(5)
        self._update_bars()
    
    def _update_bars(self):
        """更新条形图显示。"""
        if self.bar_item is not None:
            self.plot_widget.removeItem(self.bar_item)
        
        x_positions = np.arange(5)
        self.bar_item = pg.BarGraphItem(
            x=x_positions, height=self.bend_values, width=0.6,
            brush=pg.mkBrush(color=(100, 150, 255, 200))
        )
        self.plot_widget.addItem(self.bar_item)
    
    def update_bend(self, bend_data: np.ndarray):
        """
        更新手指弯曲数据。
        
        参数:
            bend_data: 弯曲传感器数据，5个值 [小指, 无名指, 中指, 食指, 拇指]
        """
        if len(bend_data) >= 5:
            self.bend_values = np.clip(bend_data[:5], 0, 1.0)
        elif len(bend_data) > 0:
            # 如果数据不足5个，填充零
            self.bend_values = np.zeros(5)
            self.bend_values[:len(bend_data)] = np.clip(bend_data, 0, 1.0)
        else:
            self.bend_values = np.zeros(5)
        
        # 更新条形图
        self._update_bars()


class FingerPressureWidget(QtWidgets.QWidget):
    """手指压力可视化组件。"""
    
    def __init__(self, title: str = "手指压力", parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.title = title
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # 标题
        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title_label)
        
        # 条形图
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', '压力')
        self.plot_widget.setLabel('bottom', '手指')
        self.plot_widget.setYRange(0, 1.0)
        self.plot_widget.setXRange(-0.5, 4.5)
        layout.addWidget(self.plot_widget)
        
        # 手指名称
        self.finger_names = ['小指', '无名指', '中指', '食指', '拇指']
        self.plot_widget.getAxis('bottom').setTicks([
            [(i, name) for i, name in enumerate(self.finger_names)]
        ])
        
        # 条形图项（使用单个BarGraphItem显示所有条形）
        self.bar_item = None
        self.pressure_values = np.zeros(5)
        self._update_bars()
    
    def _update_bars(self):
        """更新条形图显示。"""
        if self.bar_item is not None:
            self.plot_widget.removeItem(self.bar_item)
        
        x_positions = np.arange(5)
        self.bar_item = pg.BarGraphItem(
            x=x_positions, height=self.pressure_values, width=0.6,
            brush=pg.mkBrush(color=(255, 100, 100, 200))
        )
        self.plot_widget.addItem(self.bar_item)
    
    def update_pressure(self, pressure_data: np.ndarray):
        """
        更新手指压力数据。
        
        参数:
            pressure_data: 压力传感器数据，5个值 [小指, 无名指, 中指, 食指, 拇指]
        """
        if len(pressure_data) >= 5:
            self.pressure_values = np.clip(pressure_data[:5], 0, 1.0)
        elif len(pressure_data) > 0:
            # 如果数据不足5个，填充零
            self.pressure_values = np.zeros(5)
            self.pressure_values[:len(pressure_data)] = np.clip(pressure_data, 0, 1.0)
        else:
            self.pressure_values = np.zeros(5)
        
        # 更新条形图
        self._update_bars()


class SpatialVisualizer(QtWidgets.QWidget):
    """空域可视化组件（只显示手部姿态模型）。"""
    
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._init_ui()
        
        # 姿态估计器
        self._wrist_pose_estimator = None
        self._hand_pose_estimator = None
        self._last_update_time = {}
    
    def _init_ui(self):
        """初始化UI。"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # 标题
        title_label = QtWidgets.QLabel("手部姿态模型")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)
        
        # 只使用一个IMU姿态可视化组件，显示手背IMU（包含整个手部模型）
        # 使用手背IMU的姿态，因为手部模型是以手背为中心的
        self.hand_imu_widget = IMUPoseWidget("手部姿态")
        layout.addWidget(self.hand_imu_widget, stretch=1)  # 充满剩余空间
    
    def update_data(
        self,
        imu_wrist: Optional[np.ndarray] = None,  # 6通道: [accx, accy, accz, gyrx, gyry, gyrz]
        imu_hand: Optional[np.ndarray] = None,    # 6通道: [accx, accy, accz, gyrx, gyry, gyrz]
        bend_data: Optional[np.ndarray] = None,  # 5通道: [小指, 无名指, 中指, 食指, 拇指]
        pressure_data: Optional[np.ndarray] = None,  # 5通道: [小指, 无名指, 中指, 食指, 拇指]
    ):
        """
        更新空域可视化数据（只更新手部姿态模型）。
        
        参数:
            imu_wrist: 手腕IMU数据（6通道）- 未使用
            imu_hand: 手背IMU数据（6通道）- 用于显示手部姿态
            bend_data: 手指弯曲数据（5通道）- 未使用
            pressure_data: 手指压力数据（5通道）- 未使用
        """
        current_time = time.time()
        dt = 0.01  # 默认时间间隔
        
        # 只更新手背IMU姿态（手部模型以手背为中心）
        if imu_hand is not None and len(imu_hand) >= 6:
            if 'hand' in self._last_update_time:
                dt = current_time - self._last_update_time['hand']
            self._last_update_time['hand'] = current_time
            
            acc = imu_hand[:3]
            gyro = imu_hand[3:6]
            self.hand_imu_widget.update_pose(acc, gyro, dt)

