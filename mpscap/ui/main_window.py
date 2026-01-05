"""
PyQt5 主窗口定义。
"""

from __future__ import annotations

from PyQt5 import QtWidgets

from mpscap.core.data_pipeline.mqtt_gateway import MQTTConfig, MQTTGateway
from mpscap.core.data_pipeline.realtime import SignalStreamHub
from mpscap.ui.pages.data_acquisition import DataAcquisitionPage
from mpscap.ui.pages.offline_converter import OfflineConverterPage
from mpscap.ui.pages.data_processing import DataProcessingPage
from mpscap.ui.pages.task_execution import TaskExecutionPage


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MPSCAP")
        self._geom_inited = False
        self._apply_initial_geometry()
        self._mqtt_gateway = MQTTGateway(MQTTConfig())
        self._stream_hub = SignalStreamHub(self._mqtt_gateway)
        self._init_ui()

    def _init_ui(self) -> None:
        self.tabs = QtWidgets.QTabWidget()
        dashboard = DataAcquisitionPage()
        # dashboard.bind_stream(self._stream_hub)
        self.tabs.addTab(dashboard, "在线采集")
        offline_converter = OfflineConverterPage()
        self.tabs.addTab(offline_converter, "离线转换")
        data_processing = DataProcessingPage()
        self.tabs.addTab(data_processing, "数据处理")
        task_execution = TaskExecutionPage(dashboard)
        self.tabs.addTab(task_execution, "任务执行")
        self.tabs.addTab(QtWidgets.QWidget(), "结果反馈")
        self.setCentralWidget(self.tabs)

    def closeEvent(self, event):  # pragma: no cover - UI 行为
        dashboard = self.tabs.widget(0)
        if isinstance(dashboard, DataAcquisitionPage):
            dashboard.shutdown()
        super().closeEvent(event)

    def showEvent(self, event):  # pragma: no cover
        super().showEvent(event)
        # 再次调整以防首次时屏幕尺寸未获取
        if not self._geom_inited:
            self._apply_initial_geometry()
            self._geom_inited = True

    def _center_on_screen(self) -> None:
        try:
            screen = QtWidgets.QApplication.primaryScreen()
            center = screen.availableGeometry().center() if screen else None
            if center is None:
                return
            fg = self.frameGeometry()
            fg.moveCenter(center)
            self.move(fg.topLeft())
        except Exception:
            pass

    def _apply_initial_geometry(self) -> None:
        """根据屏幕可用区域设置大小并居中，避免超出分辨率。"""
        try:
            screen = QtWidgets.QApplication.primaryScreen()
            if screen:
                avail = screen.availableGeometry()
                width = min(1920, avail.width())
                height = min(1080, avail.height())
                self.resize(width, height)
            else:
                self.resize(1920, 1080)
            self._center_on_screen()
        except Exception:
            self.resize(1920, 1080)

