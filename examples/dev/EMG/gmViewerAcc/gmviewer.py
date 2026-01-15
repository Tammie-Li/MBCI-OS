#coding:utf-8

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
from gmviewerui import Ui_MainWindow
from dataplot import dataPlot
import pyqtgraph as pg
from devmanager import devManager
from shm import CreateShm
import time

'''主程序入口'''

class gmViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super(gmViewer,self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.shm = CreateShm(master=True)
        self.setWindowTitle("gmViewer by: mrTang email:810899799@qq.com")
        self._screenResize()
        self.devmanager = devManager(self.ui) # 管理设备
        # ## 初始化绘图窗口
        self.plot = dataPlot(self.ui, self.ww) # 管理绘图
        self.pgTimer = pg.QtCore.QTimer()
        self.pgTimer.timeout.connect(self.plot.update_one_frame)
        self.pgTimer.start(10)

    def closeEvent(self, event):
        self.plot.release()
        self.devmanager.release()
        self.shm.release()

    # 由initialize调用
    # 依据显示屏尺寸来初始化界面大小
    def _screenResize(self):
        # 获取显示器相关信息
        desktop = QApplication.desktop()
        # 默认在主显示屏显示
        screen_rect = desktop.screenGeometry(0)
        self.ww = screen_rect.width()
        self.hh = screen_rect.height()
        self.w = int(self.ww*0.8)
        self.h = int(self.hh*0.5)
        self.setGeometry(int((self.ww - self.w)/2), int((self.hh - self.h)/3), self.w, self.h)
 
if __name__ == '__main__':
    import sys
    import multiprocessing
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    a = gmViewer()
    a.show()
    sys.exit(app.exec_())