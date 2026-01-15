#coding:utf-8

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from mymessbox import showMessageBox
from rda1299 import RDA1299
import time

'''设备进程管理'''

class devManager(QtWidgets.QDialog):
    _sig2mesbox = pyqtSignal(str)
    def __init__(self,parentUI,mainsig,config):
        super(devManager, self).__init__()
        self.config = config
        self.mainsig = mainsig
        self.ui = parentUI
        self.RDA = RDA1299(self._sig2mesbox)
        self.ui.startacq_btn.clicked.connect(self.start_acq)
        self.ui.imp_btn.clicked.connect(self.imp_detect)
        self.ui.stop_btn.clicked.connect(self.stop_acq)
        self.ui.device_cmb.clicked.connect(self._updatedevice)
        # self.ui.save_btn.clicked.connect(self.start_save)

        self._sig2mesbox.connect(self.popmesbox)

        self.ui = parentUI
        self.datapath = './data'
        self._updatedevice()

    # def start_save(self):
    #     if self.shm.info[8] == 0:   # 未保存->保存
    #         if self.devprocess is None or not self.devprocess.is_alive():
    #             self._sig2mesbox.emit("信号采集模块未启动！")
    #             return
    #
    #         folder_path = QFileDialog.getSaveFileName(None, "设置保存文件", "./data", "Sub001 (*.dat)")
    #         if len(folder_path[0]) > 0:
    #             self.shm.setPath(folder_path[0])  # 将路径写入共享内存中
    #             self.ui.path_edit.setText(folder_path[0])
    #             # 写入保存标志
    #             self.shm.info[8] = 1
    #             self.ui.save_btn.setText('停止保存')
    #             self.ui.save_btn.setStyleSheet("font: 20px \"微软雅黑\";background-color: rgb(223, 4, 4);")
    #
    #     else:  # 保存->不保存
    #         self.shm.info[8] = 0
    #         self.ui.save_btn.setText('保存数据')
    #         self.ui.save_btn.setStyleSheet("font: 20px \"微软雅黑\";")

    def _updatedevice(self):
        device = self.RDA.getallserial()
        self.ui.device_cmb.clear()
        if len(device)>0:
            self.ui.device_cmb.addItems(device)

    def release(self):
        self.stop_acq()

    def popmesbox(self,strs):
        # if strs[0] == 'b':
        #     self.ui.save_btn.setText('保存数据')
        #     self.ui.save_btn.setStyleSheet("font: 20px \"微软雅黑\";")
        showMessageBox('设备管理器',strs)

    # 按钮事件
    def start_acq(self):
        if self.RDA.ser is None: # 当前没有激活的串口
            if self.ui.device_cmb.count() == 0: # 搜索一下串口
                self._updatedevice()

            if self.ui.device_cmb.count() == 0: # 还是没有串口
                showMessageBox('设备管理器',"没有找到设备！")
                return

            # 当前设备
            port = self.ui.device_cmb.currentText()
            self.RDA.configDev(port)
            if not self.RDA.open():
                showMessageBox("设备管理器",'设备打开失败！')
                return

        # 此时设备为打开成功
        self.RDA.writeCmd('acquireEEG')
        time.sleep(0.05)
        self.RDA.writeCmd('acquireEEG')
        self.mainsig.emit('acquireEEG')  # 激活相应的窗口

    def stop_acq(self):
        self.mainsig.emit('stop')
        if self.RDA.ser is None:
            return
        self.RDA.writeCmd('stop')
        time.sleep(0.05)
        self.RDA.writeCmd('stop')
        time.sleep(0.05)
        self.RDA.writeCmd('stop')

    def imp_detect(self):
        '''
        阻抗检测
        '''
        if self.RDA.ser is None:  # 当前没有激活的串口
            if self.ui.device_cmb.count() == 0:  # 搜索一下串口
                self._updatedevice()

            if self.ui.device_cmb.count() == 0:  # 还是没有串口
                showMessageBox('设备管理器', "没有找到设备！")
                return

            # 当前设备
            port = self.ui.device_cmb.currentText()
            self.RDA.configDev(port)
            if not self.RDA.open():
                showMessageBox("设备管理器", '设备打开失败！')
                return

        # 此时设备为打开成功
        self.RDA.writeCmd('impedanceDetect')
        time.sleep(0.05)
        self.RDA.writeCmd('impedanceDetect')
        self.mainsig.emit('impedanceDetect')
