#coding:utf-8

import multiprocessing
from multiprocessing import Queue,Event
from device import devicepro
import time
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox,QFileDialog
from PyQt5.QtCore import pyqtSignal
import serial
import serial.tools.list_ports as lp
import re
import threading
from shm import CreateShm

BAUDS = [115200,230400,460800,921600]

'''设备进程管理'''

class devManager(QtWidgets.QDialog,threading.Thread):
    _sig2mesbox = pyqtSignal(str)
    def __init__(self,parentUI):
        super(devManager, self).__init__()
        self.shm = CreateShm(master=False)
        self.ui = parentUI
        self.ui.startacq_btn.clicked.connect(self.start_acq)
        self.ui.stopacq_btn.clicked.connect(self.stop_acq)
        self.ui.updatecom_btn.clicked.connect(self._updatedevice)
        self.ui.save_btn.clicked.connect(self.start_save)

        # 屏蔽绘图操作按钮,绘图操作按钮的enable由绘图进程通过读取数据获得参数后开启
        self.ui.xrange_sbx.setEnabled(False)
        self.ui.yrange_cmb.setEnabled(False)
        self.ui.flt_cmb.setEnabled(False)
        self.ui.chssel_btn.setEnabled(False)

        self._sig2mesbox.connect(self.popmesbox)  ## 弹出消息框
        self.ui = parentUI
        self.datapath = './data'
        self.devprocess = None
        self.devprocount = 0     # 记录第几次启动设备进程
        self.stopEv = Event()
        self.devBakQue = Queue()

        self._updatedevice()
        self.threadRunning = True
        self.setDaemon(True)
        self.start()

    def _getallserial(self):  # 指定cp210x
        pp = re.compile('CP210x')
        ports = lp.comports()
        device = []
        for p in ports:
            id = p.device
            r = re.search(pp, str(p))
            if r is not None:
                device.append(id)
        return device

    def start_save(self):
        if self.shm.info[8] == 0:   # 未保存->保存
            if self.devprocess is None or not self.devprocess.is_alive():
                self._sig2mesbox.emit("信号采集模块未启动！")
                return

            folder_path = QFileDialog.getSaveFileName(None, "设置保存文件", "./data", "Sub001 (*.dat)")
            if len(folder_path[0]) > 0:
                self.shm.setPath(folder_path[0])  # 将路径写入共享内存中
                self.ui.path_edit.setText(folder_path[0])
                # 写入保存标志
                self.shm.info[8] = 1
                self.ui.save_btn.setText('停止保存')
                self.ui.save_btn.setStyleSheet("font: 20px \"微软雅黑\";background-color: rgb(223, 4, 4);")

        else:  # 保存->不保存
            self.shm.info[8] = 0
            self.ui.save_btn.setText('保存数据')
            self.ui.save_btn.setStyleSheet("font: 20px \"微软雅黑\";")

    def _updatedevice(self):
        device = self._getallserial()
        self.ui.device_cmb.clear()
        if len(device)>0:
            self.ui.device_cmb.addItems(device)

    def release(self):
        self.threadRunning = False
        self.devBakQue.put('-')  # 用来驱动子线程结束
        self.stop_acq()

    def popmesbox(self,strs):
        if strs[0] == 'b':
            self.ui.save_btn.setText('保存数据')
            self.ui.save_btn.setStyleSheet("font: 20px \"微软雅黑\";")
        QMessageBox.warning(self, 'devManager', strs, QMessageBox.Yes, QMessageBox.Yes)

    def run(self):  # 子线程监听放大器进程传递的队列消息，并以弹窗的方式显示
        while self.threadRunning:
            cmd = self.devBakQue.get()  #一般来说用来接收放大器进程的异常报警信号
            if cmd == '-': break
            self._sig2mesbox.emit(cmd)

    # 按钮事件
    def start_acq(self):
        '''启动采集进程'''
        # 有放大器正在活动,忽略
        if self.devprocess is not None and self.devprocess.is_alive():
            QMessageBox.warning(self, '提示', '设备正在运行！ ', QMessageBox.Yes, QMessageBox.Yes)
            return

        # 检查端口情况,没有可用端口
        if self.ui.device_cmb.count() == 0:
            QMessageBox.warning(self, '提示', '没有可用设备！ ', QMessageBox.Yes, QMessageBox.Yes)
            return

        # 有端口的情况也要检查，因为可能拔出但没有及时刷新
        port = self.ui.device_cmb.currentText()
        device = self._getallserial()
        if port not in device: # 当前端口不再可用设备中，说明设备拔出了，但没有及时更新
            self._updatedevice()    # 再次更新
            if self.ui.device_cmb.count() == 0:     # 没有设备
                QMessageBox.warning(self, '提示', '没有可用设备！ ', QMessageBox.Yes, QMessageBox.Yes)
                return
            else:
                QMessageBox.warning(self, '提示', '请重新选择设备！ ', QMessageBox.Yes, QMessageBox.Yes)
                return

        try:
            ser = serial.Serial(port=port, baudrate=115200)
            time.sleep(0.2)
            ser.close()
        except:
            QMessageBox.warning(self, '提示', '无效端口！ ', QMessageBox.Yes, QMessageBox.Yes)
            self._updatedevice()
            return

        # 经过了验证和检查，port是有效端口
        self.stopEv.clear()
        ctrprm = {}
        ctrprm['stopEv'] = self.stopEv
        ctrprm['backQue']= self.devBakQue
        deviceinfo = {}
        deviceinfo['type'] = 's'
        deviceinfo['param'] = {}
        deviceinfo['param']['port'] = port
        deviceinfo['param']['baudrate'] = BAUDS[self.ui.baud_cmb.currentIndex()]
        self.devprocount += 1
        deviceinfo['devprocount'] = self.devprocount

        self.devprocess = multiprocessing.Process(target=devicepro, args=(deviceinfo, ctrprm))
        self.devprocess.start()

    def stop_acq(self):
        '''
        停止采集进程
        '''
        # 屏蔽绘图操作按钮
        self.ui.xrange_sbx.setEnabled(False)
        self.ui.yrange_cmb.setEnabled(False)
        self.ui.flt_cmb.setEnabled(False)
        self.ui.chssel_btn.setEnabled(False)

        self.shm.info[8] = 0  # 同步通知停止保存
        self.ui.save_btn.setText('保存数据')
        self.ui.save_btn.setStyleSheet("font: 20px \"微软雅黑\";")

        if self.devprocess is None: return  #没有进程
        if self.devprocess.is_alive():    #尝试通知关闭
            self.stopEv.set()

        time.sleep(0.3)
        if self.devprocess.is_alive():  self.devprocess.terminate()
        self.stopEv.clear()
