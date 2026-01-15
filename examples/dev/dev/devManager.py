#coding:utf-8

import multiprocessing
from multiprocessing import Queue,Event
import time
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox,QFileDialog
from PyQt5.QtCore import pyqtSignal
import threading
from dataserverbp import devicepro as bpdevicepro
from dataserverneuracle import devicepro as ncdevicepro


'''设备进程管理'''

ampList = ['bp','neuracle','biosemi']

class devManager(QtWidgets.QDialog,threading.Thread):
    _sig2mesbox = pyqtSignal(str)
    def __init__(self,parentUI):
        super(devManager, self).__init__()
        self.ui = parentUI
        # 开始停止按钮
        self.ui.startAcqBtn.clicked.connect(self.start_acq)
        self.ui.stopAcqBtn.clicked.connect(self.stop_acq)
        self._sig2mesbox.connect(self.updateStatus)
        self.ui.ampCBox.currentIndexChanged.connect(self.ampcboxChange)
        self.ampindex = -1
        self.ampcboxChange(0)

        self.stopEv = Event()
        self.devBakQue = Queue()
        self.devprocess = None

        self.threadRunning = True
        self.setDaemon(True)
        self.start()

    def run(self):  # 子线程监听放大器进程传递的队列消息，并以弹窗的方式显示
        while self.threadRunning:
            cmd = self.devBakQue.get()  # 一般来说用来接收放大器进程的异常报警信号
            print(cmd)
            if cmd == '-': break
            self._sig2mesbox.emit(cmd)

    def updateStatus(self,mess):
        self.ui.statusEdit.append(mess)

    def ampcboxChange(self,index):
        ampindex = self.ui.ampCBox.currentIndex()
        amp = ampList[ampindex]
        if amp == 'bp':
            self.ui.eegchEdit.setText('32')
            self.ui.eegchEdit.setEnabled(False)
            self.ui.devPort.setText('51244')
            self.ui.devPort.setEnabled(False)
            self.ui.srateEdit.setText('500')
            self.ui.srateEdit.setEnabled(True)
            self.ui.TriggerCBox.setChecked(True)
        elif amp == 'neuracle':
            self.ui.eegchEdit.setText('64')
            self.ui.eegchEdit.setEnabled(False)
            self.ui.devPort.setText('8712')
            self.ui.devPort.setEnabled(False)
            self.ui.srateEdit.setText('1000')
            self.ui.srateEdit.setEnabled(False)
            self.ui.TriggerCBox.setChecked(True)
        elif amp == 'biosemi':
            self.ui.eegchEdit.setText('32')
            self.ui.eegchEdit.setEnabled(True)
            self.ui.devPort.setText('988')
            self.ui.devPort.setEnabled(True)
            self.ui.srateEdit.setText('1024')
            self.ui.srateEdit.setEnabled(True)
            self.ui.TriggerCBox.setChecked(True)
        else:
            self.ui.eegchEdit.setText('')
            self.ui.eegchEdit.setEnabled(True)
            self.ui.devPort.setText('')
            self.ui.devPort.setEnabled(True)
            self.ui.srateEdit.setText('')
            self.ui.srateEdit.setEnabled(True)
            self.ui.TriggerCBox.setChecked(True)

    # 按钮事件
    def start_acq(self):
        ampindex = self.ui.ampCBox.currentIndex()
        udpateAmp = False

        if self.devprocess == None: # 无进程
            updateAmp = True
        else:
            if self.devprocess.is_alive(): # 设备在运行
                if ampindex != self.ampindex:  # 切换了设备
                    updateAmp = True
                else:   # 没有切换设备
                    updateAmp = False
                    QMessageBox.warning(self, '提示', '设备正在运行！ ', QMessageBox.Yes, QMessageBox.Yes)
            else:
                updateAmp = True

        if updateAmp == False:
            return

        self.stopEv.clear()
        ctrprm = {}
        ctrprm['stopEv'] = self.stopEv
        ctrprm['backQue'] = self.devBakQue

        deviceinfo = {}
        dev = ampList[ampindex]
        deviceinfo['devName'] = dev
        try:
            deviceinfo['eegChs'] = int(self.ui.eegchEdit.text())
        except:
            QMessageBox.warning(self, '提示', 'eeg通道未正确配置！ ', QMessageBox.Yes, QMessageBox.Yes)
            return

        try:
            deviceinfo['srate'] = int(self.ui.srateEdit.text())
        except:
            QMessageBox.warning(self, '提示', '采样率未正确配置！ ', QMessageBox.Yes, QMessageBox.Yes)
            return

        deviceinfo['devAddr'] = self.ui.devIP.text()

        try:
            deviceinfo['devPort'] = int(self.ui.devPort.text())
        except:
            QMessageBox.warning(self, '提示', '设备端口未正确配置！ ', QMessageBox.Yes, QMessageBox.Yes)
            return

        deviceinfo['mqAddr'] = self.ui.mqbrokerIPEdit.text()
        deviceinfo['mqPort'] = 1883
        deviceinfo['topic'] = self.ui.topicnameEdit.text()
        deviceinfo['includeTrigger'] = 1

        if dev == 'bp':
            self.devprocess = multiprocessing.Process(target=bpdevicepro, args=(deviceinfo, ctrprm))
            self.devprocess.start()
            self.ampindex = ampList.index(dev)
        elif dev == 'neuracle':
            self.devprocess = multiprocessing.Process(target=ncdevicepro, args=(deviceinfo, ctrprm))
            self.devprocess.start()
            self.ampindex = ampList.index(dev)

    def stop_acq(self):
        if self.devprocess == None: return  # 没有进程
        if self.devprocess.is_alive():  # 尝试通知关闭
            self.stopEv.set()

        time.sleep(0.3)
        if self.devprocess.is_alive():  self.devprocess.terminate()
        self.stopEv.clear()