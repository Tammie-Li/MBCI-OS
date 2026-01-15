#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pyqtgraph as pg
pg.setConfigOption('background', (50, 55, 62))
from datamanager import DataManager
from shm2 import CreateShm,EEGTYPE
from butterfilter import ButterFilter,FirFilter
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget
import numpy as np


COLORS = [(255,0,0),(0,0,255),(0,0,0),(255,0,255)]
yscale = [20,50,100,200,500,2,10,100,1,5,0]
#         uv,uv,uv,  uv,uv, mv,mv,mv,v,v,?
ygain =  [1,1,1,1,1,1e-3,1e-3,1e-3,1e-6,1e-6,1]

'''负责绘图
绘图fs: 250
绘图进程从共享内存区读取数据，获得设备的相关信息，进行绘图曲线初始化
后，对实时数据进行绘制 '''

class EEGDisplay(QWidget):
    _psig = pyqtSignal(str)
    def __init__(self,parentUI,config):
        super(EEGDisplay, self).__init__()
        self.shm = CreateShm(master = False)

        # 控制电量更新频率
        self.batC = 80 # 大约4秒更新一次
        self.batUC = 0

        # 绘图相关
        self.flttype = 2  # 0-None, 1-high, 2-band
        self.scale = 10
        self.ygain = 1
        self.config = config
        rawsrate = config['srate']
        self.downsampleScale = rawsrate // 250
        self.localSrate = rawsrate // self.downsampleScale
        self.chsNum = len(self.config['eegchs'])
        self.curves = []    # 绘图曲线组
        self.period = 0     # 绘图时长
        self.prepare = True  # True： update_one_frame跳过

        # # 绘图数据配置
        self.dm = DataManager()  # 绘图数据管理器
        self.ui = parentUI
        self.ui.xrange_sbx.valueChanged.connect(self.relayout)          # x范围调整
        self.ui.yrange_cmb.currentIndexChanged.connect(self.relayout)   # y尺度调整

        # 绘图控制
        self.pgplot = pg.PlotWidget()
        self.pgplot.showGrid(True,True)

        # 初始化绘图
        self.virgin = True
        self.relayout()

        # 数据读取和滤波控制
        self.index = 0
        # self.filter = ButterFilter()
        self.filter = FirFilter()
        self.filter.reset(srate=self.localSrate, chs=self.chsNum,
                           fltparam=[(49, 51), (1, 45), (1, 0), None],eegtype=EEGTYPE)
        self.ui.displayLayout.addWidget(self.pgplot)
        self.pgTimer = pg.QtCore.QTimer()
        self.pgTimer.timeout.connect(self.update_one_frame)

    def startPloting(self,flg):
        if flg:   self.pgTimer.start(5)
        else:     self.pgTimer.stop()

    def addToMainWin(self,flg):
        if flg:
            self.ui.displayLayout.addWidget(self.pgplot)
            self.pgplot.show()
        else:
            self.pgplot.hide()
            self.ui.displayLayout.removeWidget(self.pgplot)

    def relayout(self):
        self.prepare = True  #暂时屏蔽绘图更新
        scale = yscale[self.ui.yrange_cmb.currentIndex()]   # 当前y方向上的尺度
        self.ygain = ygain[self.ui.yrange_cmb.currentIndex()]
        self.pgplot.setYRange(0,scale*self.chsNum)
        period = self.ui.xrange_sbx.value()

        if self.period != period:   # 时长改变了
            self.dm.config(self.localSrate,self.chsNum,period,EEGTYPE)
            self.pgplot.setXRange(0, self.localSrate * period)
            self.period = period

        if self.virgin:
            self.virgin = False
            self.curves = []
            for idx, ch in enumerate(self.config['eegchs']):
                curve = pg.PlotCurveItem(pen=pg.mkPen(color=(50,255,50), width=1))
                self.pgplot.addItem(curve)
                curve.setPos(0, idx * scale + 0.5 * scale)
                self.curves.append(curve)

        if scale != self.scale:
            self.scale = scale
            for idx, ch in enumerate(self.config['eegchs']):
                self.curves[idx].setPos(0, idx * scale + 0.5 * scale)

        self.prepare = False

    def update_one_frame(self):
        # 读数据
        ind = self.shm.info[0]
        if ind == 0:    return  # 设备未启动
        if self.prepare:    return      # 准备状态不更新

        if ind != self.index:  # eeg数据有更新
            self.index = ind
            if self.shm.getvalue('mode') != 1: # 确保在正确的模式下
                return

            self.shm.setvalue('plotting', 1)

            curdataindx = int(self.shm.getvalue('curdataindex'))
            pp = int(curdataindx/self.chsNum)

            dat = self.shm.eeg[:curdataindx].reshape(pp, self.chsNum).transpose()
            self.shm.setvalue('curbyteindex', 0)
            self.shm.setvalue('curdataindex', 0)
            self.shm.setvalue('plotting', 0)

            eeg = dat[:, ::self.downsampleScale]

            self.filter.update(eeg)  # 滤波
            if self.flttype == 0:  # none
                self.dm.update(self.filter.rawdata)
            # elif self.flttype == 1:  # high pass
            #     self.dm.update(self.filter.hdata)
            elif self.flttype == 2:  # band pass
                self.dm.update(self.filter.bdata)
            else:
                self.dm.update(self.filter.rawdata)

            if self.dm.data is None:    return
            for id in range(self.chsNum):
                self.curves[id].setData(self.dm.data[id, :] * self.ygain)

            # 该版本没有电量反馈
            # self.batUC += 1
            # self.batUC %= self.batC
            # if self.batUC == 0:
            #     self.ui.batLevel.setValue(self.shm.getvalue('batlevel'))


    def release(self):
        self.shm.release()

    def closeEvent(self, e):
        self.release()