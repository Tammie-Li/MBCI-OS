#!/usr/bin/python
# -*- coding: utf-8 -*-

import pyqtgraph as pg
pg.setConfigOption('background', (255, 255, 255))
from datamanager import DataManager
from shm import CreateShm
from butterfilter import ButterFilter
from itertools import cycle
from chselmanager import chselManager
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal

COLORS = [(255,0,0),(0,255,0),(0,0,255),(0,0,0),(255,0,255)]
yscale = [20,50,100,200,500,2,10,100,1,5,0]
#         uv,uv,uv,  uv,uv, mv,mv,mv,v,v,?
ygain =  [1,1,1,1,1,1e-3,1e-3,1e-3,1e-6,1e-6,1]

'''负责绘图
绘图fs: 250
绘图进程从共享内存区读取数据，获得设备的相关信息，进行绘图曲线初始化
后，对实时数据进行绘制 '''

class dataPlot(QWidget):
    _psig = pyqtSignal(str)
    def __init__(self,parentUI,width):
        super(dataPlot, self).__init__()
        self.shm = CreateShm(master = False)
        self.screenwidth = width

        # 控制电量更新频率
        self.batC = 80 # 大约4秒更新一次
        self.batUC = 0

        # info是一个字典，描述了放大器的相关参数
        # 例如：{'name':'GSR','srate':250,'sigchs':['GSR','ECG'],'testch':1,'format':'int24'}
        self.deviceinfo = {}
        self.devprocount = -1

        # 绘图相关
        self.flttype = 0  # 0-None, 1-high, 2-band
        self.scale = 10
        self.ygain = 1
        self.chs2plot = []  # 待绘图的通道
        self.curves = []    # 绘图曲线组
        self.period = 0     # 绘图时长
        self.prepare = True  # True： update_one_frame跳过
        # # 绘图数据配置
        self.dm = DataManager()
        self.dmAcc = DataManager()
        self.ui = parentUI
        self.ui.flt_cmb.currentIndexChanged.connect(self._updateFlt)  # 更新绘图滤波

        self.ui.xrange_sbx.valueChanged.connect(self.relayout)          # x范围改变
        self.ui.yrange_cmb.currentIndexChanged.connect(self.relayout)   # y尺度改变
        self.chselMgr = chselManager(self.ui)  # 绘图通道切换管理
        self._psig.connect(self.updateDeivce) # 在update_one_frame中通过信号触发设备更新

        # 绘图控制
        self.pgplot = pg.PlotWidget()
        self.pgplot.showGrid(True,True)
        self.ui.EEGLayout.addWidget(self.pgplot)

        self.pgplotAcc = pg.PlotWidget()
        self.pgplotAcc.showGrid(True, True)
        self.ui.accLayout.addWidget(self.pgplotAcc)

        # 数据读取和滤波控制
        self.index = 0
        self.filters = ButterFilter()

    def _updateFlt(self):
        self.flttype = self.ui.flt_cmb.currentIndex()

    def initAccplot(self):
        self.curvesAcc = []
        self.pgplotAcc.setYRange(-50, 50)
        cols = cycle(COLORS)
        for i in range(3):
            color = next(cols)
            curve = pg.PlotCurveItem(pen=pg.mkPen(color=color, width=1))
            self.pgplotAcc.addItem(curve)
            curve.setPos(0, 0)
            self.curvesAcc.append(curve)

    # 通道选择小窗激活该函数
    # 通道改变重新布局
    def relayout(self):
        self.prepare = True  #暂时屏蔽绘图更新
        scale = yscale[self.ui.yrange_cmb.currentIndex()]   # 当前y方向上的尺度
        self.ygain = ygain[self.ui.yrange_cmb.currentIndex()]
        chs2plot = self.chselMgr.selectedchs['sigchs']
        self.pgplot.setYRange(0,scale*len(chs2plot))
        period = self.ui.xrange_sbx.value()                 # 设置的时间长度

        if self.period != period:   # 时长改变了
            self.dm.config(self.deviceinfo['srate'],self.deviceinfo['sigchs'],period)
            self.dmAcc.config(self.deviceinfo['srate'], 3, period)
            self.pgplot.setXRange(0, 250 * period)
            self.pgplotAcc.setXRange(0, 250 * period)
            self.period = period

        if self.chs2plot != chs2plot:   # 需要重新布局
            self.chs2plot = chs2plot
            for cvs in self.curves:
                cvs.clear()
                self.pgplot.removeItem(cvs)
            self.curves = []

            cols = cycle(COLORS)
            for idx, ch in enumerate(self.chs2plot):
                color = next(cols)
                curve = pg.PlotCurveItem(pen=pg.mkPen(color=color, width=1))
                self.pgplot.addItem(curve)
                curve.setPos(0, idx * scale + 0.5 * scale)
                self.curves.append(curve)

        if scale != self.scale:
            self.scale = scale
            for idx, ch in enumerate(self.chs2plot):
                self.curves[idx].setPos(0, idx * scale + 0.5 * scale)

        self.prepare = False

    def updateDeivce(self):
        # 读到了数据，开启绘图控制按钮
        # 屏蔽绘图操作按钮
        self.ui.xrange_sbx.setEnabled(True)
        self.ui.yrange_cmb.setEnabled(True)
        self.ui.flt_cmb.setEnabled(True)
        self.ui.chssel_btn.setEnabled(True)

        # 绘图通道选择窗
        self.filters.reset(srate = self.deviceinfo['srate'], chs = self.deviceinfo['sigchs'], fltparam = [(49,51),(2,45),(1,0),None])
        self.chselMgr.reset(self.deviceinfo)
        self.chselMgr.csig.connect(self.relayout)
        self.relayout()
        self.initAccplot()

    def update_one_frame(self):
        # 读数据
        ind = self.shm.info[0]
        if ind == 0:    return  # 设备未启动

        if ind != self.index:   # 说明设备在运行
            if self.shm.info[3] != self.devprocount:    # 设备重启了
                self.prepare = True                     # 标记为准备状态
                self.devprocount = self.shm.info[3]
                self.deviceinfo['srate'] = self.shm.info[4]
                self.deviceinfo['sigchs'] = self.shm.info[5]
                self.deviceinfo['testch'] = self.shm.info[6]
                self.deviceinfo['trich'] = self.shm.info[7]
                self._psig.emit('->')  # 不占用update_one_frame函数的运行时间

        if self.prepare:    return      # 准备状态不更新

        if ind != self.index:  # eeg数据有更新
            self.shm.pinfo[0] = 1
            self.index = ind


            N = self.shm.info[1]     # 最新待写入数据的地方，也是这次读取数据的末尾后面的一个点
            pN = self.shm.info[2]    # 当前缓存中有多少有用数据
            sampleN = int(pN/self.deviceinfo['sigchs'])
            newdat = self.shm.eeg0[N-pN:N].reshape(sampleN, self.deviceinfo['sigchs']).transpose()  # 整理新数据
            # newdat = self.shm.eeg0[:N].reshape(pN, self.deviceinfo['sigchs']).transpose()  # 整理新数据

            self.shm.info[1] = 0     # 读完数据后复位
            self.shm.info[2] = 0     # 读完数据后复位
            self.shm.pinfo[0] = 0
            self.filters.update(newdat)  # 滤波
            if self.flttype == 0:   # none
                self.dm.update(self.filters.rawdata)
            elif self.flttype == 1: # high pass
                self.dm.update(self.filters.hdata)
            elif self.flttype == 2: # band pass
                self.dm.update(self.filters.bdata)
            else:
                self.dm.update(self.filters.rawdata)

            if self.dm.data is None:    return
            for id,ch in enumerate(self.chs2plot):
                self.curves[id].setData(self.dm.data[ch,:]*self.ygain)

            accN = int(self.shm.info[14]/2)  # 字节数 -> 数据点数
            pan = int(accN/3)

            accdata = self.shm.acc[:accN].astype('float32')*9.8/4096
            accdata = accdata.reshape(pan,3).transpose()
            self.shm.info[14] = 0
            self.dmAcc.update(accdata)

            for i in range(3):
                self.curvesAcc[i].setData(self.dmAcc.data[i,:])

            self.batUC += 1
            self.batUC %= self.batC
            if self.batUC == 0:
                self.ui.batLevel.setValue(self.shm.info[10])

    def release(self):
        self.shm.release()