#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pyqtgraph as pg
from bcis.core import BCIs
import bcis.core as bciscore
import numpy as np
from .calfft import calFFT
import threading
import time
from itertools import cycle

FFTLEN = 2

COLORS = [(255,125,255),(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(125,255,255)]
cycleCOLORS = cycle([(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)])

class fftPlot(threading.Thread):
    def __init__(self):
        # 绘图
        self.plotmode = 'band' # or eeg
        self.curves = []
        self.plots = []
        self.band = 0
        self.bars = []
        for i in range(8):
            plot = pg.plot()
            self.plots.append(plot)
            curve = pg.PlotCurveItem(pen=next(cycleCOLORS))  # 白色画笔
            # plot.addItem(curve)
            self.curves.append(curve)
            bar = pg.BarGraphItem(x=range(6), height=0, width=0.7, brushes=COLORS)
            self.bars.append(bar)

        self.plotOK = True      #全局控制是否绘图
        self.chs_to_plot = []
        self.chs = 64
        self.index = 0
        self.ffteeg = None
        self.fftok = False

        threading.Thread.__init__(self)
        self.lock = threading.Lock()
        self.setDaemon(True)

    def run(self):
        lst = time.perf_counter()
        while True:
            clk = time.perf_counter()
            if clk - lst >= 0.1:
                self.lock.acquire()
                data = self.fdata
                self.lock.release()
                self.ffteeg = self.calfft.calculate_fft(data)
                self.fftok = True
                lst = clk
            else:
                time.sleep(0.01)

    def init_data_reader(self):
        self.bcis = BCIs(master=False)
        parm = self.bcis.parameters.get()
        self.fdata = np.zeros((parm['eegchs'], FFTLEN * parm['srate']), bciscore.EEGTYPE)
        self.calfft = calFFT()
        self.calfft.reset(parm['srate'],FFTLEN*parm['srate'])
        self.start()
        return True

    def reset(self):
        parm = self.bcis.parameters.get()
        self.chs = parm['eegchs']   #i.e 64, 信号通道总数
        self.calfft.reset(parm['srate'],FFTLEN*parm['srate'])
        for i in range(8):
            try:
                self.plots[i].removeItem(self.curves[i])
                self.plots[i].removeItem(self.bars[i])
            except:  pass

        # 检查通道合法性
        for ch in self.chs_to_plot:
            if ch >= self.chs:  #有非法通道
                self.plotOK = False
                return 1
        # 是否有通道被绘制
        if len(self.chs_to_plot)==0:
            self.plotOK = False
            return 2

        self.plotOK = True

        for chh in self.chs_to_plot:
            ch = chh % 8
            if self.band in [0,1]:
                self.plots[ch].addItem(self.curves[ch])
                self.plots[ch].showGrid(True,True)
                self.plots[ch].setLabel('bottom',"Hz")
                rang = [45,90]
                self.plots[ch].setXRange(0, rang[self.band])

            elif self.band == 2:
                self.plots[ch].addItem(self.bars[ch])
                # self.plots[ch].showGrid(True, True)
                # self.plots[ch].setLabel('bottom', "Hz")
                self.plots[ch].setXRange(0,5.5)
                self.plots[ch].setLabel('bottom', "deta__theta__alphaA__alphaB__betaA__betaB")
            else:
                raise Exception('invalide plotmode %s'%(self.plotmode))
        return 0

    def update_one_frame(self):
        if not self.plotOK: return
        parm = self.bcis.parameters.get()
        ind = parm['indx']
        if ind != self.index:  # eeg有更新
            N = parm['eegsamples']
            chs = parm['eegchs']
            srate = parm['srate']
            dat = self.bcis.raweeg.data[:N * chs].reshape(N, chs).transpose()  # 得到数据
            r, c = dat.shape
            self.fdata = np.hstack((self.fdata, dat))[:, -FFTLEN*srate:]  # 新数据包添加到末尾
            self.index = ind

            if self.fftok:
                self.fftok = False
                tem = self.ffteeg
                if self.band in [0,1]:
                    if self.band == 0:
                        y = tem['band45']
                        xx = self.calfft.xf[:self.calfft.band1]
                    elif self.band == 1:
                        y = tem['band90']
                        xx = self.calfft.xf[:self.calfft.band2]
                    else:
                        return

                    for i in self.chs_to_plot:
                        ii = i%8
                        self.curves[ii].setData(x =xx, y = y[i,:])

                elif self.band == 2:
                    for i in self.chs_to_plot:
                        ii = i % 8
                        # self.bars[ii].y1 = 10
                        self.bars[ii].setOpts(height = tem['bands'][i,:])
