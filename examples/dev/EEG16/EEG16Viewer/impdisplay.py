#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Time    : 2024/11/28 9:07
Author  : mrtang
Email   : 810899799@qq.com
"""

import sys
from PyQt5.QtWidgets import QApplication
import pyqtgraph as pg
pg.setConfigOption('background', (50, 55, 62))
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPolygonF
from PyQt5.QtCore import Qt, QRect, QTimer
from eeglocs import nodelocs
import math
import json
from scipy.fft import fft, fftfreq
import numpy as np
from shm2 import CreateShm,EEGTYPE
from datamanager import DataManager

class ImpDisplay(QWidget):
    def __init__(self,parentUI,config):
        self.ui = parentUI
        super(ImpDisplay, self).__init__()
        self.shm = CreateShm(master=False)

        # 控制电量更新频率
        self.batC = 80 # 大约4秒更新一次
        self.batUC = 0

        self.index = 0
        self.config = config
        self.eegchs = config['eegchs']
        rawsrate = config['srate']
        self.downsampleScale = rawsrate // 250
        self.localSrate = rawsrate // self.downsampleScale
        self.chsNum = len(self.config['eegchs'])
        self.res = [1000]*self.chsNum
        self.dm = DataManager()  # 绘图数据管理器
        self.dm.config(self.localSrate, self.chsNum, 4,EEGTYPE)
        self.impcal = ImpCal(self.chsNum,self.localSrate,4)

        # 设置画笔、画刷
        self.blackPen = QPen(Qt.black, 2)
        self.grayPen = QPen(QColor(150, 150, 150, 100), 2)
        self.redPen = QPen(QColor(255,127,39), 2)
        self.greenPen = QPen(QColor(60,130,75), 2)
        self.bluePen = QPen(QColor(0,162,232), 2)

        self.textPen = QPen(QColor(255,255,255),3)

        self.redBrush = QBrush(QColor(255,127,39))
        self.greenBrush = QBrush(QColor(60,130,75))
        self.blueBrush = QBrush(QColor(0,162,232))
        self.grayBrush = QBrush(QColor(150, 150, 150, 100))

        self.counter = 0
        self.geo = QRect(0, 0, 0, 0)
        self.updateSize()

        self.ui.displayLayout.addWidget(self)

        # 定时器启动定时更新
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._updateAll)

    def startPloting(self,flg):
        if flg:   self.timer.start(5)
        else:     self.timer.stop()

    def addToMainWin(self,flg):
        if flg:
            self.ui.displayLayout.addWidget(self)
            self.show()
        else:
            self.ui.displayLayout.removeWidget(self)
            self.hide()

    def release(self):
        try:
            self.timer.stop()
            self.timer = None
        except:
            pass

    def closeEvent(self, e):
        self.release()

    def updateSize(self): # 窗口尺寸发生改变
        geo = self.geometry()
        if (geo.x() != self.geo.x()) or (geo.y() != self.geo.y()) or (geo.width() != self.geo.width()) or (geo.height() != self.geo.height()):
            self.geo = geo
            cx = self.geo.x()
            cy = self.geo.y()
            cw = self.geo.width()
            ch = self.geo.height()

            self.R = ch * 0.7  # 电极极坐标半径
            self.txtSize = int(self.R * 0.03) #字体大小
            self.eleR = int(self.R * 0.075) # 点击圆圈半径

            self.centerX = int(cw / 2 + cx)
            self.centerY = int(ch / 2)
            return True
        else:
            return False

    def _updateAll(self):
        # 5ms调用一次
        # 读数据
        ind = self.shm.info[0]
        if ind == 0:    return  # 设备未启动

        if ind != self.index:  # eeg数据有更新
            self.index = ind
            if self.shm.getvalue('mode') != 2: # 确保在正确的模式下
                return

            self.shm.setvalue('plotting', 1)

            curdataindx = int(self.shm.getvalue('curdataindex'))
            pp = int(curdataindx / self.chsNum)

            dat = self.shm.eeg[:curdataindx].reshape(pp, self.chsNum).transpose()
            self.shm.setvalue('curbyteindex', 0)
            self.shm.setvalue('curdataindex', 0)
            self.shm.setvalue('plotting', 0)

            eeg = dat[:, ::self.downsampleScale]
            self.dm.update(eeg)
            if self.dm.data is None:    return

            self.res = self.impcal.cal(self.dm.data)

            # self.batUC += 1
            # self.batUC %= self.batC
            # if self.batUC == 0:
            #     self.ui.batLevel.setValue(self.shm.getvalue('batlevel'))

        self.counter += 1
        self.counter %= 100
        if self.counter == 1:
            self.update()      # 调用父类的update API, update会触发paintEvent进行更新绘图

    def paintEvent(self,e):
        if self.counter < 1:    return  # 首次启动时，获取的尺寸为原始尺寸，不是自适应主窗口之后的尺寸

        self.updateSize()
        painter = QPainter(self)
        painter.setPen(self.grayPen)
        # 绘制两个圆
        r = int(self.R * 0.63889)
        painter.drawEllipse(int(self.centerX - r), int(self.centerY - r), 2 * r, 2 * r)
        r = int(self.R * 0.511111)
        painter.drawEllipse(int(self.centerX - r), int(self.centerY - r), 2 * r, 2 * r)

        nid = 0
        # 绘制电极
        for key in nodelocs:
            node = nodelocs[key]
            angle = node[0]
            coe = node[1]
            x = self.centerX + math.sin(angle * math.pi / 180) * self.R * coe
            y = self.centerY - math.cos(angle * math.pi / 180) * self.R * coe

            if key in self.eegchs:  # 本次设备的电极
                resNode = self.res[nid]
                nid += 1
                if resNode <= 5:
                    # 绘制电极
                    pp = self.greenPen
                    pb = self.greenBrush
                elif resNode <= 10:
                    pp = self.bluePen
                    pb = self.blueBrush
                else:
                    pp = self.redPen
                    pb = self.redBrush

                painter.setPen(pp)
                painter.setBrush(pb)

                painter.drawEllipse(int(x - self.eleR / 2), int(y - self.eleR / 2), self.eleR,
                                         self.eleR)  # 椭圆的左上角坐标和宽度、高度
                # 标注电极名称
                painter.setPen(self.textPen)
                painter.drawText(int(x - self.txtSize), int(y - self.eleR / 2 - self.txtSize / 2), key)

                txtstr = '%.1fK'%(resNode)
                painter.drawText(int(x - 0.15*self.txtSize*len(txtstr)), int(y), txtstr)

            elif key == 'Ref':
                if self.shm.getvalue('refconnect'):
                    pp = self.greenPen
                    pb = self.greenBrush
                else:
                    pp = self.redPen
                    pb = self.redBrush

                painter.setPen(pp)
                painter.setBrush(pb)
                painter.drawEllipse(int(x - self.eleR / 2), int(y - self.eleR / 2), self.eleR,
                                    self.eleR)  # 椭圆的左上角坐标和宽度、高度
                painter.setPen(self.textPen)
                painter.drawText(int(x - self.txtSize), int(y - self.eleR / 2 - self.txtSize / 2), key)

            elif key == 'Gnd':
                if self.shm.getvalue('biasconnect'):
                    pp = self.greenPen
                    pb = self.greenBrush
                else:
                    pp = self.redPen
                    pb = self.redBrush

                painter.setPen(pp)
                painter.setBrush(pb)
                painter.drawEllipse(int(x - self.eleR / 2), int(y - self.eleR / 2), self.eleR,
                                    self.eleR)  # 椭圆的左上角坐标和宽度、高度
                painter.setPen(self.textPen)
                painter.drawText(int(x - self.txtSize), int(y - self.eleR / 2 - self.txtSize / 2), key)

            else:
                # 其他电极用灰色标注
                painter.setPen(self.grayPen)
                painter.setBrush(self.grayBrush)
                painter.drawEllipse(int(x - self.eleR / 2), int(y - self.eleR / 2), self.eleR,
                                    self.eleR)  # 椭圆的左上角坐标和宽度、高度
                painter.drawText(int(x - self.txtSize), int(y - self.eleR / 2 - self.txtSize / 2), key)


class ImpCal():
    def __init__(self,chs=16,srate=250,tLen=4):
        self.N = srate*tLen
        self.T = 1/srate
        xf = fftfreq(self.N,self.T)[:self.N//2]
        self.ind0 = np.where(xf>6)[0][0]
        self.ind1 = np.where(xf<10)[0][-1]
        self.chs = chs

    def cal(self,data):
        y = fft(data)
        imp = []
        for i in range(self.chs):
            y_ = 2.0 / self.N * np.abs(y[i, :])
            m = np.max(y_[self.ind0:self.ind1])
            if m < 2400:
                res = 1
            elif m < 20000:
                res = (m - 2400) / 181.13
            else:
                res = (m - 13900) / 66.11
            imp.append(res)
        return imp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImpDisplay()
    ex.show()
    sys.exit(app.exec_())


