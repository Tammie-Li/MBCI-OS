import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QMessageBox

from shm import CreateShm,EEGTYPE
import pyqtgraph as pg
pg.setConfigOption('background', (255, 255, 255))
from scipy.fft import fft, fftfreq
import numpy as np
from itertools import cycle
import numpy as np
from fftUI import Ui_Dialog
import time
import sys

cycleCOLORS = cycle([(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,0,255)])

class FFTViewer(QWidget):
    def __init__(self):
        super().__init__()
        try:
            self.shm = CreateShm(master=False)
        except:
            QMessageBox.warning(self,"错误","采集端未启动！",QMessageBox.Yes)
            return

        self.srate = self.shm.info[4]
        self.chs = self.shm.info[5]
        if self.srate == 0 or self.chs == 0:
            QMessageBox.warning(self, "错误", "没有读取到信息", QMessageBox.Yes)
            return

        self.initUI()

    def initUI(self):
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowTitle("fftViewer by: mrTang email:810899799@qq.com")
        self._screenResize()

        self.plots = []
        self.curves = []
        for i in range(self.chs):
            self.plots.append(pg.plot())
            self.plots[i].showGrid(True,True)
            self.plots[i].setLabel('left','FFT')
            self.plots[i].setLabel('bottom', 'Hz')

            self.curves.append(pg.PlotCurveItem(pen=next(cycleCOLORS)))
            self.ui.vlayout.addWidget(self.plots[i])
            self.plots[i].addItem(self.curves[i])

        self.index = 0
        self.count = 0
        self.data = np.zeros((self.chs, 100), dtype=EEGTYPE)
        self.ploting = False
        self.ui.timeEdit.setText("1")
        self.ui.flEdit.setText("0")
        self.ui.fhEdit.setText(str(int(self.srate//2-1)))
        self.ui.updateBtn.clicked.connect(self.reset)
        self.ui.notchBtn.clicked.connect(self.setNotch)

        self.reset()


        self.pgTimer = pg.QtCore.QTimer()
        self.pgTimer.timeout.connect(self.updateOneFrame)
        self.pgTimer.start(10)

        self.show()

    # 由initialize调用
    # 依据显示屏尺寸来初始化界面大小
    def _screenResize(self):
        # 获取显示器相关信息
        desktop = QApplication.desktop()
        # 默认在主显示屏显示
        screen_rect = desktop.screenGeometry(0)
        self.ww = screen_rect.width()
        self.hh = screen_rect.height()
        self.w = 1000
        self.h = 800
        self.setGeometry(int((self.ww - self.w)/2), int((self.hh - self.h)/3), self.w, self.h)

    def reset(self):
        self.ploting = False
        self.notch = False
        try:
            self.timerange = int(self.ui.timeEdit.text())
        except:
            self.timerange = 1
            self.ui.timeEdit.setText("1")

        try:
            self.fl = int(self.ui.flEdit.text())
        except:
            self.fl = 0

        try:
            self.fh = int(self.ui.fhEdit.text())
        except:
            self.fh = self.srate//2 - 1

        if self.fl < 0:
            self.fl = 0
        if self.fh > self.srate//2 -1:
            self.fh = self.srate//2 -1
        if self.fl > self.fh:
            self.fl = 0
            self.fh = self.srate//2 -1

        self.ui.flEdit.setText("%d"%(self.fl))
        self.ui.fhEdit.setText("%d" % (self.fh))

        self.updateYRangeCount = int(0.2*self.timerange/0.01)

        self.N = self.timerange * self.srate
        self.T = 1 / self.srate
        self.x = np.linspace(0.0, self.timerange, self.N, endpoint=False)
        self.xf = fftfreq(self.N, self.T)[:self.N // 2]

        s1 = np.where((self.xf >=49)&(self.xf<=51))[0]
        s2 = np.where((self.xf >= 99) & (self.xf <= 101))[0]
        s3 = np.where((self.xf >= 149) & (self.xf <= 151))[0]
        self.ncoe = np.hstack((s1,s2,s3))

        self.flind = np.argmin(np.abs(self.xf - self.fl))
        self.fhind = np.argmin(np.abs(self.xf - self.fh))+1

        self.ploting = True

    def setNotch(self):
        self.notch = ~self.notch

    def updateOneFrame(self):
        if not self.ploting:
            return

        ind = self.shm.info[0]
        if ind == 0:    return
        if ind != self.index:
            self.index = ind
            N = self.shm.info[11]  # 当前更新的数据字节数
            pN = self.shm.info[12]  # 当前更新的数据点数
            newdat = self.shm.eeg1[:N].reshape(pN, self.chs).transpose()  # 整理新数据
            self.shm.info[11] = 0  # 读完数据后复位
            self.shm.info[12] = 0  # 读完数据后复位

            self.data = np.hstack((self.data,newdat))[:,-self.N:]
            r,c = self.data.shape
            if c < self.N:
                return
            # if len(X.shape) == 1:   X = X[np.newaxis, :]
            yf = fft(self.data)
            if self.notch:
                yf[:,self.ncoe] = 0
            yf = yf[:,self.flind:self.fhind]
            yf = 2.0/self.N*np.abs(yf)
            ymax = np.max(yf,axis=1)
            self.count += 1
            self.count %= self.updateYRangeCount

            for i in range(self.chs):
                # if self.count == 0:
                #     self.plots[i].setYRange(0,ymax[i])
                self.curves[i].setData(x = self.xf[self.flind:self.fhind], y = yf[i,:])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FFTViewer()
    # ex.show()
    sys.exit(app.exec_())