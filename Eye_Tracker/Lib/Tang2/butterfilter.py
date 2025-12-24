#coding:utf-8

'''用于在绘图前对EEG数据进行滤波'''

from scipy import signal
import numpy as np
from shm import EEGTYPE

class ButterFilter:
    def __init__(self):
        pass

    def reset(self,srate = 250, chs = 8, fltparam = [(49,51),(0.5,45),(1,0),None]):
        # filters依次指的是：陷波，带通，高通，平滑滤波的点数，如果该选项为None,则没有这个滤波器

        print(srate)
        self.srate = srate
        self.chs = chs
        self.fltparam = fltparam
        self.padL = int(self.srate)
        self.cache = np.zeros((self.chs, self.padL), dtype=EEGTYPE)

        self.rawdata = None
        self.ndata = None
        self.bdata = None
        self.hdata = None
        self._genFilters()

    def _genFilters(self):
        '''
        构造指定频率的滤波器,陷波，0.5hz高通，0.5-45带通
        '''
        fs = self.srate/2.
        # 陷波
        self.nflt = signal.butter(N=2, Wn=[self.fltparam[0][0]/ fs, self.fltparam[0][1]/ fs], btype='stop')
        self.hflt = signal.butter(N=2, Wn=self.fltparam[2][0] / fs, btype='highpass')
        self.bflt = signal.butter(N=2, Wn=[self.fltparam[1][0] / fs, self.fltparam[1][1] / fs], btype='bandpass')

    def update(self,fdat): # fdat为一个多行序列
        if self.cache is None:   return False

        r,c = fdat.shape
        self.cache = np.hstack((self.cache,fdat)) # 将新数据拼接到末尾
        dat = self.cache.copy()
        self.ndata = signal.lfilter(self.nflt[0],self.nflt[1],dat)         # notch filter
        self.hdata = signal.lfilter(self.hflt[0],self.hflt[1],self.ndata)   # high pass
        self.bdata = signal.lfilter(self.bflt[0], self.bflt[1], self.ndata)  # band pass

        self.rawdata = self.cache[:,-c:]
        self.ndata = self.ndata[:,-c:]
        self.hdata = self.hdata[:,-c:]
        self.bdata = self.bdata[:,-c:]

        self.cache = self.cache[:,-self.padL:]
        return True