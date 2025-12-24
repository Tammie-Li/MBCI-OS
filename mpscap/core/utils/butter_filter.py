"""
Butterworth滤波器（参考采集程序/butterfilter.py）。
用于在绘图前对EEG数据进行滤波。
"""

from scipy import signal
import numpy as np


class ButterFilter:
    def __init__(self):
        self.srate = None
        self.chs = None
        self.fltparam = None
        self.padL = None
        self.cache = None
        self.rawdata = None
        self.ndata = None
        self.bdata = None
        self.hdata = None
        self.nflt = None
        self.hflt = None
        self.bflt = None

    def reset(self, srate=250, chs=8, fltparam=[(49, 51), (20, 150), (1, 0), None], eegtype='float32'):
        """
        重置滤波器。
        
        参数:
            srate: 采样率
            chs: 通道数
            fltparam: 滤波器参数，依次指的是：陷波，带通，高通，平滑滤波的点数
            eegtype: 数据类型
        """
        self.srate = srate
        self.chs = chs
        self.fltparam = fltparam
        self.padL = int(self.srate)
        self.cache = np.zeros((self.chs, self.padL), dtype=eegtype)
        self.rawdata = None
        self.ndata = None
        self.bdata = None
        self.hdata = None
        self._genFilters()

    def _genFilters(self):
        """
        构造指定频率的滤波器,陷波，高通，带通。
        """
        fs = self.srate / 2.0
        
        # 陷波滤波器
        notch_low = self.fltparam[0][0] / fs
        notch_high = self.fltparam[0][1] / fs
        # 确保频率在有效范围内
        notch_low = max(0.01, min(notch_low, 0.99))
        notch_high = max(0.01, min(notch_high, 0.99))
        if notch_low < notch_high:
            self.nflt = signal.butter(N=2, Wn=[notch_low, notch_high], btype='stop')
        else:
            self.nflt = None
        
        # 高通滤波器
        high_cut = self.fltparam[2][0] / fs
        high_cut = max(0.01, min(high_cut, 0.99))
        self.hflt = signal.butter(N=2, Wn=high_cut, btype='highpass')
        
        # 带通滤波器
        band_low = self.fltparam[1][0] / fs
        band_high = self.fltparam[1][1] / fs
        band_low = max(0.01, min(band_low, 0.99))
        band_high = max(0.01, min(band_high, 0.99))
        if band_low < band_high:
            self.bflt = signal.butter(N=2, Wn=[band_low, band_high], btype='bandpass')
        else:
            self.bflt = None

    def update(self, fdat):
        """
        更新滤波器数据。
        
        参数:
            fdat: 输入数据，形状为 (chs, samples)
        
        返回:
            是否更新成功
        """
        if self.cache is None:
            return False

        r, c = fdat.shape
        self.cache = np.hstack((self.cache, fdat))  # 将新数据拼接到末尾
        dat = self.cache.copy()
        
        # 陷波滤波
        if self.nflt is not None:
            self.ndata = signal.filtfilt(self.nflt[0], self.nflt[1], dat)
        else:
            self.ndata = dat
        
        # 高通滤波
        self.hdata = signal.filtfilt(self.hflt[0], self.hflt[1], self.ndata)
        
        # 带通滤波
        if self.bflt is not None:
            self.bdata = signal.filtfilt(self.bflt[0], self.bflt[1], self.ndata)
        else:
            self.bdata = self.ndata

        self.rawdata = self.cache[:, -c:]
        self.ndata = self.ndata[:, -c:]
        self.hdata = self.hdata[:, -c:]
        self.bdata = self.bdata[:, -c:]

        self.cache = self.cache[:, -self.padL:]
        return True












