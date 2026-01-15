#coding:utf-8

from scipy.fft import fft, fftfreq
import numpy as np

class calFFT():
    def __init__(self):
        pass

    def reset(self,fs,N):
        self.N = N
        self.T = 1/fs
        self.x = np.x = np.linspace(0.0, self.N*self.T, N, endpoint=False)
        self.xf = fftfreq(self.N,self.T)[:self.N//2]

        self.ind0 = np.where(self.xf<=1)[0][-1]
        self.notch0 = np.where(self.xf>=49)[0][0]
        self.notch1 = np.where(self.xf>=51)[0][0]
        self.band1 = np.where(self.xf>=45)[0][0]
        self.band2 = np.where(self.xf>=90)[0][0]

        self.bands = []
        self.deta_band = np.where((self.xf>1) & (self.xf<4))[0]
        self.theta = np.where((self.xf>=4) & (self.xf<8))[0]
        self.alphaa = np.where((self.xf>=8) & (self.xf<=10))[0]
        self.alphab = np.where((self.xf>10) & (self.xf<13))[0]
        self.betaa = np.where((self.xf>=13) & (self.xf<20))[0]
        self.betab = np.where((self.xf>=20) & (self.xf<30))[0]

        self.bands.append(self.deta_band)
        self.bands.append(self.theta)
        self.bands.append(self.alphaa)
        self.bands.append(self.alphab)
        self.bands.append(self.betaa)
        self.bands.append(self.betab)


        self.xf = self.xf[:self.band2]

    def calculate_fft(self,X):  # x 是行向量
        if len(X.shape) == 1:   X = X[np.newaxis,:]
        X = X[:,-self.N:]
        yf = fft(X)
        yf = 2.0/self.N*np.abs(yf)

        sss = []
        for band in self.bands:
            ss = np.sum(yf[:,band],axis=1)/30
            sss.append(ss)

        bands = np.vstack(sss).transpose()
        names = ['deta','theta','alpha','alphab','betaa','betab']

        # yf[:,:self.ind0] = 0    # 去直流
        # yf[:,self.notch0:self.notch1]=0  #去工频

        eeg = {}
        eeg['band45'] = yf[:,:self.band1]
        eeg['band90'] = yf[:,:self.band2]
        eeg['bands'] = bands
        eeg['names'] = names
        return eeg
