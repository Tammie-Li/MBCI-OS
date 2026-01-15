# -*- coding: utf-8 -*-
import os
import numpy as np
import time
import datetime
from shm import CreateShm,EEGTYPE

'''调试用'''

class sinGen:
    def __init__(self,chs,intervalms,srate,sinf,dtype):
        self.chs = chs
        self.intervalsec = intervalms/1000
        self.srate = srate
        self.t = 0
        self.N = int(srate*self.intervalsec)   #点数
        self.sinf = sinf
        self.dt = np.dtype(dtype)
        self.lstclkid = 0

    def get(self):
        self.tt = np.arange(0,self.N,dtype = self.dt)/self.srate
        self.tt += self.t
        data = 25*np.sin(2*np.pi*self.sinf*self.tt)
        self.t += self.intervalsec
        data = np.repeat(data, self.chs)
        n = self.tt.size
        clkid = np.arange(n)+self.lstclkid
        clkid = clkid.astype(np.uint8)
        self.lstclkid = clkid[-1] + 1
        return data,clkid,n

class signalGenerator:
    def __init__(self):
        self.shm = CreateShm(master = False)
        self.intervalsec = 0.05
        self.chs = 8
        self.srate = 250
        self.singen = sinGen(chs=self.chs, intervalms= 50,srate=self.srate,sinf=6,dtype=EEGTYPE)

    def start(self):
        count = 0
        index = 0
        t = time.perf_counter()
        while True:
            tt = time.perf_counter()
            if tt - t >= self.intervalsec:
                eeg,clkid,sampleN = self.singen.get()
                L = eeg.size
                self.shm.eeg[:L] = eeg[:]
                self.shm.info[2] = sampleN
                self.shm.info[0] = index
                self.shm.info[4] = self.chs
                self.shm.info[3] = self.srate
                self.shm.id[:sampleN] = clkid[:]
                index += 1
                t += self.intervalsec
                count += 1
                count %= 40
                if count == 0:
                    print('[signalgenerator][%s] running...'%(str(datetime.datetime.now()),))
            time.sleep(0.010)
        self.shm.release()

def device():
    s = signalGenerator()
    s.start()

def device_test():
    a = sinGen()
    for i in range(100):
        data,clkid = a.get()
        print(clkid)
        time.sleep(0.05)

if __name__ == '__main__':
    device()