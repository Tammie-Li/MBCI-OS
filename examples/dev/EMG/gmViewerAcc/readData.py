
from shm import EEGTYPE
import numpy as np
import matplotlib.pyplot as plt

datapath = r'D:\newGITEE\gmviewer\data.dat'

class ReadGmData():
    def __init__(self,eegtype = EEGTYPE,path = datapath):
        self.path = path
        if eegtype == 'float64':
            self.byteu = 8
            self.dtype = np.float64
        elif eegtype == 'float32':
            self.byteu = 4
            self.dtype = np.float32
        else:
            raise LookupError("unkown eegtype %s"%(eegtype))

    def readfile(self):
        buffer = b''
        with open(self.path,'rb') as f:
            buffer = f.read()

        head = np.frombuffer(buffer[0:3*self.byteu],dtype=self.dtype).astype(np.int32)
        datatype = head[0]
        self.srate = head[1]
        self.chs = head[2]
        if datatype != 1:
            raise LookupError("only eegtype supported!")

        buffer = buffer[3*self.byteu:]
        data = np.frombuffer(buffer,dtype=self.dtype)
        sampleN = int(data.size/(self.chs+1))
        data = data[:sampleN*(self.chs+1)]  #确保没有不完整数据
        data = data.reshape(sampleN, self.chs + 1)
        data = data.transpose()

        # 重新整理时间戳
        stamp = data[-1,:]
        newstamp = stamp.copy()
        ds = np.diff(stamp)
        ind = np.where(ds>0.01)[0]+1  # 找到数据包分割的起点,以20毫秒为分割标准，差值超过20毫秒的均认为是包分割的起点
        ind = np.hstack((ind, stamp.size - 1))
        # ds分割起点相对于上一个包的时间差认为是采集时间差，用以消除传输误差
        newstamp[ind] = stamp[ind] - ds[ind-1]
        ind = np.hstack((0,ind))

        allstamps = []
        for i in range(ind.size-1):
            si = ind[i]
            ei = ind[i+1]
            tmstamp = np.linspace(stamp[si],stamp[ei],ei-si+1)
            newstamp[si:ei] = tmstamp[:-1]

        self.data = data.copy()
        self.data[-1,:] = newstamp[:]
        self.data = self.data[:,10:-10] # 收尾裁切掉一些点
        plt.plot(self.data[0,:2000])
        plt.show()
        # print(np.max(data[0,:2500]) - np.min(data[0,:2500]))
        return {'srate':self.srate,'chs':self.chs,'appendstamp':True,'data':self.data}

if __name__ == '__main__':
    rd = ReadGmData(EEGTYPE,datapath)
    data = rd.readfile()