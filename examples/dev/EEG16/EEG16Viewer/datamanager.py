#coding:utf-8
import numpy as np
import math

'''管理覆盖式绘图数据管理'''

class DataManager():
    def __init__(self):
        pass

    # 依据时间长度等要求来确定数据的管理
    def config(self,srate,chs,period,eegtype):
        self.dmL = srate*period
        self.chs = chs
        self.data = np.zeros((self.chs, self.dmL), dtype=eegtype)
        self.packcache = np.zeros((self.chs,1))
        self.pack = None
        self.ptr = 0
        # 一般以20Hz速度更新
        self.updateCof = int(0.05*srate)

    # 接收数据包，将其更新在成员data中，采取滚动式更新
    # 将data直接显示到gui即可
    def update(self,pack):
        self.packcache = np.hstack((self.packcache, pack))  # 将数据拼接到末尾
        r,c = self.packcache.shape
        if c < self.updateCof:   # 要求不少于20个数据点更新一次
            return 0

        self._update(self.packcache[:,1:])  # 注意头部添加了头
        self.packcache = np.zeros((self.chs, 1))
        return 1

    def _update(self,pack):
        r,c = pack.shape
        sp = self.dmL - self.ptr  # 距离末尾的点数
        if sp > c:  # 末尾足够容纳一个数据包
            self.data[:, self.ptr:self.ptr + c] = pack  # 直接添加到末尾
            self.ptr += c  # 指针增加
        elif sp == c:
            self.data[:, self.ptr:self.ptr + c] = pack  # 直接添加到末尾
            self.ptr = 0  # 指针复位
        else:  # 末尾不足容纳一个数据包
            self.data[:, self.ptr:self.ptr + sp] = pack[:, :sp]  # 一部分添加到末尾
            self.data[:, 0:c - sp] = pack[:, sp:]  # 一部分添加到头部
            self.ptr = c - sp
        return 1


if __name__ == "__main__":
    dm = DataManager()
    dm.config(40,1,4,55)
    s = 0
    for i in range(200):
        pack = np.arange(s,s+5)
        s += 5
        pack = pack[np.newaxis,:]
        if dm.update(pack):
            print(dm.data[0,:])

