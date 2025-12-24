"""
数据管理器（参考采集程序/datamanager.py）。
管理覆盖式绘图数据管理。
"""

import numpy as np


class DataManager:
    def __init__(self):
        self.data = None
        self.dmL = 0
        self.chs = 0
        self.ptr = 0
        self.packcache = None
        self.updateCof = 0

    def config(self, srate, chs, period, eegtype, fixed_points=None):
        """
        依据时间长度等要求来确定数据的管理。
        
        参数:
            srate: 采样率
            chs: 通道数
            period: 时间长度（秒）
            eegtype: 数据类型
            fixed_points: 固定点数（如果提供，则忽略srate*period）
        """
        if fixed_points is not None:
            self.dmL = int(fixed_points)
        else:
            self.dmL = int(srate * period)
        self.chs = chs
        self.data = np.zeros((self.chs, self.dmL), dtype=eegtype)
        self.packcache = np.zeros((self.chs, 1), dtype=eegtype)
        self.ptr = 0
        # 一般以20Hz速度更新
        self.updateCof = int(0.05 * srate)

    def update(self, pack):
        """
        接收数据包，将其更新在成员data中，采取滚动式更新。
        将data直接显示到gui即可。
        
        参数:
            pack: 数据包，形状为 (chs, samples)
        
        返回:
            是否更新成功
        """
        self.packcache = np.hstack((self.packcache, pack))  # 将数据拼接到末尾
        r, c = self.packcache.shape
        if c < self.updateCof:   # 要求不少于20个数据点更新一次
            return 0

        self._update(self.packcache[:, 1:])  # 注意头部添加了头
        self.packcache = np.zeros((self.chs, 1), dtype=self.packcache.dtype)
        return 1

    def _update(self, pack):
        """内部更新方法。"""
        r, c = pack.shape
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












