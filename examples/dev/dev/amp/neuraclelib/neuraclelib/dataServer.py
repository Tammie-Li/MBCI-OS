# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import socket
from struct import unpack
import numpy as np
import time
from multiprocessing import Event,Queue
from mqclient import mqClient

EEGCHS = 64
TRIGGERCHS = 1

class DataServer():
    _update_interval = 0.04  #博瑞康设备数据传输间隔
    def __init__(self,param,ctrprm):
        self.srate = param['srate']
        self.devAddr = param['devAddr']
        self.devPort= param['devPort'] # 8712
        self.stopEv = ctrprm['stopEv']
        self.backQue = ctrprm['backQue']
        self.mqconnect = None

        self.mqclient = mqClient(param, ctrprm)
        if self.mqclient.start():
            self.workloop()
        else:
            self.backQue.put('进程退出！')

    #主工作循环
    def workloop(self):
        if not self.deviceConnect():
            self.backQue.put('设备连接失败！进程退出！')
            self.sock.close()
            self.mqclient.stop()
            return

        self.backQue.put('设备连接成功！')
        print("设备连接成功")
        self._work()

    #设备连接，尝试连接3次
    def deviceConnect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setblocking(True)
        connected = False
        for i in range(3):
            try:
                # print(self.devAddr)
                # print(self.devPort)
                self.sock.connect((self.devAddr, self.devPort))
                print('链接成功')
                self.backQue.put('连接中...')
                connected = True
                break
            except:
                time.sleep(1)

        return connected

    def _work(self):
        n_chan = EEGCHS + TRIGGERCHS

        self.backQue.put('数据传输中...')
        self.bufsize = int(self._update_interval * 4 * n_chan * self.srate)  # set buffer size
        self.buffer = b''  ## binary buffer used to collect binary array from data server

        # head: srate, number of eeg, number of trigger
        # 博瑞康的
        head = np.array((self.srate,EEGCHS,TRIGGERCHS),dtype=np.float32).tobytes()

        while not self.stopEv.is_set():
            try:
                raw = self.sock.recv(self.bufsize)
            except:
                self.sock.close()
                self.backQue.put('连接断开')
                return

            # # 通过mqtt转发出去
            ## 转发数据格式
            ## head + raw
            ## head: srate + nchan: 转换为float32字节序列（小端在前）
            ## raw: float32字节序列（小端在前），按照如下顺序排列[ch1,ch2,..chn](采样点1)，[ch1,ch2,..chn](采样点2)...
            buf = head + raw
            self.mqclient.pub(buf)
            print(len(buf))

        self.sock.close()

def devicepro(param,ctrparam):
    ds = DataServer(param, ctrparam)
    ds.workloop()


if __name__ == '__main__':
    param = {'devName':'Neuracle','srate':1000,'devAddr':'127.0.0.1','devPort':8712,'mqAddr':'192.168.218.50','mqPort':1883,'topic':'/bcios/eeg'}
    ctrparam = {'stopEv':Event(),'backQue':Queue()}
    devicepro(param,ctrparam)