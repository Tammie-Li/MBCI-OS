#coding:utf-8

import serial
import time

# 封装的读取数据的客户端，它可以是udp socket, 也可以是uart串口
# param: 字典
# 示例
# param = {'type':'s','param':{'port':'COM5','baudrate':460800}} # 串口
# param = {'type':'u','param':{'ip':'127.0.0.1','port':8989}}    # udp socket

class RDA():
    def __init__(self,param):
        self.param = param
        self.readOncelen = 128

    def open(self):
        if self.param['type'] == 's':
            try:
                print(self.param)
                self._rda = serial.Serial(port=self.param['param']['port'], baudrate=self.param['param']['baudrate'])
                self.readOncelen = 128
                return True
            except:
                self._rda = None # 打开失败
                return False
        else:
            self.readOncelen = 65535 # udp
            return False

    def updateReadOncelen(self,len):
        if self.param['type'] == 's':
            self.readOncelen = len
        else:
            self.readOncelen = 65535

    def read(self):
        if self.param['type'] == 's':
            try:
                buffer = self._rda.read(self.readOncelen)
                stamp = time.time()
                return None,buffer,stamp
            except:
                return u'接收器被拔出',None,None

    def close(self):
        if self.param['type'] == 's':
            try:
                self._rda.close()
            except:
                pass


