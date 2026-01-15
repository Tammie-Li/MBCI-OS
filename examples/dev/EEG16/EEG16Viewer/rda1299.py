#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Time    : 2024/11/28 12:23
Author  : mrtang
Email   : 810899799@qq.com
"""

import threading
from threading import Event
import serial
import time
from datadecoder import DataDecoder
import serial.tools.list_ports as lp
import re

class RDA1299(threading.Thread):
    def __init__(self,pysig):
        self.ser = None
        self.threadRunning = False
        self.pysig = pysig
        self.reading = True
        self.stpEv = Event()
        self.dec = DataDecoder()
        super(RDA1299,self).__init__()
        self.setDaemon(True)

    def getallserial(self):  # 指定cp210x
        pp = re.compile('CP210x')
        ports = lp.comports()
        device = []
        for p in ports:
            id = p.device
            r = re.search(pp, str(p))
            if r is not None:
                device.append(id)
        return device

    def configDev(self,port='COM3',baudrate=460800):
        self.port = port
        self.baudrate = baudrate

    def open(self): # 注意通过None来判断串口是否有效的前提是：线程会不断读取串口，如果发现错误，立即将ser置为None
        if self.ser is None:
            try:
                self.ser = serial.Serial(port=self.port, baudrate=self.baudrate)
                if not self.threadRunning:
                    self.start()
                return True
            except:
                return False
        else:
            return True

    def close(self):
        self.stpEv.set()
        self.dec.release()

    def run(self):
        self.threadRunning = True
        clk = time.time()
        ok = False
        while not self.stpEv.is_set():
            cclk = time.time()
            rk = cclk - clk
            if rk > 0.025:  # 控制大约25ms读取一次数据
                if self.reading:
                    if self.ser is not None:
                        try:
                            buf = self.ser.read(self.ser.inWaiting())
                            stamp = time.time()
                            clk = cclk
                            ok = True
                        except:
                            ok = False
                            self.ser = None # 接收器被拔出
                            self.pysig.emit(u'接收器断开!')
                        if ok:
                            self.dec.parseData(buf,stamp)
            else:
                time.sleep(rk)
        self.threadRunning = False

    def writeCmd(self,cmd):
        self.reading = False
        self.ser.flushOutput()
        self.ser.flushInput()  # 会清空掉输入缓存中的数据

        if cmd == 'stop':
            self.ser.write('SCM'.encode('utf-8'))
        elif cmd == 'acquireEEG':
            self.ser.write('ACM'.encode('utf-8'))
        elif cmd == 'impedanceDetect':
            self.ser.write('ICM'.encode('utf-8'))
        else:
            pass

        self.reading = True

if __name__ == '__main__':
    import time
    rda = RDA1299()
    rda.open()
    time.sleep(1)
    rda.writeCmd('stop')
    print('stop')
    time.sleep(3)
    rda.writeCmd('acquireEEG')
    print('acq')
    time.sleep(5)
    rda.writeCmd('stop')
    print('stop')
    time.sleep(3)
    rda.writeCmd('impedanceDetect')
    print('imp')
    time.sleep(5)
    rda.writeCmd('stop')
    print('stop')
    time.sleep(3)
