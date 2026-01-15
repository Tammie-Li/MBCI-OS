# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import socket
from struct import unpack
import numpy as np
import paho.mqtt.client as mqtt
import time
from multiprocessing import Event,Queue
from mqclient import mqClient
BPresolution = 0.0488

class DataServer():
    _update_interval = 0.04  #博瑞康设备数据传输间隔
    def __init__(self,param,ctrprm):
        self.eegChs = param['eegChs']
        self.includeTrigger = param['includeTrigger']
        self.srate = param['srate']
        self.devAddr = param['devAddr']
        self.devPort = param['devPort']  # 51244
        self.stopEv = ctrprm['stopEv']
        self.backQue = ctrprm['backQue']

        self.mqclient = mqClient(param,ctrprm)
        if self.mqclient.start():
            self.workloop()
        else:
            self.backQue.put('进程退出！')

    #主工作循环
    def workloop(self):
        if not self.deviceConnect():
            print("设备连接失败！进程退出！")
            self.backQue.put('设备连接失败！进程退出！')
            self.sock.close()
            self.mqclient.stop()
            return

        self.backQue.put('设备连接成功！')
        self._work()

    #设备连接，尝试连接3次
    def deviceConnect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setblocking(True)
        connected = False
        for i in range(3):
            try:
                self.sock.connect((self.devAddr, self.devPort))
                self.backQue.put('连接中...')
                connected = True
                break
            except:
                time.sleep(1)

        return connected

    def GetProperties(self, rawdata):
        (channelCount, samplingInterval) = unpack('<Ld', rawdata[:12])
        # Extract resolutions
        resolutions = []
        for c in range(channelCount):
            index = 12 + c * 8
            restuple = unpack('<d', rawdata[index:index + 8])
            resolutions.append(restuple[0])
        return (channelCount, samplingInterval)

    def splitString(self, raw):
        stringlist = []
        s = b""
        for i in range(len(raw)):
            if raw[i:i+1] != b'\x00':  # 使用字节字符串作为空字节的比较
                s += raw[i:i+1]  # 将字节添加到字节字符串中
            else:
                stringlist.append(s.decode())
                s = b""
        return stringlist

    def getEEG(self, rawdata, channelCount):
        # Extract numerical data
        (block, points, markerCount) = unpack('<LLL', rawdata[:12])

        # Extract eeg data as array of floats
        data = np.frombuffer(rawdata[12:12+4*points*channelCount],dtype=np.float32)*BPresolution
        data = data.reshape(points,channelCount)

        # Extract markers
        markeray = np.zeros((points,1), dtype=np.float32)
        markers = []
        index = 12 + 4 * points * channelCount
        for m in range(markerCount):
            markersize = unpack('<L', rawdata[index:index+4])
            position, points, channel = unpack('<LLl', rawdata[index + 4:index + 16])
            typedesc = self.splitString(rawdata[index+16:index+markersize[0]])
            markeray[position,0] = float(typedesc[1][1:])
            index = index + markersize[0]

        eegay = np.hstack((data,markeray)).flatten()
        return (block, points, eegay)

    def recvData(self, requestedSize):
        disconnect = False
        returnStream = b''
        while len(returnStream) < requestedSize:
            databytes = self.sock.recv(requestedSize - len(returnStream))
            if databytes == b'':
                self.backQue.put('连接断开')
                disconnect = True
            returnStream += databytes

        return disconnect,returnStream

    def _work(self):
        self.backQue.put('数据传输中...')
        if self.includeTrigger:
            head = np.array((self.srate,self.eegChs,1),dtype=np.float32)
        else:
            head = np.array((self.srate, self.eegChs, 0), dtype=np.float32)

        headstr = head.tobytes()

        channelCount = 0
        while not self.stopEv.is_set():
            disconnect,rawhdr = self.recvData(24)
            if disconnect:
                return

            (id1, id2, id3, id4, msgsize, msgtype) = unpack('<llllLL', rawhdr)
            disconnect, rawdata = self.recvData(msgsize - 24)
            if disconnect:
                return

            if msgtype == 1:
                # Start message, extract eeg properties and display them
                (channelCount, samplingInterval) = self.GetProperties(rawdata)
                if channelCount != self.eegChs:
                    self.eegChs = channelCount
                    head[1] = channelCount
                    headstr = head.tobytes()
            elif msgtype == 4:
                if channelCount != 0:
                    (block, points, eeg) = self.getEEG(rawdata, channelCount)
                    # # 通过mqtt转发出去
                    ## 转发数据格式
                    ## head + raw
                    ## head: srate + nchan: 转换为float32字节序列（小端在前）
                    ## raw: float32字节序列（小端在前），按照如下顺序排列[ch1,ch2,..chn](采样点1)，[ch1,ch2,..chn](采样点2)...
                    buf = headstr + eeg.tobytes()
                    self.mqclient.pub(buf)
                    print(len(buf))
            elif msgtype == 3:
                self.backQue.put('放大器关闭，进程退出...')
                break

        self.sock.close()

def devicepro(param,ctrparam):
    ds = DataServer(param, ctrparam)
    ds.workloop()


if __name__ == '__main__':
    param = {'devName':'BP','eegChs':10,'includeTrigger':True,'srate':250,'devAddr':'127.0.0.1','devPort':51244,'mqAddr':'127.0.0.1','mqPort':1883,'topic':'/bcios/eeg'}
    ctrparam = {'stopEv':Event(),'backQue':Queue()}
    devicepro(param,ctrparam)