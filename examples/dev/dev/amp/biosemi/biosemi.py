# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import socket
from struct import unpack
import numpy as np
import paho.mqtt.client as mqtt
import time
from multiprocessing import Event,Queue

class DataServer():
    _update_interval = 0.04  #博瑞康设备数据传输间隔
    def __init__(self,param,ctrprm):
        self.device = param['device']
        self.n_chan = param['n_chan']
        self.srate = param['srate']
        self.devip = param['devip']
        self.devport = param['devport'] # 1111 for biosemi
        self.mqaddr = param['mqaddr']
        self.mqport = param['mqport']
        self.topicname = param['topic']
        self.stopEv = ctrprm['stopEv']
        self.backQue = ctrprm['backQue']
        self.mqconnect = None
        self.tcpsamples = 4
        self.buffer_size = self.n_chan * self.tcpsamples * 3

        self.createMqttClient()
        for i in range(100):
            if self.mqconnect != None:
                if self.mqconnect == 0:
                    # print('连接成功')
                    self.backQue.put('mqtt连接到服务器成功！')
                else:
                    # print('连接失败')
                    self.backQue.put('mqtt连接到服务器失败！')
                break
            time.sleep(0.1)

        if self.mqconnect != None and self.mqconnect == 0:
            self.workloop()
        self.backQue.put('进程退出！')

    #创建mqtt client，并连接到服务器
    def createMqttClient(self):
        # 利用时间戳产生唯一的节点名称
        timestampStr = str(int(time.time()*100))
        nodename = self.device + '-' + timestampStr
        self.mqClient = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,nodename)
        self.mqClient.on_connect = self.on_connectMq
        self.mqClient.connect(self.mqaddr,self.mqport,60)
        self.mqClient.loop_start()

    #mqttl连接回调
    def on_connectMq(self,client,userdata,flags,rc,properties):
        self.mqconnect = rc
        # if rc != 0:
        #     # self.backQue.put('mqtt连接服务器失败！')
        #     print('mq 失败')
        # else:
        #     print('mq 成功')

    #主工作循环
    def workloop(self):
        if not self.deviceConnect():
            self.backQue.put('设备连接失败！进程退出！')
            self.sock.close()
            self.mqClient.loop_stop()
            self.mqClient.disconnect()
            return

        self.backQue.put('设备连接成功！')
        self.read()

    #设备连接，尝试连接3次
    def deviceConnect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setblocking(True)
        connected = False
        for i in range(3):
            try:
                self.sock.connect((self.devip, self.devport))
                self.backQue.put('连接中...')
                connected = True
                break
            except:
                time.sleep(1)

        return connected

    def read(self):
        """
        Read signal from the EEG device
        :param duration: How long to read in seconds
        :return: Signal in the matrix form: (samples, channels)
        """
        self.backQue.put('数据传输中...')
        self.bufsize = int(self._update_interval * 4 * self.n_chan * self.srate * 10)  # set buffer size

        head = np.array((self.srate,self.n_chan),dtype=np.float32).tobytes()

        # initialize final data array
        rawdata = np.empty((0, self.n_chan))

        # The reader process will run until requested amount of data is collected
        samples = 0
        while samples < self._update_interval * self.srate and (not self.stopEv.is_set()):
            try:
                # Create a 16-sample signal_buffer
                signal_buffer = np.zeros((self.n_chan, self.tcpsamples))

                # Read the next packet from the network
                # sometimes there is an error and packet is smaller than needed
                # read until get a good one
                data = []
                while len(data) != self.buffer_size:
                    data = self.sock.recv(self.buffer_size)


                # Extract 16 samples from the packet (ActiView sends them in 16-sample chunks)
                for m in range(self.tcpsamples):
                    # extract samples for each channel
                    for ch in range(self.n_chan):
                        offset = m * 3 * self.n_chan + (ch * 3)

                        # The 3 bytes of each sample arrive in reverse order
                        sample = (data[offset+2] << 16)
                        sample += (data[offset+1] << 8)
                        sample += data[offset]

                        # Store sample to signal buffer
                        signal_buffer[ch, m] = sample

                # update sample counter
                samples += self.tcpsamples

                # transpose matrix so that rows are samples
                signal_buffer = np.transpose(signal_buffer)

                # add to the final dataset
                rawdata = np.concatenate((rawdata, signal_buffer), axis=0)
            except:
                self.sock.close()
                self.backQue.put('连接断开')
                return  
        
        # # 通过mqtt转发出去
        ## 转发数据格式
        ## head + raw
        ## head: srate + nchan: 转换为float32字节序列（小端在前）
        ## raw: float32字节序列（小端在前），按照如下顺序排列[ch1,ch2,..chn](采样点1)，[ch1,ch2,..chn](采样点2)...
        buf = head + rawdata
        self.mqClient.publish(self.topicname, buf, 0)
            
        self.sock.close()
        return rawdata

def devicepro(param,ctrparam):
    ds = DataServer(param, ctrparam)
    ds.start()


if __name__ == '__main__':
    param = {'device':'Biosemi','n_chan':65,'srate':1024,'devip':'127.0.0.1','devport':1111,'mqaddr':'127.0.0.1','mqport':1883,'topic':'/bcios/eeg'}
    ctrparam = {'stopEv':Event(),'backQue':Queue()}
    ds = DataServer(param,ctrparam)
    ds.start()