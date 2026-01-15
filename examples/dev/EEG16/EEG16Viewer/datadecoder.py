#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from multiprocessing import Event, Queue
import time
import math
from protocol import ProtocolV3
from shm2 import EEGTYPE,CreateShm,EEGMAXLEN

from multiprocessing import Queue,Event
from mqclient import mqClient

p = {}
p['srate'] = 500
p['mqAddr'] = '192.168.45.50'
p['mqPort'] = 1883
p['devName'] = 'EEGCH16'
p['topic'] = '/bcios/eeg'
c = {}
c['stopEv'] = Event()
c['backQue'] = Queue()

devconfig = {'vref':4.5,
          'bits':24,
          # 'gain':[24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24]}
            'gain':[12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12]}

class DataDecoder():
    def __init__(self):
        self.shm = CreateShm(master = False)
        self.decoder = Ads1299Decoder()
        self.protocol = ProtocolV3()
        self.mode = 0
        self.sampleCount = 0
        self.payloads = b''
        self.buffer = b''
        self.ids = b''
        self.tris = b''
        self.batLevel = 0

        if EEGTYPE == 'float32':
            self.typeLen = 4
        elif EEGTYPE == 'float64':
            self.typeLen = 8
            
        head = np.array((500, 16, 0), dtype=np.float32)
        self.headstr = head.tobytes()
        self.mqclient = mqClient(p, c)
        self.mqclient.start()

    def parseData(self, buffer,*args):
        stamp = args[0]
        self.buffer += buffer
        Len = len(self.buffer)
        indx = 0
        while indx < Len-7:
            self.protocol.loadBuffer(self.buffer[indx:])
            if self.protocol.headVerify(): # 头部校验成功
                includePak, pakLen = self.protocol.paklenVerify()  # 校验包长度
                if includePak:  # 长度足够容纳一个数据包
                    if self.protocol.getEpochAndVerify():  # 截取数据包并校验
                        devData = self.protocol.parsePak()
                        indx += pakLen
                        # 给当前数据包分配时间戳
                        faraway = math.ceil((Len - indx) / pakLen)  # 倒数第几个包
                        st = stamp - faraway * devData.sampleInterval # 对齐的时间戳
                        self.collectAll(devData,st)
                    else:
                        indx += 1
                else: #长度不够，跳出，下次再来
                    break
            else: #继续向后寻找
                indx += 1

        self.buffer = self.buffer[indx:]
        self.dataarange()  # 整理数据

    def collectAll(self,dat,stamp):  # 拿到一个新包，进行解析
        self.batLevel = dat.batLevel
        self.mode = dat.devMode
        if dat.devMode == 0:
            return

        self.devID = dat.devID
        self.payloads += dat.payload
        self.ids += dat.pakID
        self.tris += dat.trigger
        self.chs = dat.chs
        self.srate = dat.srate
        self.sampleCount += dat.sampleN
        self.biasconnect = dat.biasconnect
        self.refconnect = dat.refconnect

        # self.timestamps.append(stamp)

    def dataarange(self): # 调用者已经控制了节奏
        self.shm.setvalue('batlevel', self.batLevel)
        self.shm.setvalue('mode',self.mode)
        # print(self.mode)
        if self.mode == 0:
            return

        while self.shm.getvalue('plotting'):
            time.sleep(0.001)

        if len(self.payloads) == 0:
            return

        dataay = self.decoder.decode(self.payloads, self.sampleCount, self.chs)
        
        buf = self.headstr + dataay.astype(np.float32).tobytes()
        self.mqclient.pub(buf)
        
        
        L = dataay.size

        self.shm.setvalue('biasconnect', self.biasconnect)
        self.shm.setvalue('refconnect', self.refconnect)
        self.shm.setvalue('srate', self.srate)
        self.shm.setvalue('eegchs', self.chs)

        curdataindex = self.shm.getvalue('curdataindex')
        if curdataindex + L > EEGMAXLEN:
            curdataindex = 0

        self.shm.eeg[curdataindex:curdataindex+L] = dataay[:]
        self.shm.setvalue('curdataindex', curdataindex + L)
        self.shm.setvalue('curbyteindex', (curdataindex + L)*self.typeLen)
        self.shm.info[0] += 1

        self.sampleCount = 0
        self.payloads = b''
        self.timestamps = []
        self.ids = b''
        self.tris = b''

class Ads1299Decoder():
    def __init__(self,config = devconfig):
        self.rawdt = np.dtype('int32')
        self.rawdt = self.rawdt.newbyteorder('>')
        vref = config['vref']
        bits = config['bits']
        gain = config['gain']
        self.facs = np.array([self._calFac(vref,bits,g) for g in gain])
        self.facs = self.facs[np.newaxis,:]  # 组织为二维数组

    def _calFac(self,vref,bits,gain):
        return vref / (gain*(2**bits - 1)) * 1e6

    def getchs(self,payloads,N): # payloads必须是一个采样数据包的
        return int(len(payloads)/3/N)

    def _tobuf32(self, buf24):
        if buf24[0] > 127:
            return b'\xff' + buf24[:3]
        else:
            return b'\x00' + buf24[:3]

    def decode(self,payloads,sampleN,chs):
        tmbuf = [self._tobuf32(payloads[i:i + 3]) for i in range(0, len(payloads), 3)]
        buf = b''.join(tmbuf)
        eeg = np.frombuffer(buf, dtype=self.rawdt).astype(EEGTYPE).reshape(sampleN,chs) #组织为列向量
        fac = np.repeat(self.facs,sampleN,axis=0)
        eeg = eeg * fac
        return eeg.flatten()
