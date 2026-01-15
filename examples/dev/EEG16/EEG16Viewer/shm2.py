#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Time    : 2024/11/28 11:34
Author  : mrtang
Email   : 810899799@qq.com
"""

#coding:utf-8
import sys
if sys.version_info.major >=3 and sys.version_info.minor >= 8:    pass
else:    raise Exception('[BCIs error] Python >=3.8 is required！')
from multiprocessing import shared_memory
import numpy as np
import sys

EEG_SHM_ = "_MQ_EEG_"
INFO_SHM = "_MQ_SHM_"
EEGTYPE = 'float32'      #eeg数据类型统一为float32
INFOTYPE = 'int32'       #infos数据类型
# 一次信息更新设定50个点，按1000Hz,50ms计算。
# 考虑到读取端可能延迟，这里设置10倍冗余长度。
# 当读取端没有及时读走数据时，新数据添加到末尾。
MAXPOINTS = 50*10
EEGMAXLEN = 65*MAXPOINTS

if EEGTYPE == 'float32':
    EEGMAXBYTES = EEGMAXLEN*4
if EEGTYPE == 'float64':
    EEGMAXBYTES = EEGMAXLEN*8

INFOMAXLEN = 128

'''
eeg: float32, 专门用来存放eeg数据
info: int32, 用来存放一些参数,依次为：
      0-每次新存入数据，则该位+1, 
      1-srate
      2-eegChs
      3-includeTrigger
      4-内存区下一个新数据放置的起点位置（按照字节计算），绘图进程取完数据后应及时将该位置为0
      5-内存区下一个新数据放置的起点位置（按数据点计算），和上一个应同步修改
      6-绘图进程正在读数据，此时写进程应当等待
      7-mode #下位机模式
      8-biasconnect bias电极连接情况
      9-refconnect ref电极连接情况
      10-batlevel
      
'''

KEYS = {'index':0,'srate':1,'eegchs':2,'includetrigger':3,'curbyteindex':4,'curdataindex':5,
        'plotting':6,'mode':7,'biasconnect':8,'refconnect':9,'batlevel':10}

class BcisError(Exception):
    def __init__(self,err = 'bcis error'):
        Exception.__init__(self,err)

class CreateShm():
    def __init__(self,master = False):
        self.master = master
        self.shmEEG = None
        self.shmInfo = None
        self.shms = []
        self.eegdtype = np.dtype(EEGTYPE)
        self.infodtype = np.dtype(INFOTYPE)

        if self.master: #创建
            self.shmEEG = shared_memory.SharedMemory(create=True, size=self.eegdtype.itemsize * EEGMAXLEN,
                                                       name=EEG_SHM_)  # 申请内存
            self.shmInfo = shared_memory.SharedMemory(create=True, size=self.infodtype.itemsize * INFOMAXLEN,
                                                       name=INFO_SHM)  # 申请内存

        else:  #连接
            try:
                self.shmEEG = shared_memory.SharedMemory(name=EEG_SHM_)
                self.shmInfo = shared_memory.SharedMemory(name=INFO_SHM)
            except(FileNotFoundError):
                raise BcisError("no shm master!")

        self.shms = [self.shmEEG, self.shmInfo]
        self.eeg = np.ndarray((EEGMAXLEN,), dtype=self.eegdtype, buffer=self.shmEEG.buf)
        self.info = np.ndarray((INFOMAXLEN,), dtype=self.infodtype, buffer=self.shmInfo.buf)

    def getvalue(self,key):
        if key not in KEYS:
            raise KeyError('invalide key: %s'%(key))

        return self.info[KEYS[key]]

    def setvalue(self,key,value):
        if key not in KEYS:
            raise KeyError('invalide key: %s' % (key))

        self.info[KEYS[key]] = value

    def release(self):
        if self.master:
            for sh in self.shms:
                sh.close()
                sh.unlink()
        else:
            for sh in self.shms:
                sh.close()


