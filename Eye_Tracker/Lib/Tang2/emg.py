# coding:utf-8
import os
import numpy as np
from multiprocessing import Event, Queue
import time
import datetime
import re
import serial
from Lib.Tang2.shm import CreateShm,EEGTYPE,EEGMAXBYTES,EEGMAXLEN,MAXPOINTS,ACCTYPE,ACCMAXBYTES
import time
import math
import csv
from Lib.Tang2.rdaclient import RDA
from Lib.Tang2.adcdecoder import DevDecoder
from Lib.Tang2.protocol import Protocol
import threading

SRATES = [250,500,1000,2000]
ULEN = [2,2,3,4]

class EMGRecorder(threading.Thread):
    def __init__(self, param, ctrprm):
        '''
        e.g. param:
        param = {'type':'s','param':{'port':'COM5','baudrate':460800}}
        param = {'type':'u','param':{'ip':'127.0.0.1','port':8989}}

        e.g. ctrprm:
        '''
        super().__init__()
        self.setDaemon(True)

        self.shm = CreateShm(master = True)
        self.stopEv = ctrprm['stopEv']
        self.backQue = ctrprm['backQue']
        self.devprocount = param['devprocount']

        self.saveFlg = 0
        self.newEegUpdate = 0
        self.buffer = b''
        self.payloads = b''
        self.accbytes = b''
        self.IDbytes = b''
        self.triggerbytes = b''
        self.timestamps = []

        self.status = 0
        self.sampleCount = 0
        self.batLevel = 10
        self.lsttimeclk = 0
        self.chs = 0
        self.srate = 0
        self.virgin = True

        self.temporal_data = None        


        self.rda = RDA(param)
        self.rdaOK = self.rda.open()
        self.protocol = Protocol()
        self.devDec = DevDecoder()

    def release(self):
        # 考虑意外退出
        try:
            self.file.close()
        except:
            pass

        self.shm.info[8] = 0

        self.rda.close()
        self.shm.release()
        self.stopEv.clear()

    def run(self):
        if not self.rdaOK:
            self.release()
            return

        self.shm.info[3] = self.devprocount  # 记录dev第几次启动

        self.timestamps = []
        self.sampleCount = 0
        self.payloads = b''
        self.chs = 0

        while not self.stopEv.is_set():
            err, buf, stamp = self.rda.read()
            if err is not None:
                self.backQue.put(err)
                break

            self.unpack(buf,stamp)

        self.release()

    def unpack(self, buffer,*args):
        stamp = args[0]
        self.buffer += buffer      # 拼接到末尾
        Len = len(self.buffer)
        indx = 0
        while indx < Len-9:
            self.protocol.loadBuffer(self.buffer[indx:])
            if self.protocol.headVerify(): # 头部校验成功
                includePak, pakLen = self.protocol.paklenVerify()  # 校验包长度
                if includePak:  # 长度足够容纳一个数据包
                    if self.protocol.getEpochAndVerify(pakLen):  # 截取数据包并校验
                        self.protocol.parsePak()
                        indx += pakLen
                        # 给当前数据包分配时间戳
                        faraway = math.ceil((Len - indx) / pakLen)  # 倒数第几个包
                        st = stamp - faraway * self.protocol.devData.sampleInterval # 对齐的时间戳
                        self.collectAll(self.protocol.devData,st)
                    else:
                        indx += 1
                else: #长度不够，跳出，下次再来
                    break
            else: #继续向后寻找
                indx += 1

        self.buffer = self.buffer[indx:]
        self.dataarange()  # 整理数据

    def collectAll(self,dat,stamp):  # 拿到一个新包，进行解析
        self.sampleCount += dat.sampleN
        self.payloads += dat.payload
        self.batLevel = dat.batLevel
        self.IDbytes += dat.pakID
        self.triggerbytes += dat.trigger
        self.accbytes += dat.accBytes
        self.timestamps.append(stamp)

    def dataarange(self):
        if self.sampleCount <= 0:
            return

        clk = time.perf_counter()
        if clk - self.lsttimeclk < 0.05:  return   # 控制50ms左右更新一次
        self.lsttimeclk = clk

        # 初次写入数据参数
        if self.virgin:
            self.shm.info[7] = self.protocol.devData.includeTri
            self.shm.info[6] = self.protocol.devData.includeID
            self.chs = self.devDec.decoder.getchs(self.protocol.devData.payload, self.protocol.devData.sampleN)
            self.shm.info[5] = self.chs
            self.shm.info[4] = self.protocol.devData.srate
            self.shm.info[2] = 0   # 新数据包含的数据包个数
            self.shm.info[1] = 0   # 新数据的字节数
            self.shm.info[11] = 0  # 新数据包含的采样点数
            self.shm.info[12] = 0  # 新数据的字节数
            self.shm.info[14] = 0  # acc数据字节数
            self.shm.info[0] = 0   # 新数据更新标志
            self.virgin = False
            return

        self.shm.info[13] = self.protocol.devData.includeAcc

        dataay = self.devDec.decoder.decode(self.payloads,self.sampleCount,self.chs)


        # 如果绘图端正在读数据，则这里应当等待
        while self.shm.pinfo[0]:
            time.sleep(0.001)

        # 开始写数据
        Leeg = dataay.size         # 数据序列总长度
        Lid = len(self.IDbytes)
        Ltri = len(self.triggerbytes)
        Lacc = len(self.accbytes)
        self.newEegUpdate += 1
        curPoint = self.shm.info[1]  # 当前存储序列中末尾点位
        curpakN = self.shm.info[2]   # 当前存储序列中数据包个数
        curaccN = self.shm.info[14]

        # fft绘图端
        curPointFFT = self.shm.info[11]  # 当前存储序列中末尾点位
        curpakFFTN = self.shm.info[12]  # 当前存储序列中采样点数

        if curPoint + Leeg > EEGMAXLEN:
            curPoint = 0      # 超过了缓存最大长度，强制赋0
            curpakN = 0

        # fft绘图端
        if curPointFFT + Leeg > EEGMAXLEN:
            curPointFFT = 0      # 超过了缓存最大长度，强制赋0
            curpakFFTN = 0

        if curaccN + Lacc > ACCMAXBYTES:
            curaccN = 0
        
        curaccN = 0
        curpakN = 0

        self.shm.eeg0[curPoint:curPoint+Leeg] = dataay[:]
        self.shm.eeg1[curPointFFT:curPointFFT + Leeg] = dataay[:]
        self.shm.shm_acc.buf[curaccN:curaccN+Lacc] = self.accbytes
        self.shm.shm_id.buf[curpakN:curpakN+Lid] = self.IDbytes  # 丢包测试id
        self.shm.shm_tri.buf[curpakN:curpakN+Ltri] = self.triggerbytes    # trigger

        # pas = int((curaccN + Lacc)/2)
        # pns = int(pas/3)
        # accay = self.shm.acc[:pas].astype(EEGTYPE)*9.8/4096
        # accay = accay.reshape(pns, 3)
        
        accay = np.frombuffer(self.accbytes,dtype=np.int16).astype(EEGTYPE)*9.8/4096
        pas = accay.size
        accay = accay.reshape(int(pas/3),3)


        # 保存数据相关
        if self.saveFlg == 0:
            if self.shm.info[8] == 1:  # 开启保存
                pth = self.shm.getPath()
                # 写二进制文件
                # 依次写入eegtype:1-eeg 2-evt,srate,chs,
                try:
                    self.file.close()
                except:
                    pass

                self.file = open(pth, 'wb')
                ay = np.array([1, self.srate, self.chs], dtype=EEGTYPE)
                self.file.write(ay.tobytes())  # 头信息
                self.saveFlg = self.shm.info[8]

        else:  #self.saveFlg == 1:
            if self.shm.info[8] == 0:  # 结束保存
                self.file.close()
                self.saveFlg = self.shm.info[8]

            else:  # 正常保存
                ndataay = dataay.reshape(self.sampleCount,self.chs)
                stamp = np.array([self.timestamps]).transpose()
                ndata_col_ay = np.hstack((ndataay, accay, stamp))        # 将时间戳添加到最后一列
                ndata_col_ay = ndata_col_ay.flatten()
                self.file.write(ndata_col_ay.tobytes())
                self.saveFlg = self.shm.info[8]

        emg_data = dataay.reshape(self.sampleCount, self.chs)
        acc_data = accay.reshape(self.sampleCount, 3)
        tmp_data = np.concatenate((emg_data, acc_data), axis=1)

        if self.temporal_data is None: 
            self.temporal_data = tmp_data
        else: 
            if (self.temporal_data.shape[0] + self.sampleCount) < 256:
                self.temporal_data = np.concatenate((self.temporal_data, tmp_data))
            else:
                start = self.temporal_data.shape[0] + self.sampleCount - 256
                self.temporal_data = np.concatenate((self.temporal_data[start: ], tmp_data))

        self.shm.info[10] = self.batLevel
        self.shm.info[1] = curPoint + Leeg
        self.shm.info[2] = curpakN + self.sampleCount
        self.shm.info[11] = curPointFFT + Leeg
        self.shm.info[12] = curpakFFTN + self.sampleCount
        self.shm.info[14] = curaccN + Lacc
        self.shm.info[0] = self.newEegUpdate
        self.payloads = b''
        self.accbytes = b''
        self.IDbytes = b''
        self.triggerbytes = b''
        self.sampleCount = 0
        self.timestamps = []


    def get_data(self, time):
        tmp_point = int(time * 512)
        data_len = self.temporal_data.shape[0]
        if data_len < tmp_point:
            print("目前采集数据时间过短,需要再等待...") #  一般不会出现这种情况
            return np.random.randint(tmp_point, 3)
        return self.temporal_data.T

def devicepro(parm,ctrprm):
    am = EMGRecorder(parm,ctrprm)
    am.start()

if __name__ == '__main__':
    parm = {'type':'s','param':{'port':'COM5','baudrate':460800,'devprocount':0},'devprocount':0}
    ctrparm = {}
    ctrparm['stopEv'] = Event()
    ctrparm['backQue'] = Queue()
    devicepro(parm,ctrparm)
