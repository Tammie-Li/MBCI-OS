#coding:utf-8
import sys
if sys.version_info.major >=3 and sys.version_info.minor >= 8:    pass
else:    raise Exception('[BCIs error] Python >=3.8 is required！')
from multiprocessing import shared_memory
import numpy as np
import sys

EEG0_SHM_ = "_BCIs_EEG0_"
EEG1_SHM_ = "_BCIs_EEG1_"
ACC_SHM_ = "_BCIs_ACC_"
ID_SHM_ = "_BCIs_ID_"
TRI_SHM = "_BCIs_TRI_"
INFO_SHM = "_BCIs_INFOS_"
PINFO_SHM = "_BCIs_PINFOS_"
EEGTYPE = 'float64'      #eeg数据类型统一为float64
INFOTYPE = 'int32'       #infos数据类型
ACCTYPE = 'int16'
IDTRITYPE = 'uint8'
PTH_SHM_ = "_BCIs_PATH_"

# 一次信息更新设定50个点，按1000Hz,50ms计算。
# 考虑到读取端可能延迟，这里设置10倍冗余长度。
# 当读取端没有及时读走数据时，新数据添加到末尾。
MAXPOINTS = 50*10
EEGMAXLEN = 32*MAXPOINTS   #最多支持32通道
ACCMAXLEN = 9*MAXPOINTS
ACCMAXBYTES = 2*ACCMAXLEN
INFOMAXLEN = 128
EEGMAXBYTES = EEGMAXLEN * 4

'''
eeg: float64, 专门用来存放eeg数据
id:  uint8, 用来存放丢包测试数据
trigger: uint8, 用来存放trigger
info: int32, 用来存放一些参数,依次为：
      0-ampindex, 
      1-当前共享内存区中存储的EEG数据的字节数。消费者在使用完数据后，应当及时将该标志位清零。生产者每次将新产生的数据依据该标志位来添加到旧数据末尾。
      2-当前共享内存区中存储的EEG数据的采样点数。作用同上。
      3-device进程第几次启动
      4-srate
      5-chs
      6-includeID
      7-includeTri
      8-savedata or not
      9-pathLen
      10-电量, 0-10代表0%~100%
      11-作用同1，用于和fft绘图端交互
      12-作用同2，用于和fft绘图端交互
      13-包含ACC数据否

plotinfo: uint8
      0-plot进程正在读数据
'''

class BcisError(Exception):
    def __init__(self,err = 'bcis error'):
        Exception.__init__(self,err)

class CreateShm():
    def __init__(self,master = False):
        self.master = master
        self.shm_eeg0 = None
        self.shm_eeg1 = None
        self.shm_acc = None
        self.shm_info = None
        self.shm_pinfo = None
        self.shm_id = None
        self.shm_tri = None
        self.shm_pth = None
        self.shms = []
        self.eegdtype = np.dtype(EEGTYPE)
        self.accdtype = np.dtype(ACCTYPE)
        self.infodtype = np.dtype(INFOTYPE)
        self.idtritype = np.dtype(IDTRITYPE)

        if self.master: #创建
            self.shm_eeg0 = shared_memory.SharedMemory(create=True, size=self.eegdtype.itemsize * EEGMAXLEN,
                                                       name=EEG0_SHM_)  # 申请内存
            self.shm_eeg1 = shared_memory.SharedMemory(create=True, size=self.eegdtype.itemsize * EEGMAXLEN,
                                                       name=EEG1_SHM_)  # 申请内存
            self.shm_acc = shared_memory.SharedMemory(create=True, size=self.accdtype.itemsize * ACCMAXLEN,
                                                      name=ACC_SHM_)  # 申请内存
            self.shm_info = shared_memory.SharedMemory(create=True, size=self.infodtype.itemsize * INFOMAXLEN,
                                                       name=INFO_SHM)  # 申请内存
            self.shm_id = shared_memory.SharedMemory(create=True, size=self.idtritype.itemsize * MAXPOINTS,
                                                       name=ID_SHM_)  # 申请内存
            self.shm_tri = shared_memory.SharedMemory(create=True, size=self.idtritype.itemsize * MAXPOINTS,
                                                       name=TRI_SHM)  # 申请内存
            self.shm_pinfo = shared_memory.SharedMemory(create=True, size=8,
                                                      name=PINFO_SHM)  # 申请内存
            self.shm_pth = shared_memory.SharedMemory(create=True, size=512,
                                                      name=PTH_SHM_)  # 申请内存

        else:  #连接
            try:
                self.shm_eeg0 = shared_memory.SharedMemory(name=EEG0_SHM_)
                self.shm_eeg1 = shared_memory.SharedMemory(name=EEG1_SHM_)
                self.shm_acc = shared_memory.SharedMemory(name=ACC_SHM_)
                self.shm_info = shared_memory.SharedMemory(name=INFO_SHM)
                self.shm_pinfo = shared_memory.SharedMemory(name=PINFO_SHM)
                self.shm_id = shared_memory.SharedMemory(name=ID_SHM_)
                self.shm_tri = shared_memory.SharedMemory(name=TRI_SHM)
                self.shm_pth = shared_memory.SharedMemory(name=PTH_SHM_)
            except(FileNotFoundError):
                raise BcisError("no shm master!")

        self.shms = [self.shm_eeg0, self.shm_eeg1, self.shm_acc, self.shm_info, self.shm_pinfo, self.shm_id, self.shm_tri, self.shm_pth]
        self._mapAy2Shm()

    def _mapAy2Shm(self):
        self.eeg0 = np.ndarray((EEGMAXLEN,), dtype=self.eegdtype, buffer=self.shm_eeg0.buf)
        self.eeg1 = np.ndarray((EEGMAXLEN,), dtype=self.eegdtype, buffer=self.shm_eeg1.buf)
        self.acc = np.ndarray((ACCMAXLEN,), dtype=self.accdtype, buffer=self.shm_acc.buf)
        self.info = np.ndarray((INFOMAXLEN,), dtype=self.infodtype, buffer=self.shm_info.buf)
        self.pinfo = np.ndarray((8,), dtype=self.idtritype, buffer=self.shm_pinfo.buf)
        self.id = np.ndarray((MAXPOINTS,), dtype=self.idtritype, buffer=self.shm_id.buf)
        self.tri = np.ndarray((MAXPOINTS,), dtype=self.idtritype, buffer=self.shm_tri.buf)

    def setPath(self,pth):
        pth = pth.encode('utf-8')
        L = len(pth)
        self.info[9] = L
        self.shm_pth.buf[:L] = pth

    def getPath(self):
        L = self.info[9]
        pth = bytearray(self.shm_pth.buf[:L]).decode('utf-8')
        return pth

    def release(self):
        if self.master:
            for sh in self.shms:
                sh.close()
                sh.unlink()
        else:
            for sh in self.shms:
                sh.close()

