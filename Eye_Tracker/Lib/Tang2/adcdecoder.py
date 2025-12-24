# coding:utf-8
import numpy as np
from Lib.Tang2.shm import EEGTYPE
import json

decPth = './dec.js'

class DevDecoder():
    def __init__(self):
        with open(decPth,'rb') as f:
            buf = f.read()
        config = json.loads(buf)

        self._decoders = [Ads1299Decoder(config['Ads1299']),Ads1284Decoder(config['Ads1284']),Ads1263Decoder(config['Ads1263'])]
        self.decoder = self._decoders[0]

    def setDecoder(self,devID = 0):
        self.decoder = self._decoders[devID]

class Ads1299Decoder():
    def __init__(self,config):
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

class Ads1284Decoder():
    def __init__(self,config):
        self.rawdt = np.dtype('int32')
        self.rawdt = self.rawdt.newbyteorder('>')
        vref = config['vref']
        bits = config['bits']
        gain = config['gain'][0] # 单通道
        self.fac = self._calFac(vref,bits,gain)

    def _calFac(self,vref,bits,gain):
        return vref / (gain*(2**bits - 1)) * 1e6

    def getchs(self,payloads,N): # payloads必须是一个采样数据包的
        return int(len(payloads)/4/N)

    def decode(self,payloads,sampleN,chs):
        eeg = np.frombuffer(payloads, dtype=self.rawdt).astype(EEGTYPE)*self.fac # 1284为单通道
        return eeg

class Ads1263Decoder():
    def __init__(self,config):
        self.rawdt = np.dtype('int32')
        self.rawdt = self.rawdt.newbyteorder('>')
        vref = config['vref']
        bits = config['bits']
        gain = config['gain'][0] # 单通道
        self.fac = self._calFac(vref, bits, gain)

    def _calFac(self, vref, bits, gain):
        return vref / (gain * (2 ** bits - 1)) * 1e6

    def getchs(self,payloads,N): # payloads必须是一个采样数据包的
        return int(len(payloads)/4/N)

    def decode(self,payloads):
        eeg = np.frombuffer(payloads, dtype=self.rawdt).astype(EEGTYPE)*self.fac # 1263为单通道
        return eeg