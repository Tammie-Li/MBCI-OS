#coding:utf-8

# * 通讯协议v2.0
# * 2023.12.25
# * 2024.5.7更新
# * 包头： AB 55
# * 包长度：2字节, uint16(np.uint16)
# * 标志符1（1字节）：
#      * bit7-bit4: 设备标识符 0-ads1299(24位ADC), 1-ads1284(31位ADC), 2-ads1263(32位ADC)
#      * bit3-bit0: 电量0-10,表示0~100%
# //* 标志符2（1字节）：
# //     * bit0:带/不带丢包测试, 如果带，在可解释字段添加一个字节,为sampleID
# //     * bit1:带/不带trigger,  如果带，在可解释字段增加一个字节,为trigger
# //     * bit2:带/不带sampleN,  如果带，在可解释字段增加一个字节，为sampleN
# //     * bit3:带/不带ACC,      如果带，在可解释字段增加6字节，为ACCX,ACCY,ACCZ
# //     * bit4:带/不带glove,    如果带, 在可解释字段增加28字节，为数据手套的数据
# //     * bit5:带/不带?
# //     * bit6:带/不带?
# //     * bit7:带/不带?
# //* 采样率：2字节, uint16((numpy.uint16))
# //* 一下为可解释字段
# //* [1字节丢包测试] 0-255循环    【bit0控制】
# //* [1字节trigger] uint8         【bit1控制】
# //* [1字节sampleN] uint8         【bit2控制】
# //* [6字节ACC] int16x3           【bit3控制】
# //* [28字节glove] uint16x14      【bit4控制】
# //* [可解释字节]                【bit5控制】
# //* [可解释字节]                【bit6控制】
# //* [可解释字节]                【bit7控制】
# //* 数据序列
# //* 1字节校验
# //* 特别说明：默认一个数据包包含一个采样点。需要包含多个采样点时，可使用可解释字节补充说明

import numpy as np

class DevData:
    pakLen = 0
    devID = 0
    pgaGain = 0
    srate = 0
    batLevel = 10
    sampleN = 0
    sampleInterval = 0
    pakID = b''
    trigger = b''
    includeID = False
    includeTri = False
    includeAcc = False
    includeGlove = False
    optionBytes = b'' # bit3~bit7
    accBytes = b'\x00'*6
    gloveBytes = b''
    payload = b''

class Protocol():
    def __init__(self):
        self.u16dt = np.dtype('uint16')
        self.version = 2.0
        self.devData = DevData()
        self.package = b''
        self.pakBuf = b''
        self.devData.accBytes = b'\x00'*6
        self.payloadStartIndex = 0

    def loadBuffer(self,buf): # 加载buffer
        self.pakBuf = buf

    def headVerify(self):
        return self.pakBuf[0] == 0xAB and self.pakBuf[1] == 0x55

    def paklenVerify(self):
        self.devData.pakLen = int(np.frombuffer(self.pakBuf[2:4], dtype=self.u16dt)[0])
        return len(self.pakBuf) >= self.devData.pakLen, self.devData.pakLen

    def getEpochAndVerify(self,len):
        self.package = self.pakBuf[:self.devData.pakLen]
        return (sum(self.package[:-1]) & 0xff) == self.package[-1]

    def parsePak(self):
        self.devData.srate = int(np.frombuffer(self.package[6:8], dtype=self.u16dt)[0])
        self.devData.sampleInterval = 1/self.devData.srate

        identifier1 = self.package[4]
        self.devData.devID = identifier1 >> 4
        self.devData.batLevel = identifier1 & 0x0f

        identifier2 = self.package[5]
        ind = 8
        self.devData.includeID = identifier2 & 0x01
        if self.devData.includeID:
            self.devData.pakID = self.package[ind:ind+1]
            ind += 1

        self.devData.includeTri = (identifier2>>1) & 0x01
        if self.devData.includeTri:
            self.devData.trigger = self.package[ind:ind+1]
            ind += 1

        if (identifier2>>2) & 0x01:
            sampleN = self.package[ind]
            ind += 1
        else:
            sampleN = 1

        includeAcc = (identifier2>>3) & 0x01
        if includeAcc:
            self.devData.includeAcc = True
            self.devData.accBytes = self.package[ind:ind+6]
            ind += 6

        self.devData.includeGlove = (identifier2>>4) & 0x01
        if self.devData.includeGlove:
            self.devData.gloveBytes = self.package[ind:ind + 28]
            ind += 28

        self.devData.optionBytes = b''
        if (identifier2>>5) & 0x01:
            self.devData.optionBytes += self.package[ind:ind + 1]
            ind += 1

        if (identifier2>>6) & 0x01:
            self.devData.optionBytes += self.package[ind:ind + 1]
            ind += 1

        if (identifier2>>7) & 0x01:
            self.devData.optionBytes += self.package[ind:ind + 1]
            ind += 1

        self.devData.payload = self.package[ind:-1]
        self.devData.sampleN = sampleN
        return self.devData