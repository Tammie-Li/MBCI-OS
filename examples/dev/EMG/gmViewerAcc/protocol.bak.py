#coding:utf-8

# * 通讯协议v2.0
# * 2023.12.25
# * 包头： AB 55
# * 包长度：2字节, uint16(np.uint16)
# * 标志符1（1字节）：
#      * bit7-bit4: 设备标识符 0-ads1299(24位ADC), 1-ads1284(31位ADC), 2-ads1263(32位ADC)
#      * bit3-bit0: 电量0-10,表示0~100%
# * 标志符2（1字节）：
#      * bit0:带/不带丢包测试
#      * bit1:带/不带trigger
#      * bit2:带/不带sampleN
#      * bit3:带/不带?
#      * bit4:带/不带?
#      * bit5:带/不带?
#      * bit6:带/不带?
#      * bit7:带/不带?
# * 采样率：2字节, uint16(np.uint16)
# * [1字节丢包测试] 0-255循环    【对应bit0控制的可选字节】
# * [1字节trigger] uint8       【对应bit1控制的可选字节】
# * [可解释字节]                【对应bit2控制的可选字节】
# * [可解释字节]                【对应bit3控制的可选字节】
# * [可解释字节]                【对应bit4控制的可选字节】
# * [可解释字节]                【对应bit5控制的可选字节】
# * [可解释字节]                【对应bit6控制的可选字节】
# * [可解释字节]                【对应bit7控制的可选字节】
# * 数据序列
# * 1字节校验
# * 特别说明：默认一个数据包包含一个采样点。需要包含多个采样点时，可使用可解释字节补充说明

# // * protocol v2.0
# // * 2023.12.25
# // * Head： AB 55
# // * PackageLen：2Bytes, uint16(little endian)
# // * IndicateByte1（1Byte）：
# //      * bit7-bit4: device ID 0-ads1299, 1-ads1284, 2-ads1263
# //      * bit3-bit0: battery power level: 0-10 -> 0~100%
# // * IndicateByte2（1字节）：
# //      * bit0:include/noninclude sampleID
# //      * bit1:include/noninclude trigger
# //      * bit2:include/noninclude sampleN
# //      * bit3:include/noninclude ?
# //      * bit4:include/noninclude ?
# //      * bit5:include/noninclude ?
# //      * bit6:include/noninclude ?
# //      * bit7:include/noninclude ?
# // * Srate：2字节, uint16(little endian)
# // * [1Byte sampleID] 0-255 cycle   【optional depends on bit0】
# // * [1Byte trigger]                【optional depends on bit1】
# // * [1Byte sampleN]               【optional depends on bit2】
# // * [1Byte optional]               【optional depends on bit3】
# // * [1Byte optional]               【optional depends on bit4】
# // * [1Byte optional]               【optional depends on bit5】
# // * [1Byte optional]               【optional depends on bit6】
# // * [1Byte optional]               【optional depends on bit7】
# // * data array
# // * 1Byte check (sum check)
# // * specially: one sample included in one package in default, the optional bytes may
#      indicate how many samples include in one package if needed

import numpy as np

class DevData:
    pakLen = 0
    devID = 0
    pgaGain = 0
    srate = 0
    batLevel = 10
    sampleN = 1
    sampleInterval = 0
    pakID = b''
    trigger = b''
    includeID = False
    includeTri = False
    optionBytes = b'' # bit3~bit7
    payload = b''

class Protocol():
    def __init__(self):
        self.u16dt = np.dtype('uint16')
        self.version = 2.0
        self.devData = DevData()
        self.package = b''
        self.pakBuf = b''
        self.payloadStartIndex = 0

    def loadBuffer(self,buf): # 加载buffer
        self.pakBuf = buf

    def headVerify(self):
        return self.pakBuf[0] == 0xAB and self.pakBuf[1] == 0x55

    def paklenVerify(self):
        self.devData.pakLen = int(np.frombuffer(self.pakBuf[2:4], dtype=self.u16dt)[0])
        return len(self.pakBuf) >= self.devData.pakLen, self.devData.pakLen

    def getEpochAndVerify(self):
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

        if (identifier2>>2) & 0x04:
            self.devData.sampleN = self.package[ind]
            ind += 1
        else:
            self.devData.sampleN = 1

        self.devData.optionBytes = b''
        if (identifier2>>3) & 0x08:
            self.devData.optionBytes += self.package[ind:ind+1]
            ind += 1

        if (identifier2>>4) & 0x10:
            self.devData.optionBytes += self.package[ind:ind + 1]
            ind += 1

        if (identifier2>>5) & 0x20:
            self.devData.optionBytes += self.package[ind:ind + 1]
            ind += 1

        if (identifier2>>6) & 0x40:
            self.devData.optionBytes += self.package[ind:ind + 1]
            ind += 1

        if (identifier2>>7) & 0x80:
            self.devData.optionBytes += self.package[ind:ind + 1]
            ind += 1

        self.devData.payload = self.package[ind:-1]
        return self.devData