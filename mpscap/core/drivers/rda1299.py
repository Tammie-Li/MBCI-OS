"""
设备驱动（参考采集程序/rda1299.py）。
负责串口通信和数据采集。
"""

from __future__ import annotations

import re
import serial
import serial.tools.list_ports as lp
import threading
import time
from threading import Event
from typing import Optional

from .data_decoder import DataDecoder
from ..data_pipeline.kafka_producer import KafkaConfig

PROTOCOL_UUID = 'emg-gloveV2'


class RDA1299(threading.Thread):
    """RDA1299设备驱动（参考采集程序/rda1299.py/RDA1299）。"""
    
    def __init__(self, pysig=None, master: bool = True, kafka_config: Optional[KafkaConfig] = None):
        super(RDA1299, self).__init__()
        self.ser: Optional[serial.Serial] = None
        self.threadRunning = False
        self.pysig = pysig
        self.reading = False
        self.stpEv = Event()
        self.dec = DataDecoder(PROTOCOL_UUID, master=master, kafka_config=kafka_config)
        self.setDaemon(True)
        self.port = 'COM3'
        self.baudrate = 460800
    
    @staticmethod
    def getallserial() -> list:
        """获取所有CP210x串口设备（参考rda1299.py/getallserial）。"""
        pp = re.compile('CP210x')
        ports = lp.comports()
        device = []
        for p in ports:
            id = p.device
            r = re.search(pp, str(p))
            if r is not None:
                device.append(id)
        return device
    
    def configDev(self, port: str = 'COM3', baudrate: int = 460800) -> None:
        """配置设备（参考rda1299.py/configDev）。"""
        self.port = port
        self.baudrate = baudrate
    
    def open(self) -> bool:
        """打开设备（参考rda1299.py/open）。"""
        if self.ser is None:
            try:
                self.ser = serial.Serial(port=self.port, baudrate=self.baudrate)
                if not self.threadRunning:
                    self.start()
                    self.reading = True
                return True
            except Exception:
                return False
        else:
            return True
    
    def close(self) -> None:
        """关闭设备（参考rda1299.py/close）。"""
        self.stpEv.set()
        self.reading = False
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
        self.dec.release()
    
    def run(self) -> None:
        """数据读取线程（参考rda1299.py/run）。"""
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
                        except Exception:
                            ok = False
                            self.ser = None  # 接收器被拔出
                            if self.pysig:
                                self.pysig.emit('接收器断开!')
                        if ok and len(buf) > 0:
                            self.dec.parseData(buf, stamp)
                else:
                    time.sleep(0.01)
            else:
                time.sleep(rk)
        
        self.threadRunning = False
    
    def writeCmd(self, cmd: str) -> None:
        """写入命令（参考rda1299.py/writeCmd）。"""
        self.reading = False
        if self.ser is not None:
            self.ser.flushOutput()
            self.ser.flushInput()  # 会清空掉输入缓存中的数据
        
        if cmd == 'stop':
            self.reading = False
        elif cmd == 'acquireEEG':
            self.reading = True
        elif cmd == 'impedanceDetect':
            pass
        else:
            pass

