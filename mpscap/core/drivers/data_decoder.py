"""
数据解码器（参考采集程序/datadecoder.py）。
负责解析数据包并将数据写入共享内存。
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..utils.shm import CreateShm, EEGTYPE, EEGMAXLEN
from .emg_wristband import Protocol, ADC24Decoder, QmiDecoder, GloveDecoder, PROTOCOL_UUID
from ..data_pipeline.kafka_producer import KafkaDataProducer, KafkaConfig

# 设备配置（参考datadecoder.py）
devconfig = {
    'vref': 4.5,
    'bits': 24,
    'gain': [24, 24, 24, 24, 24, 24, 24, 24],
    'accrang': 8,
    'gyrrang': 1024
}


class DataDecoder:
    """数据解码器（参考datadecoder.py/DataDecoder）。"""
    
    def __init__(self, uuid: str = PROTOCOL_UUID, master: bool = True, kafka_config: Optional[KafkaConfig] = None):
        self.shm = CreateShm(master=master)
        self.decoder24 = ADC24Decoder(max_channels=34)
        self.decoderQmi = QmiDecoder()
        self.decoderGlove = GloveDecoder()
        self.protocol = Protocol(uuid)
        self.mode = 1
        self.sampleCount = 0
        self.payloads = b''
        self.accBytes = b''
        self.gloveBytes = b''
        self.buffer = b''
        self.ids = b''
        self.tris = b''
        self.triggerbytes = b''
        
        self.batLevel = 0
        self.saveFlg = 0
        self.file: Optional[object] = None
        self.timestamps: list = []
        self.triggers: list = []
        self.curr_trigger = 0
        self.trig_file: Optional[object] = None
        self.trig_path: Optional[str] = None
        
        self.emgChs = 0
        self.accChs = 0
        self.gloveChs = 0
        self.srate = 0
        self.biasconnect = 0
        self.refconnect = 0
        
        # Kafka生产者（可选）
        self.kafka_producer: Optional[KafkaDataProducer] = None
        if kafka_config is not None:
            try:
                self.kafka_producer = KafkaDataProducer(kafka_config)
                if self.kafka_producer.connect():
                    print(f"[Kafka] DataDecoder: Kafka生产者已初始化")
                else:
                    print(f"[Kafka] DataDecoder: Kafka生产者初始化失败")
                    self.kafka_producer = None
            except Exception as e:
                print(f"[Kafka] DataDecoder: Kafka初始化异常: {e}")
                self.kafka_producer = None
        
        if EEGTYPE == 'float32':
            self.typeLen = 4
        elif EEGTYPE == 'float64':
            self.typeLen = 8
    
    def set_kafka_config(self, kafka_config: Optional[KafkaConfig]) -> None:
        """动态设置Kafka配置（用于运行时启用/禁用Kafka）。"""
        # 先关闭现有的producer
        if self.kafka_producer is not None:
            try:
                self.kafka_producer.flush()
                self.kafka_producer.disconnect()
            except:
                pass
            self.kafka_producer = None
        
        # 如果提供了新配置，创建新的producer
        if kafka_config is not None:
            try:
                self.kafka_producer = KafkaDataProducer(kafka_config)
                if self.kafka_producer.connect():
                    print(f"[Kafka] DataDecoder: Kafka生产者已重新初始化")
                else:
                    print(f"[Kafka] DataDecoder: Kafka生产者重新初始化失败")
                    self.kafka_producer = None
            except Exception as e:
                print(f"[Kafka] DataDecoder: Kafka重新初始化异常: {e}")
                self.kafka_producer = None
    
    def update_trigger(self, trigger: int) -> None:
        """更新trigger值。"""
        self.curr_trigger = trigger
    
    def parseData(self, buffer: bytes, stamp: float) -> None:
        """解析接收到的数据（参考datadecoder.py/parseData）。"""
        self.buffer += buffer
        Len = len(self.buffer)
        indx = 0
        
        while indx < Len - 7:
            self.protocol.loadBuffer(self.buffer[indx:])
            if self.protocol.headVerify():  # 头部校验成功
                includePak, pakLen = self.protocol.paklenVerify()  # 校验包长度
                if includePak:  # 长度足够容纳一个数据包
                    if self.protocol.getEpochAndVerify():  # 截取数据包并校验
                        devData = self.protocol.parsePak()
                        indx += pakLen
                        self.collectAll(devData, stamp)
                    else:
                        indx += 1
                else:  # 长度不够，跳出，下次再来
                    break
            else:  # 继续向后寻找
                indx += 1
        
        self.buffer = self.buffer[indx:]
        self.dataarange()  # 整理数据
    
    def collectAll(self, dat: dict, stamp: float) -> None:
        """收集所有数据包（参考datadecoder.py/collectAll）。"""
        # dat是字典类型（从Protocol.parsePak返回）
        self.batLevel = dat.get('batLevel', 0)
        self.devID = dat.get('devID', 0)
        self.payloads += dat.get('emgpayload', b'')
        self.accBytes += dat.get('accBytes', b'')
        self.gloveBytes += dat.get('gloveBytes', b'')
        # trigger字段可能不存在
        trigger_bytes = dat.get('trigger', b'')
        self.triggerbytes += trigger_bytes
        
        # pakID字段可能不存在
        pak_id = dat.get('pakID', b'')
        self.ids += pak_id
        self.tris += trigger_bytes
        # 只有在接收到非零值时才更新通道数，避免被重置为0
        emg_chs = dat.get('emgChs', 0)
        acc_chs = dat.get('accChs', 0)
        glove_chs = dat.get('gloveChs', 0)
        if emg_chs > 0:
            self.emgChs = emg_chs
        if acc_chs > 0:
            self.accChs = acc_chs
        if glove_chs > 0:
            self.gloveChs = glove_chs
        srate = dat.get('srate', 0)
        if srate > 0:
            self.srate = srate
        self.sampleCount += dat.get('sampleN', 0)
        # biasconnect和refconnect字段可能不存在
        self.biasconnect = dat.get('biasconnect', 0)
        self.refconnect = dat.get('refconnect', 0)
        self.timestamps.append(stamp)
        self.triggers.append(self.curr_trigger)
    
    def dataarange(self) -> None:
        """整理数据并写入共享内存（参考datadecoder.py/dataarange）。"""
        self.shm.setvalue('batlevel', self.batLevel)
        self.shm.setvalue('mode', self.mode)
        
        # 等待绘图进程完成读取
        while self.shm.getvalue('plotting'):
            time.sleep(0.001)
        
        if len(self.payloads) == 0:
            return
        
        # 解码所有数据（参考datadecoder.py第121-125行）
        # 通道组成：8(EMG) + 12(ACC: 6 IMU1 + 6 IMU2) + 14(Glove: 5压力 + 5弯曲 + 其他) = 34通道
        
        # n×8,8-8通道（EMG）
        emgdataay = self.decoder24.decode(self.payloads, self.sampleCount, self.emgChs)
        
        # n×12,6（主机腕部的imu芯片，accx,accy,accz,gyrx,gyry,gryz）+ 6（手套上的imu芯片，accx,accy,accz,gyrx,gyry,gryz）
        # ACC数据：12通道 = 6通道(IMU1) + 6通道(IMU2)
        if len(self.accBytes) > 0 and self.accChs > 0:
            accdataay = self.decoderQmi.decode(self.accBytes, self.sampleCount, self.accChs)
        else:
            accdataay = np.zeros((self.sampleCount, self.accChs), dtype=np.float32)
        
        # n×14,0-4:压力传感器依次对应小拇指，无名指，中指，食指，大拇指，6-10：弯曲传感器依次对应小拇指，无名指，中指，食指，大拇指
        # Glove数据：14通道 = 5通道(压力传感器) + 其他通道(弯曲传感器等)
        if len(self.gloveBytes) > 0 and self.gloveChs > 0:
            glovedataay = self.decoderGlove.decode(self.gloveBytes, self.sampleCount, self.gloveChs)
        else:
            glovedataay = np.zeros((self.sampleCount, self.gloveChs), dtype=np.float32)
        

        
        # 合并所有数据（参考datadecoder.py第132行）
        # 8(EMG) + 12(ACC) + 14(Glove) = 34通道
        alldataRwt = np.hstack((emgdataay, accdataay, glovedataay))
        
        # 验证通道数（参考用户说明：8+12+14=34）
        total_channels = self.emgChs + self.accChs + self.gloveChs
        if alldataRwt.shape[1] != 34 or total_channels != 34:
            print(f"[ERROR] 通道数不正确！期望34，实际{alldataRwt.shape[1]}")
            print(f"[ERROR] EMG={self.emgChs}, ACC={self.accChs}, Glove={self.gloveChs}, 总计={total_channels}")
            print(f"[ERROR] EMG shape={emgdataay.shape}, ACC shape={accdataay.shape}, Glove shape={glovedataay.shape}")
        else:
            pass
            # print(f"[DEBUG] : {alldataRwt.shape[1]}通道 (EMG={self.emgChs}, ACC={self.accChs}, Glove={self.gloveChs})")
        
        # 发布到Kafka（如果已启用）
        if self.kafka_producer is not None:
            try:
                # 转置数据为 (channels, samples) 格式
                data_for_kafka = alldataRwt.transpose()
                # 发布数据
                self.kafka_producer.publish_data(
                    data=data_for_kafka,
                    timestamps=self.timestamps if len(self.timestamps) == self.sampleCount else None,
                    device_id=f"device_{self.devID}" if hasattr(self, 'devID') else None,
                )
            except Exception as e:
                print(f"[Kafka] 发布数据失败: {e}")
        
        # 展平数据（参考datadecoder.py第136行）
        alldataFlatten = alldataRwt.flatten()
        L = alldataFlatten.size
        
        # 更新共享内存中的参数（参考datadecoder.py第139-142行）
        self.shm.setvalue('emgchs', self.emgChs)
        self.shm.setvalue('accchs', self.accChs)
        self.shm.setvalue('glovechs', self.gloveChs)
        self.shm.setvalue('srate', self.srate)
        # 同步触发值到共享内存（供前端显示竖线），单位：当前触发码
        try:
            self.shm.setvalue('includetrigger', int(self.curr_trigger))
        except Exception:
            pass
        
        # 写入共享内存（参考datadecoder.py第144-151行）
        curdataindex = self.shm.getvalue('curdataindex')
        if curdataindex + L > EEGMAXLEN:
            curdataindex = 0
        
        self.shm.eeg[curdataindex:curdataindex + L] = alldataFlatten[:]
        self.shm.setvalue('curdataindex', curdataindex + L)
        self.shm.setvalue('curbyteindex', (curdataindex + L) * self.typeLen)
        self.shm.info[0] += 1  # 更新index
        
        # 保存数据相关（参考datadecoder.py第153-196行）
        if self.saveFlg == 0:
            if self.shm.getvalue('savedata') == 1:  # 开启保存
                pth = self.shm.getPath()
                try:
                    if self.file is not None:
                        self.file.close()
                except:
                    pass
                
                self.file = open(pth, 'wb')
                if self.typeLen == 4:
                    ay = np.array([7, 2, 2, self.srate, self.emgChs, self.accChs, self.gloveChs], dtype=np.int32)
                else:
                    ay = np.array([7, 2, 3, self.srate, self.emgChs, self.accChs, self.gloveChs], dtype=np.int32)
                
                self.file.write(ay.tobytes())  # 头信息
                self.saveFlg = 1
                # 不再保存 .trig 文件
                self.trig_file = None
                self.trig_path = None
        else:  # self.saveFlg == 1:
            if self.shm.getvalue('savedata') == 0:  # 结束保存
                if self.file is not None:
                    self.file.close()
                self.saveFlg = 0
                if self.trig_file is not None:
                    try:
                        self.trig_file.close()
                    except Exception:
                        pass
                    self.trig_file = None
                    self.trig_path = None
            else:  # 正常保存
                stamp = np.array([self.timestamps]).transpose()
                # 按 emg+acc+glove+1（时间戳）写入，避免通道错位
                ay = np.hstack((alldataRwt, stamp)).astype(EEGTYPE).flatten()
                if self.file is not None:
                    self.file.write(ay.tobytes())
                # 触发文件写入：绝对时间戳 + trigger（浮点写入，简化为两列 float64）
                # 不写入 .trig
                self.saveFlg = 1
        
        # 清空累积数据（参考datadecoder.py第197-208行）
        self.sampleCount = 0
        self.payloads = b''
        self.accBytes = b''
        self.gloveBytes = b''
        self.triggerbytes = b''
        self.timestamps = []
        self.triggers = []
        self.ids = b''
        self.tris = b''
    
    def release(self) -> None:
        """释放资源。"""
        # 关闭Kafka producer
        if self.kafka_producer is not None:
            try:
                self.kafka_producer.flush()
                self.kafka_producer.disconnect()
            except:
                pass
            self.kafka_producer = None
        
        if self.file is not None:
            try:
                self.file.close()
            except:
                pass
        if self.trig_file is not None:
            try:
                self.trig_file.close()
            except:
                pass
        self.shm.release()

