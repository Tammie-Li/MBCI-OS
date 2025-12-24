"""
自研肌电腕带设备适配器。

参考采集程序中的 rda1299.py、datadecoder.py、protocol.py 实现。
"""

from __future__ import annotations

import re
import serial
import serial.tools.list_ports as lp
import threading
import time
from typing import Callable, Optional

import numpy as np

from ...core.data_pipeline.mqtt_gateway import MQTTGateway
from ...core.data_pipeline.protocol import SignalFrame, SignalPacketHeader
from ...core.subsystems.acquisition import DeviceAdapter, FrameCallback


# 协议UUID
PROTOCOL_UUID = 'emg-gloveV2'


class Protocol:
    """数据包协议解析器（简化版，参考采集程序/protocol.py）。"""
    
    def __init__(self, uuid: str):
        if PROTOCOL_UUID != uuid:
            raise IOError("protocol match error!")
        self.u16dt = np.dtype('uint16')
        self.buffer = b''
        self.pakLen = 0
        self.package = b''
    
    def loadBuffer(self, buf: bytes) -> None:
        self.buffer = buf
    
    def headVerify(self) -> bool:
        return len(self.buffer) >= 2 and self.buffer[0] == 0xAB and self.buffer[1] == 0x55
    
    def paklenVerify(self) -> tuple[bool, int]:
        if len(self.buffer) < 4:
            return False, 0
        self.pakLen = int(np.frombuffer(self.buffer[2:4], dtype=self.u16dt)[0])
        return len(self.buffer) >= self.pakLen, self.pakLen
    
    def getEpochAndVerify(self) -> bool:
        if len(self.buffer) < self.pakLen:
            return False
        self.package = self.buffer[:self.pakLen]
        return (sum(self.package[:-1]) & 0xff) == self.package[-1]
    
    def parsePak(self) -> dict:
        """解析数据包，返回设备数据字典。"""
        identifier1 = self.package[4]
        devID = identifier1 >> 4
        batLevel = identifier1 & 0x0f
        
        srate = np.frombuffer(self.package[6:8], dtype=self.u16dt)[0]
        
        identifier2 = self.package[5]
        ind = 8
        
        includeID = identifier2 & 0x01
        if includeID:
            ind += 1
        
        includeTri = (identifier2 >> 1) & 0x01
        if includeTri:
            ind += 1
        
        if (identifier2 >> 2) & 0x01:
            sampleN = self.package[ind]
            ind += 1
        else:
            sampleN = 1
        
        includeAcc = (identifier2 >> 3) & 0x01
        accBytesWan = b''
        accBytesHand = b''
        if includeAcc:
            # 参考protocol.py第200行：ACC数据是12字节（固定，不乘以sampleN）
            # 注意：参考代码中ACC数据是固定的12字节，不是12*sampleN
            accBytesWan = self.package[ind:ind + 12]
            ind += 12
        
        includeGlove = (identifier2 >> 4) & 0x01
        gloveBytes = b''
        if includeGlove:
            # 参考protocol.py第205-206行：
            # gloveBytes = 28字节（固定，不乘以sampleN）
            # accBytesHand = 12字节（固定，不乘以sampleN）
            # 总共40字节，不是40*sampleN
            gloveBytes = self.package[ind:ind + 28]
            accBytesHand = self.package[ind + 28:ind + 40]
            ind += 40
        
        emgpayload = self.package[ind:-1]
        
        # 计算EMG通道数（每个采样点每个通道3字节）
        # 根据参考代码，默认EMG=8通道，但需要根据实际数据计算
        if len(emgpayload) > 0 and sampleN > 0:
            emgChs = len(emgpayload) // (3 * sampleN)
        else:
            emgChs = 8  # 默认值
        
        # ACC通道数：12（6+6）
        accChs = 12 if includeAcc else 0
        
        # Glove通道数：14
        gloveChs = 14 if includeGlove else 0
        
        return {
            'devID': devID,
            'batLevel': batLevel,
            'srate': srate,
            'sampleN': sampleN,
            'emgChs': emgChs,
            'accChs': accChs,
            'gloveChs': gloveChs,
            'emgpayload': emgpayload,
            'accBytes': accBytesWan + accBytesHand,
            'gloveBytes': gloveBytes,
            'includeAcc': includeAcc,
            'includeGlove': includeGlove,
        }


class ADC24Decoder:
    """24位ADC解码器（参考采集程序/datadecoder.py）。"""
    
    def __init__(self, max_channels: int = 34):
        vref = 4.5
        bits = 24
        # 支持最多34个通道，但EMG通道使用gain=24
        gain = [24] * max_channels  # 默认所有通道gain=24
        self.rawdt = np.dtype('int32').newbyteorder('>')
        self.facs = np.array([self._calFac(vref, bits, g) for g in gain])
        self.facs = self.facs[np.newaxis, :]
    
    def _calFac(self, vref: float, bits: int, gain: int) -> float:
        return vref / (gain * (2**bits - 1)) * 1e6
    
    def _tobuf32(self, buf24: bytes) -> bytes:
        if len(buf24) < 3:
            return b'\x00\x00\x00\x00'
        if buf24[0] > 127:
            return b'\xff' + buf24[:3]
        else:
            return b'\x00' + buf24[:3]
    
    def decode(self, payloads: bytes, sampleN: int, chs: int) -> np.ndarray:
        """解码EMG数据。"""
        if len(payloads) == 0 or sampleN == 0 or chs == 0:
            return np.zeros((sampleN, chs), dtype=np.float32)
        
        # 确保payloads长度正确
        expected_len = 3 * sampleN * chs
        if len(payloads) != expected_len:
            print(f"[WARNING] EMG数据长度不匹配: 期望 {expected_len} 字节，实际 {len(payloads)} 字节")
            # 截断或填充
            if len(payloads) < expected_len:
                payloads = payloads + b'\x00' * (expected_len - len(payloads))
            else:
                payloads = payloads[:expected_len]
        
        tmbuf = [self._tobuf32(payloads[i:i + 3]) for i in range(0, len(payloads), 3)]
        buf = b''.join(tmbuf)
        eeg = np.frombuffer(buf, dtype=self.rawdt).astype(np.float32).reshape(sampleN, chs)
        
        # 只使用前chs个通道的增益因子
        fac = np.repeat(self.facs[:, :chs], sampleN, axis=0)
        eeg = eeg * fac
        return eeg


class QmiDecoder:
    """QMI解码器（用于ACC数据）。"""
    
    def __init__(self):
        self.rawdt = np.dtype('int16')
        accrang = 8  # ±8g
        accfac = 9.8 * 2 * accrang / 65536
        gyrrang = 1024  # ±1024dps
        gyrfac = 2 * gyrrang / 65536
        # 12通道：6个ACC+GYR（手腕）+ 6个ACC+GYR（手部）
        self.facs = np.array([accfac]*3 + [gyrfac]*3 + [accfac]*3 + [gyrfac]*3)
        self.facs = self.facs[np.newaxis, :]
    
    def decode(self, payloads: bytes, sampleN: int, chs: int) -> np.ndarray:
        """解码ACC数据（参考datadecoder.py/QmiDecoder.decode）。"""
        if len(payloads) == 0 or sampleN == 0 or chs == 0:
            return np.zeros((sampleN, chs), dtype=np.float32) if sampleN > 0 and chs > 0 else np.zeros((1, chs), dtype=np.float32)
        
        # ACC数据：每个数据包固定12字节（手腕）或24字节（手腕+手部），累积后总字节数
        # 解码时直接使用sampleN，不根据实际数据长度计算样本数
        # 如果数据长度不够，重复最后一个样本或填充零
        bytes_per_sample = chs * 2  # 每个通道2字节，每个样本chs*2字节
        expected_len = bytes_per_sample * sampleN
        
        if len(payloads) < expected_len:
            # 数据不够，填充零或重复最后一个样本
            if len(payloads) > 0:
                # 重复最后一个样本
                last_sample = payloads[-bytes_per_sample:] if len(payloads) >= bytes_per_sample else payloads + b'\x00' * (bytes_per_sample - len(payloads))
                padding_len = expected_len - len(payloads)
                payloads = payloads + last_sample * (padding_len // bytes_per_sample) + last_sample[:padding_len % bytes_per_sample]
            else:
                payloads = b'\x00' * expected_len
        elif len(payloads) > expected_len:
            # 数据太多，截断
            payloads = payloads[:expected_len]
        
        data = np.frombuffer(payloads, dtype=self.rawdt).astype(np.float32).reshape(sampleN, chs)
        fac = np.repeat(self.facs[:, :chs], sampleN, axis=0)
        data = data * fac
        return data


class GloveDecoder:
    """手套数据解码器。"""
    
    def __init__(self):
        self.rawdt = np.dtype('uint16')
        self.fac = 0.025
    
    def decode(self, payloads: bytes, sampleN: int, chs: int) -> np.ndarray:
        """解码Glove数据（参考datadecoder.py/GloveDecoder.decode）。"""
        if len(payloads) == 0 or sampleN == 0 or chs == 0:
            return np.zeros((sampleN, chs), dtype=np.float32) if sampleN > 0 and chs > 0 else np.zeros((1, chs), dtype=np.float32)
        
        # Glove数据：每个数据包固定28字节，累积后总字节数
        # 解码时直接使用sampleN，不根据实际数据长度计算样本数
        # 如果数据长度不够，重复最后一个样本或填充零
        bytes_per_sample = chs * 2  # 每个通道2字节，每个样本chs*2字节
        expected_len = bytes_per_sample * sampleN
        
        if len(payloads) < expected_len:
            # 数据不够，填充零或重复最后一个样本
            if len(payloads) > 0:
                # 重复最后一个样本
                last_sample = payloads[-bytes_per_sample:] if len(payloads) >= bytes_per_sample else payloads + b'\x00' * (bytes_per_sample - len(payloads))
                padding_len = expected_len - len(payloads)
                payloads = payloads + last_sample * (padding_len // bytes_per_sample) + last_sample[:padding_len % bytes_per_sample]
            else:
                payloads = b'\x00' * expected_len
        elif len(payloads) > expected_len:
            # 数据太多，截断
            payloads = payloads[:expected_len]
        
        data = np.frombuffer(payloads, dtype=self.rawdt).astype(np.float32).reshape(sampleN, chs)
        data = data * self.fac
        return data


class EmgWristbandAdapter(DeviceAdapter, threading.Thread):
    """自研肌电腕带设备适配器。"""
    
    device_id: str = "emg_wristband"
    modality: str = "EMG"
    
    def __init__(self, mqtt_gateway: MQTTGateway, config: dict):
        DeviceAdapter.__init__(self, mqtt_gateway)
        threading.Thread.__init__(self)
        self.port = config.get('port', 'COM3')
        self.baudrate = config.get('baudrate', 460800)
        self.ser: Optional[serial.Serial] = None
        self.thread_running = False
        self.reading = False
        self.stop_event = threading.Event()
        self.protocol = Protocol(PROTOCOL_UUID)
        self.decoder_emg = ADC24Decoder(max_channels=34)
        self.decoder_acc = QmiDecoder()
        self.decoder_glove = GloveDecoder()
        self.buffer = b''
        # _on_frame 由父类 DeviceAdapter 初始化，不需要重复定义
        self.setDaemon(True)
    
    def connect(self) -> None:
        """连接设备。"""
        if self.ser is not None:
            raise ConnectionError(f"设备 {self.port} 已连接")
        try:
            print(f"[DEBUG] 尝试打开串口: {self.port}, 波特率: {self.baudrate}")
            self.ser = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=0.1)
            print(f"[DEBUG] 串口打开成功: {self.port}")
            print(f"[DEBUG] 串口状态 - is_open: {self.ser.is_open}, port: {self.ser.port}, baudrate: {self.ser.baudrate}")
            # 清空输入输出缓冲区
            self.ser.flushInput()
            self.ser.flushOutput()
            print(f"[DEBUG] 已清空串口缓冲区")
            if not self.thread_running:
                self.start()
            self.reading = True
        except Exception as e:
            if self.ser is not None:
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None
            raise ConnectionError(f"无法打开串口 {self.port}: {str(e)}")
    
    def start_stream(self, on_frame: FrameCallback) -> None:
        """开始数据流。"""
        if self.ser is None:
            raise RuntimeError("设备未连接")
        print(f"[DEBUG] 设置回调函数: on_frame={on_frame is not None}")
        self._on_frame = on_frame
        print(f"[DEBUG] 回调函数已设置: _on_frame={self._on_frame is not None}")
        # 清空输入输出缓冲区
        self.ser.flushOutput()
        self.ser.flushInput()
        # 发送采集命令（参考rda1299.py，实际设备可能需要特定命令）
        # 注意：根据实际设备协议，可能需要发送特定命令
        self.reading = True
        print(f"[DEBUG] 设备 {self.port} 开始采集，reading={self.reading}")
    
    def stop_stream(self) -> None:
        """停止数据流。"""
        self.reading = False
        # 清理回调，防止回调访问已删除的UI控件
        self._on_frame = None
    
    def disconnect(self) -> None:
        """断开设备。"""
        self.stop_event.set()
        self.reading = False
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
    
    def run(self) -> None:
        """数据读取线程。"""
        self.thread_running = True
        clk = time.time()
        read_count = 0
        no_data_count = 0
        
        print(f"[DEBUG] 数据读取线程启动，端口: {self.port}, reading={self.reading}")
        
        while not self.stop_event.is_set():
            cclk = time.time()
            rk = cclk - clk
            
            if rk > 0.025:  # 约25ms读取一次
                if self.reading and self.ser is not None:
                    try:
                        if not self.ser.is_open:
                            print("[ERROR] 串口未打开")
                            self.ser = None
                            self.reading = False
                            break
                        
                        available = self.ser.inWaiting()
                        if available > 0:
                            buf = self.ser.read(available)
                            if len(buf) > 0:
                                stamp = time.time()
                                clk = cclk
                                read_count += 1
                                # 显示原始数据的十六进制表示（前32字节）
                                hex_preview = buf[:32].hex() if len(buf) >= 32 else buf.hex()
                                if read_count % 10 == 0:  # 每10次打印一次
                                    print(f"[DEBUG] 从COM口读取到 {len(buf)} 字节数据 (总计 {read_count} 次)")
                                    print(f"[DEBUG] 原始数据前32字节(hex): {hex_preview}")
                                    # 检查是否有数据包头标识
                                    if b'\xAB\x55' in buf[:100]:
                                        head_pos = buf.find(b'\xAB\x55')
                                        print(f"[DEBUG] 找到数据包头 0xAB55 在位置 {head_pos}")
                                    else:
                                        print(f"[WARNING] 未找到数据包头 0xAB55，可能数据格式不对")
                                self._parse_data(buf, stamp)
                                no_data_count = 0
                        else:
                            no_data_count += 1
                            if no_data_count % 200 == 0:  # 每5秒打印一次（200*25ms）
                                print(f"[DEBUG] 等待数据中... (reading={self.reading}, ser={self.ser is not None})")
                    except Exception as e:
                        print(f"[ERROR] 读取串口数据失败: {e}")
                        import traceback
                        traceback.print_exc()
                        self.ser = None
                        self.reading = False
                        if self._on_frame:
                            # 可以发送错误通知
                            pass
                else:
                    if not self.reading:
                        print(f"[DEBUG] reading=False，等待中...")
                    if self.ser is None:
                        print(f"[DEBUG] ser=None，等待中...")
                    time.sleep(0.01)
            else:
                time.sleep(max(0, rk))
        
        print(f"[DEBUG] 数据读取线程结束")
        self.thread_running = False
    
    def _parse_data(self, buf: bytes, stamp: float) -> None:
        """解析接收到的数据。"""
        self.buffer += buf
        Len = len(self.buffer)
        indx = 0
        packet_count = 0
        
        while indx < Len - 7:
            self.protocol.loadBuffer(self.buffer[indx:])
            if self.protocol.headVerify():
                includePak, pakLen = self.protocol.paklenVerify()
                if includePak:
                    if self.protocol.getEpochAndVerify():
                        devData = self.protocol.parsePak()
                        indx += pakLen
                        packet_count += 1
                        self._process_packet(devData, stamp)
                    else:
                        print(f"[DEBUG] 数据包校验失败，包长度: {pakLen}")
                        indx += 1
                else:
                    # 数据不够，等待更多数据
                    break
            else:
                indx += 1
        
        if packet_count > 0:
            print(f"[DEBUG] 成功解析 {packet_count} 个数据包")
        
        # 如果缓冲区太大，可能数据格式不对，清空一部分
        if len(self.buffer) > 10000:
            print(f"[WARNING] 缓冲区过大 ({len(self.buffer)} 字节)，清空缓冲区")
            # 尝试找到最后一个有效包头
            last_head = self.buffer.rfind(b'\xAB\x55')
            if last_head > 0:
                self.buffer = self.buffer[last_head:]
            else:
                self.buffer = b''
        else:
            self.buffer = self.buffer[indx:]
    
    def _process_packet(self, devData: dict, stamp: float) -> None:
        """处理解析后的数据包。"""
        if not self._on_frame:
            print("[WARNING] _on_frame 回调未设置")
            return
        
        emgpayload = devData['emgpayload']
        accBytes = devData.get('accBytes', b'')
        gloveBytes = devData.get('gloveBytes', b'')
        sampleN = devData['sampleN']
        emgChs = devData['emgChs']
        accChs = devData.get('accChs', 0)
        gloveChs = devData.get('gloveChs', 0)
        srate = devData['srate']
        
        devID = devData.get('devID', 0)
        batLevel = devData.get('batLevel', 0)
        print(f"[DEBUG] 处理数据包: sampleN={sampleN}, emgChs={emgChs}, accChs={accChs}, gloveChs={gloveChs}, srate={srate}")
        print(f"[DEBUG] 设备ID: {devID}, 电池电量: {batLevel}%")
        print(f"[DEBUG] EMG payload长度: {len(emgpayload)}, ACC长度: {len(accBytes)}, Glove长度: {len(gloveBytes)}")
        # 显示EMG数据的原始字节（前12字节，即前4个通道的3字节数据）
        if len(emgpayload) >= 12:
            emg_preview = emgpayload[:12].hex()
            print(f"[DEBUG] EMG原始数据前12字节(hex): {emg_preview}")
        
        # 解码所有数据
        all_channels = []
        total_channels = 0
        
        try:
            # 解码EMG数据
            if len(emgpayload) > 0 and emgChs > 0:
                emg_data = self.decoder_emg.decode(emgpayload, sampleN, emgChs)
                # emg_data shape: (sampleN, emgChs)
                all_channels.append(emg_data)
                total_channels += emgChs
                print(f"[DEBUG] EMG数据解码成功: {emg_data.shape}")
            
            # 解码ACC数据
            if len(accBytes) > 0 and accChs > 0:
                acc_data = self.decoder_acc.decode(accBytes, sampleN, accChs)
                all_channels.append(acc_data)
                total_channels += accChs
                print(f"[DEBUG] ACC数据解码成功: {acc_data.shape}")
            
            # 解码Glove数据
            if len(gloveBytes) > 0 and gloveChs > 0:
                glove_data = self.decoder_glove.decode(gloveBytes, sampleN, gloveChs)
                all_channels.append(glove_data)
                total_channels += gloveChs
                print(f"[DEBUG] Glove数据解码成功: {glove_data.shape}")
            
            if not all_channels:
                print("[WARNING] 没有可用的数据通道")
                return
            
            # 合并所有通道数据
            # 注意：每个解码器返回的shape是 (sampleN, chs)
            # 需要确保所有解码器的sampleN相同
            sample_counts = [ch.shape[0] for ch in all_channels]
            if len(set(sample_counts)) > 1:
                print(f"[WARNING] 不同通道类型的样本数不一致: {sample_counts}")
                # 使用最小的样本数
                min_samples = min(sample_counts)
                all_channels = [ch[:min_samples, :] for ch in all_channels]
                sampleN = min_samples
            
            # 合并所有通道数据 (sampleN, total_channels)
            combined_data = np.hstack(all_channels)
            
            # 转置为 (channels, samples) 用于显示
            combined_data = combined_data.T
            
            print(f"[DEBUG] 合并后数据形状: {combined_data.shape}, 数据范围: [{combined_data.min():.2f}, {combined_data.max():.2f}]")
            print(f"[DEBUG] 期望形状: (34, {sampleN}), 实际形状: {combined_data.shape}")
            
            # 创建信号帧（只发送EMG通道用于显示，或者发送所有通道）
            # 根据需求，这里发送所有通道，但UI可以只显示EMG部分
            header = SignalPacketHeader(
                modality="EMG",  # 保持EMG类型，但包含所有通道
                device_id=self.device_id,
                channel_count=total_channels,
                samples_per_channel=sampleN,
                sample_rate=float(srate),
                timestamp=stamp,
                extra={
                    'emg_channels': emgChs,
                    'acc_channels': accChs,
                    'glove_channels': gloveChs,
                }
            )
            frame = SignalFrame(header=header, data=combined_data.astype(np.float32))
            
            # 发布数据
            print(f"[DEBUG] 准备发布数据帧: {total_channels}通道 x {sampleN}样本, _on_frame={self._on_frame is not None}")
            self._publish(frame)
            print(f"[DEBUG] 数据帧已发布: {total_channels}通道 x {sampleN}样本 (EMG:{emgChs}, ACC:{accChs}, Glove:{gloveChs})")
        except Exception as e:
            # 解码失败，打印错误信息
            print(f"[ERROR] 解码数据失败: {e}")
            import traceback
            traceback.print_exc()
    
    @staticmethod
    def get_available_ports() -> list[str]:
        """获取可用的串口列表（CP210x设备）。"""
        pp = re.compile('CP210x')
        ports = lp.comports()
        device = []
        for p in ports:
            id = p.device
            r = re.search(pp, str(p))
            if r is not None:
                device.append(id)
        return device

