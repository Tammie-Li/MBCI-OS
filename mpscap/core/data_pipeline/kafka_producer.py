"""
Kafka数据发布模块。
负责将采集到的数据实时发布到Kafka主题。
参考：https://github.com/qq1790537742 的脑电在线采集和算法实现
"""

from __future__ import annotations

import json
import time
from typing import Optional

import numpy as np

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    KafkaProducer = None
    KafkaError = Exception


class KafkaConfig:
    """Kafka配置类。"""
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "emg_data",
        batch_size: int = 50,  # 每个数据包包含的样本数
        acks: int = 1,  # 等待确认的副本数
        compression_type: str = "gzip",  # 压缩类型
        value_serializer: Optional[callable] = None,
    ) -> None:
        """
        初始化Kafka配置。
        
        参数:
            bootstrap_servers: Kafka服务器地址，格式为 "host:port"
            topic: Kafka主题名称
            batch_size: 每个数据包包含的样本数（默认50）
            acks: 等待确认的副本数（0=不等待，1=等待leader，-1=等待所有副本）
            compression_type: 压缩类型（'gzip', 'snappy', 'lz4', 'none'）
            value_serializer: 值序列化器，如果为None则使用默认的JSON序列化
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.batch_size = batch_size
        self.acks = acks
        self.compression_type = compression_type
        self.value_serializer = value_serializer or self._default_serializer
    
    @staticmethod
    def _default_serializer(data: dict) -> bytes:
        """默认序列化器：将字典转换为JSON字节串。"""
        return json.dumps(data, ensure_ascii=False).encode('utf-8')


class KafkaDataProducer:
    """Kafka数据生产者。
    
    负责将采集到的数据实时发布到Kafka主题。
    数据包格式：通道数 * batch_size（默认50个样本）
    """
    
    def __init__(self, config: KafkaConfig) -> None:
        """
        初始化Kafka生产者。
        
        参数:
            config: Kafka配置对象
        """
        if not KAFKA_AVAILABLE:
            raise ImportError(
                "kafka-python未安装。请运行: pip install kafka-python"
            )
        
        self.config = config
        self.producer: Optional[KafkaProducer] = None
        self._buffer: list = []  # 数据缓冲区（FIFO队列，严格按照时间顺序）
        self._buffer_samples: int = 0  # 缓冲区中的样本数
        self._device_id: Optional[str] = None
        self._session_id: Optional[str] = None
        self._is_connected: bool = False
        
        # 数据包统计
        self._packet_count: int = 0
        self._error_count: int = 0
        self._total_samples_received: int = 0  # 总共接收的样本数（用于验证不重复不丢失）
        self._total_samples_sent: int = 0  # 总共发送的样本数
    
    def connect(self) -> bool:
        """连接到Kafka服务器。"""
        if self._is_connected:
            return True
        
        try:
            # 解析服务器地址
            servers = self.config.bootstrap_servers.split(',') if isinstance(self.config.bootstrap_servers, str) else self.config.bootstrap_servers
            servers = [s.strip() for s in servers]
            
            # 创建KafkaProducer
            # 注意：KafkaProducer的连接是延迟的，会在第一次发送消息时建立
            # 如果配置有问题，会在创建时或首次发送时抛出异常
            self.producer = KafkaProducer(
                bootstrap_servers=servers,
                acks=self.config.acks,
                compression_type=self.config.compression_type,
                value_serializer=self.config.value_serializer,
                # 性能优化参数
                batch_size=16384,  # 16KB批次大小
                linger_ms=10,  # 等待10ms以批量发送
                max_in_flight_requests_per_connection=5,
                # 连接超时设置
                request_timeout_ms=10000,  # 10秒超时
                connections_max_idle_ms=540000,  # 9分钟空闲超时
            )
            
            # Producer创建成功，标记为已连接
            # 实际连接会在首次发送消息时建立，如果连接失败会在发送时抛出异常
            self._is_connected = True
            self._total_samples_received = 0
            self._total_samples_sent = 0
            print(f"[Kafka] Producer已创建: {self.config.bootstrap_servers}, topic: {self.config.topic}")
            print(f"[Kafka] 注意: 连接将在首次发送消息时建立")
            return True
        except Exception as e:
            error_msg = str(e)
            print(f"[Kafka] 连接失败: {error_msg}")
            print(f"[Kafka] 服务器地址: {self.config.bootstrap_servers}")
            print(f"[Kafka] Topic: {self.config.topic}")
            
            # 提供诊断建议
            if "NoBrokersAvailable" in error_msg or "Unable to connect" in error_msg:
                print("\n[Kafka] 诊断建议:")
                print("  1. 检查Kafka服务是否正在运行:")
                print("     - Docker方式: docker-compose ps")
                print("     - 本地方式: 检查Kafka进程")
                print("  2. 检查服务器地址和端口是否正确")
                print("  3. 检查防火墙设置")
                print("  4. 运行诊断脚本: python scripts/check_kafka.py")
                print("  5. 查看Kafka日志: docker-compose logs kafka")
            
            self._is_connected = False
            return False
    
    def disconnect(self) -> None:
        """断开Kafka连接。"""
        if self.producer:
            try:
                self.producer.flush(timeout=5.0)  # 等待所有消息发送完成
                self.producer.close(timeout=5.0)
            except Exception as e:
                print(f"[Kafka] 断开连接时出错: {e}")
            finally:
                self.producer = None
                self._is_connected = False
        
        # 清空缓冲区
        self._buffer.clear()
        self._buffer_samples = 0
    
    def set_device_id(self, device_id: str) -> None:
        """设置设备ID。"""
        self._device_id = device_id
    
    def set_session_id(self, session_id: str) -> None:
        """设置会话ID。"""
        self._session_id = session_id
    
    def publish_data(
        self,
        data: np.ndarray,
        timestamps: Optional[list] = None,
        device_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """
        发布数据到Kafka。
        
        参数:
            data: 数据数组，形状为 (通道数, 样本数)
            timestamps: 时间戳列表，长度应与样本数一致
            device_id: 设备ID，如果提供则覆盖之前设置的设备ID
            session_id: 会话ID，如果提供则覆盖之前设置的会话ID
        
        返回:
            成功发布的数据包数量
        """
        if not self._is_connected or self.producer is None:
            return 0
        
        if data.size == 0:
            return 0
        
        channels, samples = data.shape
        
        # 使用提供的device_id和session_id，或使用之前设置的
        current_device_id = device_id or self._device_id or "unknown"
        current_session_id = session_id or self._session_id
        
        # 将数据添加到缓冲区
        if timestamps is None:
            # 如果没有提供时间戳，使用当前时间生成
            base_time = time.time()
            timestamps = [base_time + i / 1000.0 for i in range(samples)]
        
        # 将数据按样本组织：每个样本包含所有通道的数据
        # 严格按照时间顺序追加到缓冲区（FIFO队列）
        for i in range(samples):
            sample_data = {
                'device_id': current_device_id,
                'session_id': current_session_id,
                'timestamp': timestamps[i] if i < len(timestamps) else time.time(),
                'channels': channels,
                'samples': 1,
                'data': data[:, i].tolist(),  # 该样本所有通道的数据
            }
            self._buffer.append(sample_data)  # 追加到队列末尾
            self._buffer_samples += 1
            self._total_samples_received += 1
        
        # 严格按照batch_size发送：只有当缓冲区达到batch_size时才发送
        # 例如：10个样本 -> 等待；50个样本 -> 发送；62个样本 -> 发送50个，剩余12个
        packets_sent = 0
        while self._buffer_samples >= self.config.batch_size:
            # 从队列头部提取batch_size个样本（FIFO：先入先出）
            packet_samples = self._buffer[:self.config.batch_size]
            self._buffer = self._buffer[self.config.batch_size:]  # 移除已发送的样本
            self._buffer_samples -= self.config.batch_size
            self._total_samples_sent += len(packet_samples)
            
            # 构建数据包：通道数 * batch_size
            # 格式：{channels: N, samples: batch_size, data: [[ch1_sample1, ch2_sample1, ...], [ch1_sample2, ...], ...]}
            packet_data = {
                'device_id': current_device_id,
                'session_id': current_session_id,
                'timestamp': time.time(),
                'channels': channels,
                'samples': len(packet_samples),
                'data': [sample['data'] for sample in packet_samples],  # 形状: (batch_size, channels)
                'timestamps': [sample['timestamp'] for sample in packet_samples],
            }
            
            # 发送到Kafka
            try:
                # 首次发送时，尝试同步等待以捕获连接错误
                if self._packet_count == 0:
                    try:
                        # 使用get()方法同步等待，可以捕获连接错误
                        future = self.producer.send(
                            self.config.topic,
                            value=packet_data,
                            key=current_device_id.encode('utf-8') if current_device_id else None,
                        )
                        # 等待最多5秒以建立连接
                        record_metadata = future.get(timeout=5)
                        print(f"[Kafka] ✓ 连接已建立，首次数据包已发送 (Offset: {record_metadata.offset})")
                        packets_sent += 1
                        self._packet_count += 1
                    except Exception as e:
                        print(f"[Kafka] ✗ 首次发送失败: {e}")
                        import traceback
                        traceback.print_exc()
                        self._error_count += 1
                        # 标记连接失败
                        self._is_connected = False
                else:
                    # 后续发送使用异步方式
                    future = self.producer.send(
                        self.config.topic,
                        value=packet_data,
                        key=current_device_id.encode('utf-8') if current_device_id else None,
                    )
                    # 异步发送，不阻塞
                    future.add_callback(self._on_send_success)
                    future.add_errback(self._on_send_error)
                    packets_sent += 1
                    self._packet_count += 1
            except Exception as e:
                print(f"[Kafka] ✗ 发送失败: {e}")
                import traceback
                traceback.print_exc()
                self._error_count += 1
                self._is_connected = False
        
        # 注意：不在这里进行定期刷新，严格按照batch_size发送
        # 只有在程序关闭或显式调用flush()时才会发送剩余数据
        # 这确保了严格的时序关系：数据必须累积到batch_size才发送
        
        return packets_sent
    
    def _flush_buffer_immediate(self) -> None:
        """立即刷新缓冲区，发送剩余数据（内部方法，仅在flush()时调用）。"""
        if not self._is_connected or self.producer is None:
            return
        
        if self._buffer_samples == 0:
            return
        
        channels = self._buffer[0]['channels'] if self._buffer else 0
        current_device_id = self._device_id or "unknown"
        current_session_id = self._session_id
        
        packet_data = {
            'device_id': current_device_id,
            'session_id': current_session_id,
            'timestamp': time.time(),
            'channels': channels,
            'samples': len(self._buffer),
            'data': [sample['data'] for sample in self._buffer],
            'timestamps': [sample['timestamp'] for sample in self._buffer],
        }
        
        try:
            future = self.producer.send(
                self.config.topic,
                value=packet_data,
                key=current_device_id.encode('utf-8') if current_device_id else None,
            )
            future.add_callback(self._on_send_success)
            future.add_errback(self._on_send_error)
            self._packet_count += 1
            self._total_samples_sent += len(self._buffer)
        except Exception as e:
            print(f"[Kafka] ✗ 刷新缓冲区失败: {e}")
            import traceback
            traceback.print_exc()
            self._error_count += 1
        
        self._buffer.clear()
        self._buffer_samples = 0
    
    def flush(self) -> None:
        """刷新缓冲区，发送剩余数据（仅在程序关闭时调用）。"""
        if not self._is_connected or self.producer is None:
            return
        
        # 发送缓冲区中剩余的数据（即使不足batch_size）
        if self._buffer_samples > 0:
            self._flush_buffer_immediate()
        
        # 验证数据完整性：检查是否有样本丢失或重复
        if self._total_samples_received > 0:
            pending = self._total_samples_received - self._total_samples_sent
            if pending != 0:
                print(f"[Kafka] ⚠ 数据完整性检查: 接收={self._total_samples_received}, 发送={self._total_samples_sent}, 待发送={pending}")
            else:
                print(f"[Kafka] ✓ 数据完整性验证通过: 所有 {self._total_samples_received} 个样本已发送")
        
        # 确保所有消息都已发送
        if self.producer:
            self.producer.flush(timeout=1.0)
    
    def _on_send_success(self, record_metadata) -> None:
        """发送成功回调。"""
        # 每10个数据包打印一次成功信息，避免刷屏
        if self._packet_count % 10 == 0:
            print(f"[Kafka] ✓ 已发送 {self._packet_count} 个数据包 (最新Offset: {record_metadata.offset})")
    
    def _on_send_error(self, exception) -> None:
        """发送失败回调。"""
        print(f"[Kafka] ✗ 异步发送失败: {exception}")
        print(f"     错误类型: {type(exception).__name__}")
        import traceback
        traceback.print_exc()
        self._error_count += 1
        # 如果是连接错误，标记为未连接
        error_str = str(exception).lower()
        if "nobrokersavailable" in error_str or "connection" in error_str or "timeout" in error_str:
            print(f"[Kafka] ⚠ 检测到连接错误，标记为未连接状态")
            self._is_connected = False
    
    def get_stats(self) -> dict:
        """获取统计信息。"""
        return {
            'packet_count': self._packet_count,
            'error_count': self._error_count,
            'buffer_samples': self._buffer_samples,
            'is_connected': self._is_connected,
            'total_samples_received': self._total_samples_received,
            'total_samples_sent': self._total_samples_sent,
            'pending_samples': self._total_samples_received - self._total_samples_sent,
        }
    
    def reset_stats(self) -> None:
        """重置统计信息。"""
        self._packet_count = 0
        self._error_count = 0




