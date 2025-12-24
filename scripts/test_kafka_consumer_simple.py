#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kafka数据消费者简单测试程序（快速验证版本）。
用于快速验证Kafka数据包接收是否正常。

使用方法:
    python scripts/test_kafka_consumer_simple.py
"""

import json
import sys
import time
from datetime import datetime
from typing import List, Dict, Any

try:
    from kafka import KafkaConsumer
except ImportError:
    print("错误: 未安装 kafka-python")
    print("请运行: pip install kafka-python")
    sys.exit(1)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def main():
    """主函数。"""
    print("=" * 60)
    print("Kafka数据消费者 - 简单测试")
    print("=" * 60)
    print("配置: localhost:9092, topic: emg_data")
    print("按 Ctrl+C 退出\n")
    
    try:
        print("正在连接到Kafka服务器...")
        consumer = KafkaConsumer(
            'et_data',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',  # 改为 'earliest' 以接收所有消息（包括历史消息）
            consumer_timeout_ms=5000,
            request_timeout_ms=30000,  # 增加请求超时时间，必须大于session_timeout_ms
            session_timeout_ms=10000,  # 会话超时时间
            enable_auto_commit=True,
            group_id='test_consumer_group',  # 添加消费者组ID
        )
        print("✓ Kafka连接成功")
    except Exception as e:
        error_msg = str(e)
        print(f"\n✗ 错误: 无法连接到Kafka: {error_msg}")
        print("\n请确保:")
        print("  1. Kafka服务器正在运行")
        print("     - 检查Docker容器: docker-compose ps")
        print("     - 或运行诊断: python scripts/check_kafka.py")
        print("  2. 服务器地址正确 (默认: localhost:9092)")
        print("  3. 防火墙未阻止9092端口")
        print("\n如果Kafka未运行，请先启动:")
        print("  python scripts/start_kafka_docker.py")
        print()
        sys.exit(1)
    
    print("等待数据包...\n")
    
    packet_count = 0
    start_time = time.time()
    total_samples = 0
    
    try:
        for message in consumer:
            packet_count += 1
            packet = message.value
            
            # 基本信息
            device_id = packet.get('device_id', 'N/A')
            channels = packet.get('channels', 'N/A')
            samples = packet.get('samples', 'N/A')
            timestamp = packet.get('timestamp', 'N/A')
            session_id = packet.get('session_id', 'N/A')
            
            # 验证数据
            is_valid = True
            errors = []
            
            if 'data' not in packet:
                is_valid = False
                errors.append("缺少data字段")
            elif len(packet['data']) != samples:
                is_valid = False
                errors.append(f"数据样本数不匹配: {len(packet['data'])} != {samples}")
            elif len(packet['data']) > 0 and len(packet['data'][0]) != channels:
                is_valid = False
                errors.append(f"通道数不匹配: {len(packet['data'][0])} != {channels}")
            
            # 打印数据包基本信息
            status = "✓" if is_valid else "✗"
            print("\n" + "=" * 70)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {status} 数据包 #{packet_count}")
            print("=" * 70)
            print(f"设备ID:     {device_id}")
            print(f"会话ID:     {session_id}")
            print(f"时间戳:     {timestamp}")
            print(f"通道数:     {channels}")
            print(f"样本数:     {samples}")
            print(f"Kafka Offset: {message.offset}")
            print(f"Partition:    {message.partition}")
            
            if not is_valid:
                print("\n错误:")
                for error in errors:
                    print(f"  ✗ {error}")
            else:
                total_samples += samples
                
                # 显示数据内容
                if 'data' in packet and len(packet['data']) > 0:
                    data = packet['data']
                    
                    # 转换为numpy数组以便计算统计信息
                    if NUMPY_AVAILABLE:
                        try:
                            data_array = np.array(data)
                            print(f"\n数据统计:")
                            print(f"  形状:        {data_array.shape} (样本数 × 通道数)")
                            print(f"  数据类型:    {data_array.dtype}")
                            print(f"  最小值:      {data_array.min():.4f}")
                            print(f"  最大值:      {data_array.max():.4f}")
                            print(f"  平均值:      {data_array.mean():.4f}")
                            print(f"  标准差:      {data_array.std():.4f}")
                            
                            # 显示每个通道的统计信息（前8个通道）
                            print(f"\n通道统计 (前8个通道):")
                            for ch in range(min(8, channels)):
                                ch_data = data_array[:, ch]
                                print(f"  通道 {ch:2d}: 最小={ch_data.min():8.4f}, "
                                      f"最大={ch_data.max():8.4f}, "
                                      f"平均={ch_data.mean():8.4f}")
                            
                            # 显示前3个样本的数据（前8个通道）
                            print(f"\n前3个样本的数据 (前8个通道):")
                            for i in range(min(3, samples)):
                                sample_data = data_array[i, :min(8, channels)]
                                values_str = ", ".join([f"{v:8.4f}" for v in sample_data])
                                print(f"  样本 {i}: [{values_str}]")
                        except Exception as e:
                            print(f"  数据解析错误: {e}")
                    else:
                        # 没有numpy，显示原始数据
                        print(f"\n数据预览 (前3个样本, 前8个通道):")
                        for i in range(min(3, len(data))):
                            sample = data[i][:min(8, len(data[i]))]
                            values_str = ", ".join([f"{v:.4f}" for v in sample])
                            print(f"  样本 {i}: [{values_str}]")
                    
                    # 显示时间戳信息
                    if 'timestamps' in packet and len(packet['timestamps']) > 0:
                        timestamps = packet['timestamps']
                        if len(timestamps) > 1:
                            time_span = timestamps[-1] - timestamps[0]
                            sample_rate = (len(timestamps) - 1) / time_span if time_span > 0 else 0
                            print(f"\n时间戳信息:")
                            print(f"  时间范围:    {timestamps[0]:.6f} - {timestamps[-1]:.6f}")
                            print(f"  时间跨度:    {time_span:.6f} 秒")
                            print(f"  采样率:      {sample_rate:.2f} Hz")
            
            # 每10个数据包打印一次统计
            if packet_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = packet_count / elapsed if elapsed > 0 else 0
                sample_rate = total_samples / elapsed if elapsed > 0 else 0
                print(f"\n累计统计: {packet_count} 个数据包, {total_samples} 个样本, "
                      f"{rate:.2f} 包/秒, {sample_rate:.2f} 样本/秒")
    
    except KeyboardInterrupt:
        print("\n\n正在退出...")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        consumer.close()
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("最终统计")
        print("=" * 70)
        print(f"总数据包数:   {packet_count}")
        print(f"总样本数:     {total_samples}")
        print(f"运行时间:     {elapsed:.1f} 秒")
        if elapsed > 0:
            print(f"数据包速率:   {packet_count / elapsed:.2f} 包/秒")
            print(f"样本速率:     {total_samples / elapsed:.2f} 样本/秒")

if __name__ == '__main__':
    main()

