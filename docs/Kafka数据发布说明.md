# Kafka数据发布功能说明

## 概述

本系统支持在数据采集的同时，实时将数据发布到Kafka消息队列。数据包格式为：**通道数 × 50个样本**（默认配置）。

## 功能特性

1. **实时数据发布**：采集数据的同时，自动将数据发布到Kafka主题
2. **数据包格式**：每个数据包包含50个样本（可配置），每个样本包含所有通道的数据
3. **自动缓冲**：系统自动管理数据缓冲区，确保数据包格式正确
4. **会话管理**：支持会话ID，便于数据追踪和管理
5. **压缩支持**：支持gzip、snappy、lz4等压缩格式，减少网络传输量

## 安装依赖

```bash
pip install kafka-python
```

## 启动Kafka服务器

**重要**：在使用Kafka数据发布功能之前，必须先启动Kafka服务器。

### 快速启动（需自行准备Kafka）

> 说明：仓库内的 `start_kafka_docker.py`、批处理脚本和 `docker-compose.yml` 已移除，请自行启动本地或远程的Kafka服务。

**方案A：使用已有Kafka集群**  
如果公司/实验室已有Kafka服务，直接在UI里配置 `bootstrap_servers`（如 `10.0.0.5:9092`）即可。

**方案B：本机快速体验（Windows官方二进制）**
1. 下载Kafka二进制包（示例：kafka_2.13-3.7.0）并解压。
2. 启动Zookeeper（若已有独立ZK可跳过）：
   ```bat
   .\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties
   ```
3. 另开终端启动Kafka：
   ```bat
   .\bin\windows\kafka-server-start.bat .\config\server.properties
   ```
4. 验证Topic（可选）：
   ```bat
   .\bin\windows\kafka-topics.bat --bootstrap-server localhost:9092 --list
   ```

**方案C：自行使用Docker/Compose**  
可参考官方/社区镜像自行编写 `docker run` 或 `docker-compose.yml`，保持监听端口 `9092` 可被UI访问。

### 详细安装和启动指南

请参考：[Kafka安装和启动指南](Kafka安装和启动指南.md)

## 配置Kafka

### 1. 启动Kafka服务器

确保Kafka服务器正在运行。默认配置为 `localhost:9092`。

### 2. 在UI中配置Kafka

1. 点击工具栏中的 **"Kafka发布"** 按钮
2. 在弹出的对话框中配置：
   - **Kafka服务器**：Kafka服务器地址，格式为 `host:port`（默认：`localhost:9092`）
   - **Topic名称**：Kafka主题名称（默认：`emg_data`）
   - **数据包大小**：每个数据包包含的样本数（默认：50）
   - **压缩类型**：数据压缩方式（none/gzip/snappy/lz4，默认：gzip）
3. 点击 **"确定"** 完成配置

### 3. 启用Kafka发布

配置完成后，Kafka发布功能会自动启用。按钮会显示为 **"Kafka发布 ✓"**，并变为绿色。

## 数据包格式

### JSON格式

每个数据包是一个JSON对象，包含以下字段：

```json
{
    "device_id": "设备ID",
    "session_id": "会话ID（可选）",
    "timestamp": 1234567890.123,
    "channels": 34,
    "samples": 50,
    "data": [
        [ch1_sample1, ch2_sample1, ..., ch34_sample1],
        [ch1_sample2, ch2_sample2, ..., ch34_sample2],
        ...
        [ch1_sample50, ch2_sample50, ..., ch34_sample50]
    ],
    "timestamps": [ts1, ts2, ..., ts50]
}
```

### 数据说明

- **device_id**：设备标识符
- **session_id**：会话标识符（如果设置了数据保存会话）
- **timestamp**：数据包生成时间戳（Unix时间戳，秒）
- **channels**：通道数（例如：34 = 8 EMG + 12 IMU + 14 Glove）
- **samples**：该数据包包含的样本数（默认50）
- **data**：二维数组，形状为 `(samples, channels)`
  - 第一维：样本索引（0-49）
  - 第二维：通道索引（0-33）
- **timestamps**：每个样本的时间戳列表

## 使用示例

### Python消费者示例

```python
from kafka import KafkaConsumer
import json

# 创建消费者
consumer = KafkaConsumer(
    'emg_data',  # Topic名称
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest',  # 从最新消息开始
    consumer_timeout_ms=1000
)

# 消费消息
for message in consumer:
    packet = message.value
    print(f"设备: {packet['device_id']}")
    print(f"通道数: {packet['channels']}")
    print(f"样本数: {packet['samples']}")
    print(f"数据形状: {len(packet['data'])} x {len(packet['data'][0])}")
    
    # 处理数据
    # packet['data'] 是形状为 (50, 34) 的二维数组
    for sample_idx, sample_data in enumerate(packet['data']):
        # sample_data 是长度为34的一维数组，包含所有通道的数据
        emg_channels = sample_data[:8]  # 前8个通道是EMG
        imu_channels = sample_data[8:20]  # 接下来12个通道是IMU
        glove_channels = sample_data[20:]  # 最后14个通道是Glove
```

### 使用numpy处理数据

```python
import numpy as np
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'emg_data',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest',
    consumer_timeout_ms=1000
)

for message in consumer:
    packet = message.value
    
    # 转换为numpy数组
    data_array = np.array(packet['data'])  # 形状: (50, 34)
    
    # 转置为 (通道数, 样本数) 格式
    data_transposed = data_array.T  # 形状: (34, 50)
    
    # 分离不同通道类型
    emg_data = data_transposed[:8, :]      # EMG: (8, 50)
    imu_data = data_transposed[8:20, :]    # IMU: (12, 50)
    glove_data = data_transposed[20:, :]   # Glove: (14, 50)
    
    # 处理数据...
```

## 性能优化

1. **批量发送**：系统会自动将数据缓冲到50个样本后发送，减少网络开销
2. **异步发送**：Kafka生产者使用异步发送，不阻塞数据采集
3. **压缩**：启用压缩（如gzip）可以显著减少网络传输量
4. **批次大小**：Kafka生产者配置了16KB的批次大小，提高吞吐量

## 故障排除

### 1. 连接失败

**问题**：无法连接到Kafka服务器

**解决方案**：
- 检查Kafka服务器是否正在运行
- 检查服务器地址和端口是否正确
- 检查防火墙设置

### 2. 数据包格式错误

**问题**：接收到的数据包格式不正确

**解决方案**：
- 确保使用正确的反序列化器（JSON）
- 检查数据包中的字段名称和类型

### 3. 性能问题

**问题**：Kafka发布导致采集性能下降

**解决方案**：
- 增加Kafka服务器的资源
- 调整数据包大小（减少batch_size）
- 使用更高效的压缩算法（如snappy）

## 测试消费者程序

系统提供了两个测试用的消费者程序，用于验证Kafka数据接收：

### 1. 简单测试程序（推荐快速验证）

```bash
python scripts/test_kafka_consumer_simple.py
```

功能：
- 快速验证数据包接收
- 显示基本信息和验证结果
- 每10个数据包显示统计信息

### 2. 完整测试程序（详细验证）

```bash
# 基本用法
python scripts/test_kafka_consumer.py

# 自定义配置
python scripts/test_kafka_consumer.py --server localhost:9092 --topic emg_data

# 详细模式（显示每个数据包的完整信息）
python scripts/test_kafka_consumer.py --verbose

# 安静模式（只显示统计信息）
python scripts/test_kafka_consumer.py --quiet

# 从最早的消息开始消费
python scripts/test_kafka_consumer.py --offset earliest
```

功能：
- 完整的数据包验证
- 详细的错误报告
- 实时统计信息
- 可配置的统计间隔

### 测试步骤

1. **启动Kafka服务器**
   ```bash
   # 确保Kafka和Zookeeper正在运行
   ```

2. **启动数据采集并启用Kafka发布**
   - 在UI中点击"Kafka发布"按钮
   - 配置Kafka服务器和Topic
   - 开始数据采集

3. **运行测试消费者**
   ```bash
   python scripts/test_kafka_consumer_simple.py
   ```

4. **验证输出**
   - 应该看到数据包不断接收
   - 检查数据包格式是否正确
   - 验证通道数和样本数

### 预期输出示例

```
============================================================
Kafka数据消费者 - 简单测试
============================================================
配置: localhost:9092, topic: emg_data
按 Ctrl+C 退出

等待数据包...

[14:30:15] ✓ 数据包 #1: 设备=device_001, 通道=34, 样本=50
[14:30:15] ✓ 数据包 #2: 设备=device_001, 通道=34, 样本=50
...
[14:30:16] ✓ 数据包 #10: 设备=device_001, 通道=34, 样本=50
  统计: 10 个数据包, 5.23 包/秒
```

## 参考

- [Kafka官方文档](https://kafka.apache.org/documentation/)
- [kafka-python文档](https://kafka-python.readthedocs.io/)
- [GitHub参考项目](https://github.com/qq1790537742)

## 注意事项

1. **数据包大小**：默认50个样本，可以根据实际需求调整
2. **会话ID**：如果启用了数据保存功能，会自动使用相同的会话ID
3. **设备ID**：自动使用设备的实际ID
4. **时间戳**：每个样本都有对应的时间戳，便于后续分析
5. **测试程序**：使用测试消费者程序验证数据接收，确保数据格式正确

