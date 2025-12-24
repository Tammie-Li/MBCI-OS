# Kafka安装和启动指南

## 概述

在使用Kafka数据发布功能之前，需要先安装并启动Kafka服务器。本指南提供Windows环境下的Kafka安装和启动方法。

## 方法一：使用Docker（推荐，最简单）

### 1. 安装Docker Desktop

如果还没有安装Docker，请先下载并安装：
- 下载地址：https://www.docker.com/products/docker-desktop
- 安装后启动Docker Desktop

### 2. 使用Docker Compose启动Kafka

创建 `docker-compose.yml` 文件（项目根目录下）：

```yaml
version: '3.8'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
```

### 3. 启动Kafka

在项目根目录下运行：

```bash
docker-compose up -d
```

### 4. 验证Kafka是否运行

```bash
docker-compose ps
```

应该看到zookeeper和kafka两个服务都在运行。

### 5. 停止Kafka

```bash
docker-compose down
```

## 方法二：本地安装Kafka（Windows）

### 1. 下载Kafka

1. 访问Apache Kafka官网：https://kafka.apache.org/downloads
2. 下载最新版本的二进制文件（例如：kafka_2.13-3.5.0.tgz）
3. 解压到本地目录，例如：`C:\kafka`

### 2. 安装Java

Kafka需要Java运行环境：
- 下载并安装Java JDK 8或更高版本
- 设置JAVA_HOME环境变量

### 3. 启动Zookeeper

打开命令提示符（CMD）或PowerShell，进入Kafka目录：

```bash
cd C:\kafka
.\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties
```

保持这个窗口打开。

### 4. 启动Kafka服务器

打开**另一个**命令提示符窗口：

```bash
cd C:\kafka
.\bin\windows\kafka-server-start.bat .\config\server.properties
```

保持这个窗口打开。

### 5. 创建Topic（可选）

如果需要手动创建Topic：

```bash
cd C:\kafka
.\bin\windows\kafka-topics.bat --create --topic emg_data --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
```

### 6. 验证Topic

```bash
.\bin\windows\kafka-topics.bat --list --bootstrap-server localhost:9092
```

应该能看到 `emg_data` 主题。

## 方法三：使用Kafka控制脚本（Windows批处理）

项目提供了便捷的启动脚本（见下文）。

## 快速验证Kafka是否运行

### 方法1：使用测试脚本

运行项目提供的测试脚本：

```bash
python scripts/test_kafka_consumer_simple.py
```

如果Kafka正在运行，应该看到"等待数据包..."的提示。

### 方法2：使用Kafka命令行工具

```bash
# 列出所有Topic
.\bin\windows\kafka-topics.bat --list --bootstrap-server localhost:9092

# 查看Topic详情
.\bin\windows\kafka-topics.bat --describe --topic emg_data --bootstrap-server localhost:9092
```

### 方法3：使用Python测试连接

```python
from kafka import KafkaConsumer

try:
    consumer = KafkaConsumer(
        'emg_data',
        bootstrap_servers=['localhost:9092'],
        consumer_timeout_ms=1000
    )
    print("Kafka连接成功！")
    consumer.close()
except Exception as e:
    print(f"Kafka连接失败: {e}")
```

## 常见问题

### 1. 端口被占用

**错误**: `Address already in use` 或 `端口已被占用`

**解决方案**:
- 检查9092端口是否被其他程序占用
- 修改Kafka配置文件中的端口号
- 或者停止占用端口的程序

### 2. Zookeeper连接失败

**错误**: `Connection refused` 或 `无法连接到Zookeeper`

**解决方案**:
- 确保Zookeeper已启动
- 检查Zookeeper端口（默认2181）是否可访问
- 检查防火墙设置

### 3. Topic不存在

**错误**: `Topic不存在` 或 `UnknownTopicOrPartitionException`

**解决方案**:
- Kafka会自动创建Topic（如果配置允许）
- 或者手动创建Topic（见上面的命令）

### 4. Java版本问题

**错误**: `UnsupportedClassVersionError`

**解决方案**:
- 确保安装了Java 8或更高版本
- 检查JAVA_HOME环境变量

## 推荐配置

### Docker方式（最简单）

- ✅ 无需手动配置
- ✅ 自动管理依赖
- ✅ 易于启动和停止
- ✅ 跨平台兼容

### 本地安装方式

- ✅ 更灵活的控制
- ✅ 可以自定义配置
- ⚠️ 需要手动管理Zookeeper和Kafka
- ⚠️ 需要安装Java

## 下一步

Kafka启动成功后：

1. 在数据采集UI中点击"Kafka发布"按钮
2. 配置Kafka服务器地址（默认：localhost:9092）
3. 配置Topic名称（默认：emg_data）
4. 开始数据采集
5. 运行测试消费者程序验证数据接收

## 参考资源

- [Apache Kafka官方文档](https://kafka.apache.org/documentation/)
- [Docker Compose文档](https://docs.docker.com/compose/)
- [Confluent Platform文档](https://docs.confluent.io/)

