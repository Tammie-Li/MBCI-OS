# Docker Desktop 安装指南（Windows）

## 概述

Docker Desktop是Docker在Windows和Mac上的官方桌面应用程序，提供了图形界面来管理Docker容器和镜像。

## 系统要求

在安装Docker Desktop之前，请确保您的系统满足以下要求：

### Windows 10/11 要求

1. **Windows版本**
   - Windows 10 64位：专业版、企业版或教育版（版本1903或更高，带Build 18362或更高）
   - Windows 11 64位：家庭版或专业版
   - **不支持Windows 10家庭版（除非满足WSL 2要求）**

2. **硬件要求**
   - 64位处理器，支持二级地址转换（SLAT）
   - 4GB系统RAM（推荐8GB或更多）
   - 在BIOS中启用虚拟化（VT-x/AMD-V）

3. **WSL 2功能（Windows 10/11）**
   - 启用WSL 2功能
   - 安装WSL 2 Linux内核更新包

## 安装步骤

### 方法一：官方安装程序（推荐）

#### 步骤1：下载Docker Desktop

1. 访问Docker官网：
   - 官网：https://www.docker.com/products/docker-desktop
   - 或直接下载：https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe

2. 点击"Download for Windows"按钮

3. 下载完成后，您会得到一个名为 `Docker Desktop Installer.exe` 的文件

#### 步骤2：运行安装程序

1. **以管理员身份运行**安装程序
   - 右键点击 `Docker Desktop Installer.exe`
   - 选择"以管理员身份运行"

2. **安装选项**
   - 勾选"Use WSL 2 instead of Hyper-V"（如果可用，推荐）
   - 点击"OK"继续

3. **等待安装完成**
   - 安装过程可能需要几分钟
   - 安装完成后，会提示"Close and restart"

4. **重启计算机**
   - 按照提示重启计算机（如果要求）

#### 步骤3：启动Docker Desktop

1. 重启后，在开始菜单中搜索"Docker Desktop"
2. 点击启动Docker Desktop
3. 首次启动会显示服务协议，点击"Accept"接受
4. 等待Docker Desktop完全启动
   - 系统托盘会出现Docker图标（鲸鱼图标）
   - 图标停止闪烁表示启动完成
   - 通常需要1-2分钟

#### 步骤4：验证安装

打开命令提示符（CMD）或PowerShell，运行：

```bash
docker --version
```

如果显示版本号，说明安装成功。

### 方法二：使用Winget（Windows 11/Windows 10 1809+）

如果您使用Windows 11或较新的Windows 10，可以使用Winget包管理器：

```bash
winget install Docker.DockerDesktop
```

## 启用WSL 2（Windows 10/11）

Docker Desktop需要WSL 2支持。如果您的系统未启用WSL 2，请按照以下步骤操作：

### 步骤1：启用WSL功能

1. 以管理员身份打开PowerShell
   - 右键点击"开始"菜单
   - 选择"Windows PowerShell (管理员)"或"终端 (管理员)"

2. 运行以下命令启用WSL功能：

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

3. **重要**：运行后应该看到"操作成功完成"的提示

### 步骤2：启用虚拟机平台

在同一个PowerShell窗口中运行：

```powershell
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

### 步骤3：重启计算机

**必须重启计算机**以使更改生效。重启后继续下一步。

### 步骤4：安装WSL 2 Linux内核更新包

⚠️ **重要提示**：只有在完成步骤1-3并重启后，才能安装WSL 2更新包。如果直接安装会提示错误："this update only applies to machines with the windows subsystem for linux"

**方法一：手动下载安装（推荐）**

1. 下载WSL 2更新包：
   - 官方下载链接：https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi
   - 或访问：https://aka.ms/wsl2kernel

2. 运行下载的`.msi`文件
   - 双击运行，按照提示完成安装
   - 如果提示需要管理员权限，右键选择"以管理员身份运行"

**方法二：使用Windows更新（Windows 11推荐）**

Windows 11通常已经内置WSL 2支持，可以跳过此步骤，直接进行步骤5。

**方法三：使用wsl --install命令（最简单）**

在管理员PowerShell中运行：

```powershell
wsl --install
```

这个命令会自动完成所有WSL 2的安装步骤，包括下载和安装内核更新包。

### 步骤5：将WSL 2设置为默认版本

在PowerShell中运行：

```powershell
wsl --set-default-version 2
```

如果提示成功，说明WSL 2已正确配置。

### 步骤6：验证WSL 2安装

运行以下命令检查WSL状态：

```powershell
wsl --status
```

应该看到类似输出：
```
默认版本: 2
```

如果显示"默认版本: 1"，说明WSL 2未正确设置，需要检查前面的步骤。

## 验证安装

### 1. 检查Docker版本

```bash
docker --version
```

预期输出示例：
```
Docker version 24.0.0, build abc123
```

### 2. 检查Docker Compose版本

```bash
docker-compose --version
```

或（新版本）：
```bash
docker compose version
```

### 3. 运行测试容器

```bash
docker run hello-world
```

如果看到"Hello from Docker!"消息，说明安装成功。

### 4. 使用项目检查脚本

运行项目提供的检查脚本：

```bash
python scripts/check_docker.py
```

## 常见问题

### 1. 安装失败：需要WSL 2

**问题**：安装时提示需要WSL 2支持

**解决方案**：
- 按照上面的"启用WSL 2"步骤操作
- **必须按顺序执行**：先启用WSL功能 → 重启 → 再安装WSL 2更新包
- 确保已安装WSL 2 Linux内核更新包
- 重启计算机后重试

### 1.1. WSL 2更新包安装失败："this update only applies to machines with the windows subsystem for linux"

**问题**：安装WSL 2更新包时提示此错误

**原因**：在启用WSL功能之前就尝试安装WSL 2更新包

**解决方案**：
1. **先启用WSL功能**（必须完成）：
   ```powershell
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   ```
2. **重启计算机**（必须重启）
3. **然后**再安装WSL 2更新包
4. 或者使用更简单的方法：运行 `wsl --install`（会自动处理所有步骤）

### 2. Docker Desktop启动失败

**问题**：Docker Desktop无法启动

**解决方案**：
- 检查虚拟化是否在BIOS中启用
- 确保Hyper-V或WSL 2已正确安装
- 以管理员身份运行Docker Desktop
- 查看Docker Desktop日志：`%LOCALAPPDATA%\Docker\log.txt`

### 3. 虚拟化未启用

**问题**：提示虚拟化未启用

**解决方案**：
1. 重启计算机进入BIOS设置
2. 找到虚拟化选项（通常名为"Virtualization Technology"、"Intel VT-x"或"AMD-V"）
3. 启用该选项
4. 保存并退出BIOS
5. 重启计算机

### 4. 端口被占用

**问题**：Docker使用的端口被其他程序占用

**解决方案**：
- 检查端口占用：`netstat -ano | findstr :9092`
- 停止占用端口的程序
- 或在Docker Desktop设置中更改端口配置

### 5. 防火墙阻止

**问题**：防火墙阻止Docker连接

**解决方案**：
- 在Windows防火墙中允许Docker Desktop
- 或临时禁用防火墙测试

## 配置Docker Desktop

### 基本设置

1. 右键点击系统托盘的Docker图标
2. 选择"Settings"
3. 在"General"中：
   - 可以设置启动时自动启动Docker
   - 可以设置资源限制（CPU、内存）

### 资源分配建议

- **CPU**：至少2核，推荐4核或更多
- **内存**：至少4GB，推荐8GB或更多
- **磁盘**：至少20GB可用空间

## 卸载Docker Desktop

如果需要卸载Docker Desktop：

1. 打开"设置" > "应用" > "应用和功能"
2. 搜索"Docker Desktop"
3. 点击"卸载"
4. 按照提示完成卸载

## 下一步

安装并启动Docker Desktop后：

1. **验证安装**：
   ```bash
   python scripts/check_docker.py
   ```

2. **启动Kafka**：
   ```bash
   python scripts/start_kafka_docker.py
   ```

3. **检查Kafka连接**：
   ```bash
   python scripts/check_kafka.py
   ```

## 参考资源

- [Docker Desktop官方文档](https://docs.docker.com/desktop/)
- [Docker Desktop for Windows文档](https://docs.docker.com/desktop/install/windows-install/)
- [WSL 2安装指南](https://docs.microsoft.com/zh-cn/windows/wsl/install)
- [Docker官网](https://www.docker.com/)

## 技术支持

如果遇到安装问题：

1. 查看Docker Desktop日志
2. 访问Docker社区论坛
3. 查看GitHub Issues
4. 联系Docker支持



