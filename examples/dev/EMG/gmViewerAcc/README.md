# gmviewer

#### 介绍
用于绘制来自串口的采样数据

#### 软件架构
主程序 + devmanager管理设备 + dataplot管理绘图， 设备和绘图之间通过shm共享数据

#### 特点和改变
设备为串口设备，在通讯协议中添加进了标志符字节来标记数据内容。绘图管理通过读取这些数据来解析设备的信息，如通道数，adc位数，采样率等。通过这种方式，绘图进程自动识别设备，不需要在前端手动指定设备信息。兼容性更好，更加方便。

#### 通讯协议
* 最新协议 2023.07.01
* 包头： AB 55
* 标志符（1字节）：bit1-bit0: 0-12位ADC 1-16位ADC 2-24位ADC 3-32位ADC bit2:不带/带丢包测试 bit3:不带/带trigger
*        bit5-bit4: srate, 0-250, 1-500, 2-1000, 3-2000, 其他位全0，预留
* packlen 包长度
* 数据序列
* 1字节电量（一般为12位ADC所得数据除4而来）
* [1字节丢包测试] 0-255循环
* [1字节trigger] 0-255
* 1字节校验

* 通讯协议v2.0
* 2023.12.01
* 包头： AB 55
* 包长度：2字节, uint16(小端在前)
* 标志符（1字节）：bit4-bit0: 设备标识符 0-ads1299(24位ADC), 1-ads1284(30位ADC), 2-ads1263(32位ADC), 其他依据解码器解释；bit5:包含/不包含status；bit6:带/不带丢包测试 bit7:带/不带trigger
* 增益（1字节）：uint8
* 采样率：2字节, uint16(小端在前)
* 1字节电量, uint8
* [status状态] uint8
* [1字节丢包测试] 0-255循环
* [1字节trigger] uint8
* 数据序列
* 1字节校验

#### 依赖的环境
1. python3.8以上
2.  scipy
    numpy
    pyinstaller
    PyQt5
    pyqtgraph
    pyserial

#### 使用说明

1.  python gmviewer.py

#### 打包
使用pyinstaller, 注意安排upx可以在打包过程中压缩文件

pyinstaller -F gmviewer.py --noconsole  (-F打包成单独文件，--noconsole无控制台)

### 在qt中使用multiprocessing.Process进程时，打包后运行可能会遇到问题
应该在main函数中，比如在if __name__ == '__main__':    下面添加如下语句：
multiprocessing.freeze_support()
