
from psychopy import visual, event, core
from loguru import logger
from Lib.Boruikang.neuracle import Neuracle
import math
import numpy as np
import serial
import os
from scipy import signal


class Trigger:
    def __init__(self, com):
        self.ser = serial.Serial(com, 115200, timeout=0.5)
        if not self.ser.isOpen():
            self.ser.open()

    def send_trigger(self, num):
        trigger_list = [0x01, 0xe1, 0x01, 0x00, 0x01]
        trigger_list[-1] = num

        self.ser.write(trigger_list)

    def close_trigger(self):
        self.ser.close()


class SSVEPParadigm:
    def __init__(self, win, refresh_rate = 75):

        self.window = win
        # 屏幕刷新率
        self.refresh_rate = refresh_rate
        
        # 40个刺激（可选择区域）4x10排列
        self.stim_c_num = 10
        self.stim_r_num = 4
        self.stim_num = self.stim_c_num * self.stim_r_num

        # 定义刺激频率
        # 频率为8，8.2，8.4，......, 15.8
        self.stim_frequency = np.array([8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8,
                    10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 11.6, 11.8,
                    12.0, 12.2, 12.4, 12.6, 12.8, 13.0, 13.2, 13.4, 13.6, 13.8,
                    14.0, 14.2, 14.4, 14.6, 14.8, 15.0, 15.2, 15.4, 15.6, 15.8])
        
        # 刺激初始相位
        self.stim_phase_0 = np.array([0, 0.5, 1.0, 1.5, 0, 0.5, 1.0, 1.5, 0, 0.5, 
                                      1.0, 1.5, 0, 0.5, 1.0, 1.5, 0, 0.5, 1.0, 1.5,
                                      0, 0.5, 1.0, 1.5, 0, 0.5, 1.0, 1.5, 0, 0.5, 
                                      1.0, 1.5, 0, 0.5, 1.0, 1.5, 0, 0.5, 1.0, 1.5]) / 2
        logger.info('[paradigm] stimuli frequency: {}'.format(self.stim_frequency))

        # 刺激尺寸
        self.stim_radius = 160

        # 刺激块背景颜色
        self.stim_colors = (1, 1, 1)

        # 刺激块文字颜色
        self.stim_text_colors = (0, 0, 0)

        # 结束等待标志位
        self.stim_flag, self.show_flag = False, False

        x, y = np.meshgrid(np.arange(0, self.stim_c_num), np.arange(0, self.stim_r_num))
        
        # 留给文本显示的高度
        self.init_ver_hegith = 60

        # 计算刺激的位置
        self._calculate_pos()

        # 刺激初始化
        self.stim_size = [self.stim_radius, self.stim_radius]
        globForm = visual.ElementArrayStim(self.window, nElements=self.stim_num, sfs=0, sizes=self.stim_size,
                                           xys=self.stim_pos, phases=0, colors=self.stim_colors, elementTex='sin',
                                           elementMask=None)
        self.stim = globForm

        # 红三角目标提示初始化
        self.triangle_tip_radius = 30
        self.triangle_tip = visual.Polygon(win=self.window, edges=3, units='pix', radius=self.triangle_tip_radius, fillColor='red',
                                      lineColor='red', pos = [0, 0])

        # 刺激文本初始化
        text_list = [str(i + 1) for i in range(self.stim_num)]
        self.stim_text = []
        for i in range(self.stim_num):
            self.stim_text.append(visual.TextStim(self.window, text=text_list[i], pos=self.stim_pos[i], colorSpace="rgb255", color=self.stim_text_colors, height=70,
                                     autoLog=False))
        
        # 红色框结果提示
        self.rectangle = visual.Rect(win=self.window, width=self.stim_radius+25, height=self.stim_radius+25, lineWidth=7,
                                units='pix', lineColor='red', pos=[0,0], fillColor=None)

    def show_init_ssvep_gui(self, target_idx):
        # target_idx: 使用红三角提示被试注视的位置

        # 显示系统初始化
        self.init_txt = visual.TextStim(win=self.window, text=f'请注视{target_idx+1}块', pos=(400, 440), height=60, color='red')
        self.init_txt.setPos([-20, 440])
        self.init_txt.draw()

        # 生成初始化帧
        globForm_init = visual.ElementArrayStim(self.window, nElements=self.stim_num, sfs=0, sizes=self.stim_size,
                                           xys=self.stim_pos, phases=0, colors=self.stim_colors, elementTex='sin',
                                           elementMask=None)
        self.stim_init = globForm_init

        x, y = self.stim_pos[target_idx]
        self.triangle_tip.fillColor = 'red'
        self.triangle_tip.lineColor = 'red'
        self.triangle_tip.setPos([x, y - self.stim_radius * 0.5 - self.triangle_tip_radius])
        self.triangle_tip.draw()

        self.stim_init.draw()
        for i in range(self.stim_num):
            self.stim_text[i].draw()
        self.window.flip()
        core.wait(0.5)

    def start_one_stim(self, time_length=4.2):
        start_stim_time = core.getTime()
        frame_num = 0
        while frame_num <= time_length * self.refresh_rate:
            self.stim.phases = self.stim_phase_0 + frame_num / self.refresh_rate * self.stim_frequency
            self.stim.draw()
            for i in range(self.stim_num):
                self.stim_text[i].draw()
            self.window.flip()
            frame_num = frame_num + 1
        end_stim_time = core.getTime()
        print(end_stim_time - start_stim_time)

    
    def show_result(self, res):
        # 根据用户的选择结果绘制结果提示
        res = res - 1 
        x, y = self.stim_pos[res]

        pos = self.stim_pos[res]
        self.rectangle.setPos(pos)

        self.init_txt.text = "SSVEP脑电系统的综合判定为区域为---" + str(res+1) + ""
        self.init_txt.setPos([-345, 440])
        self.init_txt.draw()
        self.stim.phases = [0 for i in range(self.stim_num)]
        self.stim.draw()
        for i in range(self.stim_num):
            self.stim_text[i].draw()
        self.rectangle.draw()
        self.window.flip()
        event.waitKeys()
    

    def _calculate_pos(self):
        # 后续需要修改为根据刺激排列，计算刺激的位置
        self.stim_pos = []
        for idx in range(self.stim_num):
            # 屏幕最左端 + 左边留白 + 位置
            x_tmp = -960 + 105 + (self.stim_radius + 30) * (idx % 10)
            y_tmp = 540 - 280 - (self.stim_radius + 70) * (idx // 10)
            self.stim_pos.append([x_tmp, y_tmp])


class CCAClass:
    def __init__(self):
        srate = 250
        multiplicateTime = 5
        # 计算时间
        calTime = 4
        # 计算偏移时间（s）
        offsetTime = 0
        # 偏移长度
        self.offsetLength = math.floor(offsetTime * srate)
        # 计算长度
        self.sampleCount = calTime * srate
        
        frequencySet = [8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8,
                    10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 11.6, 11.8,
                    12.0, 12.2, 12.4, 12.6, 12.8, 13.0, 13.2, 13.4, 13.6, 13.8,
                    14.0, 14.2, 14.4, 14.6, 14.8, 15.0, 15.2, 15.4, 15.6, 15.8]

        # 预处理滤波器设置
        self.filterB, self.filterA = self.getPreFilter(srate)
        # 正余弦参考信号
        targetTemplateSet = []
        # 采样点
        t = np.linspace(0, (self.sampleCount - 1) / srate, int(self.sampleCount), endpoint=True)
        t = t.reshape(1, len(t))
        # 对于每个频率
        for freIndex in range(0, len(frequencySet)):
            frequency = frequencySet[freIndex]
            testFre = np.linspace(frequency, frequency * multiplicateTime, int(multiplicateTime), endpoint=True)
            testFre = testFre.reshape(1, len(testFre))
            numMatrix = 2 * np.pi * np.dot(testFre.T, t)
            cosSet = np.cos(numMatrix)
            sinSet = np.sin(numMatrix)
            csSet = np.append(cosSet, sinSet, axis=0)
            targetTemplateSet.append(csSet)
        # 初始化算法
        self.targetTemplateSet = targetTemplateSet
        
    def getPreFilter(self, srate):
        fs = srate
        f0 = 50
        Q = 35
        b, a = signal.iircomb(f0, Q, ftype='notch', fs=fs)
        return b, a

    def recognize(self, x):
        # 数据预处理
        channels = [50, 51, 52, 53, 54, 57, 58, 59]
        # channels = [50, 51, 52, 53, 57, 56, 57, 54]
        x = x[:, ::4]
        x = x[channels, self.offsetLength: self.offsetLength+self.sampleCount]


        data = signal.filtfilt(self.filterB, self.filterA, x)

        # CCA算法
        p = []
        data = data.T
        # qr分解,data:length*channel
        [Q_temp, R_temp] = np.linalg.qr(data)
        for frequencyIndex in range(0,len(self.targetTemplateSet)):
            template = self.targetTemplateSet[frequencyIndex]
            template = template[:, 0:data.shape[0]]
            template = template.T
            [Q_cs,R_cs] = np.linalg.qr(template)
            data_svd = np.dot(Q_temp.T,Q_cs)
            [U,S,V] = np.linalg.svd(data_svd)
            rho = 1.25 * S[0] + 0.67 * S[1] + 0.5 * S[2]
            p.append(rho)
        result = p.index(max(p))
        result = result+1
        return result




if __name__ == "__main__":
    win = visual.Window(size=(1920, 1080), units='pix', color=(-1, -1, -1),
                        allowGUI=True, allowStencil=True, screen=-1)
    
    # 定义实验范式
    para = SSVEPParadigm(win)

    # 定义在线的脑电信号采集设备
    # neuracle = Neuracle(time_buffer=4.2)

    # 定义SSVEP脑电信号处理算法
    algorithm = CCAClass()

    # 初始化
    trigger_list = {"Block_Start": 101, "Block_Start": 102}
    # trigger_manage = Trigger(com="com6")

    BLOCK = 1
    TARGET = 40

    x, y = [], []
    # neuracle.start()


    for block in range(BLOCK):
        print("Block: ", block+1)
        # trigger_manage.send_trigger(101)
        for target in range(TARGET):
            # 每显示10次，被试休息一次，防止疲劳

            para.show_init_ssvep_gui(target)

            # 前0.2s用于去基线，后4s用于训练样本
            # trigger_manage.send_trigger(120+target)
            para.start_one_stim(time_length=0.1)

            # 实时获取数据包
            # data = neuracle.get_data()

            # 脑电信号处理，获取处理结果
            # result = algorithm.recognize(data)

            result = target + 1

            # 实时反馈最终预测结果
            para.show_result(result)

            # 刺激程序端，在线数据收集
            # x.append(data)
            y.append(target)

            # neuracle.clear_buffer()
        # trigger_manage.send_trigger(102)
    
    # trigger_manage.close_trigger()
    # neuracle.stop()
    
    print(np.array(x).shape, np.array(y).shape)
    np.save("x.npy", np.array(x))
    np.save("y.npy", np.array(y))

        