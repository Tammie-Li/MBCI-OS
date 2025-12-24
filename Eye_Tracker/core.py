#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/20 23:54
# @Version : 1.0
# @File    : py4c.py
# @Author  : Jingsheng Tang
# @Version : 1.0
# @Contact : mrtang@nudt.edu.cn   mrtang_cs@163.com
# @License : (C) All Rights Reserved

import socket
import threading
import struct
import time
import win32api,win32con
import numpy as np
import pygame

import win32com.client
import os

def check_exsit(process_name):
    '''
    check if a process is exist
    '''
    WMI = win32com.client.GetObject('winmgmts:')
    processCodeCov = WMI.ExecQuery('select * from Win32_Process where Name="%s"' % process_name)
    if len(processCodeCov) > 0:return 1
    else:return 0

def kill_process(process_name):
    '''
    kill a process by name
    '''
    if os.system('taskkill /f /im ' + process_name)==0:return 1
    else:return 0

    
class TobiiPy(threading.Thread):
    def __init__(self,filter = 3,userport = 5150,userip = '127.0.0.1',tobii4Cport=5151,tobii4Cip='127.0.0.1'):
        threading.Thread.__init__(self)

        dirpath,_ = os.path.split(os.path.abspath(__file__))
        self.useraddr = (userip,userport)
        self.tobii4caddr = (tobii4Cip,tobii4Cport)

        self.SCRW = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        self.SCRH = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)

        self.filter = filter
        self._xl = [0]*filter
        self._yl = [0]*filter

        self._lock = threading.Lock()

        self._gazexy = [-1,-1]
        self._q = False

        app = 'tobii_4c_app.exe'
        if check_exsit(app):
            kill_process(app)
            time.sleep(0.1)

        #启动服务，不显示黑框
        win32api.ShellExecute(0, 'open', os.path.join(dirpath,'data_files','tobii_4c_app.exe'), '%i %s %i %s'%(userport,userip,tobii4Cport,tobii4Cip), '', 0)

        self.setDaemon(True)
        self.start()

    def quit(self):
        self._q = True
        for i in range(10): #等待子线程结束，未等到也强制退出
            if not self._q:
                break
            time.sleep(0.5)

    @property
    def gazeXY(self):
        return self._gazexy

    @property
    def gazepos(self):
        '''
        get gaze position on the screen
        '''
        x,y = self.gazeXY
        x = int(x * self.SCRW)
        y = int(y * self.SCRH)
        return x,y

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(self.useraddr)

        while True:
            buf, addr = sock.recvfrom(128)
            x, y = struct.unpack('2f', buf)
            self._xl.append(x)
            self._xl.pop(0)
            self._yl.append(y)
            self._yl.pop(0)

            self._lock.acquire()
            self._gazexy = (sum(self._xl)/self.filter,sum(self._yl)/self.filter)
            self._lock.release()
            if self._q:
                sock.sendto(b'\x01\x03\x7d\x7f', self.tobii4caddr)
                break
        self._q = False

class ROIDetector():
    '''
    ROIDetector目的是设定一个rect区域，通过update函数传入当前的点来返回该点是否在
    定义的rect区域内，以及在该区域内超过设定的时间。
    初始化参数：
    size: rect的尺寸，（w,h）
    adjust: rect尺寸调整(wf,hf), 在size的基础上对长宽进行再次调整，最后定义的rect尺寸为（w*wf, h*hf）
    position: rect的位置，默认为中心定位
    t: 时间阈值，即update传入的点落在rect区域内超过时间t则返回True(gazelong)

    方法：
    isin, gazelong = update(self,point)
    每次向update函数传入一个坐标点，返回的isin代表point在定义的rect区域中
    返回的gazelong代表point落在定义的rect区域中超过时间t
    '''
    def __init__(self, size, adjust, position,t):
        w,h = size
        wf,hf = adjust
        ss = (int(w*wf),int(h*hf))
        self.rect = pygame.rect.Rect((0, 0), ss)
        self.rect.center = position
        self.focus = False
        self.t = t

        self.isin = False
        self.stamp = float('inf')

    def setDetectTimeRange(self,r = 1): #1秒
        self.t = r

    def update(self, point):
        isin = self.rect.collidepoint(point)

        if not isin:
            self.stamp = float('inf')

        if (not self.isin) and isin: #新获得注视
            self.stamp = time.time()

        if time.time() - self.stamp > self.t:
            gazelong = True
        else:
            gazelong = False

        self.isin = isin
        return isin,gazelong

class GazeDetector():
    '''
    该类用于判断注视。
    参数有：
        radius: 即判断注视点是否收敛到该半径之内
        secs: 注视点持续收敛到该半径内的时间
        step: 调用update的时间间隔，即更新注视点的时间间隔
    方法：
        setGazeTime：重新设置注视时间的阈值
        update：每获得一个新的注视点调用一次进行更新
        isgaze：返回在secs定义的时间内注视点是否收敛在radius之内
    '''
    def __init__(self,radius = 30, secs = 0.05, step = 0.05):
        self.radius = radius
        self.step = step
        self.setGazeTime(secs)

    def setGazeTime(self,secs=1.):
        self.t = secs
        self.N = int(self.t / self.step)
        self.points = np.zeros((self.N, 2))
        self.index = 0

    def update(self,point):
        # 缓存数据
        self.points[self.index,:] = point
        self.index += 1
        self.index %= self.N
        return self.isgaze()

    def isgaze(self):
        # 判断数据点是否收敛在指定半径范围内
        # 计算几何中心
        x,y = center = np.mean(self.points,axis=0)
        _ = self.points - center
        dis = np.linalg.norm(_,axis=1)
        mdis = np.max(dis)
        return mdis<self.radius,(int(x),int(y))
        
def demo():
    tb = TobiiPy()
    for i in range(100):
        print(tb.gazepos)
        time.sleep(0.1)

def demo2():
    '''
    运行后鼠标在窗口的位置停留超过设定时间并且收敛在半径内，则会绘制出这个圆
    '''
    screen = pygame.display.set_mode((400, 400), 1, 32)
    gd = GazeDetector(30,1,0.1)
    while True:
        screen.fill((0,0,0))
        pygame.event.clear()
        pose = pygame.mouse.get_pos()
        flg,center = gd.update(pose)
        if flg:
            pygame.draw.circle(screen,(255,0,0),center,gd.radius,2)
        pygame.display.update()
        time.sleep(0.1)

def demo3():
    '''
    鼠标停留在绘制的矩形框内，会反馈是否在框内，以及是否在框内停留超过设定的时间
    '''
    screen = pygame.display.set_mode((400, 400), 1, 32)
    rd = ROIDetector((100, 100), (1,1),(200, 200),2)

    while True:
        pygame.event.clear()
        pygame.draw.rect(screen, (255, 0, 0), rd.rect)
        pygame.display.update()
        pose = pygame.mouse.get_pos()
        isin, isinlong = rd.update(pose)
        print(isin, isinlong)
        time.sleep(0.1)

if __name__ == '__main__':
    demo3()




