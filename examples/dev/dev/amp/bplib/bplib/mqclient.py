# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# bymrtang: 2024/11
# bcios project

import paho.mqtt.client as mqtt
import time

class mqClient():
    def __init__(self,param,ctrprm):
        self.mqAddr = param['mqAddr']
        self.mqPort = param['mqPort']
        self.devName = param['devName']
        self.topic = param['topic']
        self.stopEv = ctrprm['stopEv']
        self.backQue = ctrprm['backQue']
        self.mqconnect = None

    def pub(self,msg,q=0):
        self.client.publish(self.topic,msg,q)

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()

    def start(self):
        self.createMqttClient()

        for i in range(100):
            if self.mqconnect != None:
                if self.mqconnect == 0:
                    print('mqtt连接成功')
                    self.backQue.put('mqtt连接到服务器成功！')
                else:
                    print('mqtt连接失败')
                    self.backQue.put('mqtt连接到服务器失败！')
                break
            time.sleep(0.1)

        if self.mqconnect == None or self.mqconnect != 0:
            return False
        else:
            return True

    #创建mqtt client，并连接到服务器
    def createMqttClient(self):
        # 利用时间戳产生唯一的节点名称
        timestampStr = str(int(time.time()*100))
        nodename = self.devName + '-' + timestampStr
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,nodename)
        self.client.on_connect = self.on_connectMq
        self.client.connect(self.mqAddr,self.mqPort,60)
        self.client.loop_start()

    #mqtt连接回调
    def on_connectMq(self,client,userdata,flags,rc,properties):
        self.mqconnect = rc
        # if rc != 0:
        #     # self.backQue.put('mqtt连接服务器失败！')
        #     print('mq 失败')
        # else:
        #     print('mq 成功')