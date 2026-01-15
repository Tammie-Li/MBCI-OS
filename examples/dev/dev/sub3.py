# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import socket
from struct import unpack
import numpy as np
import paho.mqtt.client as mqtt
import time
from multiprocessing import Event,Queue

class DataSub():
    def __init__(self,param):
        self.mqaddr = param['mqaddr']
        self.mqport = param['mqport']
        self.topicname = param['topic']

        # 利用时间戳产生唯一的节点名称
        timestampStr = str(int(time.time() * 100))
        nodename = 'User' + '-' + timestampStr
        self.mqClient = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, nodename)
        self.mqClient.on_connect = self.on_connectMq
        self.mqClient.on_message = self.on_messageMq
        self.mqClient.on_subscribe = self.on_subscribeMq
        self.mqClient.connect(self.mqaddr, self.mqport, 60)
        self.mqClient.loop_forever()

    #mqttl连接回调,连接成功则订阅主题
    def on_connectMq(self,client,userdata,flags,rc,pro):
        if rc == 0:
            print('mqtt服务器连接成功')
            self.mqClient.subscribe(self.topicname,0)

    def on_subscribeMq(self,client,userdata,mid,qos,pro):
        print('话题订阅成功！')

    def on_messageMq(self,client,userdata,msg):
        print(msg.topic)
        print(len(msg.payload))



if __name__ == '__main__':
    param = {'mqaddr':'127.0.0.1','mqport':1883,'topic':'/bcios/result'}
    ds = DataSub(param)
    ds.start()