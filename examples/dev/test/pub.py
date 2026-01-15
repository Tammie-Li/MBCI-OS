# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import socket
from struct import unpack
import numpy as np
import paho.mqtt.client as mqtt
import time
from multiprocessing import Event,Queue

class pub():
    def __init__(self,param):
        self.mqaddr = param['mqaddr']
        self.mqport = param['mqport']
        self.topicname = param['topic']

        # 利用时间戳产生唯一的节点名称
        timestampStr = str(int(time.time() * 100))
        nodename = 'User' + '-' + timestampStr
        self.mqClient = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, nodename)
        self.mqClient.on_connect = self.on_connectMq
        # self.mqClient.on_message = self.on_messageMq
        self.mqClient.connect(self.mqaddr, self.mqport, 60)

        self.mqClient.loop_start()

    # #mqttl连接回调,连接成功则订阅主题
    def on_connectMq(self,client,userdata,flags,rc,properties):
        if rc == 0:
            print('mqtt服务器连接成功')
            self.mqClient.subscribe(self.topicname,0)
    #
    # def on_messageMq(self,client,userdata,msg):
    #     print(msg.topic)
    #     print(len(msg.payload))

    def mainloop(self):
        while True:
            print('pub...')
            self.mqClient.publish('/bcios/eeg','hahahah',0)
            time.sleep(0.5)


if __name__ == '__main__':
    param = {'mqaddr':'127.0.0.1','mqport':1883,'topic':'/bcios/eeg'}
    ds = pub(param)
    ds.mainloop()