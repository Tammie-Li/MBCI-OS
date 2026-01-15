#coding:utf-8

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
from viewerui import Ui_MainWindow
from devmanager import devManager
from shm2 import CreateShm
from PyQt5.QtCore import pyqtSignal

from impdisplay import ImpDisplay
from eegdisplay import EEGDisplay

import json
from mymessbox import showMessageBox

'''主程序入口'''

class gmViewer(QtWidgets.QMainWindow):
    mainsig = pyqtSignal(str)
    def __init__(self,configpath = './/config.js'):
        super(gmViewer,self).__init__()

        config,err = self.loadConfigs(configpath)
        if config is None:
            showMessageBox("提示",err)
            sys.exit()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self._screenResize()
        self.shm = CreateShm(master=True)
        self.devMgr = devManager(self.ui,self.mainsig,config)
        self.mainsig.connect(self.relayout)

        self.eegDis = EEGDisplay(self.ui,config)
        self.impDis = ImpDisplay(self.ui,config)

        self.relayout('stop')

    def loadConfigs(self,path):
        try:
            with open(path,'r') as f:
                buf = f.read()
        except:
            return None,"配置文件丢失！"

        try:
            config = json.loads(buf)
        except:
            return None,"配置文件不符合json规范！"

        if 'eegchs' not in config or 'srate' not in config:
            return None, "eegchs或srate参数缺失!"
        else:
            if config['eegchs']==[] or config['srate'] not in [250,500,1000]:
                return None,"参数非法！"

        return config,''


    def relayout(self,mess):
        if mess == 'acquireEEG':
            self.impDis.addToMainWin(False)
            self.impDis.startPloting(False)
            self.eegDis.addToMainWin(True)
            self.eegDis.startPloting(True)
        elif mess == 'impedanceDetect':
            self.eegDis.addToMainWin(False)
            self.eegDis.startPloting(False)
            self.impDis.addToMainWin(True)
            self.impDis.startPloting(True)
        else:
            self.eegDis.addToMainWin(False)
            self.eegDis.startPloting(False)
            self.impDis.addToMainWin(False)
            self.impDis.startPloting(False)

    def closeEvent(self, event):
        self.devMgr.release()
        self.eegDis.release()
        self.impDis.release()
        self.shm.release()

    # 由initialize调用
    # 依据显示屏尺寸来初始化界面大小
    def _screenResize(self):
        # 获取显示器相关信息
        desktop = QApplication.desktop()
        # 默认在主显示屏显示
        screen_rect = desktop.screenGeometry(0)
        self.ww = screen_rect.width()
        self.hh = screen_rect.height()
        self.w = int(self.ww*0.92)
        self.h = int(self.hh*0.8)
        self.setGeometry(int((self.ww - self.w)/2), int((self.hh - self.h)/2), self.w, self.h)
 
if __name__ == '__main__':
    import sys
    # import multiprocessing
    # multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    a = gmViewer()
    a.show()
    sys.exit(app.exec_())