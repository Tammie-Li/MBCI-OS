# coding:utf-8

from PyQt5 import QtWidgets,QtCore
from PyQt5.QtWidgets import QApplication,QMessageBox,QLabel
from sockampui import Ui_MainWindow
import time
from devManager import devManager

import paho.mqtt.client as mqtt
PORT = 1883

class DevClient(QtWidgets.QMainWindow):
    def __init__(self):
        super(DevClient, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("博瑞康接入终端")
        self.devMgr = devManager(self.ui)

    def closeEvent(self, event):
        pass


if __name__ == '__main__':
    import sys
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    a = DevClient()
    a.show()
    sys.exit(app.exec_())