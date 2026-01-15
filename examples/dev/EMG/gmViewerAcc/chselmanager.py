import sys
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QCheckBox,QVBoxLayout, QGroupBox,QHBoxLayout
from PyQt5.QtCore import Qt
from math import ceil
from PyQt5.QtCore import pyqtSignal

'''用于管理选择绘图通道弹窗'''

class chselManager(QWidget):
    csig = pyqtSignal(str)
    def __init__(self,parenUI):
        super(chselManager, self).__init__()
        self.ui = parenUI
        self.setWindowTitle('勾选绘图通道')
        self.sigQCboxs = []
        self.testchQCboxs = []
        self.selectedchs = {'sigchs': [], 'testch': []}
        self.ui.chssel_btn.clicked.connect(self.show)  # ui按钮按键弹出本窗口

    def reset(self,info):
        self.info = info
        # 默认都选上
        self.selectedchs = {'sigchs': [], 'testch': []}
        for i in range(self.info['sigchs']):
            self.selectedchs['sigchs'].append(i)
        if self.info['testch'] > 0:
            self.selectedchs['testch'] = [0]

        # 清空界面上所有东西
        try:
            for i in range(self.layout().count()):
                self.layout().itemAt(i).widget().deleteLater()
        except:
            pass

        self.initUI()

    def closeEvent(self, evt):
        self.selectedchs['sigchs'] = []
        self.selectedchs['testch'] = []

        for id,cmb in enumerate(self.sigQCboxs):
            if cmb.isChecked():
                self.selectedchs['sigchs'].append(id)

        for id, cmb in enumerate(self.testchQCboxs):
            if cmb.isChecked():
                self.selectedchs['testch'].append(id)

        self.csig.emit('->')

    def initUI(self):
        gbox0 = QGroupBox("SigChs")
        gbox0.setStyleSheet("font: 24px \"times new roman\";")
        vbox = QVBoxLayout()
        grid = QGridLayout()
        signum = self.info['sigchs']
        row = ceil(signum/8)
        ch = 0
        for r in range(row):
            for c in range(8):
                if ch < signum:
                    self.sigQCboxs.append(QCheckBox('ch'+str(ch)))
                    self.sigQCboxs[-1].setStyleSheet("font: 24px \"times new roman\";")
                    self.sigQCboxs[-1].setMinimumWidth(100)
                    self.sigQCboxs[-1].setMinimumHeight(50)
                    self.sigQCboxs[-1].setChecked(True)
                    grid.addWidget(self.sigQCboxs[-1],r,c)
                    ch += 1

        gbox0.setLayout(grid)
        vbox.addWidget(gbox0)

        if self.info['testch'] != 0:
            gbox1 = QGroupBox("TestCh")
            gbox1.setStyleSheet("font: 24px \"times new roman\";")
            hbox = QHBoxLayout()

            QcbTest = QCheckBox('testCh')
            QcbTest.setStyleSheet("font: 24px \"times new roman\";")
            QcbTest.setMinimumWidth(100)
            QcbTest.setMinimumHeight(50)
            QcbTest.setChecked(True)
            hbox.addWidget(QcbTest)
            self.testchQCboxs.append(QcbTest)

            gbox1.setLayout(hbox)
            vbox.addWidget(gbox1)

        self.setLayout(vbox)
        self.setMinimumWidth(600)
        self.setMinimumHeight(150)
        self.setWindowFlags(Qt.WindowCloseButtonHint)

    def resizeEvent(self,evt):
        self.setFixedSize(self.width(),self.height())


if __name__ == "__main__":
    import time
    app = QApplication(sys.argv)
    # info = {'name':'GSR','srate':250,'sigchs':['GSR','ECG','ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8','ch9','ch10',],'testch':1,'format':'int24'}
    info = {'name': 'GSR', 'srate': 250,
            'sigchs': ['GSR', 'ECG'],
            'testch': 1, 'format': 'int24'}
    form = chselManager()
    form.reset(info)
    form.show()
    sys.exit(app.exec_())