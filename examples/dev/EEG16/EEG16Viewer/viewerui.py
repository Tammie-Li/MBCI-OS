from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import pyqtSignal

class MyComboBox(QtWidgets.QComboBox):
    clicked = pyqtSignal()
    def showPopup(self):
        self.clicked.emit()
        super(MyComboBox, self).showPopup()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1500, 999)
        MainWindow.setStyleSheet("background-color: rgb(40, 44, 52);")

        # 动态设置字体
        font = QtGui.QFont()
        font.setPointSize(10)  # 基础字体大小
        MainWindow.setFont(font)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # 创建主布局
        self.mainLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.mainLayout.setContentsMargins(10, 10, 10, 10)
        self.mainLayout.setSpacing(10)

        # 顶部菜单栏
        self.topMenuBar = QtWidgets.QFrame(self.centralwidget)
        self.topMenuBar.setStyleSheet("background-color: rgb(58, 64, 73); border-radius: 10px;")
        self.topMenuBar.setObjectName("topMenuBar")
        self.topMenuLayout = QtWidgets.QHBoxLayout(self.topMenuBar)
        self.topMenuLayout.setContentsMargins(10, 10, 10, 10)
        self.topMenuLayout.setSpacing(10)

        # 设备选择复选框
        self.device_cmb = MyComboBox(self.topMenuBar)
        # self.device_cmb = QtWidgets.QComboBox(self.topMenuBar)
        self.device_cmb.setFixedSize(110,30)
        self.device_cmb.setStyleSheet(
            "background-color: rgb(50, 55, 62); color: white; border-radius: 5px; font-size: 1.2em;")
        # self.device_cmb.addItem("Device 15")
        # self.device_cmb.addItem("Device 2")
        # self.device_cmb.addItem("Device 3")
        self.topMenuLayout.addWidget(self.device_cmb)

        # 开始采集按钮
        self.startacq_btn = QtWidgets.QPushButton("开始采集", self.topMenuBar)
        self.startacq_btn.setFixedSize(100,30)
        self.startacq_btn.setStyleSheet("""
        QPushButton {
            background-color: #409eff;
            color: white;
            border-radius: 8px;
            font-size: 1.6em;

        }
        QPushButton:hover {
            background-color: #66b1ff;
        }
        QPushButton:pressed {
            background-color: #3a8ee6;
        }
        """)
        # self.startacq_btn.clicked.connect(self.toggleStartAcquisition)
        self.topMenuLayout.addWidget(self.startacq_btn)

        # 阻抗检测按钮
        self.imp_btn = QtWidgets.QPushButton("阻抗检测", self.topMenuBar)
        self.imp_btn.setFixedSize(100, 30)
        self.imp_btn.setStyleSheet("""
        QPushButton {
            background-color: #67c23a; 
            color: white; 
            border-radius: 8px;
            font-size: 1.2em;
        
        }
        QPushButton:hover {
            background-color: #85ce61;
        }
        QPushButton:pressed {
            background-color: #5daf34;
        }
        """)
        self.topMenuLayout.addWidget(self.imp_btn)

        # 停止检测按钮
        self.stop_btn = QtWidgets.QPushButton("停止", self.topMenuBar)
        self.stop_btn.setFixedSize(100, 30)
        self.stop_btn.setStyleSheet("""
                QPushButton {
                    background-color: #b10303; 
                    color: white; 
                    border-radius: 8px;
                    font-size: 1.2em;

                }
                QPushButton:hover {
                    background-color: #ff7875;
                }
                QPushButton:pressed {
                    background-color: #d9363e;
                }
                """)

        self.topMenuLayout.addWidget(self.stop_btn)

        # X轴（秒）的 SpinBox
        self.label_xrange = QtWidgets.QLabel("X轴(秒)：", self.topMenuBar)
        self.label_xrange.setFixedSize(100,30)
        self.label_xrange.setStyleSheet("color: white; font-size: 1.2em;")
        self.topMenuLayout.addWidget(self.label_xrange)

        self.xrange_sbx = QtWidgets.QSpinBox(self.topMenuBar)
        self.xrange_sbx.setFixedSize(100,30)
        self.xrange_sbx.setStyleSheet(
            "background-color: rgb(50, 55, 62); color: white; border-radius: 5px; font-size: 1.2em;")
        self.xrange_sbx.setMinimum(4)
        self.xrange_sbx.setMaximum(60)
        self.xrange_sbx.setSingleStep(2)
        self.xrange_sbx.setValue(4)
        self.topMenuLayout.addWidget(self.xrange_sbx)

        # Y轴的 ComboBox
        self.label_yrange = QtWidgets.QLabel("Y轴：", self.topMenuBar)
        self.label_yrange.setFixedSize(60,30)
        self.label_yrange.setStyleSheet("color: white; font-size: 1.2em;")
        self.topMenuLayout.addWidget(self.label_yrange)

        self.yrange_cmb = QtWidgets.QComboBox(self.topMenuBar)
        self.yrange_cmb.setFixedSize(100,30)
        self.yrange_cmb.setStyleSheet(
            "background-color: rgb(50, 55, 62); color: white; border-radius: 5px; font-size: 1.2em;")
        self.yrange_cmb.addItem("20uV")
        self.yrange_cmb.addItem("50uV")
        self.yrange_cmb.addItem("100uV")
        self.yrange_cmb.addItem("200uV")
        self.yrange_cmb.addItem("500uV")
        self.yrange_cmb.addItem("2mV")
        self.yrange_cmb.addItem("10mV")
        self.yrange_cmb.addItem("100mV")
        self.yrange_cmb.addItem("1V")
        self.yrange_cmb.addItem("5V")
        self.yrange_cmb.addItem("Auto")
        self.yrange_cmb.setCurrentIndex(4)
        self.topMenuLayout.addWidget(self.yrange_cmb)

        # # 滤波的 ComboBox
        # self.label_filter = QtWidgets.QLabel("滤波：", self.topMenuBar)
        # self.label_filter.setFixedSize(60,30)
        # self.label_filter.setStyleSheet("color: white; font-size: 1.2em;")
        # self.topMenuLayout.addWidget(self.label_filter)
        #
        # self.flt_cmb = QtWidgets.QComboBox(self.topMenuBar)
        # self.flt_cmb.setFixedSize(100,30)
        # self.flt_cmb.setStyleSheet(
        #     "background-color: rgb(50, 55, 62); color: white; border-radius: 5px; font-size: 1.2em;")
        # self.flt_cmb.addItem("None")
        # self.flt_cmb.addItem("> 1Hz")
        # self.flt_cmb.addItem("1-45Hz")
        # self.topMenuLayout.addWidget(self.flt_cmb)

        # 路径的 LineEdit
        self.label_path = QtWidgets.QLabel("路径：", self.topMenuBar)
        self.label_path.setFixedSize(70,30)
        self.label_path.setStyleSheet("color: white; font-size: 1.2em;")
        self.topMenuLayout.addWidget(self.label_path)

        self.path_edit = QtWidgets.QLineEdit(self.topMenuBar)
        self.path_edit.setFixedSize(200,30)
        self.path_edit.setStyleSheet(
            "background-color: rgb(50, 55, 62); color: white; border-radius: 5px; font-size: 1.2em;")
        self.path_edit.setText("./data")
        self.topMenuLayout.addWidget(self.path_edit)

        # 保存数据按钮
        self.save_btn = QtWidgets.QPushButton("保存数据", self.topMenuBar)
        self.save_btn.setFixedSize(100, 30)
        self.save_btn.setStyleSheet("""
        QPushButton {
            background-color: #e6a23c; 
            color: white; 
            border-radius: 8px;
            font-size: 1.2em;
    
        }
        QPushButton:hover {
            background-color: #f0c567;
        }
        QPushButton:pressed {
            background-color: #d89c1b;
        }
        """)
        self.save_btn.clicked.connect(self.toggleSaveData)
        self.topMenuLayout.addWidget(self.save_btn)

        # 添加一个弹簧，使得后续控件靠右对齐
        spacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.topMenuLayout.addItem(spacer)

        # # 设备电量的 ProgressBar
        # self.label_battery = QtWidgets.QLabel("设备电量", self.topMenuBar)
        # self.label_battery.setFixedSize(70, 30)
        # self.label_battery.setStyleSheet("color: white; font-size: 1.2em;")
        # self.topMenuLayout.addWidget(self.label_battery)
        #
        # self.batLevel = QtWidgets.QProgressBar(self.topMenuBar)
        # self.batLevel.setFixedWidth(100)  # 设置固定宽度
        # self.batLevel.setFixedHeight(25)  # 设置固定高度
        # self.batLevel.setAlignment(QtCore.Qt.AlignCenter)  # 文本垂直居中
        # self.batLevel.setStyleSheet(
        #     "background-color: rgb(50, 55, 62); color: white; border-radius: 5px; font-size: 1em;")
        # self.batLevel.setMinimum(0)
        # self.batLevel.setMaximum(100)
        # self.batLevel.setValue(75)
        # self.topMenuLayout.addWidget(self.batLevel)

        # 将顶部菜单栏添加到主布局
        self.mainLayout.addWidget(self.topMenuBar)

        # 中间显示区
        self.displayArea = QtWidgets.QFrame(self.centralwidget)
        self.displayArea.setStyleSheet("background-color: rgb(50, 55, 62); border-radius: 10px;")
        self.displayArea.setObjectName("displayArea")
        self.displayLayout = QtWidgets.QVBoxLayout(self.displayArea)
        self.displayLayout.setContentsMargins(10, 10, 10, 10)
        self.mainLayout.addWidget(self.displayArea, stretch=1)  # 设置中间区域自动扩展

        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

    # def toggleStartAcquisition(self):
    #     if self.startacq_btn.styleSheet().find("background-color: #409eff;") != -1:
    #         self.startacq_btn.setStyleSheet("""
    #         QPushButton {
    #             background-color: #b10303;
    #             color: white;
    #             border-radius: 8px;
    #             font-size: 1em;
    #
    #         }
    #         QPushButton:hover {
    #             background-color: #ff7875;
    #         }
    #         QPushButton:pressed {
    #             background-color: #d9363e;
    #         }
    #         """)
    #     else:
    #         self.startacq_btn.setStyleSheet("""
    #         QPushButton {
    #             background-color: #409eff;
    #             color: white;
    #             border-radius: 8px;
    #             font-size: 1em;
    #
    #         }
    #         QPushButton:hover {
    #             background-color: #66b1ff;
    #         }
    #         QPushButton:pressed {
    #             background-color: #3a8ee6;
    #         }
    #         """)

    def toggleSaveData(self):
        if self.save_btn.styleSheet().find("background-color: #e6a23c;") != -1:
            self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #b10303; 
                color: white; 
                border-radius: 8px;
                font-size: 1.2em;
             
            }
            QPushButton:hover {
                background-color: #ff7875;
            }
            QPushButton:pressed {
                background-color: #d9363e;
            }
            """)
        else:
            self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #e6a23c; 
                color: white; 
                border-radius: 8px;
                font-size: 1.2em;
   
            }
            QPushButton:hover {
                background-color: #f0c567;
            }
            QPushButton:pressed {
                background-color: #d89c1b;
            }
            """)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    
    
    sys.exit(app.exec_())

'''
/* QMessageBox 样式 */
QMessageBox {
    background-color: rgb(50, 55, 62);
    border: 1px solid #2979ff;
    border-radius: 10px;
}

QMessageBox QLabel {
    color: white;
    font-size: 1.2em;
    padding: 10px;
}

QMessageBox QPushButton {
    background-color: #409eff;
    color: white;
    border: 1px solid #2979ff;
    border-radius: 5px;
    padding: 10px;
    font-size: 1.2em;
    font-weight: bold;
    min-width: 80px;
}

QMessageBox QPushButton:hover {
    background-color: #66b1ff;
}

QMessageBox QPushButton:pressed {
    background-color: #3a8ee6;
    padding-left: 12px;
    padding-top: 12px;
}

/* QFileDialog 样式 */
QFileDialog {
    background-color: rgb(50, 55, 62);
    border: 1px solid #2979ff;
    border-radius: 10px;
}

QFileDialog QLabel {
    color: white;
    font-size: 1.2em;
    padding: 10px;
}

QFileDialog QPushButton {
    background-color: #409eff;
    color: white;
    border: 1px solid #2979ff;
    border-radius: 5px;
    padding: 10px;
    font-size: 1.2em;
    font-weight: bold;
    min-width: 80px;
}

QFileDialog QPushButton:hover {
    background-color: #66b1ff;
}

QFileDialog QPushButton:pressed {
    background-color: #3a8ee6;
    padding-left: 12px;
    padding-top: 12px;
}

QFileDialog QLineEdit {
    background-color: rgb(58, 64, 73);
    color: white;
    border: 1px solid #2979ff;
    border-radius: 5px;
    padding: 5px;
    font-size: 1.2em;
}

'''