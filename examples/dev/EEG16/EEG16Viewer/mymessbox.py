#!/usr/bin/env python
# -*- coding: utf-8 -*-


from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QFont

def showMessageBox(title,text):
    msg_box = QMessageBox()
    msg_box.setWindowTitle(title)
    msg_box.setText(text)
    msg_box.setIcon(QMessageBox.Warning)
    # 设置消息框的字体
    msg_box.setFont(QFont('SansSerif', 10))

    # # 应用SSH风格样式表
    # msg_box.setStyleSheet("""
    #     QMessageBox {
    #         color: rgb(220,220,220);
    #         background-color: rgb(85, 85, 85);
    #         border: none;
    #     }
    #     QMessageBox QLabel {
    #         color: rgb(220,220,220);
    #     }
    #     QMessageBox QPushButton {
    #         color: rgb(20,230,20);
    #         background-color: rgb(85, 85, 85);
    #         border: 1px solid rgb(50, 50, 50);
    #         padding: 5px;
    #     }
    #     QMessageBox QPushButton:hover {
    #         background-color: rgb(100, 100, 100);
    #     }
    #     QMessageBox QPushButton:pressed {
    #         background-color: rgb(50, 50, 50);
    #     }
    # """)

    msg_box.setStyleSheet("""
        QMessageBox {
            background-color: rgb(50, 55, 62);
            border: 1px solid #2979ff;
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
    """)
    # 显示消息框并返回用户的选择
    msg_box.exec_()
    return True
