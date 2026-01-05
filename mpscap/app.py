"""
桌面应用入口。
"""

from __future__ import annotations

import sys

from PyQt5 import QtCore, QtWidgets

from mpscap.ui.main_window import MainWindow


def main():
    # 高分屏自适应
    # QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    # QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    # try:
    #     # 避免缩放因子取整导致窗口超出屏幕（如150%缩放）
    #     QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
    #         QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    #     )
    # except Exception:
    #     pass
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

