from __future__ import annotations

import time
from pathlib import Path
from typing import List

from PyQt5 import QtWidgets, QtGui, QtCore


def run_gesture_sequence(
    imgs: List[Path],
    names: List[str],
    show_sec: float,
    rest_between_gestures: float,
    rest_between_cycles: float,
    cycles: int,
    progress_cb=None,
) -> None:
    """
    手势范式：按 names 顺序呈现图片（文件夹下按顺序取，与 names 对应），每张显示 show_sec 秒，间隔 rest_sec 秒，循环 cycles 轮。
    在顶部显示黑色文字提示（手势名称），图片居中，含 3->1 倒计时。
    """
    imgs = [p for p in imgs if p.exists() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    if not imgs:
        QtWidgets.QMessageBox.information(None, "未找到图片", "请选择至少一张 png/jpg 图片")
        return
    # 按 names 长度截取或重复，若不足 6 张也重复最后一张
    seq: List[Path] = []
    for i, _ in enumerate(names):
        if i < len(imgs):
            seq.append(imgs[i])
        else:
            seq.append(imgs[-1])

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dlg = QtWidgets.QDialog()
    dlg.setWindowTitle("手势范式呈现")
    dlg.resize(1920, 1080)
    layout = QtWidgets.QVBoxLayout(dlg)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    cycle_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
    cycle_label.setStyleSheet("font-size: 28px; color: black;")
    title = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
    title.setStyleSheet("font-size: 36px; color: black;")
    progress = QtWidgets.QProgressBar()
    progress.setAlignment(QtCore.Qt.AlignCenter)
    pic = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
    layout.addWidget(cycle_label, stretch=1)
    layout.addWidget(title, stretch=1)
    layout.addWidget(pic, stretch=5)
    layout.addWidget(progress, stretch=0)

    total_steps = cycles * len(names)
    step = 0

    for c_idx in range(cycles):
        cycle_label.setText(f"第 {c_idx + 1} / {cycles} 轮")
        for idx, name in enumerate(names):
            title.setText(name)
            p = seq[idx]
            pix = QtGui.QPixmap(str(p))
            if not pix.isNull():
                pix = pix.scaledToHeight(600, QtCore.Qt.SmoothTransformation)
                pic.setPixmap(pix)
            dlg.repaint()
            QtWidgets.QApplication.processEvents()
            # 倒计时 3->1
            for t in [3, 2, 1]:
                title.setText(f"{name} （{t}）")
                dlg.repaint()
                QtWidgets.QApplication.processEvents()
                time.sleep(1)
            # 显示
            title.setText(name)
            dlg.repaint()
            QtWidgets.QApplication.processEvents()
            steps = max(1, int(show_sec * 10))
            progress.setRange(0, steps)
            for s in range(steps):
                progress.setValue(s + 1)
                QtWidgets.QApplication.processEvents()
                time.sleep(show_sec / steps)
            # 休息
            title.setText("休息")
            pic.clear()
            progress.setValue(0)
            dlg.repaint()
            QtWidgets.QApplication.processEvents()
            time.sleep(max(0.0, rest_between_gestures))
            step += 1
            if progress_cb:
                try:
                    progress_cb(step, total_steps)
                except Exception:
                    pass
        # 轮间休息
        if rest_between_cycles > 0:
            title.setText("轮间休息")
            pic.clear()
            progress.setValue(0)
            dlg.repaint()
            QtWidgets.QApplication.processEvents()
            time.sleep(rest_between_cycles)
    dlg.close()

