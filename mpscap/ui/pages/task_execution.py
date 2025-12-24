from __future__ import annotations

from typing import Dict, List, Optional
from pathlib import Path

from PyQt5 import QtCore, QtWidgets, QtGui

from ...paradigms.ssvep_demo import run_ssvep_demo
from ...paradigms.gesture_demo import run_gesture_sequence


class TaskExecutionPage(QtWidgets.QWidget):
    """任务执行：上半部分范式采集/触发反馈，下半部分在线端到端演示。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._paradigm_status = QtWidgets.QLabel("未选择范式")
        self._trigger_status = QtWidgets.QLabel("当前标签（trigger）：无")
        self._model_status = QtWidgets.QLabel("未选择模型")
        self._feedback_status = QtWidgets.QLabel("反馈结果：暂无")
        self._trigger_com_edit = QtWidgets.QLineEdit()
        self._trigger_com_edit.setPlaceholderText("可选：触发串口，如 COM3")
        self._selected_paradigm: Optional[str] = None
        self._selected_paradigm_params: Dict[str, str] = {}
        self._selected_model: Optional[str] = None
        self._gesture_files: List[Path] = []
        self._progress = QtWidgets.QProgressBar()
        self._progress.setVisible(False)
        self._init_ui()

    def _init_ui(self) -> None:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(self._build_paradigm_panel())
        splitter.addWidget(self._build_demo_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(splitter)

    def _build_paradigm_panel(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        title = QtWidgets.QLabel("数据采集（范式选择 & Trigger 反馈）")
        title.setStyleSheet("font-weight: bold;")
        vbox.addWidget(title)
        vbox.addWidget(self._paradigm_status)
        vbox.addWidget(self._trigger_status)

        self._paradigm_group = QtWidgets.QButtonGroup(self)
        self._paradigm_group.setExclusive(True)

        cards = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(cards)
        grid.setSpacing(12)

        paradigms: List[Dict] = [
            {
                "name": "SSVEP",
                "desc": "稳态视觉诱发电位，多频闪烁刺激。",
                "params": [
                    ("刺激频率(Hz)", "15, 12, 10"),
                    ("单轮时长(s)", "5"),
                    ("重复次数", "10"),
                ],
            },
            {
                "name": "RSVP",
                "desc": "快速序列视觉呈现，适用于目标检测。",
                "params": [
                    ("呈现速率(Hz)", "8"),
                    ("目标比例(%)", "10"),
                    ("轮数", "5"),
                ],
            },
            {
                "name": "手势识别",
                "desc": "基于肌电的多手势范式（按配置依次呈现图片+提示词）。",
                "params": [
                    ("手势名称列表(逗号分隔，对应6张图片)", "拳,张,捏,勾,指,摆"),
                    ("每张图片显示时长(s)", "1.0"),
                    ("图片间休息时长(s)", "3.0"),
                    ("每轮间休息时长(s)", "5.0"),
                    ("循环次数", "2"),
                ],
            },
        ]

        for idx, paradigm in enumerate(paradigms):
            card, radio = self._create_paradigm_card(paradigm)
            self._paradigm_group.addButton(radio)
            r = idx // 2
            c = idx % 2
            grid.addWidget(card, r, c)

        vbox.addWidget(cards)
        vbox.addStretch()
        return container

    def _create_paradigm_card(self, paradigm: Dict) -> (QtWidgets.QGroupBox, QtWidgets.QRadioButton):
        box = QtWidgets.QGroupBox(paradigm["name"])
        v = QtWidgets.QVBoxLayout(box)
        title_row = QtWidgets.QHBoxLayout()
        radio = QtWidgets.QRadioButton("选择")
        title_row.addWidget(radio)
        title_row.addWidget(QtWidgets.QLabel(paradigm["desc"]))
        title_row.addStretch()
        v.addLayout(title_row)
        form = QtWidgets.QFormLayout()
        edits = {}
        for label, default in paradigm.get("params", []):
            edit = QtWidgets.QLineEdit()
            edit.setText(str(default))
            form.addRow(label, edit)
            edits[label] = edit
        v.addLayout(form)
        apply_btn = QtWidgets.QPushButton("应用配置")
        apply_btn.clicked.connect(lambda _=None, name=paradigm["name"], edits=edits, r=radio, box=box: self._on_paradigm_selected(name, edits, r, box))
        v.addWidget(apply_btn)
        if paradigm["name"] == "手势识别":
            img_btn = QtWidgets.QPushButton("选择手势图片（可多选）")
            img_btn.clicked.connect(self._choose_gesture_files)
            v.addWidget(img_btn)
        return box, radio

    def _on_paradigm_selected(self, name: str, edits: Dict[str, QtWidgets.QLineEdit], radio: QtWidgets.QRadioButton, box: QtWidgets.QGroupBox) -> None:
        params = {k: e.text().strip() for k, e in edits.items()}
        self._paradigm_status.setText(f"当前范式：{name}，参数：{params}")
        # 将范式名称作为 trigger 标签提示
        self._trigger_status.setText(f"当前标签（trigger）：{name}")
        self._selected_paradigm = name
        self._selected_paradigm_params = params
        radio.setChecked(True)
        # 高亮当前卡片，其他恢复默认
        parent = box.parentWidget()
        if parent:
            for gb in parent.findChildren(QtWidgets.QGroupBox):
                if gb is box:
                    gb.setStyleSheet("QGroupBox { border: 2px solid rgb(0,150,0); } QPushButton { background-color: rgb(0,150,0); color: white; }")
                else:
                    gb.setStyleSheet("")

    def _build_demo_panel(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        title = QtWidgets.QLabel("在线端到端演示（模型输出作为反馈）")
        title.setStyleSheet("font-weight: bold;")
        vbox.addWidget(title)

        model_row = QtWidgets.QHBoxLayout()
        model_row.addWidget(QtWidgets.QLabel("选择模型:"))
        self._model_combo = QtWidgets.QComboBox()
        self._model_combo.addItems(["SSVEP-ResNet", "RSVP-Transformer", "Gesture-CNN"])
        model_row.addWidget(self._model_combo)
        apply_btn = QtWidgets.QPushButton("加载模型")
        apply_btn.clicked.connect(self._on_model_selected)
        model_row.addWidget(apply_btn)
        model_row.addStretch()
        vbox.addLayout(model_row)

        # 启动范式：根据上方选择与配置决定
        run_present_btn = QtWidgets.QPushButton("启动范式（有模型用模型输出，否则用真实标签）")
        run_present_btn.clicked.connect(self._on_run_presentation)
        vbox.addWidget(self._trigger_com_edit)
        vbox.addWidget(run_present_btn)
        vbox.addWidget(self._progress)

        vbox.addWidget(self._model_status)
        vbox.addWidget(self._feedback_status)
        vbox.addStretch()
        return container

    def _on_model_selected(self) -> None:
        name = self._model_combo.currentText()
        self._model_status.setText(f"当前模型：{name}")
        # 以模型名作为示例反馈输出
        self._feedback_status.setText(f"反馈结果：已加载模型 {name}，等待呈现")
        self._selected_model = name

    def _on_run_presentation(self) -> None:
        """根据已选范式执行呈现；有模型则用模型输出，否则用真实标签。"""
        if not self._selected_paradigm:
            QtWidgets.QMessageBox.information(self, "提示", "请先在上方选择并应用一个范式。")
            return
        paradigm = self._selected_paradigm
        params = self._selected_paradigm_params or {}
        try:
            if paradigm == "SSVEP":
                # 仅启动演示占位；可对接真实呈现
                run_ssvep_demo(trigger_com=self._trigger_com_edit.text().strip() or None)
            elif paradigm == "手势识别":
                names_raw = params.get("手势名称列表(逗号分隔，对应6张图片)", "")
                names = [n.strip() for n in names_raw.split(",") if n.strip()]
                if not names:
                    names = ["手势1", "手势2", "手势3", "手势4", "手势5", "手势6"]
                show_sec = float(params.get("每张图片显示时长(s)", "1.0") or 1.0)
                rest_gesture = float(params.get("图片间休息时长(s)", "3.0") or 3.0)
                rest_cycle = float(params.get("每轮间休息时长(s)", "5.0") or 5.0)
                cycles = int(params.get("循环次数", "1") or 1)
                if self._gesture_files:
                    imgs = self._gesture_files
                else:
                    img_dir = Path(__file__).resolve().parents[3] / "mpscap" / "frame" / "image"
                    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
                # 校验名称数量与图片数量一致
                if len(names) != len(imgs):
                    QtWidgets.QMessageBox.warning(self, "配置不一致", f"手势名称数量({len(names)})需与图片数量({len(imgs)})一致。")
                    return
                # 多线程运行，避免阻塞 UI
                from threading import Thread

                def run_task():
                    run_gesture_sequence(
                        imgs=imgs,
                        names=names,
                        show_sec=show_sec,
                        rest_between_gestures=rest_gesture,
                        rest_between_cycles=rest_cycle,
                        cycles=cycles,
                        progress_cb=None,
                    )

                Thread(target=run_task, daemon=True).start()
            # 反馈显示
            if self._selected_model:
                self._feedback_status.setText(f"反馈结果：模型 {self._selected_model} 对范式 {paradigm} 的输出（示例）")
            else:
                self._feedback_status.setText(f"反馈结果：使用范式真实标签 -> {paradigm}")
            self._paradigm_status.setText(f"当前范式：{paradigm}，参数：{params}（已呈现）")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "运行失败", f"范式运行失败：{e}")
        finally:
            self._progress.setVisible(False)

    def _update_progress(self, cur: int, tot: int) -> None:
        try:
            self._progress.setRange(0, tot)
            self._progress.setValue(cur)
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass

    def _choose_gesture_files(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "选择手势图片（可多选）",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )
        if files:
            self._gesture_files = [Path(f) for f in files]


