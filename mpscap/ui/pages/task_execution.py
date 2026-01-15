from __future__ import annotations

import multiprocessing as mp
from typing import Dict, List, Optional
from pathlib import Path

from PyQt5 import QtCore, QtWidgets, QtGui

from ...paradigms.paradigm_worker import (
    run_ssvep_worker,
    run_gesture_worker,
    run_rsvp_worker,
    run_eye_target_worker,
    run_eye_4class_worker,
)
from ...core.utils.shm import CreateShm


class TaskExecutionPage(QtWidgets.QWidget):
    """任务执行：上半部分范式采集/触发反馈，下半部分在线端到端演示。"""

    def __init__(self, acquisition_page=None, parent=None) -> None:
        super().__init__(parent)
        # 可选：引用在线采集页，用于自动开启数据保存
        self._acquisition_page = acquisition_page
        self._paradigm_status = QtWidgets.QLabel("未选择范式")
        self._trigger_status = QtWidgets.QLabel("当前标签（trigger）：无")
        self._model_status = QtWidgets.QLabel("未选择模型")
        self._feedback_status = QtWidgets.QLabel("反馈结果：暂无")
        self._trigger_com_edit = QtWidgets.QLineEdit()
        self._trigger_com_edit.setPlaceholderText("可选：触发串口，如 COM3")
        self._trigger_com_edit.setClearButtonEnabled(True)
        self._selected_paradigm: Optional[str] = None
        self._selected_paradigm_params: Dict[str, str] = {}
        self._selected_model: Optional[str] = None
        self._model_params: str = ""
        self._gesture_files: List[Path] = []
        self._progress = QtWidgets.QProgressBar()
        self._progress.setVisible(False)
        # 工作模式：采集模式 / 在线演示模式
        self._mode_combo = QtWidgets.QComboBox()
        self._mode_combo.addItems(["采集模式（仅采集+触发）", "在线演示模式（含模型反馈）"])
        # 反馈形式
        self._feedback_combo = QtWidgets.QComboBox()
        self._feedback_combo.addItems(["不反馈结果", "使用模型输出", "使用真实标签"])
        # 子进程上下文与句柄
        self._mp_ctx = mp.get_context("spawn")
        self._paradigm_proc: Optional[mp.Process] = None
        # Kafka 数据订阅配置（默认与采集端一致，可按需改成实际地址/Topic）
        self._kafka_bootstrap = "localhost:9092"
        self._kafka_topic = "emg_data"
        # 记忆最近保存目录
        self._last_save_dir: Optional[str] = None
        # 进程结束检测
        self._proc_timer = QtCore.QTimer(self)
        self._proc_timer.setInterval(500)
        self._proc_timer.timeout.connect(self._check_proc_done)
        self._running_paradigm: Optional[str] = None
        self._init_ui()
        self._apply_styles()

    def _init_ui(self) -> None:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(self._build_paradigm_panel())
        splitter.addWidget(self._build_demo_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(splitter)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget { font-family: "Microsoft YaHei"; font-size: 18px; }
            QLabel#sectionTitle { font-size: 22px; font-weight: bold; color: #2b2f36; }
            QLabel#sectionSub { color: #6b7280; font-size: 17px; }
            QLabel#statusBadge { background: #f3f4f6; color: #374151; border-radius: 6px; padding: 6px 8px; }
            QGroupBox#paradigmCard { border: 1px solid #e5e7eb; border-radius: 12px; margin-top: 8px; padding: 12px; background: #ffffff; }
            QGroupBox#paradigmCard:hover { border-color: #c7d2fe; }
            QGroupBox#paradigmCard QLabel { font-size: 20px; }
            QGroupBox#paradigmCard QLineEdit,
            QGroupBox#paradigmCard QComboBox,
            QGroupBox#paradigmCard QPushButton { font-size: 20px; }
            QGroupBox { border: 1px solid #e5e7eb; border-radius: 10px; margin-top: 8px; padding: 12px; background: #ffffff; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #111827; }
            QLineEdit, QComboBox { background: #ffffff; border: 1px solid #d1d5db; border-radius: 6px; padding: 6px 8px; color: #111827; }
            QPushButton { background: #f3f4f6; border: 1px solid #d1d5db; padding: 6px 12px; border-radius: 6px; }
            QPushButton:hover { background: #e5e7eb; }
            QPushButton:pressed { background: #e2e8f0; }
            QPushButton[primary="true"] { background: #2563eb; border-color: #2563eb; color: #ffffff; }
            QPushButton[toggle="true"] { background: #f87171; border-color: #ef4444; color: #ffffff; font-weight: bold; }
            QPushButton[toggle="true"]:checked { background: #22c55e; border-color: #16a34a; color: #ffffff; }
            QProgressBar { border: 1px solid #d1d5db; border-radius: 6px; text-align: center; background: #f9fafb; }
            QProgressBar::chunk { background: #2563eb; border-radius: 6px; }
            """
        )

    def _build_paradigm_panel(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(10)

        title = QtWidgets.QLabel("实验范式配置")
        title.setObjectName("sectionTitle")
        subtitle = QtWidgets.QLabel("选择范式并填写参数，点击“应用配置”生效")
        subtitle.setObjectName("sectionSub")
        vbox.addWidget(title)
        vbox.addWidget(subtitle)

        status_row = QtWidgets.QHBoxLayout()
        self._paradigm_status.setObjectName("statusBadge")
        self._trigger_status.setObjectName("statusBadge")
        status_row.addWidget(self._paradigm_status)
        status_row.addWidget(self._trigger_status)
        status_row.addStretch()
        vbox.addLayout(status_row)

        self._paradigm_group = QtWidgets.QButtonGroup(self)
        self._paradigm_group.setExclusive(True)
        self._paradigm_toggle_buttons: List[QtWidgets.QPushButton] = []

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        cards = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(cards)
        grid.setSpacing(12)
        grid.setContentsMargins(0, 6, 0, 0)
        grid.setColumnStretch(0, 1)

        paradigms: List[Dict] = [
            {
                "name": "SSVEP",
                "desc": "",
                "params": [
                    ("轮数", "5"),
                    ("每刺激时长(s)", "4.0"),
                ],
            },
            {
                "name": "RSVP",
                "desc": "",
                "params": [
                    ("轮数", "5"),
                    ("刺激频率(Hz)", "10"),
                    ("target文件夹", str(Path("Lib/EEG/rsvp/images/tar"))),
                    ("non-target文件夹", str(Path("Lib/EEG/rsvp/images/notar"))),
                ],
            },
            {
                "name": "眼动消除",
                "desc": "",
                "params": [
                    ("目标数量", "10"),
                    ("注视阈值(s)", "0.8"),
                    ("排列模式", "random"),
                ],
            },
            {
                "name": "眼动四分类",
                "desc": "",
                "params": [
                    ("次数", "200"),
                    ("阶段时长(s)", "3.0"),
                    ("环点时长(s)", "3.0"),
                ],
            },
            {
                "name": "手势识别",
                "desc": "",
                "params": [
                    ("手势名称列表", "拳,张,捏,勾,指,摆"),
                    ("每张图片显示时长(s)", "10.0"),
                    ("图片间休息时长(s)", "5.0"),
                    ("每轮间休息时长(s)", "30.0"),
                    ("循环次数", "6"),
                ],
            },
        ]

        for idx, paradigm in enumerate(paradigms):
            card, radio = self._create_paradigm_card(paradigm)
            if radio is not None:
                self._paradigm_group.addButton(radio)
            r = idx
            grid.addWidget(card, r, 0)

        scroll.setWidget(cards)
        vbox.addWidget(scroll)
        vbox.addStretch()
        return container

    def _create_paradigm_card(self, paradigm: Dict) -> (QtWidgets.QGroupBox, QtWidgets.QRadioButton):
        box = QtWidgets.QGroupBox(paradigm["name"])
        box.setObjectName("paradigmCard")
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(10)

        title = QtWidgets.QLabel(paradigm["name"])
        title.setStyleSheet("font-weight: bold;")
        title.setMinimumWidth(120)
        row.addWidget(title)

        edits = {}
        params_row = QtWidgets.QHBoxLayout()
        params_row.setSpacing(8)
        for label, default in paradigm.get("params", []):
            lbl = QtWidgets.QLabel(label)
            lbl.setObjectName("sectionSub")
            params_row.addWidget(lbl)
            if label == "排列模式":
                combo = QtWidgets.QComboBox()
                combo.addItems(["random", "grid", "circle", "triangle"])
                if default in {"random", "grid", "circle", "triangle"}:
                    combo.setCurrentText(default)
                combo.setToolTip("random=随机, grid=网格, circle=圆形, triangle=三角形")
                combo.setMinimumWidth(120)
                params_row.addWidget(combo)
                edits[label] = combo
            else:
                edit = QtWidgets.QLineEdit()
                edit.setText(str(default))
                edit.setMinimumWidth(120)
                params_row.addWidget(edit)
                edits[label] = edit
        row.addLayout(params_row)
        row.addStretch()
        if paradigm["name"] == "手势识别":
            img_btn = QtWidgets.QPushButton("选择手势图片（可多选）")
            img_btn.clicked.connect(self._choose_gesture_files)
            row.addWidget(img_btn)
        toggle_btn = QtWidgets.QPushButton("OFF")
        toggle_btn.setCheckable(True)
        toggle_btn.setProperty("toggle", True)
        toggle_btn.setMinimumHeight(36)
        radio = None
        toggle_btn.toggled.connect(
            lambda checked, name=paradigm["name"], edits=edits, btn=toggle_btn, box=box: self._on_paradigm_toggled(
                checked, name, edits, btn, box
            )
        )
        self._paradigm_toggle_buttons.append(toggle_btn)
        row.addWidget(toggle_btn)
        return box, radio

    def _on_paradigm_toggled(
        self,
        checked: bool,
        name: str,
        edits: Dict[str, QtWidgets.QWidget],
        btn: QtWidgets.QPushButton,
        box: QtWidgets.QGroupBox,
    ) -> None:
        if checked:
            # 其他范式全部关闭
            for other in self._paradigm_toggle_buttons:
                if other is not btn and other.isChecked():
                    other.blockSignals(True)
                    other.setChecked(False)
                    other.setText("OFF")
                    other.blockSignals(False)
            btn.setText("ON")
            self._on_paradigm_selected(name, edits, None, box)
        else:
            btn.setText("OFF")
            if self._selected_paradigm == name:
                self._selected_paradigm = None
                self._selected_paradigm_params = {}
                self._paradigm_status.setText("未选择范式")
                self._trigger_status.setText("当前标签（trigger）：无")

    def _on_paradigm_selected(self, name: str, edits: Dict[str, QtWidgets.QWidget], radio: Optional[QtWidgets.QRadioButton], box: QtWidgets.QGroupBox) -> None:
        params = {}
        for k, e in edits.items():
            if isinstance(e, QtWidgets.QComboBox):
                params[k] = e.currentText().strip()
            else:
                params[k] = e.text().strip()
        self._paradigm_status.setText(f"当前范式：{name}，参数：{params}")
        # 将范式名称作为 trigger 标签提示
        self._trigger_status.setText(f"当前标签（trigger）：{name}")
        self._selected_paradigm = name
        self._selected_paradigm_params = params
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
        vbox.setSpacing(10)

        title = QtWidgets.QLabel("运行控制 & 反馈")
        title.setObjectName("sectionTitle")
        subtitle = QtWidgets.QLabel("选择运行模式与反馈方式，点击“开始实验”启动")
        subtitle.setObjectName("sectionSub")
        vbox.addWidget(title)
        vbox.addWidget(subtitle)

        model_group = QtWidgets.QGroupBox("模型配置")
        model_box = QtWidgets.QVBoxLayout(model_group)
        model_row = QtWidgets.QHBoxLayout()
        model_row.addWidget(QtWidgets.QLabel("选择模型:"))
        self._model_combo = QtWidgets.QComboBox()
        self._model_combo.addItems(["SSVEP-ResNet", "RSVP-Transformer", "Gesture-CNN"])
        model_row.addWidget(self._model_combo)
        self._model_param_edit = QtWidgets.QLineEdit()
        self._model_param_edit.setPlaceholderText("可选：模型参数，如 key1=1,key2=2 或 JSON")
        model_row.addWidget(self._model_param_edit)
        apply_btn = QtWidgets.QPushButton("加载模型")
        apply_btn.setProperty("primary", True)
        apply_btn.clicked.connect(self._on_model_selected)
        model_row.addWidget(apply_btn)
        model_row.addStretch()
        model_box.addLayout(model_row)
        vbox.addWidget(model_group)

        run_group = QtWidgets.QGroupBox("运行设置")
        run_box = QtWidgets.QVBoxLayout(run_group)
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("运行模式:"))
        mode_row.addWidget(self._mode_combo)
        mode_row.addWidget(QtWidgets.QLabel("反馈方式:"))
        mode_row.addWidget(self._feedback_combo)
        mode_row.addStretch()
        run_box.addLayout(mode_row)

        # 启动范式：根据上方选择与配置决定
        run_present_btn = QtWidgets.QPushButton("开始实验")
        run_present_btn.clicked.connect(self._on_run_presentation)
        run_present_btn.setProperty("primary", True)
        run_box.addWidget(self._trigger_com_edit)
        run_box.addWidget(run_present_btn)
        run_box.addWidget(self._progress)
        vbox.addWidget(run_group)

        status_row = QtWidgets.QHBoxLayout()
        self._model_status.setObjectName("statusBadge")
        self._feedback_status.setObjectName("statusBadge")
        status_row.addWidget(self._model_status)
        status_row.addWidget(self._feedback_status)
        status_row.addStretch()
        vbox.addLayout(status_row)
        vbox.addStretch()
        return container

    def _on_model_selected(self) -> None:
        name = self._model_combo.currentText()
        self._model_params = self._model_param_edit.text().strip()
        self._model_status.setText(f"当前模型：{name}")
        # 以模型名作为示例反馈输出
        self._feedback_status.setText(f"反馈结果：已加载模型 {name}，参数：{self._model_params or '默认'}，等待呈现")
        self._selected_model = name

    def _on_run_presentation(self) -> None:
        """根据已选范式执行呈现；有模型则用模型输出，否则用真实标签。"""
        if not self._selected_paradigm:
            QtWidgets.QMessageBox.information(self, "提示", "请先在上方选择并应用一个范式。")
            return
        if self._paradigm_proc and self._paradigm_proc.is_alive():
            QtWidgets.QMessageBox.warning(self, "提示", "已有范式子进程在运行，请先结束或等待完成。")
            return
        paradigm = self._selected_paradigm
        params = self._selected_paradigm_params or {}
        # 开始范式前：自动开启采集页的数据保存（若有采集页引用）
        if self._acquisition_page is not None:
            try:
                # 先记录当前范式名，便于触发文件命名
                try:
                    self._acquisition_page.set_paradigm_name(paradigm)
                except Exception:
                    pass
                default_dir = self._last_save_dir or str(Path.home() / "mpscap_data")
                save_dir = QtWidgets.QFileDialog.getExistingDirectory(
                    self,
                    "选择数据保存文件夹",
                    default_dir
                )
                if not save_dir:
                    QtWidgets.QMessageBox.information(self, "提示", "未选择保存目录，已取消开始实验。")
                    return
                self._last_save_dir = save_dir
                ok = self._acquisition_page.start_saving_auto(save_dir)
                if not ok:
                    QtWidgets.QMessageBox.warning(self, "提示", "自动开始保存数据失败，可手动点击保存按钮后重试。")
                    return
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "提示", f"自动开始保存数据失败，可手动点击保存按钮。原因：{e}")
        trigger_com = self._trigger_com_edit.text().strip() or None
        mode_text = self._mode_combo.currentText()
        is_acquire_mode = "采集模式" in mode_text
        feedback_choice = self._feedback_combo.currentText()
        kafka_bootstrap = None if is_acquire_mode else self._kafka_bootstrap
        kafka_topic = None if is_acquire_mode else self._kafka_topic
        save_dir_for_trig = self._last_save_dir
        # 模态检查：SSVEP/RSVP 优先 EEG，若尚未集成 EEG 则允许 EMG 存在时继续；手势需 EMG
        if paradigm in ["SSVEP", "RSVP"]:
            try:
                shm = CreateShm(master=False)
                eegchs = int(shm.getvalue('eegchs'))
                emgchs = int(shm.getvalue('emgchs'))
                if eegchs <= 0 and emgchs <= 0:
                    QtWidgets.QMessageBox.warning(self, "提示", "未检测到 EEG 或 EMG 通道，请先开启采集后再运行 SSVEP/RSVP。")
                    return
            except Exception:
                # 无法读取共享内存时不再阻塞，假定外部已开启采集
                pass
        if paradigm == "手势识别":
            try:
                shm = CreateShm(master=False)
                emgchs = int(shm.getvalue('emgchs'))
                if emgchs <= 0:
                    QtWidgets.QMessageBox.warning(self, "提示", "未检测到 EMG 通道，请先开启肌电采集后再运行手势范式。")
                    return
            except Exception:
                QtWidgets.QMessageBox.warning(self, "提示", "无法读取 EMG 采集状态，请确认已开启采集。")
                return
        try:
            if paradigm == "SSVEP":
                cycles = int(params.get("轮数", "5") or 5)
                stim_dur = float(params.get("每刺激时长(s)", "4.0") or 4.0)
                self._paradigm_proc = self._mp_ctx.Process(
                    target=run_ssvep_worker,
                    args=(None, trigger_com, kafka_bootstrap, kafka_topic, paradigm, save_dir_for_trig, cycles, stim_dur),
                    daemon=False,
                )
                self._paradigm_proc.start()
                self._running_paradigm = "SSVEP"
                self._proc_timer.start()
            elif paradigm == "眼动消除":
                target_count = int(params.get("目标数量", "10") or 10)
                dwell_sec = float(params.get("注视阈值(s)", "0.8") or 0.8)
                layout = (params.get("排列模式", "random") or "random").strip().lower()
                if layout not in {"random", "grid", "circle", "triangle"}:
                    layout = "random"
                exe_path = Path(__file__).resolve().parents[3] / "Lib" / "EyeTracker" / "tobii_4c_app.exe"
                if not exe_path.exists():
                    QtWidgets.QMessageBox.warning(self, "眼动仪缺失", f"未找到眼动服务程序: {exe_path}")
                    return
                try:
                    __import__("pygame")
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "依赖缺失", f"需先安装 pygame：{e}")
                    return
                self._paradigm_proc = self._mp_ctx.Process(
                    target=run_eye_target_worker,
                    args=(
                        target_count,
                        dwell_sec,
                        layout,
                        kafka_bootstrap,
                        kafka_topic,
                        paradigm,
                        save_dir_for_trig,
                    ),
                    daemon=False,
                )
                self._paradigm_proc.start()
                self._running_paradigm = "眼动消除"
                self._proc_timer.start()
            elif paradigm == "眼动四分类":
                trials = int(params.get("次数", "200") or 200)
                phase_sec = float(params.get("阶段时长(s)", "3.0") or 3.0)
                ring_sec = float(params.get("环点时长(s)", "3.0") or 3.0)
                try:
                    __import__("psychopy")
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "依赖缺失", f"需先安装 psychopy：{e}")
                    return
                self._paradigm_proc = self._mp_ctx.Process(
                    target=run_eye_4class_worker,
                    args=(
                        trials,
                        phase_sec,
                        ring_sec,
                        kafka_bootstrap,
                        kafka_topic,
                        paradigm,
                        save_dir_for_trig,
                    ),
                    daemon=False,
                )
                self._paradigm_proc.start()
                self._running_paradigm = "眼动四分类"
                self._proc_timer.start()
            elif paradigm == "手势识别":
                names_raw = params.get("手势名称列表", "")
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
                img_paths = [str(p) for p in imgs]
                self._paradigm_proc = self._mp_ctx.Process(
                    target=run_gesture_worker,
                    args=(
                        img_paths,
                        names,
                        show_sec,
                        rest_gesture,
                        rest_cycle,
                        cycles,
                        kafka_bootstrap,
                        kafka_topic,
                        paradigm,
                        save_dir_for_trig,
                    ),
                    daemon=False,
                )
                self._paradigm_proc.start()
                self._running_paradigm = "手势识别"
                self._proc_timer.start()
            elif paradigm == "RSVP":
                cycles = int(params.get("轮数", "5") or 5)
                stim_freq = float(params.get("刺激频率(Hz)", "10") or 10)
                tar_dir = params.get("target文件夹", str(Path("Lib/EEG/rsvp/images/tar")))
                nt_dir = params.get("non-target文件夹", str(Path("Lib/EEG/rsvp/images/notar")))
                self._paradigm_proc = self._mp_ctx.Process(
                    target=run_rsvp_worker,
                    args=(
                        tar_dir,
                        nt_dir,
                        cycles,
                        kafka_bootstrap,
                        kafka_topic,
                        paradigm,
                        save_dir_for_trig,
                        stim_freq,
                    ),
                    daemon=False,
                )
                self._paradigm_proc.start()
                self._running_paradigm = "RSVP"
                self._proc_timer.start()
            else:
                QtWidgets.QMessageBox.information(self, "提示", f"{paradigm} 范式暂未实现独立进程运行。")
                return
            # 反馈显示
            if is_acquire_mode:
                self._feedback_status.setText("反馈结果：采集模式，不输出反馈（仅发送 trigger / 采集端保存数据）")
            else:
                if feedback_choice == "不反馈结果":
                    self._feedback_status.setText("反馈结果：已关闭反馈（仅呈现范式）")
                elif feedback_choice == "使用真实标签":
                    self._feedback_status.setText(f"反馈结果：使用范式真实标签 -> {paradigm}")
                else:
                    if not self._selected_model:
                        QtWidgets.QMessageBox.warning(self, "模型未选择", "请选择模型或将反馈方式改为“不反馈结果/使用真实标签”。")
                    self._feedback_status.setText(f"反馈结果：模型 {self._selected_model or '未选择'} 输出（示例），参数：{self._model_params or '默认'}")
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

    def _stop_saving_after_exp(self) -> None:
        """实验结束后自动停止数据保存。"""
        try:
            if self._acquisition_page is not None and self._acquisition_page._saving:
                # 触发采集页保存按钮，结束保存
                self._acquisition_page._save_data_btn.setChecked(False)
        except Exception as e:
            print(f"[TaskExecution] 停止保存失败: {e}")

    def _check_proc_done(self) -> None:
        """轮询子进程结束，结束后自动停保存。"""
        if self._paradigm_proc is None:
            self._proc_timer.stop()
            self._running_paradigm = None
            return
        if self._paradigm_proc.is_alive():
            return
        # 进程已结束
        self._proc_timer.stop()
        self._paradigm_proc = None
        self._running_paradigm = None
        self._stop_saving_after_exp()

    def _choose_gesture_files(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "选择手势图片",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )
        if files:
            self._gesture_files = [Path(f) for f in files]


