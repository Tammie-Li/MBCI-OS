from __future__ import annotations

import multiprocessing as mp
from typing import Dict, List, Optional
from pathlib import Path

from PyQt5 import QtCore, QtWidgets, QtGui

from ...paradigms.paradigm_worker import run_ssvep_worker, run_gesture_worker


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
        self._model_param_edit = QtWidgets.QLineEdit()
        self._model_param_edit.setPlaceholderText("可选：模型参数，如 key1=1,key2=2 或 JSON")
        model_row.addWidget(self._model_param_edit)
        apply_btn = QtWidgets.QPushButton("加载模型")
        apply_btn.clicked.connect(self._on_model_selected)
        model_row.addWidget(apply_btn)
        model_row.addStretch()
        vbox.addLayout(model_row)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("运行模式:"))
        mode_row.addWidget(self._mode_combo)
        mode_row.addStretch()
        vbox.addLayout(mode_row)

        # 启动范式：根据上方选择与配置决定
        run_present_btn = QtWidgets.QPushButton("开始实验")
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
        paradigm = self._selected_paradigm
        params = self._selected_paradigm_params or {}
        trigger_com = self._trigger_com_edit.text().strip() or None
        mode_text = self._mode_combo.currentText()
        is_acquire_mode = "采集模式" in mode_text
        feedback_choice = self._feedback_combo.currentText()
        kafka_bootstrap = None if is_acquire_mode else self._kafka_bootstrap
        kafka_topic = None if is_acquire_mode else self._kafka_topic
        save_dir_for_trig = self._last_save_dir
        try:
            if paradigm == "SSVEP":
                freq_str = params.get("刺激频率(Hz)", "")
                freqs = []
                for seg in freq_str.split(","):
                    seg = seg.strip()
                    if not seg:
                        continue
                    try:
                        freqs.append(float(seg))
                    except ValueError:
                        pass
                self._paradigm_proc = self._mp_ctx.Process(
                    target=run_ssvep_worker,
                    args=(freqs or None, trigger_com, kafka_bootstrap, kafka_topic, paradigm, save_dir_for_trig),
                    daemon=False,
                )
                self._paradigm_proc.start()
                self._running_paradigm = "SSVEP"
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


