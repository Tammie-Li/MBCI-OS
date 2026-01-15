from __future__ import annotations

import math
import sys
import time
from typing import Callable, Optional


def run_eye_4class_demo(
    trials: int = 200,
    phase_sec: float = 3.0,
    ring_sec: float = 3.0,
    trigger_cb: Optional[Callable[[int], None]] = None,
) -> None:
    """
    眼动四分类范式（PsychoPy）：
    - 每次实验：Noise/Blinks -> Saccades -> Smooth Pursuits -> Fixations（每段约 phase_sec 秒）
    - 之后展示环状 5 点（ring_sec 秒）
    - 每次结束后按空格进入下一次
    """
    try:
        from psychopy import visual, event, core
    except Exception as e:
        print(f"[Eye4Class] PsychoPy 未安装或加载失败: {e}")
        return

    def _make_process_dpi_aware() -> None:
        if not sys.platform.startswith("win"):
            return
        try:
            import ctypes

            shcore = getattr(ctypes, "windll", None) and getattr(ctypes.windll, "shcore", None)
            if shcore and hasattr(shcore, "SetProcessDpiAwareness"):
                shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
            elif hasattr(ctypes.windll.user32, "SetProcessDPIAware"):
                ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    _make_process_dpi_aware()
    win = visual.Window(
        size=(1920, 1080),
        units="pix",
        color=(1, 1, 1),
        fullscr=False,
        allowGUI=True,
        checkTiming=False,
        winType="pyglet",
    )

    center_dot = visual.Circle(win=win, radius=10, fillColor="black", lineColor="black", pos=(0, 0))
    label = visual.TextStim(win=win, text="", color="black", height=36, pos=(0, 380))
    tip = visual.TextStim(win=win, text="", color="black", height=28, pos=(0, -420))

    ring_points = []
    ring_r = 180
    for idx in range(5):
        angle = idx * 2 * math.pi / 5
        x = ring_r * math.cos(angle)
        y = ring_r * math.sin(angle)
        ring_points.append(visual.Circle(win=win, radius=12, fillColor="black", lineColor="black", pos=(x, y)))

    phases = [
        ("Noise / Blinks", 1),
        ("Saccades", 2),
        ("Smooth Pursuits", 3),
        ("Fixations", 4),
    ]

    try:
        for trial_idx in range(max(1, int(trials))):
            for name, code in phases:
                label.setText(f"{name} （第 {trial_idx + 1}/{trials} 次）")
                tip.setText("请按指令完成当前任务")
                if trigger_cb:
                    trigger_cb(code)
                t0 = time.time()
                while time.time() - t0 < max(0.5, phase_sec):
                    center_dot.draw()
                    label.draw()
                    tip.draw()
                    win.flip()
                    if "escape" in event.getKeys():
                        raise KeyboardInterrupt()

            # 环状 5 点
            if trigger_cb:
                trigger_cb(10)
            label.setText("选择点呈现")
            tip.setText("请注视环状点位")
            t1 = time.time()
            while time.time() - t1 < max(0.5, ring_sec):
                center_dot.draw()
                for p in ring_points:
                    p.draw()
                label.draw()
                tip.draw()
                win.flip()
                if "escape" in event.getKeys():
                    raise KeyboardInterrupt()

            # 等待空格进入下一次
            label.setText("本次结束，按空格进入下一次")
            tip.setText("")
            while True:
                center_dot.draw()
                label.draw()
                win.flip()
                keys = event.getKeys()
                if "space" in keys:
                    break
                if "escape" in keys:
                    raise KeyboardInterrupt()
                core.wait(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            win.close()
        except Exception:
            pass

