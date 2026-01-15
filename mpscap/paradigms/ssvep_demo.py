from __future__ import annotations

import sys
import time
from typing import List, Optional, Callable


def run_ssvep_demo(
    freqs: Optional[List[float]] = None,
    cycles: int = 1,
    refresh_rate: float = 75.0,
    stim_duration: float = 4.0,
    trigger_cb: Optional[Callable[[int], None]] = None,
    tip_time: float = 0.5,
) -> None:
    """
    参考 examples/paradigm/ssvep_stimulate.py，使用 4x10 频闪阵列。
    - 默认 40 个刺激，频率 8.0~15.8 Hz（步长 0.2）。
    - 每个刺激时长可配（stim_duration），轮次 cycles。
    - 开始每个刺激前显示红色三角提示当前目标，并在顶部显示真实标签（编号）。
    - trigger_cb 可选：每个刺激开始时发送标签（index+1），外层轮/实验触发由 worker 负责。
    """
    try:
        from psychopy import visual, event, core
        import numpy as np
    except Exception as e:
        print(f"[SSVEP] PsychoPy 未安装或加载失败: {e}")
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

    freqs = freqs or list(np.arange(8.0, 16.0, 0.2))  # 40 频点
    stim_num = len(freqs)
    stim_r_num, stim_c_num = 4, 10
    if stim_num < stim_r_num * stim_c_num:
        need = stim_r_num * stim_c_num - stim_num
        freqs = freqs + freqs[:need]
    freqs = freqs[: stim_r_num * stim_c_num]

    stim_radius = 160
    positions = []
    for idx in range(stim_r_num * stim_c_num):
        x_tmp = -960 + 105 + (stim_radius + 30) * (idx % stim_c_num)
        y_tmp = 540 - 280 - (stim_radius + 70) * (idx // stim_c_num)
        positions.append([x_tmp, y_tmp])

    _make_process_dpi_aware()
    win = visual.Window(
        size=(1920, 1080),
        units="pix",
        color=(-1, -1, -1),
        fullscr=False,
        allowGUI=True,
        checkTiming=False,
        winType="pyglet",
    )

    stim = visual.ElementArrayStim(
        win=win,
        nElements=stim_r_num * stim_c_num,
        sfs=0,
        sizes=[stim_radius, stim_radius],
        xys=positions,
        phases=0,
        colors=(1, 1, 1),
        elementTex="sin",
        elementMask=None,
    )

    texts = [
        visual.TextStim(
            win=win,
            text=str(i + 1),
            pos=positions[i],
            colorSpace="rgb255",
            color=(0, 0, 0),
            height=70,
            autoLog=False,
        )
        for i in range(stim_r_num * stim_c_num)
    ]

    triangle_tip_radius = 30
    triangle_tip = visual.Polygon(
        win=win,
        edges=3,
        units="pix",
        radius=triangle_tip_radius,
        fillColor="red",
        lineColor="red",
        pos=(0, 0),
    )
    label_txt = visual.TextStim(win=win, text="", pos=(0, 500), color="red", height=48)

    frames = int(stim_duration * refresh_rate)

    try:
        for _ in range(max(1, cycles)):
            for idx, freq in enumerate(freqs):
                # 目标提示
                x, y = positions[idx]
                triangle_tip.setPos([x, y - stim_radius * 0.5 - triangle_tip_radius])
                label_txt.setText(f"目标: {idx + 1}")
                triangle_tip.draw()
                label_txt.draw()
                win.flip()
                core.wait(max(0.1, tip_time))

                # 触发（真实标签）
                if trigger_cb:
                    trigger_cb(idx + 1)

                start_t = time.time()
                for frame_idx in range(frames):
                    phases = (np.array(freqs, dtype=float) * frame_idx / refresh_rate) % 2 * np.pi
                    stim.phases = phases
                    stim.draw()
                    for t in texts:
                        t.draw()
                    label_txt.draw()
                    triangle_tip.draw()
                    win.flip()
                    if "escape" in event.getKeys():
                        raise KeyboardInterrupt()
                rest_t = max(0.0, 1.0 - (time.time() - start_t - stim_duration))
                if rest_t > 0:
                    core.wait(rest_t)
    except KeyboardInterrupt:
        if trigger_cb:
            trigger_cb(0xFF)
    finally:
        try:
            win.close()
        except Exception:
            pass
