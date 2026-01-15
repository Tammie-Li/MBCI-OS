from __future__ import annotations

import math
import random
import sys
import time


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


def run_test_paradigm(trials: int = 5) -> None:
    """
    测试范式：
    Noise/Blinks → Saccades → Smooth Pursuits → Fixations
    - 点会移动以引导眼动行为
    - 点大一些；进入“选中/完成”阶段后缩小
    - 周围生成 5 个“叶”（小圆点），并用线连接到中心
    """
    try:
        from psychopy import visual, event, core
    except Exception as e:
        print(f"[test.py] PsychoPy 未安装或加载失败: {e}")
        return

    _make_process_dpi_aware()
    win = visual.Window(
        size=(1920, 1080),
        units="pix",
        color=(1, 1, 1),
        fullscr=True,
        allowGUI=False,
        checkTiming=False,
        winType="pyglet",
        screen=1,
    )
    refresh_hz = 60.0

    big_radius = 40
    small_radius = 16
    center_dot = visual.Circle(win=win, radius=big_radius, fillColor="black", lineColor="black", pos=(0, 0))
    label = visual.TextStim(win=win, text="", color="black", height=28, pos=(0, 260))
    tip = visual.TextStim(win=win, text="", color="black", height=22, pos=(0, -300))
    info = visual.TextStim(win=win, text="", color="black", height=20, pos=(0, -360))

    def _check_abort() -> None:
        keys = event.getKeys()
        if "escape" in keys:
            raise KeyboardInterrupt()

    def _draw_center(extra=None) -> None:
        center_dot.draw()
        if extra:
            for obj in extra:
                obj.draw()
        label.draw()
        tip.draw()
        info.draw()
        win.flip()

    def _wait_space() -> None:
        event.clearEvents()
        while True:
            _check_abort()
            _draw_center()
            keys = event.waitKeys(keyList=["space", "escape"])
            if "escape" in keys:
                raise KeyboardInterrupt()
            if "space" in keys:
                break

    # 3x3 网格目标点
    grid_spacing = 420
    grid_positions = []
    for gy in [1, 0, -1]:
        for gx in [-1, 0, 1]:
            grid_positions.append((gx * grid_spacing, gy * grid_spacing))
    usage_count = [0] * len(grid_positions)

    def _weighted_pick() -> tuple[float, float]:
        weights = []
        for u in usage_count:
            weights.append(1.0 / (1.0 + u))
        idx = random.choices(range(len(grid_positions)), weights=weights, k=1)[0]
        usage_count[idx] += 1
        return grid_positions[idx]

    # 预生成 5 个叶（小圆点）和连线
    petal_points = []
    petal_lines = []
    ring_r = 200
    for idx in range(5):
        angle = idx * 2 * math.pi / 5
        x = ring_r * math.cos(angle)
        y = ring_r * math.sin(angle)
        petal = visual.Circle(win=win, radius=16, fillColor="black", lineColor="black", pos=(x, y))
        line = visual.Line(win=win, start=(0, 0), end=(x, y), lineColor="black", lineWidth=3)
        petal_points.append(petal)
        petal_lines.append(line)

    def _frames_from_ms(ms: float) -> int:
        return max(1, int((ms / 1000.0) * refresh_hz))

    def _cubic_coeff(p0: float, p1: float, p2: float) -> tuple[float, float]:
        # f(0)=p0, f(0.5)=p1, f(1)=p2, f'(0)=0
        b = 8 * p1 - 7 * p0 - p2
        a = 2 * p2 - 8 * p1 + 6 * p0
        return a, b

    try:
        for trial_idx in range(max(1, trials)):
            info.setText(f"第 {trial_idx + 1}/{trials} 次 | 60Hz 刷新")

            # Blinks：红色提示 2000 ms
            label.setText("Noise / Blinks")
            tip.setText("目标变红时请眨眼")
            center_dot.fillColor = "red"
            center_dot.lineColor = "red"
            center_dot.pos = _weighted_pick()
            for _ in range(_frames_from_ms(2000)):
                _check_abort()
                _draw_center()
            center_dot.fillColor = "black"
            center_dot.lineColor = "black"

            # Saccades：270-300 ms 内消失并出现在新位置
            label.setText("Saccades")
            tip.setText("目标快速跳到新位置，请跟随")
            start_pos = _weighted_pick()
            end_pos = _weighted_pick()
            center_dot.pos = start_pos
            _draw_center()
            blink_ms = random.uniform(270, 300)
            center_dot.opacity = 0.0
            for _ in range(_frames_from_ms(blink_ms)):
                _check_abort()
                _draw_center()
            center_dot.opacity = 1.0
            center_dot.pos = end_pos
            _draw_center()

            # Smooth Pursuits：1500-2000 ms，三点轨迹 + 三次插值
            label.setText("Smooth Pursuits")
            tip.setText("目标平滑移动，请跟随")
            p0 = _weighted_pick()
            p1 = _weighted_pick()
            p2 = _weighted_pick()
            dur_ms = random.uniform(1500, 2000)
            frames = _frames_from_ms(dur_ms)
            ax, bx = _cubic_coeff(p0[0], p1[0], p2[0])
            ay, by = _cubic_coeff(p0[1], p1[1], p2[1])
            for f in range(frames):
                _check_abort()
                t = f / max(1, frames - 1)
                x = ax * t**3 + bx * t**2 + p0[0]
                y = ay * t**3 + by * t**2 + p0[1]
                center_dot.pos = (x, y)
                _draw_center()

            # Fixations：静止 400-700 ms
            label.setText("Fixations")
            tip.setText("目标静止，请注视")
            center_dot.pos = _weighted_pick()
            fix_ms = random.uniform(400, 700)
            for _ in range(_frames_from_ms(fix_ms)):
                _check_abort()
                _draw_center()

            # “选中后”缩小 + 生成 5 个连接的叶（3s）
            center_dot.radius = small_radius
            label.setText("目标选中")
            tip.setText("出现 5 个叶点，请注视")
            for _ in range(_frames_from_ms(3000)):
                _check_abort()
                center_dot.draw()
                for ln in petal_lines:
                    ln.draw()
                for p in petal_points:
                    p.draw()
                label.draw()
                tip.draw()
                info.draw()
                win.flip()

            center_dot.radius = big_radius
            label.setText("本次结束，按空格进入下一次")
            tip.setText("")
            _wait_space()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            win.close()
        except Exception:
            pass


if __name__ == "__main__":
    run_test_paradigm(trials=5)

