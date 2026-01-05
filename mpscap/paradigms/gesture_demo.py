from __future__ import annotations

import sys
from pathlib import Path
from typing import List


# 统一限制图片显示尺寸（占窗口宽高的 80%），避免超大分辨率溢出
IMG_FIT_RATIO = 0.8


_dpi_fixed = False


def _make_process_dpi_aware() -> None:
    """
    Windows 下关闭系统缩放，避免 PsychoPy 窗口被 150%/200% 放大。
    调用一次即可，失败时静默忽略。
    """
    global _dpi_fixed
    if _dpi_fixed:
        return
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        shcore = getattr(ctypes, "windll", None) and getattr(ctypes.windll, "shcore", None)
        if shcore and hasattr(shcore, "SetProcessDpiAwareness"):
            shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
        elif hasattr(ctypes.windll.user32, "SetProcessDPIAware"):
            ctypes.windll.user32.SetProcessDPIAware()
        _dpi_fixed = True
    except Exception:
        pass


def _fit_image_size(orig_size, max_w: float, max_h: float) -> tuple[float, float]:
    """
    等比缩放图片尺寸，限制在给定最大宽高内；不放大小图，只缩小超大图。
    - 兼容 numpy 数组或列表，避免高维数组触发布尔歧义。
    """
    try:
        ow, oh = orig_size  # 可能是 list/tuple/ndarray
    except Exception:
        return max_w, max_h
    try:
        ow = float(ow)
        oh = float(oh)
    except Exception:
        return max_w, max_h
    if ow <= 0 or oh <= 0:
        return max_w, max_h
    scale = min(max_w / ow, max_h / oh, 1.0)  # 不对小图放大
    return ow * scale, oh * scale


def run_gesture_sequence(
    imgs: List[Path],
    names: List[str],
    show_sec: float,
    rest_between_gestures: float,
    rest_between_cycles: float,
    cycles: int,
    progress_cb=None,
    trigger_cb=None,
) -> None:
    """
    手势范式：使用 PsychoPy 窗口（1920x1080，非全屏）按 names 顺序呈现图片。
    - 每张显示 show_sec 秒，图片间休息 rest_between_gestures 秒，轮间休息 rest_between_cycles 秒，循环 cycles 轮。
    - 顶部显示提示文字（手势名 + 轮次 + 倒计时），底部用简易进度条显示当前手势剩余时间。
    """
    try:
        from psychopy import visual, core, event
    except Exception as e:
        print(f"[Gesture] PsychoPy 未安装或加载失败: {e}")
        return

    imgs = [p for p in imgs if p.exists() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    if not imgs:
        print("[Gesture] 未找到图片，请提供至少一张 png/jpg/jpeg。")
        return
    if not names:
        print("[Gesture] 未提供手势名称列表。")
        return

    # 按 names 长度截取或重复，保持数量一致
    seq: List[Path] = []
    for i, _ in enumerate(names):
        if i < len(imgs):
            seq.append(imgs[i])
        else:
            seq.append(imgs[-1])

    print(f"[Gesture] 准备显示图片数: {len(seq)}, 原始图片数: {len(imgs)}, 手势名数: {len(names)}")

    _make_process_dpi_aware()

    # 创建 PsychoPy 窗口（非全屏）
    win = visual.Window(
        size=(1920, 1080),
        units="pix",
        color=(0, 0, 0),
        fullscr=False,
        allowGUI=True,
        checkTiming=False,
        winType="pyglet",
    )

    # 计算图片可用区域（留出顶部文字与底部进度条空间）
    max_img_w = win.size[0] * IMG_FIT_RATIO
    max_img_h = win.size[1] * IMG_FIT_RATIO

    # 文字与进度条元素
    title = visual.TextStim(win=win, text="", color="black", pos=(0, 400), height=48, wrapWidth=1600)
    cycle_txt = visual.TextStim(win=win, text="", color="black", pos=(0, 460), height=36)

    bar_width = 1400
    bar_height = 24
    bar_y = -480 + 40
    bar_bg = visual.Rect(win=win, width=bar_width, height=bar_height, fillColor=[-0.5, -0.5, -0.5], lineColor=None, pos=(0, bar_y))
    bar_fg = visual.Rect(win=win, width=1, height=bar_height, fillColor=[0.2, 0.8, 0.2], lineColor=None, pos=(0, bar_y))

    def _set_bar_frac(frac: float) -> None:
        w = max(1, bar_width * max(0.0, min(1.0, frac)))
        bar_fg.width = w
        # 以左侧为起点，居中到当前宽度的一半
        bar_fg.pos = (-bar_width / 2 + w / 2, bar_y)

    # 预载图片
    stim_images = []
    for p in seq:
        stim = visual.ImageStim(
            win=win,
            image=str(p),
            size=None,  # 先按原尺寸加载
            units="pix",
            pos=(0, 0),
            ori=0,
            interpolate=True,
        )
        stim.size = _fit_image_size(stim.size, max_img_w, max_img_h)
        stim_images.append(stim)

    total_steps = cycles * len(names)
    step = 0
    esc = False
    sent_start = False

    try:
        # 实验开始 trigger 254
        if trigger_cb:
            trigger_cb(254)
            sent_start = True
        for c_idx in range(cycles):
            if trigger_cb:
                trigger_cb(252)  # 轮开始
            cycle_txt.setText(f"第 {c_idx + 1} / {cycles} 轮")
            for idx, name in enumerate(names):
                if esc:
                    break
                stim = stim_images[idx]
                if trigger_cb:
                    trigger_cb(idx * 2 + 1)  # 手势开始（1,3,5,...）
                # 3-2-1 倒计时
                for t in [3, 2, 1]:
                    title.setText(f"{name} （{t}）")
                    _set_bar_frac(0.0)
                    stim.draw()
                    cycle_txt.draw()
                    title.draw()
                    bar_bg.draw()
                    bar_fg.draw()
                    win.flip()
                    core.wait(1.0)
                    if "escape" in event.getKeys():
                        esc = True
                        break
                if esc:
                    break

                # 正式显示，带进度条
                title.setText(name)
                clock = core.Clock()
                while True:
                    elapsed = clock.getTime()
                    frac = min(1.0, max(0.0, elapsed / max(0.001, show_sec)))
                    _set_bar_frac(frac)
                    stim.draw()
                    cycle_txt.draw()
                    title.draw()
                    bar_bg.draw()
                    bar_fg.draw()
                    win.flip()
                    if "escape" in event.getKeys():
                        esc = True
                        break
                    if elapsed >= show_sec:
                        break

                # 手势结束 trigger（紧接展示结束发出，不等待休息段）
                if trigger_cb:
                    trigger_cb(idx * 2 + 2)  # 手势结束（2,4,6,...）

                # 休息
                title.setText("休息")
                _set_bar_frac(0.0)
                title.draw()
                cycle_txt.draw()
                bar_bg.draw()
                bar_fg.draw()
                win.flip()
                if rest_between_gestures > 0:
                    core.wait(rest_between_gestures)

                step += 1
                if progress_cb:
                    try:
                        progress_cb(step, total_steps)
                    except Exception:
                        pass

            if esc:
                break
            # 轮间休息：最后一轮不休息
            if rest_between_cycles > 0 and c_idx != cycles - 1:
                title.setText("轮间休息")
                _set_bar_frac(0.0)
                title.draw()
                cycle_txt.draw()
                bar_bg.draw()
                bar_fg.draw()
                win.flip()
                core.wait(rest_between_cycles)
            if trigger_cb:
                trigger_cb(253)  # 轮结束
    finally:
        # 实验结束 trigger 255
        if trigger_cb and sent_start:
            trigger_cb(255)
        try:
            win.close()
        except Exception:
            pass

