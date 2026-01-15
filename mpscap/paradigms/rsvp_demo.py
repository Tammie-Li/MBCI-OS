from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Optional, Callable


def run_rsvp_demo(
    target_dir: Path,
    nontarget_dir: Path,
    cycles: int = 1,
    target_code: int = 2,
    nontarget_code: int = 1,
    trigger_cb: Optional[Callable[[int], None]] = None,
    stim_freq: float = 10.0,
) -> None:
    """
    简化版 RSVP：
    - 每轮将 target 与 non-target 图片混合随机呈现；
    - target 触发码 target_code，non-target 触发码 nontarget_code；
    - 每张展示 stim_time 秒，间隔 isi 秒。
    """
    try:
        from psychopy import visual, event, core
    except Exception as e:
        print(f"[RSVP] PsychoPy 未安装或加载失败: {e}")
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

    tg_imgs = [p for p in Path(target_dir).iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    nt_imgs = [p for p in Path(nontarget_dir).iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    if not tg_imgs or not nt_imgs:
        print("[RSVP] target 或 non-target 图片为空，请检查目录。")
        return

    _make_process_dpi_aware()
    # 与其它范式一致的窗口（非全屏），缩小以避免高 DPI 放大
    win = visual.Window(
        size=(1920, 1080),
        units="pix",
        color=(-1, -1, -1),
        fullscr=False,
        allowGUI=True,
        checkTiming=False,
        winType="pyglet",
    )

    stim_time = 1.0 / max(0.1, stim_freq)
    isi = 0.0

    def show_image(p: Path, code: int) -> None:
        stim = visual.ImageStim(win=win, image=str(p), pos=(0, 0), units="pix", size=None, interpolate=True)
        stim.draw()
        win.flip()
        if trigger_cb:
            trigger_cb(code)
        core.wait(stim_time)
        win.flip()
        if isi > 0:
            core.wait(isi)

    try:
        for _ in range(max(1, cycles)):
            seq = [(p, target_code) for p in tg_imgs] + [(p, nontarget_code) for p in nt_imgs]
            random.shuffle(seq)
            for p, code in seq:
                show_image(p, code)
                if "escape" in event.getKeys():
                    raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            win.close()
        except Exception:
            pass

