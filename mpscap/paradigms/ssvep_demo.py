from __future__ import annotations

import time
from typing import List, Optional


def run_ssvep_demo(trigger_com: Optional[str] = None, freqs: Optional[List[float]] = None) -> None:
    """
    简化版 SSVEP 范式演示（基于 PsychoPy），用于快速可视化与触发输出。

    - 若安装了 psychopy，则会弹出全屏窗口，展示 4 个频闪目标（默认频率 [8, 10, 12, 15] Hz）。
    - 可选串口触发：trigger_com 不为空时，以 115200 波特发送单字节标记（0x01 开始，0x02 结束，目标ID）。
    - 演示时长：每目标 4s，轮询播放 1 轮。
    """
    try:
        from psychopy import visual, event, core
    except Exception as e:
        print(f"[SSVEP] PsychoPy 未安装，无法运行演示: {e}")
        return

    try:
        import serial
    except Exception:
        serial = None

    # 串口触发
    trig = None
    if trigger_com and serial is not None:
        try:
            trig = serial.Serial(trigger_com, 115200, timeout=0.5)
        except Exception as e:
            print(f"[SSVEP] 打开串口 {trigger_com} 失败，继续无触发: {e}")
            trig = None

    def send_trigger(code: int) -> None:
        if trig is None:
            return
        try:
            trig.write(bytes([code & 0xFF]))
        except Exception:
            pass

    freqs = freqs or [8.0, 10.0, 12.0, 15.0]
    stim_time = 4.0
    refresh_hz = 75.0
    frames = int(stim_time * refresh_hz)

    # 创建窗口
    win = visual.Window(size=(1920, 1080), units="pix", color=(-1, -1, -1), fullscr=True, allowGUI=True)

    # 创建 4 个频闪方块
    positions = [(-400, 200), (400, 200), (-400, -200), (400, -200)]
    rects = []
    for pos in positions[: len(freqs)]:
        rects.append(visual.Rect(win=win, width=200, height=200, fillColor="white", lineColor="white", pos=pos))

    info = visual.TextStim(win=win, text="按 ESC 随时退出\n自动播放一轮 SSVEP 刺激", pos=(0, 0), color="white")
    info.draw()
    win.flip()
    core.wait(1.0)

    try:
        for idx, freq in enumerate(freqs):
            send_trigger(0x01)  # 开始
            start = time.time()
            for f_idx in range(frames):
                # 闪烁：使用正弦调制亮度
                phase = 2 * 3.14159 * freq * (f_idx / refresh_hz)
                amp = 0.5 * (1 + (1 if (f_idx * freq / refresh_hz) % 1 < 0.5 else -1))
                rects[idx].fillColor = [amp * 2 - 1] * 3
                rects[idx].lineColor = rects[idx].fillColor
                for r_i, r in enumerate(rects):
                    if r_i == idx:
                        r.draw()
                win.flip()
                if "escape" in event.getKeys():
                    raise KeyboardInterrupt()
            send_trigger(0x10 + idx + 1)  # 目标ID触发
            # 间隔 1s
            wait_t = max(0.0, 1.0 - (time.time() - start - stim_time))
            if wait_t > 0:
                core.wait(wait_t)
        send_trigger(0x02)  # 结束
    except KeyboardInterrupt:
        send_trigger(0xFF)
    finally:
        try:
            win.close()
        except Exception:
            pass
        if trig is not None:
            try:
                trig.close()
            except Exception:
                pass


