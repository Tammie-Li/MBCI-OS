from __future__ import annotations

"""
范式子进程入口。
- 为 SSVEP / 手势范式提供独立进程运行，避免阻塞主界面。
- 可选：在子进程中开启 Kafka 订阅（仅拉取数据，默认不做处理）。
"""

import os
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from .gesture_demo import run_gesture_sequence
from .eye_target_demo import run_eye_target_demo
from .eye_4class_demo import run_eye_4class_demo
from .ssvep_demo import run_ssvep_demo
from .rsvp_demo import run_rsvp_demo
from ..core.utils.shm import CreateShm

try:
    from kafka import KafkaConsumer

    KAFKA_AVAILABLE = True
except Exception:
    KafkaConsumer = None
    KAFKA_AVAILABLE = False


_dpi_fixed = False
TRIGGER_UDP_HOST = os.environ.get("MPSCAP_TRIGGER_UDP_HOST", "127.0.0.1")
TRIGGER_UDP_PORT = int(os.environ.get("MPSCAP_TRIGGER_UDP_PORT", "15000") or 15000)


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


def _show_instruction_psychopy(text: str, duration: float = 3.0) -> None:
    """在启动正式范式前，弹出基于 PsychoPy 的提示窗口，窗口固定 1920x1080。"""
    try:
        from psychopy import visual, core
    except Exception as e:
        print(f"[PsychoPy] 无法显示提示窗口: {e}")
        return

    win = None
    try:
        _make_process_dpi_aware()
        # 强制 1920x1080 窗口，禁用全屏；checkTiming=False 避免某些显卡触发警告退出
        win = visual.Window(
            size=(1920, 1080),
            units="pix",
            color=(-1, -1, -1),
            fullscr=False,
            allowGUI=True,
            checkTiming=False,
            winType="pyglet",
        )
        msg = visual.TextStim(win=win, text=text, color="white", height=36, wrapWidth=1500)
        msg.draw()
        win.flip()
        core.wait(max(1.0, duration))
    except Exception as e:
        print(f"[PsychoPy] 提示窗口显示失败: {e}")
    finally:
        if win is not None:
            try:
                win.close()
            except Exception:
                pass


def _send_trigger(code: int) -> None:
    """将 trigger 写入共享内存，供采集端落盘/前端显示。"""
    try:
        shm = CreateShm(master=False)
        shm.setvalue('includetrigger', int(code))
    except Exception as e:
        print(f"[Trigger] 写入触发失败: {e}")
    _send_trigger_udp(code)


def _send_trigger_udp(code: int) -> None:
    """辅助：通过 UDP 本地端口发送触发，作为共享内存的备选路径。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(str(int(code)).encode(), (TRIGGER_UDP_HOST, TRIGGER_UDP_PORT))
    except Exception as e:
        print(f"[Trigger UDP] 发送失败: {e}")


class TriggerFileLogger:
    """简单的触发文件记录器：每次记录绝对时间和触发值，UTF-8 文本。"""

    def __init__(self, paradigm_name: Optional[str], save_dir: Optional[str]) -> None:
        safe = paradigm_name or "paradigm"
        safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in safe).strip("_")
        if not safe:
            safe = "paradigm"
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(save_dir) if save_dir else Path.cwd()
        base_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{safe}_{ts_tag}_trigger.txt"
        self.path = base_dir / fname
        self._lock = threading.Lock()
        try:
            self._fh = open(self.path, "a", encoding="utf-8")
        except Exception as e:
            print(f"[TriggerFileLogger] 无法创建文件 {self.path}: {e}")
            self._fh = None

    def log(self, code: int) -> None:
        if self._fh is None:
            return
        try:
            t = time.time()
            line = f"{t:.6f}\t{int(code)}\n"
            with self._lock:
                self._fh.write(line)
                self._fh.flush()
        except Exception as e:
            print(f"[TriggerFileLogger] 写入失败: {e}")

    def close(self) -> None:
        if self._fh:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None


def _start_kafka_consumer(
    bootstrap_servers: Optional[str],
    topic: Optional[str],
    group_id: str = "paradigm-consumer",
) -> Tuple[Optional["KafkaConsumer"], Optional[threading.Event]]:
    """可选启动 Kafka consumer，仅订阅数据以保持与采集端一致的通路。"""
    if not (bootstrap_servers and topic and KAFKA_AVAILABLE):
        return None, None

    stop_evt = threading.Event()

    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=[s.strip() for s in bootstrap_servers.split(",")],
            group_id=group_id,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            consumer_timeout_ms=1000,
        )
    except Exception as e:
        print(f"[Kafka] 启动 consumer 失败: {e}")
        return None, None

    def _loop():
        while not stop_evt.is_set():
            try:
                consumer.poll(timeout_ms=500)
            except Exception as err:
                print(f"[Kafka] consumer 轮询异常: {err}")
                break
        try:
            consumer.close()
        except Exception:
            pass

    threading.Thread(target=_loop, daemon=True).start()
    return consumer, stop_evt


def _stop_kafka_consumer(consumer: Optional["KafkaConsumer"], stop_evt: Optional[threading.Event]) -> None:
    if stop_evt is not None:
        stop_evt.set()
    if consumer is not None:
        try:
            consumer.wakeup()
        except Exception:
            pass


def run_ssvep_worker(
    freqs: Optional[List[float]] = None,
    trigger_com: Optional[str] = None,
    kafka_bootstrap: Optional[str] = None,
    kafka_topic: Optional[str] = None,
    paradigm_name: Optional[str] = None,
    save_dir: Optional[str] = None,
    cycles: int = 1,
    stim_duration: float = 4.0,
) -> None:
    """独立进程运行 SSVEP 范式，可选开启 Kafka 订阅。"""
    logger = TriggerFileLogger(paradigm_name or "SSVEP", save_dir)

    def _send(code: int) -> None:
        _send_trigger(code)
        logger.log(code)

    _show_instruction_psychopy("即将开始 SSVEP 实验，请专注观看提示的频闪目标。")
    consumer, evt = _start_kafka_consumer(kafka_bootstrap, kafka_topic)
    try:
        _send(254)  # 实验开始
        for c in range(max(1, cycles)):
            _send(252)  # 轮开始
            run_ssvep_demo(freqs=freqs, cycles=1, stim_duration=stim_duration, trigger_cb=_send)
            _send(253)  # 轮结束
    finally:
        _send(255)  # 实验结束
        logger.close()
        _stop_kafka_consumer(consumer, evt)


def run_rsvp_worker(
    target_dir: str,
    nontarget_dir: str,
    cycles: int,
    kafka_bootstrap: Optional[str] = None,
    kafka_topic: Optional[str] = None,
    paradigm_name: Optional[str] = None,
    save_dir: Optional[str] = None,
    stim_freq: float = 10.0,
) -> None:
    """独立进程运行 RSVP 范式。"""
    logger = TriggerFileLogger(paradigm_name or "RSVP", save_dir)

    def _send(code: int) -> None:
        _send_trigger(code)
        logger.log(code)

    consumer, evt = _start_kafka_consumer(kafka_bootstrap, kafka_topic)
    try:
        _send(254)  # 实验开始
        for _ in range(max(1, cycles)):
            _send(252)  # 轮开始
            run_rsvp_demo(
                target_dir=Path(target_dir),
                nontarget_dir=Path(nontarget_dir),
                cycles=1,
                target_code=2,
                nontarget_code=1,
                trigger_cb=_send,
                stim_freq=stim_freq,
            )
            _send(253)  # 轮结束
    finally:
        _send(255)  # 实验结束
        logger.close()
        _stop_kafka_consumer(consumer, evt)


def run_gesture_worker(
    imgs: List[str],
    names: List[str],
    show_sec: float,
    rest_between_gestures: float,
    rest_between_cycles: float,
    cycles: int,
    kafka_bootstrap: Optional[str] = None,
    kafka_topic: Optional[str] = None,
    paradigm_name: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> None:
    """独立进程运行手势范式，可选开启 Kafka 订阅。"""
    logger = TriggerFileLogger(paradigm_name or "gesture", save_dir)

    def _send(code: int) -> None:
        _send_trigger(code)
        logger.log(code)

    _show_instruction_psychopy("即将开始手势实验，请按屏幕提示依次完成手势动作。")
    consumer, evt = _start_kafka_consumer(kafka_bootstrap, kafka_topic)
    try:
        img_paths = [Path(p) for p in imgs]
        run_gesture_sequence(
            imgs=img_paths,
            names=names,
            show_sec=show_sec,
            rest_between_gestures=rest_between_gestures,
            rest_between_cycles=rest_between_cycles,
            cycles=cycles,
            progress_cb=None,
            trigger_cb=_send,
        )
    finally:
        logger.close()
        _stop_kafka_consumer(consumer, evt)


def run_eye_target_worker(
    target_count: int = 10,
    dwell_time_sec: float = 0.8,
    layout: str = "random",
    kafka_bootstrap: Optional[str] = None,
    kafka_topic: Optional[str] = None,
    paradigm_name: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> None:
    """独立进程运行眼动目标消除范式。"""
    logger = TriggerFileLogger(paradigm_name or "Eye_Target", save_dir)

    def _send(code: int) -> None:
        _send_trigger(code)
        logger.log(code)

    consumer, evt = _start_kafka_consumer(kafka_bootstrap, kafka_topic)
    try:
        _send(254)  # 实验开始
        _send(252)  # 轮开始（眼动范式只有一轮）
        run_eye_target_demo(
            target_count=target_count,
            dwell_time_sec=dwell_time_sec,
            layout=layout,
            trigger_cb=_send,
            save_dir=save_dir,
        )
    finally:
        _send(253)  # 轮结束
        _send(255)  # 实验结束
        logger.close()
        _stop_kafka_consumer(consumer, evt)


def run_eye_4class_worker(
    trials: int = 200,
    phase_sec: float = 3.0,
    ring_sec: float = 3.0,
    kafka_bootstrap: Optional[str] = None,
    kafka_topic: Optional[str] = None,
    paradigm_name: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> None:
    """独立进程运行眼动四分类范式。"""
    logger = TriggerFileLogger(paradigm_name or "Eye_4Class", save_dir)

    def _send(code: int) -> None:
        _send_trigger(code)
        logger.log(code)

    consumer, evt = _start_kafka_consumer(kafka_bootstrap, kafka_topic)
    try:
        _send(254)  # 实验开始
        _send(252)  # 轮开始（连续 trials 次）
        run_eye_4class_demo(
            trials=trials,
            phase_sec=phase_sec,
            ring_sec=ring_sec,
            trigger_cb=_send,
        )
    finally:
        _send(253)  # 轮结束
        _send(255)  # 实验结束
        logger.close()
        _stop_kafka_consumer(consumer, evt)


if __name__ == "__main__":  # pragma: no cover
    # 简单自测：默认参数下启动手势范式
    run_gesture_worker(
        imgs=[],
        names=["手势1", "手势2"],
        show_sec=1.0,
        rest_between_gestures=1.0,
        rest_between_cycles=2.0,
        cycles=1,
        paradigm_name="gesture_test",
    )

