"""
Tobii 眼动仪数据获取（参考 Eye_Tracker/core.py）。
"""

from __future__ import annotations

import socket
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Tuple


class TobiiEyeTracker:
    """Tobii 4C 眼动数据获取。

    - 启动 `tobii_4c_app.exe`（来自 Eye_Tracker/data_files）作为数据源。
    - 通过 UDP 接收 (x, y) 归一化坐标，提供平滑后的最新坐标。
    """

    def __init__(
        self,
        filter_len: int = 5,
        user_port: int = 5150,
        user_ip: str = "127.0.0.1",
        tobii_port: int = 5151,
        tobii_ip: str = "127.0.0.1",
    ) -> None:
        self._worker = _TobiiWorker(
            filter_len=filter_len,
            user_port=user_port,
            user_ip=user_ip,
            tobii_port=tobii_port,
            tobii_ip=tobii_ip,
        )

    def latest_xy(self) -> Tuple[float, float]:
        """获取最新的归一化坐标 (0~1)。无效时返回 (-1, -1)。"""
        return self._worker.latest_xy()

    def stop(self) -> None:
        """停止采集并关闭子进程。"""
        self._worker.stop()


class _TobiiWorker(threading.Thread):
    def __init__(
        self,
        filter_len: int,
        user_port: int,
        user_ip: str,
        tobii_port: int,
        tobii_ip: str,
    ) -> None:
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._latest: Tuple[float, float] = (-1.0, -1.0)
        self._quit = False

        root = Path(__file__).resolve().parents[3]  # .../MPSCAP
        data_dir = root / "Eye_Tracker" / "data_files"
        exe_path = data_dir / "tobii_4c_app.exe"
        if not exe_path.exists():
            raise FileNotFoundError(f"未找到眼动服务程序: {exe_path}")

        # 启动眼动采集服务（隐藏窗口）
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen(
            [
                str(exe_path),
                str(user_port),
                user_ip,
                str(tobii_port),
                tobii_ip,
            ],
            cwd=str(data_dir),
            creationflags=creation_flags,
        )

        self._filter_len = max(1, filter_len)
        self._xs = [0.0] * self._filter_len
        self._ys = [0.0] * self._filter_len

        self._user_addr = (user_ip, user_port)
        self._tobii_addr = (tobii_ip, tobii_port)

        self.start()

    def latest_xy(self) -> Tuple[float, float]:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._quit = True
        # 发送一次退出信号给 tobii 服务端
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(b"\x01\x03\x7d\x7f", self._tobii_addr)
            sock.close()
        except Exception:
            pass

        # 等待线程退出
        self.join(timeout=1.0)

        # 终止子进程
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
        sock.bind(self._user_addr)

        while not self._quit:
            try:
                buf, _ = sock.recvfrom(128)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                x, y = struct.unpack("2f", buf)
            except Exception:
                continue

            self._xs.append(x)
            self._ys.append(y)
            self._xs.pop(0)
            self._ys.pop(0)

            with self._lock:
                self._latest = (
                    sum(self._xs) / self._filter_len,
                    sum(self._ys) / self._filter_len,
                )

        try:
            sock.close()
        except Exception:
            pass






