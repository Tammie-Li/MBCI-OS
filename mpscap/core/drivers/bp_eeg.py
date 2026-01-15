from __future__ import annotations

import socket
import struct
import threading
import time
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from ..data_pipeline.kafka_producer import KafkaConfig, KafkaDataProducer

BPresolution = 0.0488


class BPEEGClient(threading.Thread):
    """BP/BrainProducts RDA 协议接收（参考 examples/dev/dev/dataserverbp.py）。"""

    def __init__(
        self,
        dev_addr: str = "127.0.0.1",
        dev_port: int = 51244,
        srate: int = 500,
        eeg_chs: int = 32,
        include_trigger: bool = True,
        kafka_config: Optional[KafkaConfig] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.dev_addr = dev_addr
        self.dev_port = dev_port
        self.srate = srate
        self.eeg_chs = eeg_chs
        self.include_trigger = include_trigger
        self._sock: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._data_q: Deque[np.ndarray] = deque(maxlen=200)
        self._trigger_q: Deque[int] = deque(maxlen=200)
        self._kafka: Optional[KafkaDataProducer] = None
        if kafka_config is not None:
            try:
                producer = KafkaDataProducer(kafka_config)
                if producer.connect():
                    self._kafka = producer
            except Exception:
                self._kafka = None

    def connect(self) -> bool:
        if self._sock is not None:
            return True
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((self.dev_addr, self.dev_port))
            sock.settimeout(1.0)
            self._sock = sock
            return True
        except Exception:
            self._sock = None
            return False

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._kafka is not None:
            try:
                self._kafka.disconnect()
            except Exception:
                pass
            self._kafka = None

    def pop_data(self) -> list[np.ndarray]:
        items: list[np.ndarray] = []
        while self._data_q:
            items.append(self._data_q.popleft())
        return items

    def pop_trigger(self) -> Optional[int]:
        val = None
        while self._trigger_q:
            val = self._trigger_q.popleft()
        return val

    def set_kafka_config(self, kafka_config: Optional[KafkaConfig]) -> None:
        if self._kafka is not None:
            try:
                self._kafka.disconnect()
            except Exception:
                pass
            self._kafka = None
        if kafka_config is None:
            return
        try:
            producer = KafkaDataProducer(kafka_config)
            if producer.connect():
                self._kafka = producer
        except Exception:
            self._kafka = None

    def _recv_exact(self, size: int) -> Optional[bytes]:
        if self._sock is None:
            return None
        data = b""
        while len(data) < size and not self._stop.is_set():
            try:
                chunk = self._sock.recv(size - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                continue
            except OSError:
                return None
        return data

    @staticmethod
    def _split_string(raw: bytes) -> list[str]:
        items = []
        s = b""
        for i in range(len(raw)):
            if raw[i:i + 1] != b"\x00":
                s += raw[i:i + 1]
            else:
                if s:
                    try:
                        items.append(s.decode())
                    except Exception:
                        items.append("")
                s = b""
        return items

    def _parse_properties(self, raw: bytes) -> Tuple[int, float]:
        channel_count, sampling_interval = struct.unpack("<Ld", raw[:12])
        return int(channel_count), float(sampling_interval)

    def _parse_eeg(self, raw: bytes, channel_count: int) -> Tuple[int, int, np.ndarray, np.ndarray]:
        block, points, marker_count = struct.unpack("<LLL", raw[:12])
        data = np.frombuffer(raw[12:12 + 4 * points * channel_count], dtype=np.float32)
        data = data.reshape(points, channel_count) * BPresolution

        marker = np.zeros((points, 1), dtype=np.float32)
        index = 12 + 4 * points * channel_count
        for _ in range(marker_count):
            marker_size = struct.unpack("<L", raw[index:index + 4])[0]
            position, m_points, m_channel = struct.unpack("<LLl", raw[index + 4:index + 16])
            desc = self._split_string(raw[index + 16:index + marker_size])
            try:
                marker_val = float(desc[1][1:])
            except Exception:
                marker_val = 0.0
            if 0 <= position < points:
                marker[position, 0] = marker_val
            index += marker_size

        eeg = np.hstack((data, marker)) if self.include_trigger else data
        return int(block), int(points), eeg, marker

    def run(self) -> None:
        if not self.connect():
            return
        channel_count = 0

        while not self._stop.is_set():
            hdr = self._recv_exact(24)
            if not hdr:
                break
            try:
                _, _, _, _, msg_size, msg_type = struct.unpack("<llllLL", hdr)
            except Exception:
                break
            payload = self._recv_exact(msg_size - 24)
            if not payload:
                break

            if msg_type == 1:
                channel_count, _ = self._parse_properties(payload)
                if channel_count > 0:
                    self.eeg_chs = channel_count
            elif msg_type == 4 and channel_count > 0:
                _, _, eeg, marker = self._parse_eeg(payload, channel_count)
                eeg_t = eeg.T.astype(np.float64, copy=False)
                self._data_q.append(eeg_t)
                if marker.size > 0:
                    trig = int(marker[-1][0])
                    if trig:
                        self._trigger_q.append(trig)
                if self._kafka is not None:
                    try:
                        self._kafka.publish_data(eeg_t, device_id="bp_eeg")
                    except Exception:
                        pass
            elif msg_type == 3:
                break

        self.stop()

