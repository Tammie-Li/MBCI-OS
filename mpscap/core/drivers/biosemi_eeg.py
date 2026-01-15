from __future__ import annotations

import socket
import threading
import time
from collections import deque
from typing import Deque, Optional

import numpy as np

from ..data_pipeline.kafka_producer import KafkaConfig, KafkaDataProducer


class BiosemiEEGClient(threading.Thread):
    """Biosemi TCP 数据接收（参考 examples/dev/dev/biosemi.py）。"""

    def __init__(
        self,
        dev_addr: str = "127.0.0.1",
        dev_port: int = 1111,
        srate: int = 1024,
        eeg_chs: int = 65,
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

        self._tcpsamples = 4
        self._buffer_size = self.eeg_chs * self._tcpsamples * 3

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

    def run(self) -> None:
        if not self.connect():
            return
        while not self._stop.is_set():
            try:
                data = b""
                while len(data) != self._buffer_size and not self._stop.is_set():
                    chunk = self._sock.recv(self._buffer_size - len(data))
                    if not chunk:
                        time.sleep(0.01)
                        continue
                    data += chunk
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) != self._buffer_size:
                continue

            signal_buffer = np.zeros((self.eeg_chs, self._tcpsamples), dtype=np.float32)
            for m in range(self._tcpsamples):
                for ch in range(self.eeg_chs):
                    offset = m * 3 * self.eeg_chs + (ch * 3)
                    sample = (data[offset + 2] << 16) + (data[offset + 1] << 8) + data[offset]
                    signal_buffer[ch, m] = sample

            eeg_t = signal_buffer.astype(np.float64, copy=False)
            self._data_q.append(eeg_t)
            if self.include_trigger and self.eeg_chs > 0:
                trig = int(signal_buffer[-1, -1])
                if trig:
                    self._trigger_q.append(trig)
            if self._kafka is not None:
                try:
                    self._kafka.publish_data(eeg_t, device_id="biosemi_eeg")
                except Exception:
                    pass

        self.stop()

