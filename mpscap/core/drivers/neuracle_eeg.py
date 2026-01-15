from __future__ import annotations

import socket
import threading
import time
from collections import deque
from typing import Deque, Optional

import numpy as np

from ..data_pipeline.kafka_producer import KafkaConfig, KafkaDataProducer


class NeuracleEEGClient(threading.Thread):
    """Neuracle EEG TCP 数据接收（参考 examples/dev/dev/dataserverneuracle.py）。"""

    def __init__(
        self,
        dev_addr: str = "127.0.0.1",
        dev_port: int = 8712,
        srate: int = 1000,
        eeg_chs: int = 64,
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
        self._buf = b""
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

    def run(self) -> None:
        if not self.connect():
            return
        n_chan = self.eeg_chs + (1 if self.include_trigger else 0)
        bytes_per_sample = 4 * n_chan

        while not self._stop.is_set():
            try:
                raw = self._sock.recv(4096)
                if not raw:
                    time.sleep(0.01)
                    continue
                self._buf += raw
            except socket.timeout:
                continue
            except OSError:
                break

            if len(self._buf) < bytes_per_sample:
                continue

            samples = len(self._buf) // bytes_per_sample
            take = samples * bytes_per_sample
            chunk = self._buf[:take]
            self._buf = self._buf[take:]

            try:
                arr = np.frombuffer(chunk, dtype=np.float32).reshape(samples, n_chan)
            except Exception:
                continue

            if self.include_trigger:
                eeg = arr[:, : self.eeg_chs]
                trig = arr[:, -1]
                trig_val = int(trig[-1]) if trig.size else 0
                if trig_val:
                    self._trigger_q.append(trig_val)
            else:
                eeg = arr

            eeg_t = eeg.T.astype(np.float64, copy=False)
            self._data_q.append(eeg_t)

            if self._kafka is not None:
                try:
                    self._kafka.publish_data(eeg_t, device_id="neuracle_eeg")
                except Exception:
                    pass

        self.stop()

