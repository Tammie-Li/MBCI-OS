"""
实时数据订阅工具，负责桥接 MQTT 数据与上层 UI/算法模块。
"""

from __future__ import annotations

from typing import Callable, List

from .mqtt_gateway import MQTTGateway, MQTTConfig
from .protocol import SignalFrame


FrameCallback = Callable[[SignalFrame], None]


class SignalStreamHub:
    """简化 MQTT 订阅，提供观察者接口。"""

    def __init__(
        self,
        gateway: MQTTGateway,
        topic_filter: str | None = None,
    ) -> None:
        self._gateway = gateway
        self._topic_filter = topic_filter or f"{gateway.config.base_topic}/#"
        self._subscribers: List[FrameCallback] = []
        self._started = False

    def subscribe(self, callback: FrameCallback) -> None:
        self._subscribers.append(callback)

    def start(self) -> None:
        if self._started:
            return
        self._gateway.connect()
        self._gateway.subscribe(self._topic_filter, self._on_frame)
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._gateway.disconnect()
        self._started = False

    def _on_frame(self, frame: SignalFrame) -> None:
        for callback in self._subscribers:
            callback(frame)


def build_default_stream() -> SignalStreamHub:
    """供快速集成使用的默认流。"""

    gateway = MQTTGateway(MQTTConfig())
    return SignalStreamHub(gateway)

