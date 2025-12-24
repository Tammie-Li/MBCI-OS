"""
MQTT 网关封装，支持多设备数据的发布与订阅。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from paho.mqtt import client as mqtt

from .protocol import SignalFrame


DEFAULT_QOS = 1


@dataclass
class MQTTConfig:
    host: str = "127.0.0.1"
    port: int = 1883
    username: Optional[str] = None
    password: Optional[str] = None
    keepalive: int = 60
    base_topic: str = "mpscap/signals"


class MQTTGateway:
    """负责生产者/消费者的统一封装。"""

    def __init__(self, config: MQTTConfig) -> None:
        self.config = config
        self._client = mqtt.Client()
        if config.username:
            self._client.username_pw_set(config.username, config.password or "")

    def connect(self) -> None:
        self._client.connect(
            self.config.host,
            self.config.port,
            keepalive=self.config.keepalive,
        )
        self._client.loop_start()

    def disconnect(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    def publish_signal(self, frame: SignalFrame, topic_suffix: str) -> None:
        import json

        payload = json.dumps(frame.as_dict()).encode("utf-8")
        topic = f"{self.config.base_topic}/{topic_suffix}"
        self._client.publish(topic, payload, qos=DEFAULT_QOS)

    def subscribe(
        self,
        topic_filter: str,
        handler: Callable[[SignalFrame], None],
    ) -> None:
        def _on_message(_client, _userdata, message):  # type: ignore[override]
            handler(self._deserialize(message.payload))

        self._client.subscribe(topic_filter, qos=DEFAULT_QOS)
        self._client.message_callback_add(topic_filter, _on_message)

    @staticmethod
    def _deserialize(payload: bytes) -> SignalFrame:
        import json
        from .protocol import SignalPacketHeader  # 避免循环导入
        import numpy as np

        data = json.loads(payload.decode("utf-8"))
        header = SignalPacketHeader(**data["header"])
        frame = SignalFrame(
            header=header,
            data=np.asarray(data["data"]),
            annotations=data.get("annotations", {}),
        )
        return frame

