"""
数据采集子系统：统一管理设备接入、解析与发布。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

from ..data_pipeline.mqtt_gateway import MQTTGateway
from ..data_pipeline.protocol import SignalFrame, SignalPacketHeader


FrameCallback = Callable[[SignalFrame], None]


class DeviceAdapter(ABC):
    """设备驱动接口。"""

    device_id: str
    modality: str

    def __init__(self, mqtt_gateway: MQTTGateway):
        self.mqtt_gateway = mqtt_gateway
        self._on_frame: Optional[FrameCallback] = None

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def start_stream(self, on_frame: FrameCallback) -> None: ...

    @abstractmethod
    def stop_stream(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    def _publish(self, frame: SignalFrame) -> None:
        topic_suffix = f"{self.modality.lower()}/{self.device_id}"
        self.mqtt_gateway.publish_signal(frame, topic_suffix)
        if self._on_frame:
            self._on_frame(frame)


class DeviceRegistry:
    """通过配置动态加载驱动。"""

    def __init__(self) -> None:
        self._drivers: Dict[str, Callable[[MQTTGateway, Dict], DeviceAdapter]] = {}

    def register(self, name: str, factory: Callable[[MQTTGateway, Dict], DeviceAdapter]) -> None:
        self._drivers[name] = factory

    def create(self, name: str, mqtt_gateway: MQTTGateway, config: Dict) -> DeviceAdapter:
        try:
            factory = self._drivers[name]
        except KeyError as exc:
            raise KeyError(f"device driver {name} not registered") from exc
        return factory(mqtt_gateway, config)


class AcquisitionSubsystem:
    """统一管理多个设备的采集状态。"""

    def __init__(self, registry: DeviceRegistry):
        self.registry = registry
        self._devices: Dict[str, DeviceAdapter] = {}

    def attach_device(self, name: str, driver: DeviceAdapter) -> None:
        self._devices[name] = driver

    def start(self, name: str, callback: Optional[FrameCallback] = None) -> None:
        device = self._require(name)
        device.connect()
        device.start_stream(callback or (lambda _: None))

    def stop(self, name: str) -> None:
        device = self._require(name)
        device.stop_stream()
        device.disconnect()

    def _require(self, name: str) -> DeviceAdapter:
        try:
            return self._devices[name]
        except KeyError as exc:
            raise KeyError(f"device {name} not attached") from exc


class MockDeviceAdapter(DeviceAdapter):
    """供调试使用的模拟设备。"""

    device_id = "mock"
    modality = "EEG"

    def connect(self) -> None:  # pragma: no cover - 调试用
        pass

    def start_stream(self, on_frame: FrameCallback) -> None:  # pragma: no cover - 调试用
        import numpy as np
        import time

        header = SignalPacketHeader(
            modality="EEG",
            device_id="mock",
            channel_count=4,
            samples_per_channel=128,
            sample_rate=256.0,
            timestamp=time.time(),
        )
        data = np.random.randn(header.channel_count, header.samples_per_channel)
        frame = SignalFrame(header=header, data=data)
        on_frame(frame)
        self._publish(frame)

    def stop_stream(self) -> None:  # pragma: no cover - 调试用
        pass

    def disconnect(self) -> None:  # pragma: no cover - 调试用
        pass

