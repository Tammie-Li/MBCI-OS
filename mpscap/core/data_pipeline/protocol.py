"""
生理信号数据协议定义。

该模块统一描述多模态、多设备信号的数据格式，便于在线/离线流程共享。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Literal, Optional, Protocol


ModalityType = Literal["EEG", "EMG", "EOG", "EYE", "FNIRS", "ECG", "UNKNOWN"]


@dataclass(slots=True)
class MetaInfo:
    """离线数据描述信息。"""

    subject_id: str
    modality: ModalityType
    channels: int
    sample_rate: float
    duration_seconds: float
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SignalPacketHeader:
    """在线数据包头信息。"""

    modality: ModalityType
    device_id: str
    channel_count: int
    samples_per_channel: int
    sample_rate: float
    timestamp: float
    has_trigger_channel: bool = True
    has_timestamp_channel: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SignalFrame:
    """
    单帧数据结构，承载包头与矩阵数据。

    data 矩阵为 C x T，使用 float32 / int16 等 numpy dtype。
    """

    header: SignalPacketHeader
    data: "np.ndarray"  # noqa: UP037 - 延迟导入 numpy
    annotations: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典。"""
        # 使用 asdict 处理 slots=True 的 dataclass
        header_dict = asdict(self.header)
        return {
            "header": header_dict,
            "data": self.data.tolist(),
            "annotations": self.annotations,
        }


class Converter(Protocol):
    """离线数据格式转换器接口。"""

    source_formats: Iterable[str]

    def convert(self, input_paths: List[str], output_dir: str) -> MetaInfo:
        """实现具体转换逻辑。"""


class ConverterRegistry:
    """管理多设备离线转换器。"""

    def __init__(self) -> None:
        self._converters: Dict[str, Converter] = {}

    def register(self, name: str, converter: Converter) -> None:
        if name in self._converters:
            raise ValueError(f"converter {name} already registered")
        self._converters[name] = converter

    def get(self, name: str) -> Converter:
        try:
            return self._converters[name]
        except KeyError as exc:
            raise KeyError(f"converter {name} not found") from exc

    def list_available(self) -> List[str]:
        return sorted(self._converters.keys())


# 延迟导入 numpy 以减少强制依赖
try:  # pragma: no cover - 运行时导入
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

