"""
InfluxDB 与文件存储接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

from influxdb_client import InfluxDBClient, Point, WriteApi

from .protocol import SignalFrame


@dataclass
class StorageConfig:
    url: str = "http://localhost:8086"
    token: str = "mpscap-token"
    org: str = "mpscap"
    bucket: str = "signals"
    file_output: Path = Path("./data/output")


class SignalStorage:
    """同时支持 TSDB 与本地文件落地。"""

    def __init__(self, config: StorageConfig) -> None:
        self._config = config
        self._client = InfluxDBClient(
            url=config.url,
            token=config.token,
            org=config.org,
        )
        self._write_api: WriteApi = self._client.write_api()
        config.file_output.mkdir(parents=True, exist_ok=True)

    def write_frame(self, frame: SignalFrame, tags: Optional[Dict[str, str]] = None) -> None:
        point = Point("signal").time(frame.header.timestamp).field(
            "channel_count", frame.header.channel_count
        )
        if tags:
            for key, value in tags.items():
                point.tag(key, value)
        self._write_api.write(self._config.bucket, record=point)
        self._write_to_file(frame)

    def query_recent(self, limit: int = 1) -> Iterable[SignalFrame]:
        query = f'from(bucket:"{self._config.bucket}") |> range(start: -1m) |> limit(n:{limit})'
        result = self._client.query_api().query(query=query)
        return result  # TODO: 根据实际 schema 反序列化

    def _write_to_file(self, frame: SignalFrame) -> None:
        import json
        import numpy as np

        header = frame.header
        fname = (
            f"{header.device_id}_{int(header.timestamp)}_{header.modality}"
        )
        np.save(self._config.file_output / f"{fname}.npy", frame.data)
        with (self._config.file_output / f"{fname}.json").open("w", encoding="utf-8") as fp:
            json.dump(header.__dict__, fp, ensure_ascii=False, indent=2)

