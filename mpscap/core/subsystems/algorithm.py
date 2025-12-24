"""
算法子系统：统一算法接口与注册机制。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type


class AlgorithmBase(ABC):
    name: str
    modalities: list[str]

    @abstractmethod
    def configure(self, **kwargs) -> None: ...

    @abstractmethod
    def fit(self, data, labels) -> None: ...

    @abstractmethod
    def predict(self, data): ...


class AlgorithmRegistry:
    def __init__(self) -> None:
        self._algorithms: Dict[str, Type[AlgorithmBase]] = {}

    def register(self, algo_cls: Type[AlgorithmBase]) -> None:
        self._algorithms[algo_cls.name] = algo_cls

    def create(self, name: str, **kwargs) -> AlgorithmBase:
        try:
            algo_cls = self._algorithms[name]
        except KeyError as exc:
            raise KeyError(f"algorithm {name} not registered") from exc
        algo = algo_cls()
        algo.configure(**kwargs)
        return algo

    def list_available(self) -> list[str]:
        return sorted(self._algorithms.keys())

