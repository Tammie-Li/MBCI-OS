"""
范式子系统：管理多种人机交互范式及自定义扩展。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type


class ParadigmBase(ABC):
    name: str
    modalities: list[str]

    @abstractmethod
    def prepare(self, **kwargs) -> None: ...

    @abstractmethod
    def render(self) -> None: ...

    @abstractmethod
    def handle_event(self, event: Dict) -> None: ...


class ParadigmRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, Type[ParadigmBase]] = {}

    def register(self, paradigm_cls: Type[ParadigmBase]) -> None:
        self._registry[paradigm_cls.name] = paradigm_cls

    def get(self, name: str) -> Type[ParadigmBase]:
        try:
            return self._registry[name]
        except KeyError as exc:
            raise KeyError(f"paradigm {name} not found") from exc

    def list_available(self) -> list[str]:
        return sorted(self._registry.keys())

