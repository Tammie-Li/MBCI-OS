"""
结果呈现子系统。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type


class FeedbackBase(ABC):
    name: str

    @abstractmethod
    def present(self, payload: Dict) -> None: ...


class FeedbackRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, Type[FeedbackBase]] = {}

    def register(self, feedback_cls: Type[FeedbackBase]) -> None:
        self._registry[feedback_cls.name] = feedback_cls

    def create(self, name: str) -> FeedbackBase:
        try:
            cls = self._registry[name]
        except KeyError as exc:
            raise KeyError(f"feedback {name} not registered") from exc
        return cls()

