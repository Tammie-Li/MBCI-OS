"""
任务子系统：负责实验流程编排与状态管理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


class TaskState(Enum):
    IDLE = auto()
    PREPARING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class TaskConfig:
    name: str
    paradigm: str
    devices: List[str]
    algorithm: str
    preprocessors: List[str] = field(default_factory=list)
    feedback: str = "visual"


class TaskManager:
    def __init__(self) -> None:
        self._current: Optional[TaskConfig] = None
        self._state = TaskState.IDLE
        self._listeners: List = []

    def load(self, config: TaskConfig) -> None:
        self._current = config
        self._state = TaskState.PREPARING
        self._notify()

    def start(self) -> None:
        self._state = TaskState.RUNNING
        self._notify()

    def pause(self) -> None:
        self._state = TaskState.PAUSED
        self._notify()

    def complete(self) -> None:
        self._state = TaskState.COMPLETED
        self._notify()

    def fail(self, reason: str) -> None:
        self._state = TaskState.FAILED
        self._notify({"reason": reason})

    @property
    def state(self) -> TaskState:
        return self._state

    def subscribe(self, listener) -> None:
        self._listeners.append(listener)

    def _notify(self, payload: Optional[Dict] = None) -> None:
        for listener in self._listeners:
            listener(self._state, payload or {})

