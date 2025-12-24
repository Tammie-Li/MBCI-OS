"""
转换器基类与注册辅助。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from ..protocol import MetaInfo


class BaseConverter(ABC):
    """将原始文件转换成统一格式。"""

    name: str
    source_extensions: List[str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "name", None):
            raise ValueError("converter must define name")

    @abstractmethod
    def convert(self, input_paths: List[Path], output_dir: Path) -> MetaInfo:
        """执行转换流程。"""

