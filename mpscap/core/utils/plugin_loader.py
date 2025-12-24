"""
通用插件加载工具，帮助动态接入设备、范式、算法。
"""

from __future__ import annotations

import importlib
from typing import Any


def load_class(path: str) -> Any:
    """
    根据字符串路径加载类。

    :param path: 例如 ``drivers.tobii.TobiiAdapter``。
    """
    module_path, _, class_name = path.rpartition(".")
    if not module_path:
        raise ValueError(f"invalid class path: {path}")
    module = importlib.import_module(module_path)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"{class_name} not found in {module_path}") from exc

