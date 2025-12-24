"""
MPSCAP: Multi-device Physiological Signal Cross-platform Analysis Platform.

This package exposes only high-level factories for the desktop application.
具体业务逻辑位于 core 与 ui 子包中。
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mpscap")
except PackageNotFoundError:  # pragma: no cover - 未打包情况下获取失败
    __version__ = "0.0.0-dev"

__all__ = ["__version__"]

