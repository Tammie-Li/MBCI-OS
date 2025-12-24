"""
离线数据转换器集合，支持设备驱动自行注册。
"""

from .base import BaseConverter
from .emg_dat import EmgDatConverter
from .edf import EdfConverter

__all__ = ["BaseConverter", "EmgDatConverter", "EdfConverter"]

