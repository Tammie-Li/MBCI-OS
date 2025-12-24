"""
批量将离线数据转换为标准格式。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mpscap.core.data_pipeline.protocol import ConverterRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线数据转换工具")
    parser.add_argument("--converter", required=True, help="转换器名称")
    parser.add_argument("--inputs", nargs="+", required=True, help="原始文件路径")
    parser.add_argument("--output", required=True, help="输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = ConverterRegistry()
    converter = registry.get(args.converter)
    converter.convert([Path(p) for p in args.inputs], Path(args.output))


if __name__ == "__main__":
    main()

