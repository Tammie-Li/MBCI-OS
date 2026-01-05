"""
读取触发文件：
- TRIGGER_*.dat（float64，列: trigger, t_rel_sec）
- *.trig（int32，每采样一个触发值，需提供采样率估算时间）

示例：
python scripts/read_trigger.py TRIGGER_20250105_120000.dat --limit 20
python scripts/read_trigger.py data.trig --srate 500 --limit 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_dat(path: Path) -> np.ndarray:
    """读取 TRIGGER_*.dat，返回 (N,2) float64，[trigger, t_rel_s]。"""
    arr = np.fromfile(path, dtype=np.float64)
    if arr.size % 2 != 0:
        raise ValueError("数据长度不是偶数，文件可能损坏或格式不符。")
    return arr.reshape(-1, 2)


def load_trig(path: Path, srate: float | None) -> tuple[np.ndarray, np.ndarray | None]:
    """
    读取 .trig：
    - 优先尝试按 float64 成对读取（timestamp, trigger）。
    - 若长度与8对齐但无法成对，或无法解析，则回退为 int32。
    """
    raw = np.fromfile(path, dtype=np.float64)
    if raw.size >= 2 and raw.size % 2 == 0:
        arr = raw.reshape(-1, 2)
        t = arr[:, 0]
        trig = arr[:, 1]
        return trig, t
    # 回退 int32
    arr_i = np.fromfile(path, dtype=np.int32)
    t = None
    if srate and srate > 0 and arr_i.size > 0:
        t = np.arange(arr_i.size, dtype=np.float64) / float(srate)
    return arr_i, t


def main() -> None:
    parser = argparse.ArgumentParser(description="读取触发文件（.dat 或 .trig）")
    parser.add_argument("file", type=str, help="触发文件路径（TRIGGER_*.dat 或 *.trig）")
    parser.add_argument("--limit", type=int, default=20, help="打印前 N 条（默认 20，<=0 表示全部）")
    parser.add_argument("--srate", type=float, default=None, help=".trig 对应的采样率，用于估算时间轴（秒）")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    suffix = path.suffix.lower()
    if suffix == ".dat":
        data = load_dat(path)
        mask = data[:, 0] != 0
        data = data[mask]
        total = data.shape[0]
        print(f"文件: {path} (TRIGGER dat，仅输出非零触发)")
        print(f"非零触发数: {total}")
        if total == 0:
            return
        limit = args.limit
        if limit <= 0 or limit > total:
            limit = total
        print(f"前 {limit} 条非零（trigger, t_rel_sec）：")
        for i in range(limit):
            trig, t_rel = data[i]
            print(f"{i+1:4d}: {trig:8.0f}  {t_rel:10.3f}")
    elif suffix == ".trig":
        trig, t = load_trig(path, args.srate)
        nz_idx = np.nonzero(trig)[0]
        total = nz_idx.size
        print(f"文件: {path} (.trig int32, 仅输出非零触发)")
        print(f"非零触发数: {total}")
        if total == 0:
            return
        limit = args.limit
        if limit <= 0 or limit > total:
            limit = total
        if t is not None:
            print(f"前 {limit} 条非零（index, trigger, t_sec @ srate={args.srate}）：")
            for k in range(limit):
                i = nz_idx[k]
                print(f"{k+1:4d}: {trig[i]:8d}  {t[i]:10.3f}")
        else:
            print(f"前 {limit} 条非零（index, trigger），未提供采样率，时间轴省略：")
            for k in range(limit):
                i = nz_idx[k]
                print(f"{k+1:4d}: {trig[i]:8d} (idx={i})")
    else:
        raise ValueError(f"不支持的后缀: {suffix}，仅支持 .dat 或 .trig")


if __name__ == "__main__":
    main()

