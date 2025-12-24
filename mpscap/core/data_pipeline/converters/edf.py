"""
EDF/BDF 文件转换器：读取 EDF/BDF，保存为统一的 C×T numpy 数组与元信息。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from ..protocol import MetaInfo
from .base import BaseConverter


class EdfConverter(BaseConverter):
    """EDF/BDF 文件转换器。"""

    name = "edf"
    source_extensions = [".edf", ".bdf"]

    def convert(self, input_paths: List[Path], output_dir: Path) -> MetaInfo:
        if len(input_paths) != 1:
            raise ValueError("EDF/BDF converter expects exactly one input file")
        input_path = input_paths[0]
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        try:
            import pyedflib
        except ImportError as exc:  # pragma: no cover - 运行时缺依赖
            raise ImportError("未安装 pyedflib，无法读取 EDF/BDF。请执行: pip install pyedflib") from exc

        reader = pyedflib.EdfReader(str(input_path))
        try:
            chs = reader.signals_in_file
            srs = reader.getSampleFrequencies()
            if len(srs) == 0:
                raise ValueError("EDF/BDF 采样率为空")
            # 校验采样率一致性
            sr0 = float(srs[0])
            if any(abs(float(sr) - sr0) > 1e-6 for sr in srs):
                raise ValueError("当前仅支持通道采样率一致的 EDF/BDF 文件")

            nsamples = reader.getNSamples()
            total_samples = int(min(nsamples)) if len(nsamples) > 0 else 0
            if total_samples <= 0:
                raise ValueError("EDF/BDF 文件无有效样本")

            data = np.zeros((chs, total_samples), dtype=np.float32)
            for i in range(chs):
                sig = reader.readSignal(i, start=0, n=total_samples)
                data[i, :] = sig.astype(np.float32, copy=False)

            labels = reader.getSignalLabels() or []
        finally:
            try:
                reader.close()
            except Exception:
                pass

        duration = total_samples / sr0 if sr0 > 0 else 0.0

        # 清洗元信息，避免 numpy 类型导致 json 序列化失败
        labels = [str(lab) for lab in labels]
        nsamples_clean = [int(x) for x in nsamples] if nsamples is not None else []

        # 保存结果
        output_dir.mkdir(parents=True, exist_ok=True)
        data_file = output_dir / "data.npy"
        np.save(data_file, data)

        meta = MetaInfo(
            subject_id=input_path.stem,
            modality="EEG",
            channels=chs,
            sample_rate=sr0,
            duration_seconds=duration,
            extra={
                "channel_names": labels,
                "source_file": str(input_path),
                "nsamples": nsamples_clean,
            },
        )

        meta_file = output_dir / "datainfo.json"
        meta_dict = {
            "subject_id": meta.subject_id,
            "modality": meta.modality,
            "channels": meta.channels,
            "sample_rate": meta.sample_rate,
            "duration_seconds": meta.duration_seconds,
            "extra": meta.extra,
        }
        meta_file.write_text(
            __import__("json").dumps(meta_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return meta


