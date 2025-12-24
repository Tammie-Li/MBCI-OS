"""
自研肌电腕带 .dat 文件转换器。

文件格式：
- 头部：int32 数组 [headlen, version, datatype, srate, emgchs, accchs, glovechs]
- 数据：float32 或 float64，按 totalChs 列组织，转置后为 C×T
"""

from __future__ import annotations
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..protocol import MetaInfo
from .base import BaseConverter


class EmgDatConverter(BaseConverter):
    """自研肌电腕带 .dat 文件转换器。"""

    name = "emg_dat"
    source_extensions = [".dat"]

    def convert(
        self,
        input_paths: List[Path],
        output_dir: Path,
        params: Optional[Dict[str, Union[int, float, str]]] = None,
    ) -> MetaInfo:
        if len(input_paths) != 1:
            raise ValueError("EMG .dat converter expects exactly one input file")
        input_path = input_paths[0]
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # 读取头信息
        with open(input_path, 'rb') as f:
            # 读取7个int32组成的头信息
            header = np.frombuffer(f.read(7 * 4), dtype=np.int32, count=7)
            
            # 解析头信息
            header_len = int(header[0])       # 头信息长度 (固定为7)
            file_version = int(header[1])     # 文件格式版本 (2.0)
            data_type = int(header[2])        # 数据类型 (2=float32, 3=float64)
            srate = int(header[3])            # 采样率
            emg_chs = int(header[4])          # EMG通道数
            acc_chs = int(header[5])          # ACC/GYR通道数
            glove_chs = int(header[6])        # 手套传感器通道数

            # 允许用户覆盖头信息（来自 UI 输入）
            if params:
                srate = int(params.get("srate", srate))
                emg_chs = int(params.get("emg_chs", emg_chs))
                acc_chs = int(params.get("acc_chs", acc_chs))
                glove_chs = int(params.get("glove_chs", glove_chs))
                override_dtype = params.get("dtype")
                if isinstance(override_dtype, str):
                    od = override_dtype.lower()
                    if od in ("float32", "f32", "32"):
                        data_type = 2
                    elif od in ("float64", "f64", "64"):
                        data_type = 3
            
            # 眼动判定：文件名前缀 ET_ 或 params 指定 is_eye
            is_eye = False
            if params and params.get("is_eye"):
                is_eye = True
            if input_path.stem.upper().startswith("ET_"):
                is_eye = True

            # 确定数据类型
            dtype = np.float32 if data_type == 2 else np.float64
            type_size = 4 if data_type == 2 else 8
            
            if is_eye:
                # 眼动：仅 X,Y，无时间戳
                frame_size = 2  # 强制两导
                file_size = os.path.getsize(input_path)
                data_size = file_size - header_len * 4
                n_frames = data_size // (frame_size * type_size)
                if n_frames <= 0:
                    raise ValueError("数据帧数为0，文件内容不足")
                data = np.frombuffer(f.read(), dtype=dtype, count=n_frames * frame_size)
                data = data.reshape(n_frames, frame_size).T
            else:
                # 肌电/IMU/手套：通道 + 时间戳(1)
                frame_size = emg_chs + acc_chs + glove_chs + 1
                if frame_size <= 0:
                    raise ValueError("通道数无效，无法解析 .dat 文件")
                file_size = os.path.getsize(input_path)
                data_size = file_size - header_len * 4
                n_frames = data_size // (frame_size * type_size)
                if n_frames <= 0:
                    raise ValueError("数据帧数为0，文件内容不足")
                data = np.frombuffer(f.read(), dtype=dtype, count=n_frames * frame_size)
                data = data.reshape(n_frames, frame_size).T


        # 分离 EMG 数据（只取前 emgchs 通道）
        emg_data = data[:frame_size, :].astype(np.float32)
        # 保存为标准格式
        output_dir.mkdir(parents=True, exist_ok=True)
        data_file = output_dir / "data.npy"
        np.save(data_file, emg_data)

        # 生成元信息
        duration = float(n_frames) / float(srate) if srate > 0 else 0.0
        modality = "EYE" if is_eye else "EMG"
        total_chs = frame_size
        meta = MetaInfo(
            subject_id=input_path.stem,
            modality=modality,
            channels=int(total_chs),
            sample_rate=float(srate),
            duration_seconds=float(duration),
            extra={
                "acc_channels": int(acc_chs),
                "glove_channels": int(glove_chs),
                "total_channels": int(total_chs),
                "source_file": str(input_path),
                "dtype": "float32" if data_type == 2 else "float64",
                "is_eye": bool(is_eye),
            },
        )

        # 保存元信息
        import json

        meta_file = output_dir / "datainfo.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "subject_id": meta.subject_id,
                    "modality": meta.modality,
                    "channels": meta.channels,
                    "sample_rate": meta.sample_rate,
                    "duration_seconds": meta.duration_seconds,
                    "extra": meta.extra,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        return meta

