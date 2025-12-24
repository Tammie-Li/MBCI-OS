# 读取 EDF 文件并输出通道信息与前几项示例，便于快速验证保存结果。
from pathlib import Path


def main():
    try:
        import pyedflib
    except ImportError:
        print("请先安装 pyedflib：pip install pyedflib")
        return

    edf_path = Path("MPSCAP_data.edf")
    if not edf_path.exists():
        print("未找到 MPSCAP_data.edf，请确认保存目录。")
        return

    reader = pyedflib.EdfReader(str(edf_path))
    try:
        n_sig = reader.signals_in_file
        labels = reader.getSignalLabels()
        srate = reader.getSampleFrequencies()
        print(f"EDF 文件: {edf_path}")
        print(f"通道数: {n_sig}")
        for idx in range(n_sig):
            lbl = labels[idx] if idx < len(labels) else f"CH{idx+1}"
            sr = srate[idx] if idx < len(srate) else "未知"
            samples = reader.getNSamples()[idx]
            preview = reader.readSignal(idx, start=0, n=5)
            print(f"[{idx}] {lbl} | 采样率: {sr} Hz | 样本数: {samples} | 前5项: {preview}")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
