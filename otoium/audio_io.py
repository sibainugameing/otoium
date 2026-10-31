"""WAV 読み込み（soundfile があれば使用、無ければ標準 wave / scipy）"""
import wave
from pathlib import Path
import numpy as np


def _read_wave_module(path):
    with wave.open(str(path), "rb") as w:
        n, sw, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if sw == 1:
        a = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 2:
        a = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= (1 << 23), v - (1 << 24), v)
        a = v.astype(np.float32) / 8388608.0
    elif sw == 4:
        a = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"未対応のサンプル幅: {sw}")
    return a.reshape(-1, n).mean(axis=1), sr


def _read_scipy(path):
    from scipy.io import wavfile
    sr, d = wavfile.read(str(path))
    if d.dtype == np.uint8:
        d = (d.astype(np.float32) - 128.0) / 128.0
    elif d.dtype == np.int16:
        d = d.astype(np.float32) / 32768.0
    elif d.dtype == np.int32:
        d = d.astype(np.float32) / 2147483648.0
    else:
        d = d.astype(np.float32)
    if d.ndim == 2:
        d = d.mean(axis=1)
    return d, sr


def read_wav(path):
    """モノラル float32 と サンプルレートを返す"""
    path = Path(path)
    try:
        import soundfile as sf
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return x.mean(axis=1).astype(np.float32), int(sr)
    except ImportError:
        pass
    except Exception:
        pass
    try:
        x, sr = _read_wave_module(path)
    except (wave.Error, EOFError):
        x, sr = _read_scipy(path)
    return x.astype(np.float32), int(sr)
