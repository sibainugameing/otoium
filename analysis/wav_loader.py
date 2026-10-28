"""
WAV ファイルの読み込み
音声データを numpy 配列として返す。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class WavData:
    """読み込んだ WAV データのコンテナ"""

    samples: np.ndarray   # float32, shape=(samples,) or (samples, channels)
    sample_rate: int
    duration_sec: float
    path: Path

    @property
    def mono(self) -> np.ndarray:
        """モノラルに変換して返す"""
        if self.samples.ndim == 1:
            return self.samples
        return self.samples.mean(axis=1)

    def samples_to_ms(self, sample_idx: int) -> float:
        return sample_idx / self.sample_rate * 1000.0

    def ms_to_samples(self, ms: float) -> int:
        return int(ms / 1000.0 * self.sample_rate)


class WavLoader:
    """WAV ファイルを読み込むクラス"""

    TARGET_SAMPLE_RATE = 44100  # UTAU 標準

    def load(self, path: Path) -> WavData:
        """
        WAV を読み込み WavData を返す。
        読み込みに失敗した場合は例外を送出する。
        """
        if not path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {path}")

        try:
            samples, sr = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception as exc:
            raise RuntimeError(f"WAV 読み込みエラー: {path.name}") from exc

        duration = len(samples) / sr

        if sr != self.TARGET_SAMPLE_RATE:
            logger.debug(
                f"{path.name}: サンプルレート {sr} Hz（非標準）"
            )

        logger.debug(
            f"読み込み: {path.name} | {sr} Hz | {duration:.3f} s | shape={samples.shape}"
        )

        return WavData(
            samples=samples,
            sample_rate=sr,
            duration_sec=duration,
            path=path,
        )
