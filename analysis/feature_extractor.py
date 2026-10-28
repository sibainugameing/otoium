"""
音響特徴量の抽出

波形・スペクトログラム・音量エンベロープ・有声区間を計算し、
境界推定モジュールに渡すための特徴量セットを生成する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import medfilt

from analysis.wav_loader import WavData

logger = logging.getLogger(__name__)


@dataclass
class AcousticFeatures:
    """抽出した音響特徴量のコンテナ"""

    # RMS エネルギー（フレーム単位）
    rms: np.ndarray
    rms_times: np.ndarray      # 各フレームの中心時刻 [s]

    # 有声判定（フレーム単位）
    voiced_flags: np.ndarray   # bool 配列

    # スペクトル重心（フレーム単位）
    spectral_centroid: np.ndarray

    # ゼロ交差率（フレーム単位）
    zcr: np.ndarray

    # MFCC（フレーム x 係数数）
    mfcc: np.ndarray

    # 無音区間の推定（開始・終了サンプル）
    silence_regions: list[tuple[int, int]] = field(default_factory=list)

    # 有声区間の推定（開始・終了サンプル）
    voiced_regions: list[tuple[int, int]] = field(default_factory=list)

    # サンプルレートと基準情報
    sample_rate: int = 44100
    frame_length: int = 1024
    hop_length: int = 256


class FeatureExtractor:
    """
    WavData から音響特徴量を抽出する。

    librosa を使用するが、依存を局所化してここだけが import する。
    """

    FRAME_LENGTH = 1024
    HOP_LENGTH = 256
    N_MFCC = 13
    SILENCE_THRESHOLD_DB = -40.0   # これ以下を無音とみなす
    VOICED_ZCR_MAX = 0.15          # 有声音のゼロ交差率上限

    def extract(self, wav: WavData) -> AcousticFeatures:
        import librosa

        y = wav.mono.astype(np.float32)
        sr = wav.sample_rate

        # --- RMS エネルギー ---
        rms = librosa.feature.rms(
            y=y, frame_length=self.FRAME_LENGTH, hop_length=self.HOP_LENGTH
        )[0]
        rms_times = librosa.frames_to_time(
            np.arange(len(rms)), sr=sr, hop_length=self.HOP_LENGTH
        )

        # --- ゼロ交差率 ---
        zcr = librosa.feature.zero_crossing_rate(
            y, frame_length=self.FRAME_LENGTH, hop_length=self.HOP_LENGTH
        )[0]

        # --- スペクトル重心 ---
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=self.HOP_LENGTH
        )[0]

        # --- MFCC ---
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.N_MFCC, hop_length=self.HOP_LENGTH
        ).T  # shape: (frames, n_mfcc)

        # --- 有声判定 ---
        rms_db = librosa.amplitude_to_db(rms, ref=np.max(rms + 1e-9))
        energy_voiced = rms_db > self.SILENCE_THRESHOLD_DB
        zcr_voiced = zcr < self.VOICED_ZCR_MAX
        voiced_flags = energy_voiced & zcr_voiced
        voiced_flags = medfilt(voiced_flags.astype(float), kernel_size=5).astype(bool)

        # --- 無音・有声区間を抽出 ---
        silence_regions = self._extract_regions(~voiced_flags, sr)
        voiced_regions = self._extract_regions(voiced_flags, sr)

        logger.debug(
            f"特徴抽出: frames={len(rms)} voiced_regions={len(voiced_regions)}"
        )

        return AcousticFeatures(
            rms=rms,
            rms_times=rms_times,
            voiced_flags=voiced_flags,
            spectral_centroid=centroid,
            zcr=zcr,
            mfcc=mfcc,
            silence_regions=silence_regions,
            voiced_regions=voiced_regions,
            sample_rate=sr,
            frame_length=self.FRAME_LENGTH,
            hop_length=self.HOP_LENGTH,
        )

    # ------------------------------------------------------------------

    def _extract_regions(
        self, flags: np.ndarray, sr: int
    ) -> list[tuple[int, int]]:
        """
        フラグ配列からランレングス圧縮で連続区間を取り出す。
        フレームインデックスをサンプルインデックスに変換して返す。
        """
        regions = []
        in_region = False
        start_frame = 0
        for i, v in enumerate(flags):
            if v and not in_region:
                start_frame = i
                in_region = True
            elif not v and in_region:
                start_sample = start_frame * self.HOP_LENGTH
                end_sample = i * self.HOP_LENGTH
                regions.append((start_sample, end_sample))
                in_region = False
        if in_region:
            start_sample = start_frame * self.HOP_LENGTH
            regions.append((start_sample, len(flags) * self.HOP_LENGTH))
        return regions
