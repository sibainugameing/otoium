"""
CV（単独音）処理モジュール

子音開始・母音への遷移・先頭余白を中心にパラメータを推定する。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

from analysis.boundary_estimator import BoundaryEstimator, PhonemeEstimation
from analysis.feature_extractor import AcousticFeatures
from analysis.wav_loader import WavData
from core.result import OtoParams

logger = logging.getLogger(__name__)

# 日本語子音の概ねの長さ目安 [ms]
CONSONANT_DURATION_HINTS: dict[str, tuple[float, float]] = {
    "k": (50, 120),
    "s": (80, 160),
    "t": (40, 100),
    "n": (60, 120),
    "h": (40, 100),
    "m": (50, 110),
    "y": (30, 80),
    "r": (30, 70),
    "w": (30, 80),
    "g": (50, 120),
    "z": (70, 150),
    "d": (40, 100),
    "b": (50, 110),
    "p": (50, 100),
    "sh": (80, 180),
    "ch": (60, 150),
    "ts": (60, 150),
}


class CVProcessor:
    """CV 音源のパラメータ推定"""

    # oto.ini パラメータの定数
    OVERLAP_RATIO_OF_PREUTTERANCE = 0.5   # overlap = preutterance * この割合
    DEFAULT_OFFSET_PAD_MS = 10.0          # onset 前のパディング

    def __init__(self, boundary_estimator: BoundaryEstimator):
        self.estimator = boundary_estimator

    def estimate(
        self, wav_path: Path, wav: WavData, features: AcousticFeatures
    ) -> tuple[OtoParams, float]:
        """
        CV の oto.ini パラメータを推定して (OtoParams, confidence) を返す。
        """
        # 1. 音素境界推定
        phoneme = self.estimator.estimate_cv(wav, features)

        # 2. alias・ファイル名の決定
        alias = self._stem_to_alias(wav_path.stem)

        # 3. オフセット（先行発声より前の余白含む起点）
        offset_ms = self._compute_offset(phoneme, wav)

        # 4. 子音部分の長さ
        consonant_ms = self._compute_consonant(phoneme, wav_path.stem, wav)

        # 5. 先行発声（preutterance）= 子音の手前
        preutterance_ms = self._compute_preutterance(phoneme, offset_ms, wav)

        # 6. オーバーラップ：子音の立ち上がりを損なわない位置
        overlap_ms = self._compute_overlap_cv(phoneme, preutterance_ms, wav_path.stem)

        # 7. カットオフ（ファイル末尾から負値）
        cutoff_ms = self._compute_cutoff(wav, features)

        params = OtoParams(
            alias=alias,
            filename=wav_path.name,
            offset=offset_ms,
            consonant=consonant_ms,
            cutoff=cutoff_ms,
            preutterance=preutterance_ms,
            overlap=overlap_ms,
        )

        confidence = phoneme.overall_confidence
        logger.debug(f"CV params: {alias} offset={offset_ms:.1f} pre={preutterance_ms:.1f} conf={confidence:.2f}")
        return params, confidence

    # ------------------------------------------------------------------

    def _stem_to_alias(self, stem: str) -> str:
        """ファイル名ステムから alias を生成する"""
        # 数字サフィックスを除去
        alias = re.sub(r"\d+$", "", stem).strip("_- ")
        return alias or stem

    def _compute_offset(self, phoneme: PhonemeEstimation, wav: WavData) -> float:
        """
        先行発声起点より少し前にオフセットを置く。
        子音の開始から DEFAULT_OFFSET_PAD_MS 前。
        """
        if phoneme.consonant_start:
            start_ms = phoneme.consonant_start.to_ms(wav.sample_rate)
            offset = max(0.0, start_ms - self.DEFAULT_OFFSET_PAD_MS)
        elif phoneme.onset_silence_end:
            offset = phoneme.onset_silence_end.to_ms(wav.sample_rate)
        else:
            offset = 0.0
        return round(offset, 2)

    def _compute_consonant(
        self, phoneme: PhonemeEstimation, stem: str, wav: WavData
    ) -> float:
        """
        子音部分の長さ [ms]。
        子音開始〜母音開始の実測値と、子音種別ヒントで補正する。
        """
        if phoneme.consonant_start and phoneme.vowel_start:
            measured = (
                phoneme.vowel_start.to_ms(wav.sample_rate)
                - phoneme.consonant_start.to_ms(wav.sample_rate)
            )
            hint = self._consonant_hint(stem)
            if hint:
                lo, hi = hint
                consonant = float(np.clip(measured, lo, hi))
            else:
                consonant = max(measured, 10.0)
        else:
            hint = self._consonant_hint(stem)
            consonant = float(np.mean(hint)) if hint else 60.0

        return round(consonant, 2)

    def _compute_preutterance(
        self, phoneme: PhonemeEstimation, offset_ms: float, wav: WavData
    ) -> float:
        """
        先行発声 = オフセットから子音開始までの距離。
        UTAU の仕様: offset からの相対 ms。
        """
        if phoneme.consonant_start:
            cons_ms = phoneme.consonant_start.to_ms(wav.sample_rate)
            preutterance = cons_ms - offset_ms
        else:
            preutterance = self.DEFAULT_OFFSET_PAD_MS

        return round(max(preutterance, 1.0), 2)

    def _compute_overlap_cv(
        self, phoneme: PhonemeEstimation, preutterance_ms: float, stem: str
    ) -> float:
        """
        CV のオーバーラップ：子音の立ち上がりを損なわない位置。
        先行発声より手前（先行発声の約 50 〜 70% 手前）に置く。
        子音が強いほどオーバーラップを早める。
        """
        ratio = 0.5
        cons_initial = stem[0].lower() if stem else ""
        # 破裂音・摩擦音は子音が目立つので overlap を小さく（手前に置く）
        if cons_initial in "kgptdbszsh":
            ratio = 0.35
        elif cons_initial in "nmryw":
            ratio = 0.65

        overlap = preutterance_ms * ratio
        return round(max(overlap, 1.0), 2)

    def _compute_cutoff(self, wav: WavData, features: AcousticFeatures) -> float:
        """
        カットオフ：ファイル末尾の無音を切り落とす負値 [ms]。
        末尾の有声区間終端から少し余白を持たせる。
        """
        dur_ms = wav.duration_sec * 1000.0
        if features.voiced_regions:
            last_voiced_end_sample = features.voiced_regions[-1][1]
            last_voiced_ms = last_voiced_end_sample / wav.sample_rate * 1000.0
            tail_pad = 30.0
            cutoff = -(dur_ms - last_voiced_ms - tail_pad)
        else:
            cutoff = -50.0
        return round(min(cutoff, -1.0), 2)

    def _consonant_hint(self, stem: str) -> Optional[tuple[float, float]]:
        """ファイル名から子音種別のヒントを取得する"""
        s = stem.lower()
        for key in sorted(CONSONANT_DURATION_HINTS.keys(), key=len, reverse=True):
            if s.startswith(key):
                return CONSONANT_DURATION_HINTS[key]
        return None
