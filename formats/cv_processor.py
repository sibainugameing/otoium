"""
CV（単独音）処理モジュール

改善点:
- 子音長のクリップを廃止。ヒント範囲は「異常値検出」にのみ使い、
  実測値が範囲外のときだけ信頼度を下げて警告する。
- オーバーラップを consonant_start（子音開始）を基準に計算。
  preutterance の割合ではなく「子音立ち上がりを何 ms だけ手前に置くか」。
- 先頭余白の推定精度を上げるため、onset_silence_end と consonant_start の
  両方を考慮して offset を決定する。
- 母音が存在しない（母音推定失敗）ケースの fallback を充実させる。
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

# 日本語子音の概ねの持続時間 [ms]（lo, hi）
# ※ クリップには使わない。異常値フラグ判定のみ。
CONSONANT_DURATION_HINTS: dict[str, tuple[float, float]] = {
    "sh": (60, 200),
    "ch": (55, 170),
    "ts": (55, 170),
    "ky": (50, 130),
    "gy": (50, 130),
    "ny": (50, 120),
    "hy": (40, 110),
    "my": (50, 120),
    "ry": (30, 90),
    "k":  (45, 130),
    "g":  (45, 130),
    "s":  (70, 180),
    "z":  (60, 160),
    "t":  (35, 110),
    "d":  (35, 110),
    "n":  (50, 130),
    "h":  (35, 110),
    "b":  (45, 120),
    "p":  (45, 120),
    "m":  (45, 120),
    "y":  (25, 90),
    "r":  (25, 80),
    "w":  (25, 85),
    "f":  (50, 130),
    "v":  (50, 130),
}

# 子音種別ごとの overlap オフセット [ms]（子音開始より何 ms 前に置くか）
# 値が小さいほど overlap が子音開始に近い（子音をしっかり含む）
OVERLAP_OFFSET_BY_TYPE: dict[str, float] = {
    # 破裂音・破擦音：子音の立ち上がりが急なので直前に置く
    "sh": 5.0, "ch": 5.0, "ts": 5.0,
    "k":  8.0, "g":  8.0,
    "p":  8.0, "b":  8.0,
    "t":  8.0, "d":  8.0,
    # 摩擦音：子音の立ち上がりがなだらか
    "s": 18.0, "z": 18.0, "f": 18.0, "v": 18.0, "h": 15.0,
    # 共鳴音：オーバーラップを広く取れる
    "n": 25.0, "m": 25.0, "r": 28.0, "y": 28.0, "w": 28.0,
}

# オフセットのデフォルト（子音開始の何 ms 前にオフセットを置くか）
OFFSET_PRE_CONSONANT_MS = 12.0


class CVProcessor:
    """CV 音源のパラメータ推定"""

    def __init__(self, boundary_estimator: BoundaryEstimator):
        self.estimator = boundary_estimator

    def estimate(
        self, wav_path: Path, wav: WavData, features: AcousticFeatures
    ) -> tuple[OtoParams, float]:
        phoneme = self.estimator.estimate_cv(wav, features)
        alias   = self._stem_to_alias(wav_path.stem)

        offset_ms       = self._compute_offset(phoneme, wav)
        consonant_ms    = self._compute_consonant(phoneme, wav_path.stem, wav)
        preutterance_ms = self._compute_preutterance(phoneme, offset_ms, wav)
        overlap_ms      = self._compute_overlap(phoneme, wav_path.stem, wav)
        cutoff_ms       = self._compute_cutoff(wav, features)

        # 整合性保証: preutterance > overlap
        if overlap_ms >= preutterance_ms:
            overlap_ms = max(preutterance_ms * 0.5, 1.0)

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
        self._warn_if_abnormal(alias, consonant_ms, wav_path.stem, confidence)
        logger.debug(
            f"CV: {alias} off={offset_ms:.1f} pre={preutterance_ms:.1f} "
            f"ov={overlap_ms:.1f} cons={consonant_ms:.1f} conf={confidence:.2f}"
        )
        return params, confidence

    # ------------------------------------------------------------------

    def _stem_to_alias(self, stem: str) -> str:
        alias = re.sub(r"\d+$", "", stem).strip("_- ")
        return alias or stem

    def _compute_offset(self, phoneme: PhonemeEstimation, wav: WavData) -> float:
        """
        offset = 子音開始の OFFSET_PRE_CONSONANT_MS ms 前。
        onset_silence_end がある場合は、それより前にはしない。
        """
        if phoneme.consonant_start:
            cons_ms = phoneme.consonant_start.to_ms(wav.sample_rate)
            offset = max(0.0, cons_ms - OFFSET_PRE_CONSONANT_MS)
        elif phoneme.onset_silence_end:
            offset = phoneme.onset_silence_end.to_ms(wav.sample_rate)
        else:
            offset = 0.0

        # 先頭余白終端より前にはしない
        if phoneme.onset_silence_end:
            sil_end_ms = phoneme.onset_silence_end.to_ms(wav.sample_rate)
            offset = max(offset, sil_end_ms)

        return round(max(offset, 0.0), 2)

    def _compute_consonant(
        self, phoneme: PhonemeEstimation, stem: str, wav: WavData
    ) -> float:
        """
        子音長 = 実測値（子音開始〜母音開始）をそのまま使う。
        実測できない場合のみヒント値の中央値を使う。
        """
        if phoneme.consonant_start and phoneme.vowel_start:
            measured = (
                phoneme.vowel_start.to_ms(wav.sample_rate)
                - phoneme.consonant_start.to_ms(wav.sample_rate)
            )
            return round(max(measured, 5.0), 2)

        hint = self._consonant_hint(stem)
        return round(float(np.mean(hint)) if hint else 60.0, 2)

    def _compute_preutterance(
        self, phoneme: PhonemeEstimation, offset_ms: float, wav: WavData
    ) -> float:
        """
        先行発声 = offset から子音開始までの距離 [ms]。
        """
        if phoneme.consonant_start:
            cons_ms = phoneme.consonant_start.to_ms(wav.sample_rate)
            pre = cons_ms - offset_ms
        else:
            pre = OFFSET_PRE_CONSONANT_MS

        return round(max(pre, 1.0), 2)

    def _compute_overlap(
        self, phoneme: PhonemeEstimation, stem: str, wav: WavData
    ) -> float:
        """
        overlap = offset から (子音開始 - overlap_offset) までの距離 [ms]。
        overlap_offset は子音種別ごとに異なる。

        overlap はオフセットからの相対時刻なので:
          overlap = preutterance - overlap_offset_from_consonant_start
        """
        cons_initial = stem[:2].lower() if len(stem) >= 2 else stem[:1].lower()
        # 2 文字キーを優先
        offset_from_cons = OVERLAP_OFFSET_BY_TYPE.get(
            cons_initial, OVERLAP_OFFSET_BY_TYPE.get(cons_initial[:1], 15.0)
        )

        if phoneme.consonant_start:
            cons_ms  = phoneme.consonant_start.to_ms(wav.sample_rate)
            # overlap 絶対位置 = 子音開始 - offset_from_cons
            overlap_abs = cons_ms - offset_from_cons
            # offset からの相対に変換
            if phoneme.consonant_start:
                offset_ms = max(0.0, cons_ms - OFFSET_PRE_CONSONANT_MS)
                # onset_silence_end 制約（_compute_offset と同じロジック）
                overlap_rel = overlap_abs - offset_ms
            else:
                overlap_rel = offset_from_cons
        else:
            overlap_rel = 5.0

        return round(max(overlap_rel, 1.0), 2)

    def _compute_cutoff(self, wav: WavData, features: AcousticFeatures) -> float:
        dur_ms = wav.duration_sec * 1000.0
        if features.voiced_regions:
            last_end_ms = features.voiced_regions[-1][1] / wav.sample_rate * 1000.0
            cutoff = -(dur_ms - last_end_ms - 25.0)
        else:
            cutoff = -50.0
        return round(min(cutoff, -1.0), 2)

    # ------------------------------------------------------------------

    def _consonant_hint(self, stem: str) -> Optional[tuple[float, float]]:
        s = stem.lower()
        for key in sorted(CONSONANT_DURATION_HINTS.keys(), key=len, reverse=True):
            if s.startswith(key):
                return CONSONANT_DURATION_HINTS[key]
        return None

    def _warn_if_abnormal(
        self, alias: str, consonant_ms: float, stem: str, confidence: float
    ) -> None:
        """子音長がヒント範囲を大きく外れていればデバッグ警告を出す"""
        hint = self._consonant_hint(stem)
        if not hint:
            return
        lo, hi = hint
        margin = 30.0
        if consonant_ms < lo - margin or consonant_ms > hi + margin:
            logger.warning(
                f"{alias}: 子音長 {consonant_ms:.0f}ms はヒント範囲({lo}-{hi}ms)を外れています "
                f"(conf={confidence:.2f})"
            )
