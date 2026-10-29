"""
VCV（連続音）処理モジュール

改善点:
- オーバーラップを「先行母音の実際の長さ」に対する割合で決定する。
  固定 30ms 引き算ではなく、先行母音が短ければ overlap も短くなる。
- 先行母音の安定区間（先頭〜先行母音終端の 40〜80%）を overlap の上限として使う。
- VCV 先頭音（先行母音なし）の分岐をより厳密に判定する。
- reprocess 後の append 書き込みで重複 alias が生じないよう、
  VCVProcessor は純粋にパラメータを計算するだけにする（書き込みは pipeline が担う）。
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

# VCV ファイル名パターン（先行母音_後続音素）
_VCV_PATTERNS = [
    re.compile(r"^([aiueoAIUEO])[-_](.+)$"),          # a_ka, i-sa
    re.compile(r"^([aiueoAIUEO])([a-zA-Z]{2,5})$"),   # aka, isa, asha
]

# 先頭音（先行母音なし）のパターン
_NO_PRE_VOWEL_PATTERNS = [
    re.compile(r"^[-_R]"),   # -ka, _sa, R_na
]

# オーバーラップを先行母音長の何割にするか（基本値）
OVERLAP_RATIO_OF_PRE_VOWEL = 0.55
# オーバーラップの最小・最大 [ms]
OVERLAP_MIN_MS = 8.0
OVERLAP_MAX_MS = 120.0

# 先行発声は先行母音終端（=子音開始）からの offset からの相対
# offset は先行母音の概ね 5〜10% 位置に置く
OFFSET_RATIO_OF_PRE_VOWEL = 0.08


class VCVProcessor:
    """VCV 音源のパラメータ推定"""

    def __init__(self, boundary_estimator: BoundaryEstimator):
        self.estimator = boundary_estimator

    def estimate(
        self, wav_path: Path, wav: WavData, features: AcousticFeatures
    ) -> tuple[OtoParams, float]:
        has_pre_vowel = self._has_pre_vowel(wav_path.stem)

        if has_pre_vowel:
            phoneme = self.estimator.estimate_vcv(wav, features)
        else:
            phoneme = self.estimator.estimate_cv(wav, features)

        alias           = self._stem_to_alias(wav_path.stem)
        offset_ms       = self._compute_offset(phoneme, wav, has_pre_vowel)
        consonant_ms    = self._compute_consonant(phoneme, wav)
        preutterance_ms = self._compute_preutterance(phoneme, offset_ms, wav, has_pre_vowel)
        overlap_ms      = self._compute_overlap(phoneme, offset_ms, wav, has_pre_vowel, features)
        cutoff_ms       = self._compute_cutoff(wav, features)

        # 整合性保証: overlap < preutterance
        if overlap_ms >= preutterance_ms:
            overlap_ms = max(preutterance_ms * 0.55, 1.0)
        overlap_ms = round(min(overlap_ms, OVERLAP_MAX_MS), 2)

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
        logger.debug(
            f"VCV: {alias} off={offset_ms:.1f} pre={preutterance_ms:.1f} "
            f"ov={overlap_ms:.1f} cons={consonant_ms:.1f} conf={confidence:.2f}"
        )
        return params, confidence

    # ------------------------------------------------------------------

    def _has_pre_vowel(self, stem: str) -> bool:
        for pat in _NO_PRE_VOWEL_PATTERNS:
            if pat.match(stem):
                return False
        for pat in _VCV_PATTERNS:
            if pat.match(stem):
                return True
        return bool(stem) and stem[0].lower() in "aiueo"

    def _stem_to_alias(self, stem: str) -> str:
        for pat in _VCV_PATTERNS:
            m = pat.match(stem)
            if m:
                return f"{m.group(1).lower()} {m.group(2).lower()}"
        # 先頭音（先行母音なし）：ファイル名から記号を除いてそのまま
        alias = re.sub(r"^[-_R]+", "", stem).lower()
        return alias or stem.lower()

    def _compute_offset(
        self, phoneme: PhonemeEstimation, wav: WavData, has_pre_vowel: bool
    ) -> float:
        """
        先行母音あり:
          offset = 先行母音の長さ × OFFSET_RATIO_OF_PRE_VOWEL の位置
          （先行母音の前部）
        先行母音なし:
          offset = 子音開始 - 12ms（CV と同じ）
        """
        sr = wav.sample_rate
        if not has_pre_vowel:
            if phoneme.consonant_start:
                return round(max(0.0, phoneme.consonant_start.to_ms(sr) - 12.0), 2)
            return 0.0

        pre_vowel_len_ms = self._pre_vowel_length_ms(phoneme, wav)
        if pre_vowel_len_ms > 0:
            offset = pre_vowel_len_ms * OFFSET_RATIO_OF_PRE_VOWEL
        elif phoneme.onset_silence_end:
            offset = phoneme.onset_silence_end.to_ms(sr)
        else:
            offset = 0.0

        return round(max(offset, 0.0), 2)

    def _compute_consonant(self, phoneme: PhonemeEstimation, wav: WavData) -> float:
        if phoneme.consonant_start and phoneme.vowel_start:
            dur = (
                phoneme.vowel_start.to_ms(wav.sample_rate)
                - phoneme.consonant_start.to_ms(wav.sample_rate)
            )
            return round(max(dur, 5.0), 2)
        return 60.0

    def _compute_preutterance(
        self,
        phoneme: PhonemeEstimation,
        offset_ms: float,
        wav: WavData,
        has_pre_vowel: bool,
    ) -> float:
        """
        先行発声 = offset からの相対時刻で、子音開始点を指す。
        """
        sr = wav.sample_rate
        if phoneme.consonant_start:
            cons_ms = phoneme.consonant_start.to_ms(sr)
            pre = cons_ms - offset_ms
        else:
            pre = 20.0

        return round(max(pre, 1.0), 2)

    def _compute_overlap(
        self,
        phoneme: PhonemeEstimation,
        offset_ms: float,
        wav: WavData,
        has_pre_vowel: bool,
        features: AcousticFeatures,
    ) -> float:
        """
        先行母音あり:
          先行母音の長さ × OVERLAP_RATIO_OF_PRE_VOWEL
          → offset からの相対に変換
          → 子音開始より必ず手前になるよう保証
        先行母音なし（先頭音）:
          CV 的に子音種別ベースで計算
        """
        sr = wav.sample_rate

        if not has_pre_vowel:
            # CV 的: preutterance の 50%
            pre = self._compute_preutterance(phoneme, offset_ms, wav, False)
            return round(max(pre * 0.5, OVERLAP_MIN_MS), 2)

        pre_vowel_len_ms = self._pre_vowel_length_ms(phoneme, wav)

        if pre_vowel_len_ms <= 0:
            # 先行母音長が計測できない → preutterance の 55%
            pre = self._compute_preutterance(phoneme, offset_ms, wav, True)
            return round(max(pre * 0.55, OVERLAP_MIN_MS), 2)

        # overlap の絶対位置 = 先行母音長 × ratio の時点（offset からの相対）
        overlap_abs_from_offset = pre_vowel_len_ms * OVERLAP_RATIO_OF_PRE_VOWEL
        overlap_ms = overlap_abs_from_offset

        # 最大でも preutterance の 90% まで
        pre = self._compute_preutterance(phoneme, offset_ms, wav, True)
        overlap_ms = min(overlap_ms, pre * 0.90)
        overlap_ms = max(overlap_ms, OVERLAP_MIN_MS)

        return round(overlap_ms, 2)

    def _compute_cutoff(self, wav: WavData, features: AcousticFeatures) -> float:
        dur_ms = wav.duration_sec * 1000.0
        if features.voiced_regions:
            last_ms = features.voiced_regions[-1][1] / wav.sample_rate * 1000.0
            cutoff = -(dur_ms - last_ms - 25.0)
        else:
            cutoff = -50.0
        return round(min(cutoff, -1.0), 2)

    # ------------------------------------------------------------------

    def _pre_vowel_length_ms(self, phoneme: PhonemeEstimation, wav: WavData) -> float:
        """先行母音の長さ [ms] を推定する"""
        sr = wav.sample_rate
        if phoneme.pre_vowel_end:
            end_ms = phoneme.pre_vowel_end.to_ms(sr)
            start_ms = (
                phoneme.onset_silence_end.to_ms(sr)
                if phoneme.onset_silence_end
                else 0.0
            )
            return max(end_ms - start_ms, 0.0)
        if phoneme.consonant_start:
            # 先行母音終端推定がなくても子音開始から逆算
            cons_ms = phoneme.consonant_start.to_ms(sr)
            start_ms = (
                phoneme.onset_silence_end.to_ms(sr)
                if phoneme.onset_silence_end
                else 0.0
            )
            return max(cons_ms - start_ms, 0.0)
        return 0.0
