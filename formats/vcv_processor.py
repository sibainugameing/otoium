"""
VCV（連続音）処理モジュール

先行母音・子音への遷移・後続母音をまとめて解析し、
音のつながりを維持したパラメータを推定する。
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

# VCV ファイル名から先行母音・後続音素を取り出す正規表現群
_VCV_NAME_RE = [
    re.compile(r"^([aiueo])[-_]?([a-z]+)$", re.IGNORECASE),   # a_ka, a-sa
    re.compile(r"^([aiueo])([a-z]{2,4})$", re.IGNORECASE),    # aka, asa
]

# 先行母音が存在しないケース（先頭音）のステムパターン
_NO_PRE_VOWEL_RE = re.compile(r"^-|^_|^R_|^-_", re.IGNORECASE)


class VCVProcessor:
    """VCV 音源のパラメータ推定"""

    # 先行母音安定区間から子音境界までのパディング [ms]
    PRE_VOWEL_PAD_MS = 20.0
    # オーバーラップは先行母音の安定区間に食い込む
    OVERLAP_INTO_PRE_VOWEL_MS = 30.0

    def __init__(self, boundary_estimator: BoundaryEstimator):
        self.estimator = boundary_estimator

    def estimate(
        self, wav_path: Path, wav: WavData, features: AcousticFeatures
    ) -> tuple[OtoParams, float]:
        """VCV の oto.ini パラメータを推定して (OtoParams, confidence) を返す"""

        # 先行母音があるかを先に判定
        has_pre_vowel = self._has_pre_vowel(wav_path.stem)

        if has_pre_vowel:
            phoneme = self.estimator.estimate_vcv(wav, features)
        else:
            # 先行母音なし → CV に近い処理
            phoneme = self.estimator.estimate_cv(wav, features)

        alias = self._stem_to_alias(wav_path.stem)
        offset_ms = self._compute_offset(phoneme, wav, has_pre_vowel)
        consonant_ms = self._compute_consonant(phoneme, wav)
        preutterance_ms = self._compute_preutterance(phoneme, offset_ms, wav, has_pre_vowel)
        overlap_ms = self._compute_overlap_vcv(phoneme, offset_ms, wav, has_pre_vowel)
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
        logger.debug(
            f"VCV params: {alias} offset={offset_ms:.1f} pre={preutterance_ms:.1f} "
            f"overlap={overlap_ms:.1f} conf={confidence:.2f}"
        )
        return params, confidence

    # ------------------------------------------------------------------

    def _has_pre_vowel(self, stem: str) -> bool:
        """先行母音が存在するかをファイル名から判定する"""
        if _NO_PRE_VOWEL_RE.match(stem):
            return False
        for pat in _VCV_NAME_RE:
            if pat.match(stem):
                return True
        # 先頭文字が母音なら先行母音あり
        return bool(stem) and stem[0].lower() in "aiueo"

    def _stem_to_alias(self, stem: str) -> str:
        """VCV ファイル名から alias 文字列を生成する（例: "a ka"）"""
        for pat in _VCV_NAME_RE:
            m = pat.match(stem)
            if m:
                pre_vowel = m.group(1).lower()
                consonant = m.group(2).lower()
                return f"{pre_vowel} {consonant}"
        return stem

    def _compute_offset(
        self, phoneme: PhonemeEstimation, wav: WavData, has_pre_vowel: bool
    ) -> float:
        """
        VCV のオフセット：先行母音の安定区間の先頭付近に置く。
        先行母音がない場合は CV と同じ扱い。
        """
        if not has_pre_vowel:
            if phoneme.consonant_start:
                return max(0.0, phoneme.consonant_start.to_ms(wav.sample_rate) - 10.0)
            return 0.0

        # 先行母音は音声の冒頭から始まると想定
        if phoneme.onset_silence_end:
            return phoneme.onset_silence_end.to_ms(wav.sample_rate)
        return 0.0

    def _compute_consonant(
        self, phoneme: PhonemeEstimation, wav: WavData
    ) -> float:
        """子音区間の長さ [ms]"""
        if phoneme.consonant_start and phoneme.vowel_start:
            dur = (
                phoneme.vowel_start.to_ms(wav.sample_rate)
                - phoneme.consonant_start.to_ms(wav.sample_rate)
            )
            return round(max(dur, 10.0), 2)
        return 60.0

    def _compute_preutterance(
        self,
        phoneme: PhonemeEstimation,
        offset_ms: float,
        wav: WavData,
        has_pre_vowel: bool,
    ) -> float:
        """
        先行発声 [ms]（offset からの相対）。
        VCV では子音の直前（先行母音と子音の境界あたり）に置く。
        """
        if phoneme.consonant_start:
            cons_ms = phoneme.consonant_start.to_ms(wav.sample_rate)
            return round(max(cons_ms - offset_ms, 1.0), 2)
        return round(self.PRE_VOWEL_PAD_MS, 2)

    def _compute_overlap_vcv(
        self,
        phoneme: PhonemeEstimation,
        offset_ms: float,
        wav: WavData,
        has_pre_vowel: bool,
    ) -> float:
        """
        VCV のオーバーラップ：先行母音の安定区間に食い込む位置。
        先行母音がない場合は CV 的に計算する。
        """
        if not has_pre_vowel:
            # CV 的: 先行発声の約 50%
            if phoneme.consonant_start:
                cons_ms = phoneme.consonant_start.to_ms(wav.sample_rate)
                preutterance = cons_ms - offset_ms
                return round(max(preutterance * 0.5, 1.0), 2)
            return 10.0

        if phoneme.consonant_start:
            cons_ms = phoneme.consonant_start.to_ms(wav.sample_rate)
            # overlap = 子音開始より少し前（先行母音安定区間の後半）
            overlap_pos = cons_ms - offset_ms - self.OVERLAP_INTO_PRE_VOWEL_MS
            return round(max(overlap_pos, 1.0), 2)

        return round(self.OVERLAP_INTO_PRE_VOWEL_MS, 2)

    def _compute_cutoff(self, wav: WavData, features: AcousticFeatures) -> float:
        """カットオフ：末尾の無音を切り落とす負値 [ms]"""
        dur_ms = wav.duration_sec * 1000.0
        if features.voiced_regions:
            last_voiced_end = features.voiced_regions[-1][1]
            last_voiced_ms = last_voiced_end / wav.sample_rate * 1000.0
            cutoff = -(dur_ms - last_voiced_ms - 30.0)
        else:
            cutoff = -50.0
        return round(min(cutoff, -1.0), 2)
