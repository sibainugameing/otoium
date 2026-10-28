"""
音素境界推定

音響特徴量から子音開始・母音開始・音素境界を推定する。
推定値には信頼度スコアを付与し、低信頼度の場合は複数候補を返す。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import find_peaks

from analysis.feature_extractor import AcousticFeatures
from analysis.wav_loader import WavData

logger = logging.getLogger(__name__)


@dataclass
class BoundaryEstimate:
    """音素境界推定の結果（1 点）"""

    sample: int         # 境界位置 [サンプル]
    confidence: float   # 0.0 〜 1.0
    label: str = ""     # "consonant_start" | "vowel_start" | "vcv_boundary" …

    def to_ms(self, sample_rate: int) -> float:
        return self.sample / sample_rate * 1000.0


@dataclass
class PhonemeEstimation:
    """1 ファイルの全境界推定結果"""

    consonant_start: Optional[BoundaryEstimate] = None
    vowel_start: Optional[BoundaryEstimate] = None
    pre_vowel_end: Optional[BoundaryEstimate] = None   # VCV 専用
    onset_silence_end: Optional[BoundaryEstimate] = None  # 先頭余白の終端

    # 複数候補（低信頼度のとき探索）
    consonant_candidates: list[BoundaryEstimate] = field(default_factory=list)
    vowel_candidates: list[BoundaryEstimate] = field(default_factory=list)

    @property
    def overall_confidence(self) -> float:
        scores = []
        for est in [self.consonant_start, self.vowel_start]:
            if est is not None:
                scores.append(est.confidence)
        return float(np.mean(scores)) if scores else 0.0


class BoundaryEstimator:
    """
    音響特徴量から音素境界を推定する。

    優先順位:
    1. 音量の急激な変化（onset検出）
    2. スペクトル重心の変化（子音→母音の遷移）
    3. 有声区間の開始
    4. ZCR の急激な低下
    """

    # onset 検出の感度（librosa のデフォルトに近い値）
    ONSET_BACKTRACK_SAMPLES = 512

    def estimate_cv(self, wav: WavData, features: AcousticFeatures) -> PhonemeEstimation:
        """CV 用：子音開始と母音開始を推定する"""
        result = PhonemeEstimation()

        # 先頭の無音終端を取得
        result.onset_silence_end = self._estimate_onset_silence(features)

        # 子音の開始 ≒ 最初の onset
        cons_candidates = self._find_consonant_onsets(wav, features)
        result.consonant_candidates = cons_candidates
        if cons_candidates:
            result.consonant_start = cons_candidates[0]

        # 母音の開始 ≒ 最初の有声区間開始
        vowel_candidates = self._find_vowel_starts(features)
        result.vowel_candidates = vowel_candidates
        if vowel_candidates:
            result.vowel_start = vowel_candidates[0]

        # 整合性チェック：子音開始 < 母音開始
        result = self._validate_cv_order(result, wav.sample_rate)

        logger.debug(
            f"CV推定: cons={result.consonant_start}, vowel={result.vowel_start}, "
            f"conf={result.overall_confidence:.2f}"
        )
        return result

    def estimate_vcv(self, wav: WavData, features: AcousticFeatures) -> PhonemeEstimation:
        """VCV 用：先行母音・子音・後続母音の境界を推定する"""
        result = PhonemeEstimation()

        # 先行母音の終端（= 子音の開始）
        pre_vowel_end = self._find_pre_vowel_end(features)
        result.pre_vowel_end = pre_vowel_end

        if pre_vowel_end:
            result.consonant_start = BoundaryEstimate(
                sample=pre_vowel_end.sample,
                confidence=pre_vowel_end.confidence * 0.9,
                label="consonant_start",
            )

        # 後続母音の開始
        vowel_candidates = self._find_vowel_starts(features, search_from=pre_vowel_end)
        result.vowel_candidates = vowel_candidates
        if vowel_candidates:
            result.vowel_start = vowel_candidates[0]

        logger.debug(
            f"VCV推定: pre_vowel_end={result.pre_vowel_end}, vowel={result.vowel_start}, "
            f"conf={result.overall_confidence:.2f}"
        )
        return result

    # ------------------------------------------------------------------
    # 内部推定メソッド
    # ------------------------------------------------------------------

    def _estimate_onset_silence(
        self, features: AcousticFeatures
    ) -> Optional[BoundaryEstimate]:
        """先頭の無音区間の終端を推定する"""
        if not features.silence_regions:
            return None
        first_silence = features.silence_regions[0]
        # 先頭から始まる無音のみ対象
        if first_silence[0] == 0:
            return BoundaryEstimate(
                sample=first_silence[1],
                confidence=0.85,
                label="onset_silence_end",
            )
        return BoundaryEstimate(sample=0, confidence=0.7, label="onset_silence_end")

    def _find_consonant_onsets(
        self, wav: WavData, features: AcousticFeatures
    ) -> list[BoundaryEstimate]:
        """
        onset 検出で子音の候補位置を取り出す。
        librosa の onset_detect を使い、エネルギー的な立ち上がりを検出する。
        """
        try:
            import librosa

            y = wav.mono.astype(np.float32)
            onset_frames = librosa.onset.onset_detect(
                y=y,
                sr=wav.sample_rate,
                hop_length=features.hop_length,
                backtrack=True,
            )
            candidates = []
            for frame in onset_frames:
                sample = int(frame * features.hop_length)
                conf = self._onset_confidence(features, frame)
                candidates.append(
                    BoundaryEstimate(sample=sample, confidence=conf, label="consonant_start")
                )
            return candidates
        except Exception as exc:
            logger.warning(f"onset 検出失敗: {exc}")
            return self._fallback_consonant_onset(features)

    def _find_vowel_starts(
        self,
        features: AcousticFeatures,
        search_from: Optional[BoundaryEstimate] = None,
    ) -> list[BoundaryEstimate]:
        """
        有声区間の開始を母音候補として返す。
        search_from が指定された場合、その位置以降のみ探索する。
        """
        start_sample = search_from.sample if search_from else 0
        candidates = []
        for s, e in features.voiced_regions:
            if s < start_sample:
                continue
            # ZCR で有声らしさを計算
            frame = s // features.hop_length
            zcr_val = features.zcr[min(frame, len(features.zcr) - 1)]
            conf = 1.0 - min(zcr_val / 0.15, 1.0)
            conf = max(conf, 0.3)
            candidates.append(
                BoundaryEstimate(sample=s, confidence=conf, label="vowel_start")
            )
        return candidates

    def _find_pre_vowel_end(
        self, features: AcousticFeatures
    ) -> Optional[BoundaryEstimate]:
        """
        VCV の先行母音終端 = 最初の有声区間の終了点。
        スペクトル重心の急変をあわせて確認する。
        """
        if not features.voiced_regions:
            return None

        first_voiced_end = features.voiced_regions[0][1]

        # スペクトル変化でサブフレーム精度を上げる
        refined = self._refine_with_spectral_change(
            features, first_voiced_end, window_samples=1024
        )

        conf = 0.7
        if refined != first_voiced_end:
            conf = 0.65  # 修正した分だけ信頼度を下げる

        return BoundaryEstimate(sample=refined, confidence=conf, label="pre_vowel_end")

    def _refine_with_spectral_change(
        self, features: AcousticFeatures, sample: int, window_samples: int
    ) -> int:
        """スペクトル重心の変化率が最大になるフレームに位置をスナップする"""
        center_frame = sample // features.hop_length
        half = window_samples // features.hop_length
        start_f = max(0, center_frame - half)
        end_f = min(len(features.spectral_centroid) - 1, center_frame + half)

        if end_f <= start_f + 1:
            return sample

        centroid_window = features.spectral_centroid[start_f:end_f]
        diff = np.abs(np.diff(centroid_window))
        if len(diff) == 0:
            return sample

        peak_idx = int(np.argmax(diff)) + start_f
        return peak_idx * features.hop_length

    def _onset_confidence(self, features: AcousticFeatures, frame: int) -> float:
        """onset フレームの信頼度を RMS の変化量から計算する"""
        if frame <= 0 or frame >= len(features.rms) - 1:
            return 0.5
        before = features.rms[frame - 1]
        after = features.rms[min(frame + 2, len(features.rms) - 1)]
        ratio = after / (before + 1e-9)
        # 急激な立ち上がり（ratio 大）ほど信頼度が高い
        conf = min(np.log1p(ratio) / np.log1p(20), 1.0)
        return float(max(conf, 0.3))

    def _fallback_consonant_onset(
        self, features: AcousticFeatures
    ) -> list[BoundaryEstimate]:
        """onset 検出が使えないときの RMS 閾値ベースのフォールバック"""
        threshold = np.max(features.rms) * 0.05
        for i, v in enumerate(features.rms):
            if v > threshold:
                return [
                    BoundaryEstimate(
                        sample=i * features.hop_length,
                        confidence=0.45,
                        label="consonant_start",
                    )
                ]
        return []

    def _validate_cv_order(
        self, result: PhonemeEstimation, sample_rate: int
    ) -> PhonemeEstimation:
        """子音開始 < 母音開始 になっているかチェックし、逆転していれば修正する"""
        if result.consonant_start and result.vowel_start:
            if result.consonant_start.sample >= result.vowel_start.sample:
                logger.warning("子音開始 >= 母音開始：候補を入れ替えます")
                # 子音候補の中で母音開始より前のものを選び直す
                valid = [
                    c
                    for c in result.consonant_candidates
                    if c.sample < result.vowel_start.sample
                ]
                if valid:
                    result.consonant_start = max(valid, key=lambda x: x.confidence)
                else:
                    # 母音開始の少し前を子音開始として使う
                    result.consonant_start = BoundaryEstimate(
                        sample=max(0, result.vowel_start.sample - int(0.05 * sample_rate)),
                        confidence=0.3,
                        label="consonant_start",
                    )
        return result
