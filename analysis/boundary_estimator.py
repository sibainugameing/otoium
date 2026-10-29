"""
音素境界推定

精度改善の核心:

[CV]
1. スペクトルフラックスのピーク = 子音開始（音が急変する点）
2. MFCC C1-C4 の重み付き変化 = 母音開始（フォルマント遷移）
3. ファイル名から子音種別を特定し、種別ごとの探索戦略を切り替える
   - 無声破裂音(k/p/t): 短い無音後の突発的エネルギー上昇
   - 無声摩擦音(s/sh/f): 高域エネルギー比の上昇 → 下降
   - 有声子音(n/m/r/y): ZCR と重心変化を使う
4. RMS 立ち上がり勾配でフレーム精度を補正

[VCV]
1. スペクトルフラックス + MFCC C1-C4 変化 の複合スコアで先行母音終端を特定
2. 探索範囲を「ファイル全体の 15〜65%」に限定（前後を切り捨て）
3. 後続母音開始は子音開始 + 子音の最短長以降で MFCC が安定化する点

[共通]
- 各推定値に「種別ごとの期待信頼度」を掛け合わせた信頼度を持つ
- 候補が複数あるとき「子音区間長の合理性」スコアで絞る
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import find_peaks

from analysis.feature_extractor import AcousticFeatures
from analysis.wav_loader import WavData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# 子音種別の分類
# ---------------------------------------------------------------
# 無声破裂音: 短い閉鎖後の爆発。エネルギーが突然立ち上がる。
PLOSIVE_UNVOICED = {"k", "ky", "g", "gy", "p", "py", "b", "by", "t", "d"}
# 無声摩擦音: 高域エネルギーが高い区間が続く。
FRICATIVE_UNVOICED = {"s", "sh", "z", "f", "v", "h", "hy"}
# 破擦音
AFFRICATE = {"ch", "ts"}
# 有声共鳴音: 子音部分も有声。
SONORANT = {"n", "ny", "m", "my", "r", "ry", "y", "w"}

# 最低子音長 [ms]（これより短い子音区間は無効）
CONSONANT_MIN_MS: dict[str, float] = {
    "plosive":    25.0,
    "fricative":  50.0,
    "affricate":  45.0,
    "sonorant":   30.0,
    "unknown":    20.0,
}
CONSONANT_MAX_MS: dict[str, float] = {
    "plosive":   180.0,
    "fricative": 250.0,
    "affricate": 200.0,
    "sonorant":  180.0,
    "unknown":   300.0,
}

MFCC_DELTA_NORM_PERCENTILE = 80  # 上位 20% を遷移候補とする


@dataclass
class BoundaryEstimate:
    sample: int
    confidence: float
    label: str = ""

    def to_ms(self, sample_rate: int) -> float:
        return self.sample / sample_rate * 1000.0


@dataclass
class PhonemeEstimation:
    consonant_start:    Optional[BoundaryEstimate] = None
    vowel_start:        Optional[BoundaryEstimate] = None
    pre_vowel_end:      Optional[BoundaryEstimate] = None
    onset_silence_end:  Optional[BoundaryEstimate] = None
    consonant_candidates: list[BoundaryEstimate] = field(default_factory=list)
    vowel_candidates:     list[BoundaryEstimate] = field(default_factory=list)

    @property
    def overall_confidence(self) -> float:
        scores = [e.confidence for e in [self.consonant_start, self.vowel_start] if e]
        return float(np.mean(scores)) if scores else 0.0


class BoundaryEstimator:

    CONFIDENCE_THRESHOLD = 0.52

    def estimate_cv(self, wav: WavData, features: AcousticFeatures) -> PhonemeEstimation:
        result = PhonemeEstimation()
        stem = getattr(getattr(wav, "path", None), "stem", "") or ""
        cons_type = _consonant_type(stem)

        result.onset_silence_end = self._onset_silence(features)

        # --- 子音開始 ---
        cons_cands = self._find_consonant_start(wav, features, cons_type, stem)
        result.consonant_candidates = cons_cands
        result.consonant_start = cons_cands[0] if cons_cands else None

        # --- 母音開始 ---
        vowel_cands = self._find_vowel_start_cv(features, result.consonant_start, cons_type, wav.sample_rate)
        result.vowel_candidates = vowel_cands
        result.vowel_start = vowel_cands[0] if vowel_cands else None

        result = self._enforce_cv_order(result, wav.sample_rate)
        if result.overall_confidence < self.CONFIDENCE_THRESHOLD:
            result = self._refine_cv(result, wav, features, cons_type)

        logger.debug(
            f"CV [{cons_type}] {stem}: "
            f"cons={_ms(result.consonant_start, wav.sample_rate)} "
            f"vowel={_ms(result.vowel_start, wav.sample_rate)} "
            f"conf={result.overall_confidence:.2f}"
        )
        return result

    def estimate_vcv(self, wav: WavData, features: AcousticFeatures) -> PhonemeEstimation:
        result = PhonemeEstimation()
        stem = getattr(getattr(wav, "path", None), "stem", "") or ""

        pre_end = self._find_pre_vowel_end_vcv(features)
        result.pre_vowel_end = pre_end
        if pre_end:
            result.consonant_start = BoundaryEstimate(
                sample=pre_end.sample,
                confidence=pre_end.confidence * 0.92,
                label="consonant_start",
            )

        # 後続母音の子音種別は stem の後半から判定
        post_stem = _extract_post_consonant(stem)
        cons_type  = _consonant_type(post_stem)

        vowel_cands = self._find_vowel_start_vcv(features, pre_end, cons_type, wav.sample_rate)
        result.vowel_candidates = vowel_cands
        result.vowel_start = vowel_cands[0] if vowel_cands else None

        if result.overall_confidence < self.CONFIDENCE_THRESHOLD:
            result = self._refine_vcv(result, features)

        logger.debug(
            f"VCV [{cons_type}] {stem}: "
            f"pre_end={_ms(result.pre_vowel_end, wav.sample_rate)} "
            f"vowel={_ms(result.vowel_start, wav.sample_rate)} "
            f"conf={result.overall_confidence:.2f}"
        )
        return result

    # ==================================================================
    # CV: 子音開始
    # ==================================================================

    def _find_consonant_start(
        self,
        wav: WavData,
        features: AcousticFeatures,
        cons_type: str,
        stem: str,
    ) -> list[BoundaryEstimate]:
        """
        子音種別ごとに最適な手法を選ぶ。
        """
        if cons_type == "plosive":
            return self._cons_plosive(wav, features)
        elif cons_type == "fricative":
            return self._cons_fricative(features)
        elif cons_type == "affricate":
            return self._cons_affricate(wav, features)
        elif cons_type == "sonorant":
            return self._cons_sonorant(features)
        else:
            # 不明: 複数手法を試して信頼度で競わせる
            cands = (
                self._cons_plosive(wav, features)
                + self._cons_fricative(features)
                + self._cons_sonorant(features)
            )
            return _top(cands, 3)

    def _cons_plosive(self, wav: WavData, features: AcousticFeatures) -> list[BoundaryEstimate]:
        """
        無声破裂音: スペクトルフラックスの最初のピーク。
        破裂音は無音（閉鎖）→ 爆発でフラックスが急増する。
        """
        hl  = features.hop_length
        flux = features.spectral_flux

        # 先頭無音の終端以降のみ探索
        search_start = self._silence_end_frame(features)
        w = flux[search_start:]
        if len(w) < 3:
            return self._flux_fallback(features)

        # 最初の強いピーク
        thresh = np.percentile(w, 70)
        peaks, _ = find_peaks(w, height=thresh, distance=2)
        if len(peaks) == 0:
            return self._flux_fallback(features)

        cands = []
        for p in peaks[:4]:
            frame = p + search_start
            # backtrack: RMS が立ち上がり始めるフレームに補正
            frame = self._rms_rise_frame(features, frame)
            conf  = self._flux_confidence(features, p, w)
            cands.append(BoundaryEstimate(sample=frame * hl, confidence=conf, label="consonant_start"))
        return _top(cands, 3)

    def _cons_fricative(self, features: AcousticFeatures) -> list[BoundaryEstimate]:
        """
        摩擦音: 高域エネルギー比（HFR）の上昇開始点。
        HFR が閾値を超えた最初のフレームを子音開始とみなす。
        """
        hl  = features.hop_length
        hfr = features.high_freq_ratio

        search_start = self._silence_end_frame(features)
        w = hfr[search_start:]
        if len(w) < 3:
            return []

        # HFR の移動平均でノイズ除去
        kernel = min(5, len(w))
        smooth = np.convolve(w, np.ones(kernel) / kernel, mode="same")
        threshold = np.percentile(smooth, 55)

        cands = []
        for i, v in enumerate(smooth):
            if v > threshold:
                frame = i + search_start
                conf  = min(float(v / (smooth.max() + 1e-9)) * 1.2, 1.0)
                conf  = max(conf, 0.45)
                cands.append(BoundaryEstimate(
                    sample=frame * hl, confidence=conf, label="consonant_start"
                ))
                break  # 最初の上昇点のみ

        return cands

    def _cons_affricate(self, wav: WavData, features: AcousticFeatures) -> list[BoundaryEstimate]:
        """
        破擦音: 破裂 + 摩擦の複合。フラックスピーク + HFR 上昇の早い方。
        """
        plosive  = self._cons_plosive(wav, features)
        fricative = self._cons_fricative(features)
        return _top(plosive + fricative, 3)

    def _cons_sonorant(self, features: AcousticFeatures) -> list[BoundaryEstimate]:
        """
        共鳴音(n/m/r/y/w): RMS の立ち上がり + スペクトル重心の低下。
        母音と音色が近いので MFCC デルタで補助。
        """
        hl  = features.hop_length
        search_start = self._silence_end_frame(features)

        rms_grad  = np.diff(features.rms, prepend=features.rms[0])
        delta_n   = np.linalg.norm(features.mfcc_delta, axis=1)

        combined  = _normalize01(rms_grad) * 0.6 + _normalize01(delta_n) * 0.4
        w = combined[search_start:]
        if len(w) < 3:
            return self._flux_fallback(features)

        thresh = np.percentile(w, 65)
        peaks, _ = find_peaks(w, height=thresh, distance=3)
        if len(peaks) == 0:
            return self._flux_fallback(features)

        cands = []
        for p in peaks[:3]:
            frame = p + search_start
            conf  = float(min(w[p] * 1.3, 1.0))
            conf  = max(conf, 0.4)
            cands.append(BoundaryEstimate(sample=frame * hl, confidence=conf, label="consonant_start"))
        return cands

    # ==================================================================
    # CV: 母音開始
    # ==================================================================

    def _find_vowel_start_cv(
        self,
        features: AcousticFeatures,
        consonant_start: Optional[BoundaryEstimate],
        cons_type: str,
        sr: int,
    ) -> list[BoundaryEstimate]:
        """
        母音開始 = MFCC C1-C4 の変化が収まり始めるフレーム。
        C1-C4 はフォルマント（声道形状）に対応し、子音→母音で大きく変化する。

        有声子音の場合は子音区間の最低長 (cons_min_ms) 後から探索する。
        """
        hl  = features.hop_length
        sr  = features.sample_rate if hasattr(features, "sample_rate") else sr

        # 探索開始フレームを子音種別の最低長で制限
        if consonant_start:
            cons_frame  = consonant_start.sample // hl
            min_ms      = CONSONANT_MIN_MS.get(cons_type, 20.0)
            wait_frames = max(1, int(min_ms / 1000.0 * sr / hl))
            search_start = cons_frame + wait_frames
        else:
            search_start = self._silence_end_frame(features)

        # MFCC C1-C4 重み付きデルタノルム
        delta_c14 = features.mfcc_delta[:, 1:5]        # C1-C4 のみ
        weights   = np.array([2.0, 1.5, 1.2, 1.0])     # C1 を最重視
        delta_w   = np.abs(delta_c14) * weights
        delta_score = delta_w.sum(axis=1)               # (frames,)

        # スペクトルフラックス も補助に使う
        flux_score  = _normalize01(features.spectral_flux)

        combined = 0.65 * _normalize01(delta_score) + 0.35 * flux_score
        w = combined[search_start:]
        if len(w) < 3:
            return self._vowel_fallback(features, search_start)

        # ピーク直後の「安定化点」= フォルマントが落ち着いた母音開始
        thresh = np.percentile(w, MFCC_DELTA_NORM_PERCENTILE)
        peaks, _ = find_peaks(w, height=thresh, distance=3)
        if len(peaks) == 0:
            return self._vowel_fallback(features, search_start)

        cands = []
        max_val = w.max() + 1e-9
        for p in peaks[:4]:
            frame = p + search_start
            conf  = float(min(w[p] / max_val * 1.2, 1.0))
            conf  = max(conf, 0.38)
            cands.append(BoundaryEstimate(
                sample=frame * hl, confidence=conf, label="vowel_start"
            ))
        return _top(cands, 3)

    # ==================================================================
    # VCV: 先行母音終端
    # ==================================================================

    def _find_pre_vowel_end_vcv(self, features: AcousticFeatures) -> Optional[BoundaryEstimate]:
        """
        先行母音→子音 の境界。
        スペクトルフラックス・MFCC C1-C4 デルタ・RMS 変化の複合スコアで探す。
        探索範囲は音声全体の 15〜65%（前後を除外）。
        """
        hl = features.hop_length
        n  = len(features.spectral_centroid)

        start_f = max(2, int(n * 0.12))
        end_f   = min(n - 2, int(n * 0.68))
        if end_f <= start_f + 3:
            return None

        # MFCC C1-C4 重み付きデルタ
        delta_c14  = features.mfcc_delta[:, 1:5]
        weights    = np.array([2.0, 1.5, 1.2, 1.0])
        delta_score = (np.abs(delta_c14) * weights).sum(axis=1)

        # スペクトルフラックス
        flux_score  = features.spectral_flux

        # RMS 変化（下降→上昇 のパターンを検出）
        rms_diff = np.abs(np.diff(features.rms, prepend=features.rms[0]))

        combined = (
            0.50 * _normalize01(delta_score) +
            0.30 * _normalize01(flux_score)  +
            0.20 * _normalize01(rms_diff)
        )

        w = combined[start_f:end_f]
        peaks, props = find_peaks(w, prominence=0.06, distance=4)

        if len(peaks) == 0:
            # フォールバック: ウィンドウ内の最大値
            best_local = int(np.argmax(w))
            frame = best_local + start_f
            conf  = float(min(combined[frame] * 1.1, 1.0))
            conf  = max(conf, 0.40)
        else:
            # 最も顕著なピーク
            best_idx = int(peaks[np.argmax(props["prominences"])])
            frame = best_idx + start_f
            conf  = float(min(combined[frame] * 1.15, 1.0))
            conf  = max(conf, 0.45)

        return BoundaryEstimate(sample=frame * hl, confidence=conf, label="pre_vowel_end")

    # ==================================================================
    # VCV: 後続母音開始
    # ==================================================================

    def _find_vowel_start_vcv(
        self,
        features: AcousticFeatures,
        pre_vowel_end: Optional[BoundaryEstimate],
        cons_type: str,
        sr: int,
    ) -> list[BoundaryEstimate]:
        """
        先行母音終端の後から、MFCC C1-C4 の変化が収まる点（＝母音安定化）を探す。
        """
        hl = features.hop_length

        if pre_vowel_end:
            min_ms = CONSONANT_MIN_MS.get(cons_type, 20.0)
            wait   = max(2, int(min_ms / 1000.0 * sr / hl))
            search_start = pre_vowel_end.sample // hl + wait
        else:
            search_start = 0

        return self._find_vowel_start_cv(features, None, cons_type, sr)

    # ==================================================================
    # 共通ユーティリティ
    # ==================================================================

    def _onset_silence(self, features: AcousticFeatures) -> Optional[BoundaryEstimate]:
        if not features.silence_regions:
            return BoundaryEstimate(sample=0, confidence=0.60, label="onset_silence_end")
        first = features.silence_regions[0]
        if first[0] == 0:
            return BoundaryEstimate(sample=first[1], confidence=0.90, label="onset_silence_end")
        return BoundaryEstimate(sample=0, confidence=0.65, label="onset_silence_end")

    def _silence_end_frame(self, features: AcousticFeatures) -> int:
        if features.silence_regions and features.silence_regions[0][0] == 0:
            return features.silence_regions[0][1] // features.hop_length
        return 0

    def _rms_rise_frame(self, features: AcousticFeatures, frame: int) -> int:
        """onset フレーム以降で RMS 勾配が最大になるフレームへスナップ"""
        end = min(frame + 6, len(features.rms) - 1)
        if end <= frame:
            return frame
        grad = np.diff(features.rms[frame:end + 1])
        return frame + int(np.argmax(grad))

    def _flux_confidence(self, features: AcousticFeatures, local_idx: int, window: np.ndarray) -> float:
        max_v = window.max() + 1e-9
        val   = window[local_idx]
        return float(max(min(val / max_v * 1.2, 1.0), 0.40))

    def _flux_fallback(self, features: AcousticFeatures) -> list[BoundaryEstimate]:
        hl = features.hop_length
        threshold = np.max(features.rms) * 0.04
        for i, v in enumerate(features.rms):
            if v > threshold:
                return [BoundaryEstimate(sample=i * hl, confidence=0.35, label="consonant_start")]
        return []

    def _vowel_fallback(
        self, features: AcousticFeatures, search_start: int
    ) -> list[BoundaryEstimate]:
        """有声区間の最初の開始を母音候補として返す（フォールバック）"""
        hl = features.hop_length
        for s, e in features.voiced_regions:
            if s >= search_start * hl:
                zcr  = features.zcr[min(s // hl, len(features.zcr) - 1)]
                conf = max(0.35, 1.0 - zcr / 0.25)
                return [BoundaryEstimate(sample=s, confidence=conf, label="vowel_start")]
        return []

    # ==================================================================
    # 整合性 & 低信頼度再推定
    # ==================================================================

    def _enforce_cv_order(self, result: PhonemeEstimation, sr: int) -> PhonemeEstimation:
        if not (result.consonant_start and result.vowel_start):
            return result
        if result.consonant_start.sample < result.vowel_start.sample:
            return result

        logger.warning("子音 >= 母音：候補を入れ替えます")
        valid = [c for c in result.consonant_candidates if c.sample < result.vowel_start.sample]
        if valid:
            result.consonant_start = max(valid, key=lambda x: x.confidence)
        else:
            result.consonant_start = BoundaryEstimate(
                sample=max(0, result.vowel_start.sample - int(0.05 * sr)),
                confidence=0.25,
                label="consonant_start",
            )
        return result

    def _refine_cv(self, result, wav, features, cons_type) -> PhonemeEstimation:
        """全候補の組み合わせから最良ペアを選ぶ"""
        all_c = result.consonant_candidates or ([result.consonant_start] if result.consonant_start else [])
        all_v = result.vowel_candidates     or ([result.vowel_start]     if result.vowel_start     else [])

        lo = CONSONANT_MIN_MS.get(cons_type, 20.0)
        hi = CONSONANT_MAX_MS.get(cons_type, 300.0)

        best, best_score = None, -1.0
        for c in all_c:
            for v in all_v:
                if c.sample >= v.sample:
                    continue
                dur = (v.sample - c.sample) / wav.sample_rate * 1000.0
                if not (lo <= dur <= hi):
                    continue
                score = c.confidence + v.confidence
                if score > best_score:
                    best_score, best = score, (c, v)
        if best:
            result.consonant_start, result.vowel_start = best
        return result

    def _refine_vcv(self, result, features) -> PhonemeEstimation:
        """VCV 低信頼度: 探索範囲を広げて再実行"""
        n = len(features.spectral_centroid)
        if n < 5:
            return result
        delta_score = (np.abs(features.mfcc_delta[:, 1:5]) * np.array([2, 1.5, 1.2, 1])).sum(axis=1)
        combined = (
            0.55 * _normalize01(delta_score) +
            0.45 * _normalize01(features.spectral_flux)
        )
        start, end = int(n * 0.08), int(n * 0.75)
        w = combined[start:end]
        if len(w) == 0:
            return result
        best_local = int(np.argmax(w))
        frame = best_local + start
        conf  = float(min(combined[frame] * 1.1, 1.0))
        if conf > (result.pre_vowel_end.confidence if result.pre_vowel_end else 0.0):
            s = frame * features.hop_length
            result.pre_vowel_end  = BoundaryEstimate(sample=s, confidence=conf, label="pre_vowel_end")
            result.consonant_start = BoundaryEstimate(sample=s, confidence=conf * 0.9, label="consonant_start")
        return result


# ==================================================================
# モジュールユーティリティ
# ==================================================================

def _normalize01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _top(cands: list[BoundaryEstimate], n: int) -> list[BoundaryEstimate]:
    return sorted(cands, key=lambda x: x.confidence, reverse=True)[:n]


def _deduplicate(cands, hop_length, tol_ms, sr):
    if not cands:
        return []
    tol = int(tol_ms / 1000.0 * sr)
    cands = sorted(cands, key=lambda x: x.sample)
    out = [cands[0]]
    for c in cands[1:]:
        if abs(c.sample - out[-1].sample) < tol:
            if c.confidence > out[-1].confidence:
                out[-1] = c
        else:
            out.append(c)
    return out


def _consonant_type(stem: str) -> str:
    s = stem.lower().lstrip("-_ ")
    for key in sorted(
        list(PLOSIVE_UNVOICED) + list(FRICATIVE_UNVOICED) + list(AFFRICATE) + list(SONORANT),
        key=len, reverse=True
    ):
        if s.startswith(key):
            if key in PLOSIVE_UNVOICED:   return "plosive"
            if key in FRICATIVE_UNVOICED: return "fricative"
            if key in AFFRICATE:          return "affricate"
            if key in SONORANT:           return "sonorant"
    return "unknown"


def _extract_post_consonant(stem: str) -> str:
    """VCV ステムから後続子音部分を取り出す（a_ka → ka, aka → ka）"""
    m = re.match(r"^[aiueoAIUEO][-_]?(.+)$", stem)
    return m.group(1) if m else stem


def _ms(est: Optional[BoundaryEstimate], sr: int) -> str:
    if est is None:
        return "None"
    return f"{est.to_ms(sr):.1f}ms(c={est.confidence:.2f})"
