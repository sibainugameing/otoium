"""
CV プロセッサのユニットテスト（改訂版）
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from core.result import OtoParams
from analysis.boundary_estimator import BoundaryEstimate, PhonemeEstimation
from formats.cv_processor import CVProcessor, OVERLAP_OFFSET_BY_TYPE


def make_mock_wav(sr=44100, duration=1.0, stem="ka"):
    samples = np.zeros(int(sr * duration), dtype=np.float32)
    start = int(0.1 * sr)
    samples[start : start + int(0.5 * sr)] = 0.5
    wav = MagicMock()
    wav.mono = samples
    wav.sample_rate = sr
    wav.duration_sec = duration
    wav.path = MagicMock()
    wav.path.stem = stem
    return wav


def make_mock_features(sr=44100):
    n_frames = 200
    f = MagicMock()
    f.sample_rate = sr
    f.frame_length = 1024
    f.hop_length = 256
    f.rms = np.ones(n_frames) * 0.1
    f.rms_times = np.linspace(0, 1.0, n_frames)
    f.voiced_flags = np.ones(n_frames, dtype=bool)
    f.spectral_centroid = np.ones(n_frames) * 2000.0
    f.spectral_flatness = np.ones(n_frames) * 0.1
    f.zcr = np.ones(n_frames) * 0.05
    f.mfcc = np.zeros((n_frames, 13))
    f.mfcc_delta = np.zeros((n_frames, 13))
    f.silence_regions = [(0, int(0.05 * sr))]
    f.voiced_regions = [(int(0.1 * sr), int(0.8 * sr))]
    return f


def make_phoneme(sr=44100):
    return PhonemeEstimation(
        consonant_start=BoundaryEstimate(sample=int(0.1 * sr), confidence=0.82, label="consonant_start"),
        vowel_start=BoundaryEstimate(sample=int(0.18 * sr), confidence=0.78, label="vowel_start"),
        onset_silence_end=BoundaryEstimate(sample=int(0.05 * sr), confidence=0.90, label="onset_silence_end"),
    )


class TestCVProcessor:

    def test_returns_valid_params_and_confidence(self):
        estimator = MagicMock()
        estimator.estimate_cv.return_value = make_phoneme()
        proc = CVProcessor(estimator)
        params, conf = proc.estimate(Path("ka.wav"), make_mock_wav(), make_mock_features())

        assert isinstance(params, OtoParams)
        assert 0.0 <= conf <= 1.0
        assert params.alias == "ka"
        assert params.filename == "ka.wav"
        assert params.offset >= 0.0
        assert params.preutterance > 0.0
        assert params.overlap > 0.0
        assert params.cutoff < 0.0

    def test_preutterance_greater_than_overlap(self):
        estimator = MagicMock()
        estimator.estimate_cv.return_value = make_phoneme()
        proc = CVProcessor(estimator)
        params, _ = proc.estimate(Path("sa.wav"), make_mock_wav(stem="sa"), make_mock_features())
        assert params.preutterance > params.overlap

    def test_alias_strips_digit_suffix(self):
        proc = CVProcessor(MagicMock())
        assert proc._stem_to_alias("ka1") == "ka"
        assert proc._stem_to_alias("sa2") == "sa"
        assert proc._stem_to_alias("a") == "a"

    def test_consonant_not_clipped(self):
        """実測値がヒント範囲を外れてもクリップしない（異常値警告のみ）"""
        estimator = MagicMock()
        sr = 44100
        # 子音区間を意図的に 300ms に設定（ヒント上限を超える）
        phoneme = PhonemeEstimation(
            consonant_start=BoundaryEstimate(sample=int(0.1 * sr), confidence=0.8, label="c"),
            vowel_start=BoundaryEstimate(sample=int(0.4 * sr), confidence=0.8, label="v"),
        )
        estimator.estimate_cv.return_value = phoneme
        proc = CVProcessor(estimator)
        params, _ = proc.estimate(Path("ka.wav"), make_mock_wav(), make_mock_features())
        # 実測 300ms がそのまま使われる（クリップされない）
        assert params.consonant > 200.0

    def test_overlap_plosive_smaller_than_nasal(self):
        """破裂音の overlap オフセットは鼻音より小さい（手前に置く）"""
        assert OVERLAP_OFFSET_BY_TYPE.get("k", 15.0) < OVERLAP_OFFSET_BY_TYPE.get("n", 15.0)
        assert OVERLAP_OFFSET_BY_TYPE.get("p", 15.0) < OVERLAP_OFFSET_BY_TYPE.get("m", 15.0)

    def test_offset_not_before_silence_end(self):
        """offset が先頭無音終端より前にならない"""
        estimator = MagicMock()
        sr = 44100
        phoneme = PhonemeEstimation(
            consonant_start=BoundaryEstimate(sample=int(0.08 * sr), confidence=0.8, label="c"),
            onset_silence_end=BoundaryEstimate(sample=int(0.09 * sr), confidence=0.9, label="s"),
        )
        estimator.estimate_cv.return_value = phoneme
        proc = CVProcessor(estimator)
        wav = make_mock_wav()
        params, _ = proc.estimate(Path("ka.wav"), wav, make_mock_features())
        silence_end_ms = int(0.09 * sr) / sr * 1000.0
        assert params.offset >= silence_end_ms - 0.1  # 浮動小数点許容
