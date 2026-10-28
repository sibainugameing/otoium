"""
CV プロセッサのユニットテスト
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.result import OtoParams
from analysis.boundary_estimator import BoundaryEstimate, PhonemeEstimation
from formats.cv_processor import CVProcessor


def make_mock_wav(sr=44100, duration=1.0):
    samples = np.zeros(int(sr * duration), dtype=np.float32)
    # 100ms 以降に音声を配置
    start = int(0.1 * sr)
    samples[start : start + int(0.5 * sr)] = 0.5
    wav = MagicMock()
    wav.mono = samples
    wav.sample_rate = sr
    wav.duration_sec = duration
    wav.samples_to_ms = lambda s: s / sr * 1000.0
    wav.ms_to_samples = lambda ms: int(ms / 1000.0 * sr)
    return wav


def make_mock_features(sr=44100):
    n_frames = 200
    features = MagicMock()
    features.sample_rate = sr
    features.frame_length = 1024
    features.hop_length = 256
    features.rms = np.ones(n_frames) * 0.1
    features.rms_times = np.linspace(0, 1.0, n_frames)
    features.voiced_flags = np.ones(n_frames, dtype=bool)
    features.spectral_centroid = np.ones(n_frames) * 2000.0
    features.zcr = np.ones(n_frames) * 0.05
    features.mfcc = np.zeros((n_frames, 13))
    features.silence_regions = [(0, int(0.05 * sr))]
    features.voiced_regions = [(int(0.1 * sr), int(0.8 * sr))]
    return features


class TestCVProcessor:

    def test_estimate_returns_params_and_confidence(self):
        estimator = MagicMock()
        # mock phoneme estimation
        phoneme = PhonemeEstimation(
            consonant_start=BoundaryEstimate(sample=4410, confidence=0.8, label="consonant_start"),
            vowel_start=BoundaryEstimate(sample=6615, confidence=0.75, label="vowel_start"),
            onset_silence_end=BoundaryEstimate(sample=2205, confidence=0.9, label="onset_silence_end"),
        )
        estimator.estimate_cv.return_value = phoneme

        processor = CVProcessor(estimator)
        wav = make_mock_wav()
        features = make_mock_features()
        wav_path = Path("ka.wav")

        params, confidence = processor.estimate(wav_path, wav, features)

        assert isinstance(params, OtoParams)
        assert 0.0 <= confidence <= 1.0
        assert params.alias == "ka"
        assert params.filename == "ka.wav"
        assert params.offset >= 0.0
        assert params.preutterance > 0.0
        assert params.overlap > 0.0
        assert params.cutoff < 0.0

    def test_preutterance_greater_than_overlap(self):
        estimator = MagicMock()
        phoneme = PhonemeEstimation(
            consonant_start=BoundaryEstimate(sample=4410, confidence=0.8, label="consonant_start"),
            vowel_start=BoundaryEstimate(sample=8820, confidence=0.75, label="vowel_start"),
        )
        estimator.estimate_cv.return_value = phoneme

        processor = CVProcessor(estimator)
        wav = make_mock_wav()
        features = make_mock_features()

        params, _ = processor.estimate(Path("sa.wav"), wav, features)

        # 先行発声 > オーバーラップ が UTAU の基本条件
        assert params.preutterance > params.overlap

    def test_alias_strips_number_suffix(self):
        processor = CVProcessor(MagicMock())
        assert processor._stem_to_alias("ka1") == "ka"
        assert processor._stem_to_alias("sa2") == "sa"
        assert processor._stem_to_alias("a") == "a"

    def test_consonant_hint_plosive(self):
        processor = CVProcessor(MagicMock())
        hint = processor._consonant_hint("ka")
        assert hint is not None
        lo, hi = hint
        assert lo < hi

    def test_overlap_plosive_smaller_than_nasal(self):
        """破裂音のオーバーラップは鼻音より小さくなる"""
        processor = CVProcessor(MagicMock())
        pre = 50.0
        overlap_k = processor._compute_overlap_cv(MagicMock(), pre, "ka")
        overlap_n = processor._compute_overlap_cv(MagicMock(), pre, "na")
        assert overlap_k < overlap_n
