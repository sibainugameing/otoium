"""
VCV プロセッサのユニットテスト
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from core.result import OtoParams
from analysis.boundary_estimator import BoundaryEstimate, PhonemeEstimation
from formats.vcv_processor import VCVProcessor


def make_mock_wav(sr=44100, duration=1.0):
    samples = np.zeros(int(sr * duration), dtype=np.float32)
    wav = MagicMock()
    wav.mono = samples
    wav.sample_rate = sr
    wav.duration_sec = duration
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
    f.zcr = np.ones(n_frames) * 0.05
    f.mfcc = np.zeros((n_frames, 13))
    f.silence_regions = []
    f.voiced_regions = [(0, int(0.3 * sr)), (int(0.5 * sr), int(0.9 * sr))]
    return f


class TestVCVProcessor:

    def _make_phoneme_with_pre_vowel(self, sr=44100):
        return PhonemeEstimation(
            pre_vowel_end=BoundaryEstimate(sample=int(0.3 * sr), confidence=0.75, label="pre_vowel_end"),
            consonant_start=BoundaryEstimate(sample=int(0.3 * sr), confidence=0.72, label="consonant_start"),
            vowel_start=BoundaryEstimate(sample=int(0.5 * sr), confidence=0.80, label="vowel_start"),
            onset_silence_end=BoundaryEstimate(sample=0, confidence=0.9, label="onset_silence_end"),
        )

    def test_has_pre_vowel_detected(self):
        proc = VCVProcessor(MagicMock())
        assert proc._has_pre_vowel("a_ka") is True
        assert proc._has_pre_vowel("aka") is True
        assert proc._has_pre_vowel("asa") is True

    def test_no_pre_vowel_detected(self):
        proc = VCVProcessor(MagicMock())
        assert proc._has_pre_vowel("-ka") is False

    def test_alias_generation(self):
        proc = VCVProcessor(MagicMock())
        assert proc._stem_to_alias("a_ka") == "a ka"
        assert proc._stem_to_alias("asa") == "a sa"

    def test_estimate_returns_valid_params(self):
        estimator = MagicMock()
        estimator.estimate_vcv.return_value = self._make_phoneme_with_pre_vowel()

        proc = VCVProcessor(estimator)
        wav = make_mock_wav()
        features = make_mock_features()

        params, confidence = proc.estimate(Path("a_ka.wav"), wav, features)

        assert isinstance(params, OtoParams)
        assert params.alias == "a ka"
        assert params.overlap >= 0.0
        assert params.preutterance > params.overlap
        assert params.cutoff < 0.0
        assert 0.0 <= confidence <= 1.0

    def test_overlap_is_in_pre_vowel_region(self):
        """VCV のオーバーラップは先行母音区間に入ること（offset < overlap < preutterance）"""
        estimator = MagicMock()
        estimator.estimate_vcv.return_value = self._make_phoneme_with_pre_vowel()

        proc = VCVProcessor(estimator)
        wav = make_mock_wav()
        features = make_mock_features()

        params, _ = proc.estimate(Path("a_ka.wav"), wav, features)

        assert params.overlap < params.preutterance
