"""
VCV プロセッサのユニットテスト（改訂版）
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from core.result import OtoParams
from analysis.boundary_estimator import BoundaryEstimate, PhonemeEstimation
from formats.vcv_processor import VCVProcessor


def make_mock_wav(sr=44100, duration=1.2):
    samples = np.zeros(int(sr * duration), dtype=np.float32)
    wav = MagicMock()
    wav.mono = samples
    wav.sample_rate = sr
    wav.duration_sec = duration
    return wav


def make_mock_features(sr=44100):
    n = 240
    f = MagicMock()
    f.sample_rate = sr
    f.frame_length = 1024
    f.hop_length = 256
    f.rms = np.ones(n) * 0.1
    f.rms_times = np.linspace(0, 1.2, n)
    f.voiced_flags = np.ones(n, dtype=bool)
    f.spectral_centroid = np.ones(n) * 2000.0
    f.spectral_flatness = np.ones(n) * 0.1
    f.zcr = np.ones(n) * 0.05
    f.mfcc = np.zeros((n, 13))
    f.mfcc_delta = np.zeros((n, 13))
    f.silence_regions = []
    f.voiced_regions = [(0, int(0.4 * sr)), (int(0.6 * sr), int(1.0 * sr))]
    return f


def make_vcv_phoneme(sr=44100):
    return PhonemeEstimation(
        pre_vowel_end=BoundaryEstimate(sample=int(0.4 * sr), confidence=0.78, label="pre_vowel_end"),
        consonant_start=BoundaryEstimate(sample=int(0.4 * sr), confidence=0.74, label="consonant_start"),
        vowel_start=BoundaryEstimate(sample=int(0.6 * sr), confidence=0.80, label="vowel_start"),
        onset_silence_end=BoundaryEstimate(sample=0, confidence=0.85, label="onset_silence_end"),
    )


class TestVCVProcessor:

    def test_has_pre_vowel(self):
        proc = VCVProcessor(MagicMock())
        assert proc._has_pre_vowel("a_ka") is True
        assert proc._has_pre_vowel("aka") is True
        assert proc._has_pre_vowel("asa") is True
        assert proc._has_pre_vowel("-ka") is False
        assert proc._has_pre_vowel("_sa") is False

    def test_alias_generation(self):
        proc = VCVProcessor(MagicMock())
        assert proc._stem_to_alias("a_ka") == "a ka"
        assert proc._stem_to_alias("asa") == "a sa"
        assert proc._stem_to_alias("i_na") == "i na"

    def test_valid_params(self):
        estimator = MagicMock()
        estimator.estimate_vcv.return_value = make_vcv_phoneme()
        proc = VCVProcessor(estimator)
        params, conf = proc.estimate(Path("a_ka.wav"), make_mock_wav(), make_mock_features())

        assert isinstance(params, OtoParams)
        assert params.alias == "a ka"
        assert params.preutterance > params.overlap
        assert params.cutoff < 0.0
        assert 0.0 <= conf <= 1.0

    def test_overlap_proportional_to_pre_vowel(self):
        """先行母音が長いほど overlap も長くなる（割合ベース）"""
        sr = 44100
        estimator_long = MagicMock()
        phoneme_long = PhonemeEstimation(
            pre_vowel_end=BoundaryEstimate(sample=int(0.6 * sr), confidence=0.8, label="p"),
            consonant_start=BoundaryEstimate(sample=int(0.6 * sr), confidence=0.75, label="c"),
            vowel_start=BoundaryEstimate(sample=int(0.8 * sr), confidence=0.8, label="v"),
            onset_silence_end=BoundaryEstimate(sample=0, confidence=0.9, label="s"),
        )
        estimator_short = MagicMock()
        phoneme_short = PhonemeEstimation(
            pre_vowel_end=BoundaryEstimate(sample=int(0.2 * sr), confidence=0.8, label="p"),
            consonant_start=BoundaryEstimate(sample=int(0.2 * sr), confidence=0.75, label="c"),
            vowel_start=BoundaryEstimate(sample=int(0.4 * sr), confidence=0.8, label="v"),
            onset_silence_end=BoundaryEstimate(sample=0, confidence=0.9, label="s"),
        )
        estimator_long.estimate_vcv.return_value = phoneme_long
        estimator_short.estimate_vcv.return_value = phoneme_short

        proc_long  = VCVProcessor(estimator_long)
        proc_short = VCVProcessor(estimator_short)
        wav = make_mock_wav()
        feat = make_mock_features()

        params_long,  _ = proc_long.estimate(Path("a_ka.wav"),  wav, feat)
        params_short, _ = proc_short.estimate(Path("a_ka.wav"), wav, feat)

        assert params_long.overlap > params_short.overlap
