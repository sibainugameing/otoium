"""
BoundaryEstimator のユニットテスト
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

from analysis.boundary_estimator import BoundaryEstimator, _normalize01, _deduplicate, BoundaryEstimate


class TestNormalize01:
    def test_zero_range(self):
        arr = np.ones(5) * 3.0
        result = _normalize01(arr)
        assert np.all(result == 0.0)

    def test_normal(self):
        arr = np.array([0.0, 0.5, 1.0])
        result = _normalize01(arr)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)


class TestDeduplicate:
    def test_merges_close_candidates(self):
        sr = 44100
        hl = 256
        cands = [
            BoundaryEstimate(sample=1000, confidence=0.6, label="v"),
            BoundaryEstimate(sample=1100, confidence=0.8, label="v"),  # 近い → 高信頼度が勝つ
            BoundaryEstimate(sample=9000, confidence=0.5, label="v"),  # 遠い → 別候補
        ]
        result = _deduplicate(cands, hl, tol_ms=30.0, sr=sr)
        assert len(result) == 2
        # 近い 2 つのうち信頼度が高い方が残る
        assert result[0].confidence == pytest.approx(0.8)

    def test_no_merging_when_far(self):
        sr = 44100
        hl = 256
        cands = [
            BoundaryEstimate(sample=0,     confidence=0.7, label="v"),
            BoundaryEstimate(sample=44100, confidence=0.7, label="v"),  # 1s 離れている
        ]
        result = _deduplicate(cands, hl, tol_ms=30.0, sr=sr)
        assert len(result) == 2
