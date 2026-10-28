"""
フォーマット検出器のユニットテスト
"""

import pytest
from core.format_detector import FormatDetector


class TestFormatDetector:

    def setup_method(self):
        self.detector = FormatDetector()

    def test_cv_simple(self):
        assert self.detector._classify_name("ka") == "cv"
        assert self.detector._classify_name("sa") == "cv"
        assert self.detector._classify_name("a") == "cv"
        assert self.detector._classify_name("ka1") == "cv"

    def test_vcv_with_separator(self):
        assert self.detector._classify_name("a_ka") == "vcv"
        assert self.detector._classify_name("i-sa") == "vcv"

    def test_vcv_without_separator(self):
        assert self.detector._classify_name("aka") == "vcv"
        assert self.detector._classify_name("asa") == "vcv"
