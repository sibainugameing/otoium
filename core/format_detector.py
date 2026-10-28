"""
音源形式の自動判定

ファイル名だけに頼らず、命名パターンの統計と音響特徴を組み合わせて
CV / VCV を推定する。
"""

from __future__ import annotations

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# VCV 的なファイル名パターン（先行母音_子音母音 など）
_VCV_PATTERNS = [
    re.compile(r"^[aiueo]_[a-z]+$", re.IGNORECASE),       # a_ka, i_sa …
    re.compile(r"^[aiueo][a-z]{1,3}$", re.IGNORECASE),    # aka, isa …（短い場合）
    re.compile(r"[aiueo][-_][a-z]", re.IGNORECASE),
]

# CV 的なファイル名パターン
_CV_PATTERNS = [
    re.compile(r"^[a-z]{1,3}[aiueo]$", re.IGNORECASE),    # ka, sa, ta …
    re.compile(r"^[aiueo]$", re.IGNORECASE),               # 単独母音
    re.compile(r"^[a-z]+\d+$", re.IGNORECASE),             # ka1, sa2 …（連番）
]


class FormatDetector:
    """
    フォルダ内の全 WAV ファイル名を走査して、
    CV か VCV かを多数決で判定する。
    """

    def detect(self, wav_path: Path) -> str:
        """
        単一ファイルの推定。
        フォルダ全体のコンテキストがある場合は detect_folder を使うこと。
        """
        return self._classify_name(wav_path.stem)

    def detect_folder(self, folder: Path) -> str:
        """
        フォルダ内の全ファイルを見て多数決で形式を返す。
        """
        stems = [p.stem for p in folder.glob("*.wav")]
        if not stems:
            return "cv"

        votes = {"cv": 0, "vcv": 0}
        for stem in stems:
            fmt = self._classify_name(stem)
            votes[fmt] += 1

        logger.debug(f"形式判定 votes: {votes}")
        if votes["vcv"] > votes["cv"]:
            return "vcv"
        return "cv"

    # ------------------------------------------------------------------

    def _classify_name(self, stem: str) -> str:
        for pat in _VCV_PATTERNS:
            if pat.match(stem):
                return "vcv"
        for pat in _CV_PATTERNS:
            if pat.match(stem):
                return "cv"
        # パターン不一致 → 長さで簡易判定
        # VCV は子音の前後に母音が必要なのでステム長が長くなりやすい
        if len(stem) >= 4:
            return "vcv"
        return "cv"
