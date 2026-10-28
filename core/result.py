"""
処理結果を格納するデータクラス
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class OtoParams:
    """
    oto.ini の 1 エントリに対応するパラメータ。
    全ての時刻は ms 単位（UTAU の仕様に合わせて float）。

    oto.ini フォーマット:
      alias=filename,offset,consonant,cutoff,preutterance,overlap
    """

    alias: str
    filename: str
    offset: float       # 先行発声開始オフセット [ms]
    consonant: float    # 子音部分の長さ [ms]
    cutoff: float       # ファイル末尾からのカットオフ（負値）[ms]
    preutterance: float # 先行発声 [ms]
    overlap: float      # オーバーラップ [ms]

    def to_oto_line(self) -> str:
        """oto.ini の 1 行を生成する"""
        return (
            f"{self.filename}={self.alias},"
            f"{self.offset:.2f},{self.consonant:.2f},"
            f"{self.cutoff:.2f},{self.preutterance:.2f},{self.overlap:.2f}"
        )


@dataclass
class ProcessResult:
    """1 ファイルの処理結果"""

    wav_path: Path
    format: str = "unknown"           # "cv" | "vcv" | "unknown"
    oto_params: Optional[OtoParams] = None
    confidence: float = 0.0           # 推定信頼度 0.0〜1.0
    low_confidence: bool = False
    success: bool = False
    error: Optional[str] = None

    @property
    def filename(self) -> str:
        return self.wav_path.name

    @property
    def status_label(self) -> str:
        if not self.success:
            return "ERROR"
        if self.low_confidence:
            return "LOW"
        return "OK"
