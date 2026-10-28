"""
ロギング設定

ファイルとコンソールの両方に出力する。
TUI 使用時はコンソール出力を抑制してファイルのみに書く。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """
    ルートロガーを設定して返す。

    TUI モードでは StreamHandler を付けない
    （Textual が stderr を管理するため）。
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ファイルハンドラ
    if log_file is None:
        log_file = Path("utau_oto_tool.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(numeric_level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # バッチモード（--batch）のときだけ stderr にも出す
    if "--batch" in sys.argv:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(numeric_level)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    return root
