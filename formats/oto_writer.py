"""
oto.ini ファイルの書き出し

UTAU 互換フォーマットで出力する。
文字コードは Shift_JIS（cp932）が UTAU 標準。
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.result import ProcessResult

logger = logging.getLogger(__name__)

# UTAU は Shift_JIS（Windows CP932）を標準とする
OTO_ENCODING = "cp932"
OTO_ENCODING_FALLBACK = "utf-8"


class OtoWriter:
    """ProcessResult のリストから oto.ini を生成する"""

    def write(
        self,
        output_path: Path,
        results: list[ProcessResult],
        append: bool = False,
    ) -> None:
        """
        results から oto.ini を書き出す。

        append=True のとき、既存ファイルに追記する（再解析時用）。
        """
        lines = []
        for result in results:
            if result.success and result.oto_params:
                lines.append(result.oto_params.to_oto_line())

        if not lines:
            logger.warning("書き出す有効なパラメータがありません")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"

        # Shift_JIS で書き出す；失敗時は UTF-8
        for encoding in (OTO_ENCODING, OTO_ENCODING_FALLBACK):
            try:
                with output_path.open(mode, encoding=encoding, newline="\r\n") as f:
                    f.write("\n".join(lines) + "\n")
                logger.info(
                    f"oto.ini を書き出しました ({encoding}): {output_path} ({len(lines)} エントリ)"
                )
                return
            except (UnicodeEncodeError, LookupError):
                logger.debug(f"{encoding} でエンコード失敗。次を試します。")

        logger.error("oto.ini の書き出しに失敗しました（エンコードエラー）")

    def read_existing(self, oto_path: Path) -> dict[str, str]:
        """
        既存 oto.ini を読み込み {alias: line} の辞書で返す。
        マージや上書き保護に使用する。
        """
        if not oto_path.exists():
            return {}

        lines = {}
        for encoding in (OTO_ENCODING, OTO_ENCODING_FALLBACK):
            try:
                with oto_path.open("r", encoding=encoding) as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line:
                            # alias は '=' の後、最初の ',' まで
                            rhs = line.split("=", 1)[1]
                            alias = rhs.split(",")[0]
                            lines[alias] = line
                return lines
            except (UnicodeDecodeError, LookupError):
                continue

        logger.warning(f"既存 oto.ini の読み込みに失敗しました: {oto_path}")
        return {}
