"""
oto.ini ファイルの書き出し

改善点:
- write_raw_lines() を追加（reprocess での上書き更新用）。
- read_existing() の alias 取得ロジックを修正（`=` の後ろ最初の `,` まで）。
- 書き出し前に重複 alias を検出して最後のエントリのみ残す。
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.result import ProcessResult

logger = logging.getLogger(__name__)

OTO_ENCODING = "cp932"
OTO_ENCODING_FALLBACK = "utf-8"


class OtoWriter:

    def write(
        self,
        output_path: Path,
        results: list[ProcessResult],
        append: bool = False,
    ) -> None:
        """results から oto.ini を書き出す（重複 alias は最後のものが勝つ）"""
        # alias をキーにして重複を排除
        entries: dict[str, str] = {}
        for result in results:
            if result.success and result.oto_params:
                entries[result.oto_params.alias] = result.oto_params.to_oto_line()

        if not entries:
            logger.warning("書き出す有効なパラメータがありません")
            return

        if append:
            existing = self.read_existing(output_path)
            existing.update(entries)   # 新しい値で上書き
            lines = list(existing.values())
        else:
            lines = list(entries.values())

        self._write_lines(output_path, lines)
        logger.info(f"oto.ini 書き出し: {output_path} ({len(lines)} エントリ)")

    def write_raw_lines(self, output_path: Path, lines: list[str]) -> None:
        """生の行リストをそのまま書き出す（reprocess での全エントリ更新用）"""
        self._write_lines(output_path, lines)

    def read_existing(self, oto_path: Path) -> dict[str, str]:
        """既存 oto.ini を {alias: line} 辞書で返す"""
        if not oto_path.exists():
            return {}

        for encoding in (OTO_ENCODING, OTO_ENCODING_FALLBACK):
            try:
                result: dict[str, str] = {}
                with oto_path.open("r", encoding=encoding) as f:
                    for line in f:
                        line = line.strip()
                        if not line or "=" not in line:
                            continue
                        rhs = line.split("=", 1)[1]
                        alias = rhs.split(",")[0]
                        result[alias] = line
                return result
            except (UnicodeDecodeError, LookupError):
                continue

        logger.warning(f"既存 oto.ini の読み込みに失敗: {oto_path}")
        return {}

    # ------------------------------------------------------------------

    def _write_lines(self, output_path: Path, lines: list[str]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for encoding in (OTO_ENCODING, OTO_ENCODING_FALLBACK):
            try:
                with output_path.open("w", encoding=encoding, newline="\r\n") as f:
                    f.write("\r\n".join(lines) + "\r\n")
                logger.debug(f"エンコード: {encoding}")
                return
            except (UnicodeEncodeError, LookupError):
                continue
        logger.error("oto.ini の書き出しに失敗（エンコードエラー）")
