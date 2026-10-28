"""
TUI カスタムウィジェット

ResultTable: 処理結果一覧
LogPane: ログ表示
StatusBar: ステータスバー
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from textual.widgets import DataTable, RichLog, Static
from textual.reactive import reactive
from rich.text import Text

from core.result import ProcessResult


# ステータスラベルの色
STATUS_COLORS = {
    "OK": "green",
    "LOW": "yellow",
    "ERROR": "red",
    "SKIP": "dim",
}

LOG_COLORS = {
    "info": "white",
    "warning": "yellow",
    "error": "red bold",
    "success": "green",
    "debug": "dim",
}


class ResultTable(DataTable):
    """処理結果を表示する DataTable"""

    COLUMNS = [
        ("ファイル名", 30),
        ("形式", 6),
        ("状態", 6),
        ("信頼度", 7),
        ("offset", 9),
        ("子音", 8),
        ("先行発声", 8),
        ("overlap", 8),
        ("cutoff", 8),
    ]

    def on_mount(self) -> None:
        for label, width in self.COLUMNS:
            self.add_column(label, width=width)

    def populate(self, results: list[ProcessResult]) -> None:
        """結果リストでテーブルを再描画する"""
        self.clear()
        for result in results:
            self.add_row(*self._result_to_row(result), key=str(result.wav_path))

    def get_selected_wav_path(self) -> Optional[Path]:
        """現在選択中の行の WAV パスを返す"""
        if self.cursor_row < 0:
            return None
        try:
            row_key = self.get_row_at(self.cursor_row)
            # key は str(wav_path) で登録している
            for cell in row_key:
                pass  # DataTable API では key から Path を取得
            # cursor_row から key 経由で取得
            key = list(self._row_locations.keys())[self.cursor_row]
            return Path(str(key))
        except Exception:
            return None

    def _result_to_row(self, result: ProcessResult) -> list:
        status_color = STATUS_COLORS.get(result.status_label, "white")
        status = Text(result.status_label, style=status_color)

        if result.oto_params:
            p = result.oto_params
            return [
                result.filename,
                result.format.upper(),
                status,
                f"{result.confidence:.0%}",
                f"{p.offset:.1f}",
                f"{p.consonant:.1f}",
                f"{p.preutterance:.1f}",
                f"{p.overlap:.1f}",
                f"{p.cutoff:.1f}",
            ]
        else:
            err = (result.error or "")[:30]
            return [
                result.filename,
                result.format.upper(),
                status,
                "—",
                err, "", "", "", "",
            ]


class LogPane(RichLog):
    """ログメッセージを時刻付きで表示するウィジェット"""

    MAX_LINES = 500

    def on_mount(self) -> None:
        self.max_lines = self.MAX_LINES

    def append(self, message: str, level: str = "info") -> None:
        color = LOG_COLORS.get(level, "white")
        ts = datetime.now().strftime("%H:%M:%S")
        prefix = Text(f"[{ts}] ", style="dim")
        body = Text(message, style=color)
        self.write(prefix + body)


class StatusBar(Static):
    """画面下部のステータスバー"""

    status: reactive[str] = reactive("準備完了")

    def render(self) -> str:
        return f" {self.status}"

    def update_status(self, text: str) -> None:
        self.status = text
