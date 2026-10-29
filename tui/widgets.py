"""
TUI カスタムウィジェット
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from textual.widgets import DataTable, RichLog, Static
from rich.text import Text

from core.result import ProcessResult

STATUS_COLORS = {"OK": "green", "LOW": "yellow", "ERROR": "red"}
LOG_COLORS    = {"info": "white", "warning": "yellow", "error": "red bold",
                 "success": "green", "debug": "dim"}


class ResultTable(DataTable):

    COLUMNS = [
        ("ファイル名",  28),
        ("形式",         5),
        ("状態",         5),
        ("信頼度",       7),
        ("offset",       8),
        ("子音",         7),
        ("先行発声",     8),
        ("overlap",      8),
        ("cutoff",       8),
    ]

    def on_mount(self) -> None:
        for label, width in self.COLUMNS:
            self.add_column(label, width=width)

    def populate(self, results: list[ProcessResult]) -> None:
        self.clear()
        for r in results:
            self.add_row(*self._to_row(r), key=str(r.wav_path))

    def get_selected_wav_path(self) -> Optional[Path]:
        try:
            keys = list(self._row_locations.keys())
            if self.cursor_row < 0 or self.cursor_row >= len(keys):
                return None
            return Path(str(keys[self.cursor_row]))
        except Exception:
            return None

    def _to_row(self, r: ProcessResult) -> list:
        color  = STATUS_COLORS.get(r.status_label, "white")
        status = Text(r.status_label, style=color)
        if r.oto_params:
            p = r.oto_params
            return [
                r.filename, r.format.upper(), status,
                f"{r.confidence:.0%}",
                f"{p.offset:.1f}", f"{p.consonant:.1f}",
                f"{p.preutterance:.1f}", f"{p.overlap:.1f}", f"{p.cutoff:.1f}",
            ]
        return [
            r.filename, r.format.upper(), status, "—",
            (r.error or "")[:28], "", "", "", "",
        ]


class LogPane(RichLog):
    MAX_LINES = 300

    def on_mount(self) -> None:
        self.max_lines = self.MAX_LINES

    def append(self, message: str, level: str = "info") -> None:
        color = LOG_COLORS.get(level, "white")
        ts    = datetime.now().strftime("%H:%M:%S")
        self.write(Text(f"[{ts}] ", style="dim") + Text(message, style=color))
