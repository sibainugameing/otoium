"""
TUI メインアプリケーション

Textual を使用したターミナル UI。
音源フォルダ指定・形式選択・解析開始・進捗確認・結果表示を提供する。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RadioButton,
    RadioSet,
    Static,
)

from tui.widgets import StatusBar, ResultTable, LogPane
from core.pipeline import Pipeline
from core.result import ProcessResult

logger = logging.getLogger(__name__)


class TUIApp(App):
    """UTAU 原音設定自動化 TUI"""

    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("q", "quit", "終了"),
        Binding("r", "reanalyze", "再解析", show=True),
        Binding("s", "save", "保存", show=True),
        Binding("f1", "help", "ヘルプ"),
    ]

    TITLE = "UTAU 原音設定自動化ツール"
    SUB_TITLE = "CV / VCV oto.ini ジェネレーター"

    def __init__(
        self,
        initial_input: Optional[Path] = None,
        format_hint: str = "auto",
        output_dir: Optional[Path] = None,
    ):
        super().__init__()
        self.initial_input = initial_input
        self.format_hint = format_hint
        self.output_dir = output_dir
        self.results: list[ProcessResult] = []
        self._pipeline: Optional[Pipeline] = None

    # ------------------------------------------------------------------
    # UI 構成
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()

        with Vertical(id="main"):
            # --- 入力設定パネル ---
            with Container(id="settings-panel"):
                yield Label("音源フォルダ", classes="label")
                yield Input(
                    placeholder="/path/to/voice_bank",
                    id="input-dir",
                    value=str(self.initial_input) if self.initial_input else "",
                )

                yield Label("出力先 (省略可)", classes="label")
                yield Input(placeholder="省略時は音源フォルダ内", id="output-dir")

                yield Label("音源形式", classes="label")
                with RadioSet(id="format-select"):
                    yield RadioButton("自動判定", value="auto", id="fmt-auto")
                    yield RadioButton("CV（単独音）", value="cv", id="fmt-cv")
                    yield RadioButton("VCV（連続音）", value="vcv", id="fmt-vcv")

                with Horizontal(id="action-buttons"):
                    yield Button("解析開始", id="btn-analyze", variant="primary")
                    yield Button("再解析（選択）", id="btn-reanalyze", variant="default")
                    yield Button("保存", id="btn-save", variant="success")

            # --- 進捗バー ---
            with Container(id="progress-panel"):
                yield Label("", id="progress-label")
                yield ProgressBar(id="progress-bar", total=100, show_eta=False)

            # --- 結果テーブル ---
            with Container(id="result-panel"):
                yield Label("処理結果", classes="section-label")
                yield ResultTable(id="result-table")

            # --- ログペイン ---
            with Container(id="log-panel"):
                yield Label("ログ", classes="section-label")
                yield LogPane(id="log-pane")

        yield StatusBar(id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        # フォーマット初期選択
        fmt_map = {"auto": "fmt-auto", "cv": "fmt-cv", "vcv": "fmt-vcv"}
        radio_id = fmt_map.get(self.format_hint, "fmt-auto")
        try:
            self.query_one(f"#{radio_id}", RadioButton).value = True
        except Exception:
            pass

        self._log("ツールを起動しました。音源フォルダを指定して「解析開始」を押してください。")

    # ------------------------------------------------------------------
    # ボタンイベント
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-analyze":
            self._start_analysis()
        elif event.button.id == "btn-reanalyze":
            self._reanalyze_selected()
        elif event.button.id == "btn-save":
            self._save_results()

    # ------------------------------------------------------------------
    # 解析処理（バックグラウンドスレッド）
    # ------------------------------------------------------------------

    def _start_analysis(self) -> None:
        input_dir_str = self.query_one("#input-dir", Input).value.strip()
        if not input_dir_str:
            self._log("エラー: 音源フォルダを指定してください", level="error")
            return

        input_dir = Path(input_dir_str)
        if not input_dir.is_dir():
            self._log(f"エラー: フォルダが存在しません: {input_dir}", level="error")
            return

        output_dir_str = self.query_one("#output-dir", Input).value.strip()
        output_dir = Path(output_dir_str) if output_dir_str else None

        fmt = self._get_selected_format()

        self._log(f"解析開始: {input_dir} [{fmt}]")
        self._set_progress(0, 1, "初期化中...")

        self._pipeline = Pipeline(
            input_dir=input_dir,
            format_hint=fmt,
            output_dir=output_dir,
        )

        thread = threading.Thread(target=self._run_pipeline, daemon=True)
        thread.start()

    def _run_pipeline(self) -> None:
        """バックグラウンドで Pipeline を実行する"""
        try:
            results = self._pipeline.run(progress_callback=self._on_progress)
            self.results = results
            self.call_from_thread(self._on_analysis_complete, results)
        except Exception as exc:
            self.call_from_thread(self._log, f"解析エラー: {exc}", "error")

    def _on_progress(self, current: int, total: int, filename: str) -> None:
        """進捗コールバック（バックグラウンドスレッドから呼ばれる）"""
        self.call_from_thread(self._set_progress, current, total, filename)

    def _on_analysis_complete(self, results: list[ProcessResult]) -> None:
        success = sum(1 for r in results if r.success)
        low_conf = sum(1 for r in results if r.low_confidence)
        errors = len(results) - success

        self._log(
            f"解析完了: {success}/{len(results)} 成功  "
            f"低信頼度: {low_conf}  エラー: {errors}"
        )
        self._set_progress(len(results), len(results), "完了")
        self._update_result_table(results)

    def _reanalyze_selected(self) -> None:
        table: ResultTable = self.query_one("#result-table")
        selected = table.get_selected_wav_path()
        if not selected or not self._pipeline:
            self._log("再解析するファイルを選択してください", level="warning")
            return

        self._log(f"再解析: {selected.name}")
        result = self._pipeline.reprocess(selected)
        # テーブルを更新
        for i, r in enumerate(self.results):
            if r.wav_path == selected:
                self.results[i] = result
                break
        else:
            self.results.append(result)
        self._update_result_table(self.results)

    def _save_results(self) -> None:
        if not self.results:
            self._log("保存するデータがありません", level="warning")
            return
        if self._pipeline:
            self._pipeline._write_oto(self.results)
            self._log("oto.ini を保存しました")

    # ------------------------------------------------------------------
    # UI 更新ヘルパー
    # ------------------------------------------------------------------

    def _set_progress(self, current: int, total: int, label: str) -> None:
        try:
            self.query_one("#progress-label", Label).update(
                f"{label}  ({current}/{total})"
            )
            bar = self.query_one("#progress-bar", ProgressBar)
            bar.total = max(total, 1)
            bar.progress = current
        except Exception:
            pass

    def _update_result_table(self, results: list[ProcessResult]) -> None:
        try:
            table: ResultTable = self.query_one("#result-table")
            table.populate(results)
        except Exception as exc:
            logger.debug(f"テーブル更新エラー: {exc}")

    def _log(self, message: str, level: str = "info") -> None:
        try:
            pane: LogPane = self.query_one("#log-pane")
            pane.append(message, level)
        except Exception:
            pass
        getattr(logger, level, logger.info)(message)

    def _get_selected_format(self) -> str:
        try:
            rs: RadioSet = self.query_one("#format-select")
            pressed = rs.pressed_button
            if pressed:
                return str(pressed.value)
        except Exception:
            pass
        return "auto"

    # ------------------------------------------------------------------
    # キーバインド アクション
    # ------------------------------------------------------------------

    def action_reanalyze(self) -> None:
        self._reanalyze_selected()

    def action_save(self) -> None:
        self._save_results()

    def action_help(self) -> None:
        self._log(
            "操作: [解析開始] フォルダを解析  [r] 選択ファイルを再解析  "
            "[s] oto.ini を保存  [q] 終了"
        )
