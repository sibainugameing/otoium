"""
otoium TUI — ウィザード形式

ステップ:
  1. フォルダ選択   : 音源フォルダを選び、WAV 一覧と自動判定結果をプレビュー
  2. 設定確認       : 形式・出力先・オプションを確定する
  3. 解析実行       : プログレスバー + リアルタイムログ
  4. 結果確認       : 信頼度テーブル・低信頼度フィルタ・保存

各ステップは Screen クラスとして独立させる。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import Screen
from textual.widgets import (
    Button, DataTable, Footer, Header,
    Input, Label, ProgressBar,
    RadioButton, RadioSet, Static, Switch, Rule,
)

from tui.widgets import ResultTable, LogPane
from core.pipeline import Pipeline, PipelineConfig
from core.result import ProcessResult

logger = logging.getLogger(__name__)


# ======================================================================
# Step 1: フォルダ選択
# ======================================================================

class StepFolder(Screen):
    """音源フォルダを選択し、内容をプレビューする"""

    BINDINGS = [Binding("ctrl+n", "next", "次へ"), Binding("q", "app.quit", "終了")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="wizard-body"):
            yield Label("STEP 1 / 4  —  音源フォルダを選択", id="step-label")
            yield Rule()
            yield Label("フォルダのパス", classes="field-label")
            yield Input(
                placeholder="例: /Users/you/voice_bank",
                id="input-dir",
            )
            yield Button("確認", id="btn-check", variant="primary")
            yield Rule()
            yield Label("", id="preview-label")
            yield DataTable(id="wav-preview")
        yield Footer()

    def on_mount(self) -> None:
        t: DataTable = self.query_one("#wav-preview")
        t.add_column("ファイル名", width=30)
        t.add_column("長さ", width=8)
        t.add_column("推定形式", width=10)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-check":
            self._check_folder()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._check_folder()

    def _check_folder(self) -> None:
        from analysis.wav_loader import WavLoader
        from core.format_detector import FormatDetector

        path_str = self.query_one("#input-dir", Input).value.strip()
        path = Path(path_str)
        if not path.is_dir():
            self.query_one("#preview-label", Label).update(
                f"[red]フォルダが見つかりません: {path_str}[/red]"
            )
            return

        wav_files = sorted(path.glob("*.wav"))
        if not wav_files:
            self.query_one("#preview-label", Label).update(
                "[red]WAV ファイルが見つかりません[/red]"
            )
            return

        detector = FormatDetector()
        folder_fmt = detector.detect_folder(path)

        t: DataTable = self.query_one("#wav-preview")
        t.clear()

        loader = WavLoader()
        for wav_path in wav_files[:30]:  # 最大 30 件プレビュー
            try:
                wav = loader.load(wav_path)
                duration = f"{wav.duration_sec:.2f}s"
            except Exception:
                duration = "?"
            file_fmt = detector.detect(wav_path)
            t.add_row(wav_path.name, duration, file_fmt.upper())

        suffix = f"（他 {len(wav_files) - 30} 件）" if len(wav_files) > 30 else ""
        self.query_one("#preview-label", Label).update(
            f"[green]{len(wav_files)} ファイル検出{suffix}  —  自動判定: {folder_fmt.upper()}[/green]"
        )

        # アプリにフォルダと形式を記憶させて「次へ」を有効化
        self.app.wizard_input_dir = path
        self.app.wizard_folder_fmt = folder_fmt

    def action_next(self) -> None:
        if not getattr(self.app, "wizard_input_dir", None):
            self.query_one("#preview-label", Label).update(
                "[yellow]先にフォルダを確認してください[/yellow]"
            )
            return
        self.app.push_screen(StepSettings())


# ======================================================================
# Step 2: 設定確認
# ======================================================================

class StepSettings(Screen):
    """形式・出力先・オプションを確定する"""

    BINDINGS = [
        Binding("ctrl+n", "next",  "次へ"),
        Binding("ctrl+p", "prev",  "戻る"),
        Binding("q",      "app.quit", "終了"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="wizard-body"):
            yield Label("STEP 2 / 4  —  設定", id="step-label")
            yield Rule()

            yield Label("音源形式", classes="field-label")
            with RadioSet(id="format-select"):
                yield RadioButton("自動判定",   value="auto", id="fmt-auto")
                yield RadioButton("CV（単独音）", value="cv",   id="fmt-cv")
                yield RadioButton("VCV（連続音）",value="vcv",  id="fmt-vcv")

            yield Rule()
            yield Label("出力先フォルダ（省略可: 音源フォルダ内に生成）", classes="field-label")
            yield Input(placeholder="省略可", id="output-dir")

            yield Rule()
            with Horizontal(id="options-row"):
                yield Label("既存 oto.ini を上書き", classes="option-label")
                yield Switch(value=False, id="opt-overwrite")

            yield Rule()
            with Horizontal(id="nav-buttons"):
                yield Button("← 戻る", id="btn-prev", variant="default")
                yield Button("解析開始 →", id="btn-next", variant="primary")

        yield Footer()

    def on_mount(self) -> None:
        # 前ステップの自動判定結果を初期選択に反映
        fmt = getattr(self.app, "wizard_folder_fmt", "auto")
        fmt_map = {"auto": "fmt-auto", "cv": "fmt-cv", "vcv": "fmt-vcv"}
        try:
            self.query_one(f"#{fmt_map.get(fmt, 'fmt-auto')}", RadioButton).value = True
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-prev":
            self.app.pop_screen()
        elif event.button.id == "btn-next":
            self._go_next()

    def _go_next(self) -> None:
        # 設定をアプリに保存
        try:
            rs = self.query_one("#format-select", RadioSet)
            btn = rs.pressed_button
            self.app.wizard_format = str(btn.value) if btn else "auto"
        except Exception:
            self.app.wizard_format = "auto"

        out_str = self.query_one("#output-dir", Input).value.strip()
        self.app.wizard_output_dir = Path(out_str) if out_str else None
        self.app.wizard_overwrite  = self.query_one("#opt-overwrite", Switch).value

        self.app.push_screen(StepAnalyze())

    def action_next(self) -> None:
        self._go_next()

    def action_prev(self) -> None:
        self.app.pop_screen()


# ======================================================================
# Step 3: 解析実行
# ======================================================================

class StepAnalyze(Screen):
    """バックグラウンドで解析を実行し、リアルタイムで進捗を表示する"""

    BINDINGS = [Binding("q", "app.quit", "終了")]

    def __init__(self):
        super().__init__()
        self._pipeline: Optional[Pipeline] = None
        self._results: list[ProcessResult] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="wizard-body"):
            yield Label("STEP 3 / 4  —  解析中", id="step-label")
            yield Rule()
            yield Label("", id="progress-label")
            yield ProgressBar(id="progress-bar", total=100, show_eta=False)
            yield Rule()
            yield LogPane(id="log-pane")
        yield Footer()

    def on_mount(self) -> None:
        self._start()

    def _start(self) -> None:
        app = self.app
        config = PipelineConfig(
            input_dir=app.wizard_input_dir,
            output_dir=getattr(app, "wizard_output_dir", None),
            format_hint=getattr(app, "wizard_format", "auto"),
            overwrite_existing=getattr(app, "wizard_overwrite", False),
        )
        self._pipeline = Pipeline(config=config, input_dir=config.input_dir)
        self._log(f"解析開始: {config.input_dir}", "info")
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def _run(self) -> None:
        try:
            results = self._pipeline.run(progress_callback=self._on_progress)
            self._results = results
            self.call_from_thread(self._done, results)
        except Exception as exc:
            self.call_from_thread(self._log, f"エラー: {exc}", "error")

    def _on_progress(self, current: int, total: int, filename: str) -> None:
        self.call_from_thread(self._update_progress, current, total, filename)

    def _update_progress(self, current: int, total: int, filename: str) -> None:
        self.query_one("#progress-label", Label).update(
            f"{filename}  ({current}/{total})"
        )
        bar = self.query_one("#progress-bar", ProgressBar)
        bar.total    = max(total, 1)
        bar.progress = current

    def _done(self, results: list[ProcessResult]) -> None:
        ok       = sum(1 for r in results if r.success)
        low_conf = sum(1 for r in results if r.low_confidence)
        errors   = len(results) - ok
        self._log(
            f"完了: {ok}/{len(results)} 成功  低信頼度: {low_conf}  エラー: {errors}",
            "success" if errors == 0 else "warning",
        )
        self.app.wizard_results  = results
        self.app.wizard_pipeline = self._pipeline
        self.app.push_screen(StepResult())

    def _log(self, msg: str, level: str = "info") -> None:
        try:
            self.query_one("#log-pane", LogPane).append(msg, level)
        except Exception:
            pass


# ======================================================================
# Step 4: 結果確認
# ======================================================================

class StepResult(Screen):
    """処理結果を確認し、低信頼度ファイルの再解析・保存を行う"""

    BINDINGS = [
        Binding("r",      "reanalyze",    "選択を再解析"),
        Binding("s",      "save",         "保存"),
        Binding("f",      "toggle_filter","低信頼度のみ表示"),
        Binding("ctrl+p", "restart",      "最初から"),
        Binding("q",      "app.quit",     "終了"),
    ]

    def __init__(self):
        super().__init__()
        self._show_low_only = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="wizard-body"):
            yield Label("STEP 4 / 4  —  結果確認", id="step-label")
            yield Rule()

            with Horizontal(id="result-header"):
                yield Label("", id="result-summary")
                yield Button("低信頼度のみ [f]", id="btn-filter",  variant="default")
                yield Button("再解析 [r]",        id="btn-reanalyze", variant="default")
                yield Button("保存 [s]",           id="btn-save",   variant="success")
                yield Button("最初から [^p]",     id="btn-restart", variant="default")

            yield Rule()
            yield ResultTable(id="result-table")
            yield Rule()
            yield LogPane(id="log-pane")

        yield Footer()

    def on_mount(self) -> None:
        results = getattr(self.app, "wizard_results", [])
        self._populate(results)

    def _populate(self, results: list[ProcessResult]) -> None:
        shown = [r for r in results if not self._show_low_only or r.low_confidence] if results else []
        t: ResultTable = self.query_one("#result-table")
        t.populate(shown)

        ok       = sum(1 for r in results if r.success)
        low_conf = sum(1 for r in results if r.low_confidence)
        errors   = len(results) - ok
        self.query_one("#result-summary", Label).update(
            f"成功 {ok} / {len(results)}  低信頼度 [yellow]{low_conf}[/yellow]  エラー [red]{errors}[/red]"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-filter":
            self.action_toggle_filter()
        elif event.button.id == "btn-reanalyze":
            self.action_reanalyze()
        elif event.button.id == "btn-save":
            self.action_save()
        elif event.button.id == "btn-restart":
            self.action_restart()

    def action_toggle_filter(self) -> None:
        self._show_low_only = not self._show_low_only
        self._populate(getattr(self.app, "wizard_results", []))
        label = "全件表示" if self._show_low_only else "低信頼度のみ"
        self.query_one("#btn-filter", Button).label = f"{label} [f]"

    def action_reanalyze(self) -> None:
        pipeline: Pipeline = getattr(self.app, "wizard_pipeline", None)
        table: ResultTable  = self.query_one("#result-table")
        selected = table.get_selected_wav_path()
        if not selected or not pipeline:
            self._log("再解析するファイルを選択してください", "warning")
            return
        self._log(f"再解析: {selected.name}")
        result = pipeline.reprocess(selected)
        results: list[ProcessResult] = getattr(self.app, "wizard_results", [])
        for i, r in enumerate(results):
            if r.wav_path == selected:
                results[i] = result
                break
        self.app.wizard_results = results
        self._populate(results)

    def action_save(self) -> None:
        pipeline: Pipeline = getattr(self.app, "wizard_pipeline", None)
        results: list[ProcessResult] = getattr(self.app, "wizard_results", [])
        if not results:
            self._log("保存するデータがありません", "warning")
            return
        if pipeline:
            pipeline._write_oto(results)
            self._log("oto.ini を保存しました", "success")

    def action_restart(self) -> None:
        # スタック全部クリアして最初に戻る
        while len(self.app.screen_stack) > 1:
            self.app.pop_screen()

    def _log(self, msg: str, level: str = "info") -> None:
        try:
            self.query_one("#log-pane", LogPane).append(msg, level)
        except Exception:
            pass


# ======================================================================
# アプリ本体
# ======================================================================

class OtoiumApp(App):
    TITLE = "otoium"
    CSS_PATH = "styles.tcss"
    BINDINGS = [Binding("q", "quit", "終了")]

    def on_mount(self) -> None:
        # ウィザード状態を格納する属性を初期化
        self.wizard_input_dir: Optional[Path] = None
        self.wizard_folder_fmt: str = "auto"
        self.wizard_format: str = "auto"
        self.wizard_output_dir: Optional[Path] = None
        self.wizard_overwrite: bool = False
        self.wizard_results: list[ProcessResult] = []
        self.wizard_pipeline: Optional[Pipeline] = None

        self.push_screen(StepFolder())


# 外部から呼び出す際のエイリアス（main.py との互換）
TUIApp = OtoiumApp
