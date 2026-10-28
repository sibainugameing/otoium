"""
メイン処理パイプライン
音源フォルダの読み込みから oto.ini 生成までを一貫して管理する
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

from analysis.wav_loader import WavLoader
from analysis.feature_extractor import FeatureExtractor
from analysis.boundary_estimator import BoundaryEstimator
from formats.cv_processor import CVProcessor
from formats.vcv_processor import VCVProcessor
from formats.oto_writer import OtoWriter
from core.format_detector import FormatDetector
from core.result import ProcessResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Optional[Path] = None
    format_hint: str = "auto"          # "cv" | "vcv" | "auto"
    confidence_threshold: float = 0.6  # これ以下は低信頼度として警告
    overwrite_existing: bool = False   # 既存 oto.ini を上書きするか


class Pipeline:
    """
    WAV → 特徴抽出 → 境界推定 → パラメータ計算 → oto.ini 出力
    の一連の処理をファイル単位で実行する。
    """

    def __init__(
        self,
        input_dir: Path,
        format_hint: str = "auto",
        output_dir: Optional[Path] = None,
        config: Optional[PipelineConfig] = None,
    ):
        self.config = config or PipelineConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            format_hint=format_hint,
        )
        self.wav_loader = WavLoader()
        self.feature_extractor = FeatureExtractor()
        self.boundary_estimator = BoundaryEstimator()
        self.format_detector = FormatDetector()
        self.oto_writer = OtoWriter()

    # ------------------------------------------------------------------
    # 公開インターフェース
    # ------------------------------------------------------------------

    def run(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[ProcessResult]:
        """
        音源フォルダ内の全 WAV を処理して oto.ini を生成する。

        progress_callback(current, total, filename) が渡された場合、
        ファイルごとに呼び出される（TUI 向け）。
        """
        wav_files = self._collect_wav_files()
        if not wav_files:
            logger.warning(f"WAV ファイルが見つかりません: {self.config.input_dir}")
            return []

        total = len(wav_files)
        results: list[ProcessResult] = []

        for i, wav_path in enumerate(wav_files):
            if progress_callback:
                progress_callback(i, total, wav_path.name)
            result = self._process_one(wav_path)
            results.append(result)
            if result.success:
                logger.debug(f"OK  {wav_path.name}")
            else:
                logger.warning(f"NG  {wav_path.name}: {result.error}")

        # 最後にまとめて oto.ini を書き出す（フォルダ単位）
        self._write_oto(results)

        if progress_callback:
            progress_callback(total, total, "完了")

        return results

    def reprocess(self, wav_path: Path) -> ProcessResult:
        """単一ファイルを再処理する（TUI での個別再解析用）"""
        result = self._process_one(wav_path)
        if result.success:
            self._write_oto([result], append=True)
        return result

    # ------------------------------------------------------------------
    # 内部処理
    # ------------------------------------------------------------------

    def _collect_wav_files(self) -> list[Path]:
        return sorted(self.config.input_dir.glob("*.wav"))

    def _process_one(self, wav_path: Path) -> ProcessResult:
        try:
            # 1. WAV 読み込み
            wav = self.wav_loader.load(wav_path)

            # 2. 音源形式の判定
            fmt = self._detect_format(wav_path)

            # 3. 音響特徴抽出
            features = self.feature_extractor.extract(wav)

            # 4. 音素境界推定（形式別）
            if fmt == "vcv":
                processor = VCVProcessor(self.boundary_estimator)
            else:
                processor = CVProcessor(self.boundary_estimator)

            params, confidence = processor.estimate(wav_path, wav, features)

            # 5. 信頼度チェック
            low_confidence = confidence < self.config.confidence_threshold

            return ProcessResult(
                wav_path=wav_path,
                format=fmt,
                oto_params=params,
                confidence=confidence,
                low_confidence=low_confidence,
                success=True,
            )

        except Exception as exc:
            logger.exception(f"処理エラー: {wav_path.name}")
            return ProcessResult(
                wav_path=wav_path,
                format="unknown",
                success=False,
                error=str(exc),
            )

    def _detect_format(self, wav_path: Path) -> str:
        if self.config.format_hint != "auto":
            return self.config.format_hint
        return self.format_detector.detect(wav_path)

    def _write_oto(self, results: list[ProcessResult], append: bool = False) -> None:
        """成功した結果をフォルダごとにまとめて oto.ini に書き出す"""
        # 出力先フォルダを決定
        out_dir = self.config.output_dir or self.config.input_dir

        successful = [r for r in results if r.success and r.oto_params]
        if not successful:
            return

        output_path = out_dir / "oto.ini"

        # 既存ファイルの保護
        if output_path.exists() and not self.config.overwrite_existing and not append:
            backup = output_path.with_suffix(".ini.bak")
            output_path.rename(backup)
            logger.info(f"既存 oto.ini を {backup.name} にバックアップしました")

        self.oto_writer.write(output_path, successful, append=append)
        logger.info(f"oto.ini を書き出しました: {output_path}")
