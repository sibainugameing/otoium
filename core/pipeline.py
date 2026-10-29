"""
メイン処理パイプライン
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

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
    format_hint: str = "auto"
    confidence_threshold: float = 0.52
    overwrite_existing: bool = False


class Pipeline:

    def __init__(
        self,
        input_dir: Optional[Path] = None,
        format_hint: str = "auto",
        output_dir: Optional[Path] = None,
        config: Optional[PipelineConfig] = None,
    ):
        if config is not None:
            self.config = config
        elif input_dir is not None:
            self.config = PipelineConfig(
                input_dir=input_dir,
                output_dir=output_dir,
                format_hint=format_hint,
            )
        else:
            raise ValueError("input_dir か config のどちらかを指定してください")

        self.wav_loader        = WavLoader()
        self.feature_extractor = FeatureExtractor()
        self.boundary_estimator = BoundaryEstimator()
        self.format_detector   = FormatDetector()
        self.oto_writer        = OtoWriter()
        self._folder_format: Optional[str] = None

    # ------------------------------------------------------------------

    def run(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[ProcessResult]:
        wav_files = sorted(self.config.input_dir.glob("*.wav"))
        if not wav_files:
            logger.warning(f"WAV ファイルが見つかりません: {self.config.input_dir}")
            return []

        if self.config.format_hint == "auto":
            self._folder_format = self.format_detector.detect_folder(self.config.input_dir)
            logger.info(f"形式自動判定: {self._folder_format.upper()}")
        else:
            self._folder_format = self.config.format_hint

        results: list[ProcessResult] = []
        total = len(wav_files)

        for i, wav_path in enumerate(wav_files):
            if progress_callback:
                progress_callback(i, total, wav_path.name)
            result = self._process_one(wav_path)
            results.append(result)
            if result.success:
                logger.debug(f"OK  {wav_path.name}")
            else:
                logger.warning(f"NG  {wav_path.name}: {result.error}")

        self._write_oto(results)
        if progress_callback:
            progress_callback(total, total, "完了")
        return results

    def reprocess(self, wav_path: Path) -> ProcessResult:
        if self._folder_format is None:
            self._folder_format = (
                self.format_detector.detect_folder(self.config.input_dir)
                if self.config.format_hint == "auto"
                else self.config.format_hint
            )
        result = self._process_one(wav_path)
        if result.success:
            self._update_oto_entry(result)
        return result

    # ------------------------------------------------------------------

    def _process_one(self, wav_path: Path) -> ProcessResult:
        try:
            wav      = self.wav_loader.load(wav_path)
            fmt      = self._folder_format or self.format_detector.detect(wav_path)
            features = self.feature_extractor.extract(wav)

            processor = (
                VCVProcessor(self.boundary_estimator)
                if fmt == "vcv"
                else CVProcessor(self.boundary_estimator)
            )
            params, confidence = processor.estimate(wav_path, wav, features)
            return ProcessResult(
                wav_path=wav_path,
                format=fmt,
                oto_params=params,
                confidence=confidence,
                low_confidence=confidence < self.config.confidence_threshold,
                success=True,
            )
        except Exception as exc:
            logger.exception(f"処理エラー: {wav_path.name}")
            return ProcessResult(
                wav_path=wav_path, format="unknown",
                success=False, error=str(exc),
            )

    def _write_oto(self, results: list[ProcessResult], append: bool = False) -> None:
        out_dir    = self.config.output_dir or self.config.input_dir
        successful = [r for r in results if r.success and r.oto_params]
        if not successful:
            return

        output_path = out_dir / "oto.ini"
        if output_path.exists() and not self.config.overwrite_existing and not append:
            backup = output_path.with_suffix(".ini.bak")
            output_path.rename(backup)
            logger.info(f"既存 oto.ini → {backup.name}")

        self.oto_writer.write(output_path, successful, append=append)

    def _update_oto_entry(self, result: ProcessResult) -> None:
        out_dir     = self.config.output_dir or self.config.input_dir
        output_path = out_dir / "oto.ini"
        alias       = result.oto_params.alias if result.oto_params else None
        if alias is None:
            return
        if not output_path.exists():
            self.oto_writer.write(output_path, [result])
            return
        existing        = self.oto_writer.read_existing(output_path)
        existing[alias] = result.oto_params.to_oto_line()
        self.oto_writer.write_raw_lines(output_path, list(existing.values()))
        logger.info(f"oto.ini 更新: {alias}")
