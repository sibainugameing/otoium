#!/usr/bin/env python3
"""otoium — UTAU oto.ini 自動生成"""

import sys
import argparse
from pathlib import Path

from tui.app import OtoiumApp
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="otoium — UTAU oto.ini 自動生成")
    parser.add_argument("--input",  "-i", type=Path, help="音源フォルダ")
    parser.add_argument("--format", "-f", choices=["cv", "vcv", "auto"], default="auto")
    parser.add_argument("--output", "-o", type=Path, help="出力先フォルダ")
    parser.add_argument("--batch",        action="store_true", help="バッチモード（TUI なし）")
    parser.add_argument("--log-level",    default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args   = parse_args()
    logger = setup_logger(args.log_level)

    if args.batch and args.input:
        from core.pipeline import Pipeline
        pipeline = Pipeline(input_dir=args.input, format_hint=args.format, output_dir=args.output)
        results  = pipeline.run()
        success  = sum(1 for r in results if r.success)
        logger.info(f"完了: {success}/{len(results)}")
        sys.exit(0 if success == len(results) else 1)
    else:
        app = OtoiumApp()
        if args.input:
            app.wizard_input_dir = args.input
        app.run()


if __name__ == "__main__":
    main()
