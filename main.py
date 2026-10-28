#!/usr/bin/env python3
"""
UTAU原音設定自動化ツール
CV / VCV 音源に対応した oto.ini 全自動生成
"""

import sys
import argparse
from pathlib import Path

from tui.app import TUIApp
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="UTAU oto.ini 自動生成ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python main.py                        # TUI を起動
  python main.py --input ./voice_bank   # 音源フォルダを指定して起動
  python main.py --batch ./voice_bank --format cv --output ./output
        """,
    )
    parser.add_argument("--input", "-i", type=Path, help="音源フォルダのパス")
    parser.add_argument(
        "--format",
        "-f",
        choices=["cv", "vcv", "auto"],
        default="auto",
        help="音源形式 (default: auto)",
    )
    parser.add_argument("--output", "-o", type=Path, help="出力先フォルダ (省略時は入力フォルダ内)")
    parser.add_argument("--batch", action="store_true", help="バッチモード（TUI を表示せず自動処理）")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(args.log_level)

    if args.batch and args.input:
        # バッチモード：TUI なしで直接処理
        from core.pipeline import Pipeline

        pipeline = Pipeline(
            input_dir=args.input,
            format_hint=args.format,
            output_dir=args.output,
        )
        results = pipeline.run()
        success = sum(1 for r in results if r.success)
        logger.info(f"完了: {success}/{len(results)} ファイル成功")
        sys.exit(0 if success == len(results) else 1)
    else:
        # TUI モード
        app = TUIApp(
            initial_input=args.input,
            format_hint=args.format,
            output_dir=args.output,
        )
        app.run()


if __name__ == "__main__":
    main()
