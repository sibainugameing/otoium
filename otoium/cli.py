"""コマンドライン (TUI 風ログ表示)"""
import argparse
import sys
from .config import Config
from .pipeline import Pipeline


def build_parser():
    p = argparse.ArgumentParser(prog="otoium", description="otoium: UTAU 原音設定 (oto.ini) 自動生成")
    p.add_argument("folder", nargs="?", help="原音の WAV を含むフォルダ (省略で GUI 起動)")
    p.add_argument("--mode", choices=["auto", "cv", "vcv", "cvvc"], default="auto")
    p.add_argument("--vcv-style", choices=["mora", "vowel"], default="mora", help="mora: 「か さ」 / vowel: 「あ さ」")
    p.add_argument("--vc-style", choices=["romaji", "kana"], default="romaji")
    p.add_argument("--head-prefix", default="", help='先頭エイリアスの接頭辞 例: "- "')
    p.add_argument("--tail", action="store_true", help="末尾 'V -' エイリアスも生成")
    p.add_argument("--encoding", default="cp932", choices=["cp932", "utf-8"])
    p.add_argument("--script", default="", help="『wav名=テキスト』の台本ファイル (省略時はファイル名)")
    p.add_argument("--accuracy", choices=["fast", "high", "max"], default="high", help="精度プリセット")
    p.add_argument("--iters", type=int, default=0, help="学習反復回数 (0: プリセット)")
    p.add_argument("--no-tune", action="store_true", help="自動調整を無効化")
    p.add_argument("--no-rescue", action="store_true", help="失敗ファイルの救済を無効化")
    p.add_argument("--no-refine", action="store_true", help="1ms 境界微調整を無効化")
    p.add_argument("--jobs", type=int, default=1, help="特徴抽出の並列数")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--merge", action="store_true", help="既存 oto.ini の未処理ファイル行を残す")
    p.add_argument("--prefix", default="", help="エイリアス接頭辞")
    p.add_argument("--suffix", default="", help="エイリアス接尾辞")
    p.add_argument("--textgrid", action="store_true", help="Praat TextGrid を出力")
    p.add_argument("--model", default="", help="学習済みモデル(.npz)を読み込む")
    p.add_argument("--strict", action="store_true", help="WARN も出力から除外")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-backup", action="store_true")
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("--no-dedupe", action="store_true")
    p.add_argument("--labels", action="store_true", help="Audacity 用ラベルを出力")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if not a.folder:
        try:
            from .gui import run_gui
        except ImportError as ex:
            print(f"GUI を起動できません ({ex})。フォルダを指定して CLI で実行してください。", file=sys.stderr)
            return 1
        run_gui()
        return 0
    cfg = Config(mode=a.mode, vcv_style=a.vcv_style, vc_style=a.vc_style, head_prefix=a.head_prefix, add_tail=a.tail,
                 encoding=a.encoding, script_path=a.script, train_iters=a.iters, accuracy=a.accuracy, auto_tune=False if a.no_tune else None,
                 rescue=False if a.no_rescue else None, refine=not a.no_refine, jobs=a.jobs, cache=not a.no_cache,
                 merge=a.merge, alias_prefix=a.prefix, alias_suffix=a.suffix, textgrid=a.textgrid, model_path=a.model, strict=a.strict,
                 dry_run=a.dry_run, backup=not a.no_backup, recursive=not a.no_recursive, dedupe=not a.no_dedupe,
                 export_labels=a.labels)

    def prog(frac, msg=""):
        sys.stderr.write(f"\r{frac * 100:5.1f}% {msg[:40]:<40}")
        sys.stderr.flush()

    try:
        rep = Pipeline(cfg, log=print, progress=prog).run(a.folder)
    except Exception as ex:
        print(f"\nエラー: {ex}", file=sys.stderr)
        return 1
    sys.stderr.write("\n")
    return 2 if rep["failed"] else 0
