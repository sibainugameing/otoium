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
    p.add_argument("--accuracy", choices=["fast", "high", "max", "ultra", "custom"], default="high",
                   help="精度プリセット (ultra: 最高品質 / custom: 下記の回数指定を使用)")
    p.add_argument("--iters", type=int, default=0, help="ガウスモデルの学習回数 (0: プリセット)")
    p.add_argument("--tune-rounds", type=int, default=-1, help="自動調整の周回数 (0: 無効)")
    p.add_argument("--net-epochs", type=int, default=0, help="ニューラルモデルの学習エポック数")
    p.add_argument("--net-ensemble", type=int, default=0, help="ニューラルモデルのアンサンブル数")
    p.add_argument("--adapt-rounds", type=int, default=-1, help="録音ごとの適応の反復回数 (0: 無効)")
    p.add_argument("--post-iters", type=int, default=-1, help="自動調整後の再学習回数")
    p.add_argument("--confident-rounds", type=int, default=-1, help="高信頼ファイルのみでの再学習の周回数")
    p.add_argument("--no-tune", action="store_true", help="自動調整を無効化")
    p.add_argument("--no-rescue", action="store_true", help="失敗ファイルの救済を無効化")
    p.add_argument("--no-net", action="store_true", help="ニューラル音響モデルを無効化")
    p.add_argument("--no-adapt", action="store_true", help="録音ごとの適応を無効化")
    p.add_argument("--no-auto-thresholds", action="store_true", help="しきい値の自動校正を無効化")
    p.add_argument("--no-refine", action="store_true", help="1ms 境界微調整を無効化")
    p.add_argument("--jobs", type=int, default=1, help="特徴抽出の並列数")
    p.add_argument("--recheck-rounds", type=int, default=3, help="失敗箇所の再検査の最大回数 (0 で無効)")
    p.add_argument("--patience", type=int, default=-1, help="収束判定の我慢回数 (0: 早期終了しない / -1: プリセット)")
    p.add_argument("--converge-shift", type=float, default=0.03, help="境界の平均変化がこの値(フレーム)未満で「動かない」")
    p.add_argument("--no-partial-repair", action="store_true", help="失敗モーラ区間だけのやり直しを無効化")
    p.add_argument("--write-failed", action="store_true", help="失敗・保留も oto.ini に書き込む (一覧は otoium_forced.txt)")
    p.add_argument("--skip-unknown", action="store_true", help="台本・ファイル名の未対応文字を無視して解析")
    p.add_argument("--json", nargs="?", const="", default=None, metavar="PATH",
                   help="詳細な実行結果を JSON で書き出す (PATH 省略時は <フォルダ>/otoium_result.json)")
    p.add_argument("--json-detail", choices=["summary", "standard", "full"], default="full", help="JSON の詳細度")
    p.add_argument("--min-dur-ratio", type=float, default=0.0, help="母音・鼻音の最短長を学習済み平均長のこの割合に引き上げる (整列の潰れ防止)")
    p.add_argument("--no-guard", action="store_true", help="学習更新の得点ガード (悪化する更新を採用しない) を無効化")
    p.add_argument("--accept-tolerance", type=float, default=0.003, help="更新を採用する際に許容する、最高点からの得点低下幅")
    p.add_argument("--cut-dip-db", type=float, default=15.0, help="母音が連続する箇所のエネルギーの谷がこの dB 以上なら「声が途切れている」と警告 (0 で無効)")
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
                 encoding=a.encoding, script_path=a.script, train_iters=a.iters, tune_rounds=a.tune_rounds, net_epochs=a.net_epochs,
                 net_ensemble=a.net_ensemble, adapt_rounds=a.adapt_rounds, post_iters=a.post_iters,
                 confident_rounds=a.confident_rounds, accuracy=a.accuracy, auto_tune=False if a.no_tune else None,
                 rescue=False if a.no_rescue else None, refine=not a.no_refine, use_net=False if a.no_net else None,
                 adapt=False if a.no_adapt else None, auto_thresholds=not a.no_auto_thresholds, jobs=a.jobs, cache=not a.no_cache,
                 merge=a.merge, cut_dip_db=a.cut_dip_db, min_dur_ratio=a.min_dur_ratio, guarded_updates=not a.no_guard, accept_tolerance=a.accept_tolerance, result_json=a.json is not None, result_json_path=a.json or "",
                 result_json_detail=a.json_detail, skip_unknown_chars=a.skip_unknown, patience=a.patience, converge_shift=a.converge_shift,
                 partial_repair=not a.no_partial_repair, write_failed=a.write_failed, recheck_rounds=a.recheck_rounds, alias_prefix=a.prefix, alias_suffix=a.suffix, textgrid=a.textgrid, model_path=a.model, strict=a.strict,
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
