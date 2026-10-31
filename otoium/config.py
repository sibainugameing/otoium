"""設定値（CLI / GUI 共通）"""
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # ---- 出力形式 ----
    mode: str = "auto"          # auto / cv / vcv / cvvc  (auto: 1モーラ=CV, 複数モーラ=VCV)
    vcv_style: str = "mora"     # mora: 「か さ」 / vowel: 「あ さ」
    vc_style: str = "romaji"    # CVVC の VC: romaji「a k」 / kana「あ k」
    head_prefix: str = ""       # 先頭エイリアスの接頭辞 (例 "- ")
    add_tail: bool = False      # 末尾「V -」エイリアスを作る
    tail_alias: str = "-"
    encoding: str = "cp932"     # oto.ini の文字コード (cp932 / utf-8)
    # ---- 解析 ----
    hop_ms: float = 5.0
    win_ms: float = 25.0
    accuracy: str = "high"      # fast / high / max (学習回数・自動調整・救済の強さ)
    train_iters: int = 0        # 0: accuracy の既定値
    auto_tune: bool = None      # None: accuracy の既定値
    rescue: bool = None
    refine: bool = True         # 1ms 精度の境界微調整
    delta_weight: float = 0.25
    mfcc_weight: float = 0.35
    boundary_weight: float = 4.0
    duration_weight: float = 0.5
    pause_penalty: float = 2.0
    free_switch_penalty: float = 10.0
    # ---- oto パラメータ ----
    cv_lead_ms: float = 5.0
    iso_lead_ms: float = 25.0
    vcv_tail_ms: float = 90.0
    vc_tail_ms: float = 70.0
    fixed_extra_ms: float = 20.0
    # ---- 検証 ----
    thr_contrast: float = 1.5
    thr_bnd: float = 0.6
    ok_agree: float = 0.55
    fail_agree: float = 0.25
    # ---- 動作 ----
    recursive: bool = True
    backup: bool = True
    dry_run: bool = False
    strict: bool = False        # WARN も出力から除外
    dedupe: bool = True
    script_path: str = ""       # 「wav名=台本テキスト」形式の台本
    model_path: str = ""        # 学習済みモデルの読込
    save_model: bool = True
    export_labels: bool = False  # Audacity ラベルを書き出す
    textgrid: bool = False
    merge: bool = False         # 既存 oto.ini の未処理ファイル分を残す
    alias_prefix: str = ""
    alias_suffix: str = ""
    jobs: int = 1
    cache: bool = True

    def to_dict(self):
        return asdict(self)


PRESETS = {
    "fast": dict(iters=3, tune=False, rounds=0, rescue=False),
    "high": dict(iters=5, tune=True, rounds=1, rescue=True),
    "max": dict(iters=8, tune=True, rounds=2, rescue=True),
}
