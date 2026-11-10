"""設定値（CLI / GUI 共通）"""
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # ---- 出力形式 ----
    mode: str = "auto"          # auto / cv / vcv / cvvc  (auto: 1モーラ=CV, 複数モーラ=VCV)
    vcv_style: str = "mora"     # mora: 「か さ」 / vowel: 「あ さ」 / romaji: 「a さ」 (日本語VCVで一般的な表記)
    vc_style: str = "romaji"    # CVVC の VC: romaji「a k」 / kana「あ k」
    head_prefix: str = ""       # 先頭エイリアスの接頭辞 (例 "- ")
    add_tail: bool = False      # 末尾「V -」エイリアスを作る
    tail_alias: str = "-"
    encoding: str = "cp932"     # oto.ini の文字コード (cp932 / utf-8)
    # ---- 解析 ----
    hop_ms: float = 5.0
    win_ms: float = 25.0
    accuracy: str = "high"      # fast / high / max / ultra / custom
    # --- カスタム値 (0 / -1 は精度プリセットの値を使う。custom 以外でも指定すれば上書き) ---
    train_iters: int = 0        # ガウスモデルの学習回数
    tune_rounds: int = -1       # 自動調整の周回数 (0 で無効)
    net_epochs: int = 0         # ニューラルモデルの学習エポック
    net_ensemble: int = 0       # ニューラルモデルのアンサンブル数
    adapt_rounds: int = -1      # 録音ごとの適応の反復 (0 で無効)
    post_iters: int = -1        # 自動調整後の再学習回数
    confident_rounds: int = -1  # 高信頼ファイルだけでの再学習の周回数
    auto_tune: bool = None      # None: accuracy の既定値
    rescue: bool = None
    use_net: bool = None        # ニューラル音響モデル (None: プリセット)
    nn_weight: float = 2.0
    adapt: bool = None          # ファイル単位の適応 (None: プリセット)
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
    fixed_extra_sonorant_ms: float = 40.0   # な行・ま行・や行・ら行・わ行は遷移が終わる所まで固定範囲を長く取る (UTAU wiki)
    cv_overlap_margin_ms: float = 35.0      # さ行・は行・な行など: オーバーラップは先行発声の 20〜50ms 手前
    # ---- 検証 ----
    thr_contrast: float = 1.5
    thr_bnd: float = 0.6
    cons_review: float = 0.34
    cons_warn: float = 0.6
    auto_thresholds: bool = True   # しきい値を録音データから自動校正
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
    patience: int = -1              # 収束判定: 連続で「境界が動かず得点も伸びない」回数 (0: 早期終了なし, -1: プリセット)
    converge_shift: float = 0.03    # 境界の平均変化がこの値(フレーム)未満なら「動かない」
    converge_score: float = 0.002   # 得点の伸びがこの値以下なら「伸びない」
    partial_repair: bool = True     # 失敗したモーラの区間だけを再整列
    write_failed: bool = False      # 失敗・保留でも oto.ini に書き込む (一覧は otoium_forced.txt)
    skip_unknown_chars: bool = False  # 台本/ファイル名の未対応文字を無視して解析 (無視した文字は注記)
    result_json: bool = False         # 詳細な実行結果を JSON で書き出す
    result_json_path: str = ""        # 空: <フォルダ>/otoium_result.json
    result_json_detail: str = "full"  # summary / standard / full
    allow_devoiced: bool = True       # 無声子音の後の i/u を無声化母音として整列してよい
    devoiced_penalty: float = 3.0     # 無声化母音を選ぶときの罰則 (小さいほど選ばれやすい)
    cut_dip_db: float = 15.0          # 母音が連続する箇所のエネルギーの谷がこの dB 以上なら「声が途切れている」と警告 (0 で無効)
    min_dur_ratio: float = 0.0        # 母音・鼻音の最短長 = 学習済み平均長 × この値 (0 で無効。整列の潰れ防止)
    dv_penalty: float = 3.0           # 独立認識で「無声化母音」と判定するときの罰則 (誤って息の子音を奪わないため)
    guarded_updates: bool = True      # 学習の更新で得点が最高点より下がるなら、減衰した更新→更新なし の順に試す (学習で悪化しない)
    accept_tolerance: float = 0.003   # 最高点からこの幅までの低下は許容 (累積しない: 常に最高点と比べる)
    recheck_rounds: int = 3         # 失敗箇所の再検査の最大回数 (0 で無効)
    track_metrics: bool = True      # 学習の得点を記録・表示

    def to_dict(self):
        return asdict(self)


PRESETS = {
    "fast":  dict(iters=3,  rounds=0, rescue=False, net_epochs=0,  ens=0, adapt_rounds=0, post=0, conf=0, patience=1),
    "high":  dict(iters=5,  rounds=1, rescue=True,  net_epochs=10, ens=1, adapt_rounds=1, post=2, conf=0, patience=2),
    "max":   dict(iters=8,  rounds=2, rescue=True,  net_epochs=14, ens=2, adapt_rounds=2, post=3, conf=1, patience=3),
    "ultra": dict(iters=12, rounds=3, rescue=True,  net_epochs=20, ens=3, adapt_rounds=3, post=4, conf=2, patience=0),
}
ACCURACY_MODES = ["fast", "high", "max", "ultra", "custom"]


def resolve(cfg):
    """精度プリセット + カスタム値 + 個別スイッチ → 実際に使う設定値"""
    pr = dict(PRESETS["high" if cfg.accuracy == "custom" else cfg.accuracy])
    if cfg.train_iters > 0:
        pr["iters"] = cfg.train_iters
    if cfg.tune_rounds >= 0:
        pr["rounds"] = cfg.tune_rounds
    if cfg.net_epochs > 0:
        pr["net_epochs"] = cfg.net_epochs
    if cfg.net_ensemble > 0:
        pr["ens"] = cfg.net_ensemble
    if cfg.adapt_rounds >= 0:
        pr["adapt_rounds"] = cfg.adapt_rounds
    if cfg.post_iters >= 0:
        pr["post"] = cfg.post_iters
    if cfg.confident_rounds >= 0:
        pr["conf"] = cfg.confident_rounds
    if cfg.patience >= 0:
        pr["patience"] = cfg.patience
    pr["tune"] = pr["rounds"] > 0 if cfg.auto_tune is None else bool(cfg.auto_tune) and max(pr["rounds"], 1) > 0
    if cfg.auto_tune and pr["rounds"] == 0:
        pr["rounds"] = 1
    pr["net"] = (pr["net_epochs"] > 0 and pr["ens"] > 0) if cfg.use_net is None else bool(cfg.use_net)
    if pr["net"]:
        pr["net_epochs"] = pr["net_epochs"] or 10
        pr["ens"] = pr["ens"] or 1
    pr["adapt"] = pr["adapt_rounds"] > 0 if cfg.adapt is None else bool(cfg.adapt)
    if pr["adapt"] and pr["adapt_rounds"] == 0:
        pr["adapt_rounds"] = 1
    if cfg.rescue is not None:
        pr["rescue"] = bool(cfg.rescue)
    return pr
