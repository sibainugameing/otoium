# Architecture

```
batch ─ scans dirs, builds candidates, runs files (threads, catch_unwind), saves via store
  └ pipeline::process_file
       wav → dsp(coarse 5 ms, fine 1 ms)
       candidates(sources: oto.ini > index.csv > filename) → pick by acoustic cost (+0.02/rank)
       attempts (≤3): primary → alternate → fallback(heuristic, only if no error-free attempt)
          align(Aligner) → refine(1 ms, class-pair evidence) → params::compute → validate
       select first error-free attempt, else fewest errors → sanitize → re-validate (else FAILED, nothing saved)
       confidence::assess (disagreement of two engines, DP fit, SNR, clipping, warnings)
store::save  backup → tmp → verify → rename → reload verify → rollback on failure; state file for manual-edit protection
report / eval
```
Modules: `wav dsp kana format sjis oto align refine params validate confidence pipeline batch store sources report eval error`.
Boundary estimation (`align`, `refine`) and parameter math (`params`) are independent and separately tested.
Engines are swappable through `trait Aligner`; `Engine` holds primary/alternate/fallback.
No external crates.

## ディレクトリ構成 / Directory layout

```
otoium/
├── Cargo.toml                 # 依存crateなし (std only), edition 2021
├── README.md
├── docs/
│   ├── research.md            # 調査ログ (※未実施。検証チェックリストのみ)
│   ├── architecture.md        # 構成とデータフロー
│   ├── alignment.md           # アライメント方式と境界推定
│   ├── oto-parameters.md      # 5項目の仕様と計算方法 (※仕様は未検証)
│   ├── accuracy.md            # 評価プロトコル (※実測値なし)
│   ├── testing.md             # テスト一覧 (※未実行)
│   └── privacy-and-licenses.md
└── src/
    ├── main.rs                # CLI: `auto` / `eval`
    ├── lib.rs                 # モジュール宣言
    ├── error.rs               # 共通エラー型
    │ ── 入力・解析 ──────────────────────────────
    ├── wav.rs                 # WAV読み込み (読み取り専用)
    ├── dsp.rs                 # FFT, エネルギー/有声度/高域比/flux/ZCR
    ├── kana.rs                # ひらがな/カタカナ/ローマ字 → 音素列
    ├── format.rs              # エイリアス種別, CV/VCV/CVVC判定, 期待エイリアス
    ├── sources.rs             # index.csv 読み込み
    │ ── 推定エンジン (交換可能) ──────────────────
    ├── align.rs               # trait Aligner / 音響DP / 比例分割フォールバック
    ├── refine.rs              # 音素クラス別の境界精密化 (1ms)
    │ ── oto.ini 値の計算 (境界推定とは独立) ───────
    ├── params.rs              # 境界 → offset/consonant/cutoff/preutter/overlap
    ├── validate.rs            # 整合性検証
    ├── confidence.rs          # 信頼度 (測定可能な根拠, 未較正)
    │ ── 入出力・保護 ───────────────────────────
    ├── oto.rs                 # oto.ini 読み書き (コメント/行順/不明行を保持)
    ├── sjis.rs                # Shift_JIS 部分実装
    ├── store.rs               # バックアップ→一時書込→検証→置換→再検証→復元
    │ ── 統括 ─────────────────────────────────
    ├── pipeline.rs            # 1ファイルの全自動処理 (最大3回の再推定)
    ├── batch.rs               # フォルダ一括処理 (並列, panic隔離, キャンセル)
    ├── report.rs              # テキスト/JSONレポート
    ├── eval.rs                # 精度評価 (MAE/中央値/p90/大誤差率)
    └── testutil.rs            # テスト用合成音声 (cfg(test))
```

### モジュール依存関係

```
main ─► batch ─► pipeline ─► align ─► dsp
          │         │          ▲        ▲
          │         ├─► refine ┘        │
          │         ├─► params ─────────┤
          │         ├─► validate ───────┤
          │         ├─► confidence      │
          │         └─► wav ────────────┘
          ├─► format ─► kana
          ├─► sources ─► sjis
          ├─► store ─► oto ─► sjis
          └─► report
main ─► eval ─► oto
```

### 対象フォルダ側に生成されるファイル

```
<voicebank>/
├── *.wav                      # 変更しない
├── index.csv                  # 任意入力 (読み取りのみ)
├── oto.ini                    # 生成/更新 (Shift_JIS既定, 既存の文字コードは維持)
├── oto.ini.bak-<unix>-<n>     # 書き換え前の自動バックアップ
├── .otoium-state.tsv          # otoiumが生成した行の記録 (手動編集の検出用)
└── oto.ini.otoium-tmp         # 保存途中のみ存在 (成功/失敗後に消える・置換される)
```
