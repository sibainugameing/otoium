# otoium (engine-first)

Fully automatic UTAU `oto.ini` generator in Rust. **This drop contains the internal engine and CLI only; the TUI
(waveform view, playback, manual editing) is deliberately deferred.**

> ## ⚠ Verification status
> * **Compiles on macOS (user-reported)**; written without a toolchain, so only the user's build has verified this. Ubuntu not tried.
> * **Tests:** 43/46 passed on the first user run; the 3 failures were diagnosed and patched but the re-run is unconfirmed (see `docs/testing.md`).
> * **No accuracy has been measured.** There was no ground-truth data. `docs/accuracy.md` contains a protocol, not results.
> * **No research was performed** on Moresampler / OpenMoresampler (network disabled). `docs/research.md` lists what must be verified.

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

## Usage
```
cargo build --release
otoium auto <voicebank_dir> [--mode append|update|replace] [--recursive] [--jobs N] [--dry-run]
                            [--overwrite-manual] [--format cv|vcv|cvvc] [--positive-cutoff]
                            [--report FILE] [--json FILE]
otoium eval <generated oto.ini> <reference oto.ini> [--large-ms N]
```
No per-file prompts. Failures on single files never stop the batch; low-confidence results are still saved and flagged.

## Implementation status
| Area | Status |
|---|---|
| WAV decode (PCM 8/16/24/32, float, extensible), read-only | implemented, untested |
| Feature extraction (energy, voicing, HF ratio, flux, ZCR) | implemented, untested |
| Kana/romaji → phonemes (youon, sokuon, hatsuon, chouon, ふぁ/てぃ…) | implemented, untested |
| Alias kind + CV/VCV/CVVC detection from alias structure | implemented (heuristic weights, unvalidated) |
| Acoustic detection of format (hypothesis comparison by alignment quality) | **not implemented** |
| Pronunciation sources: existing oto.ini, index.csv, filename; acoustic tie-break | implemented; `index.csv` layout is an **unverified assumption** |
| Alignment: rule-based acoustic DP (2 weightings) + proportional fallback | implemented, **not a learned model**, untested |
| Learned forced aligner (CTC/HMM/ONNX) | **not implemented** (swap-in point: `Aligner` trait) |
| Class-pair boundary refinement at 1 ms | implemented, untested |
| 5-parameter computation | implemented; conventions **recalled, not verified** (`docs/oto-parameters.md`) |
| Validation, bounded re-estimation (≤3 attempts), sanitize | implemented, untested |
| Confidence from measurable evidence | implemented; **uncalibrated** |
| Safe save (backup, temp, verify, atomic rename, reload verify, rollback, manual-edit protection, idempotency) | implemented, untested |
| Shift_JIS | partial: ASCII, kana, full-width alnum only; undecodable lines preserved byte-for-byte |
| Parallel batch, panic isolation, cancellation flag | implemented, untested (no CLI hook for cancel yet) |
| Multiple aliases per single WAV (reclist-style) | **not implemented** → reported as failed, existing entries untouched |
| Evaluation: MAE/median/p90/large%, alias coverage | implemented |
| Eval by confidence / phoneme class / recording condition; Moresampler comparison | **not implemented** |
| TUI, waveform, playback, undo/redo | **not implemented** (deferred by request) |
| Training a dedicated model | **not done** |
