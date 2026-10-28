# UTAU 原音設定自動化ツール

CV（単独音）と VCV（連続音）の音源に対応した `oto.ini` 全自動生成ツールです。

---

## 特徴

- **CV / VCV を自動判定**（またはユーザー指定）
- **音素境界を多段階で推定**：onset 検出 → 有声区間 → スペクトル変化
- **推定値に信頼度を付与**：低信頼度ファイルを TUI で一覧表示
- **元音声を一切変更しない**：出力は `oto.ini` のみ
- **既存 oto.ini を自動バックアップ**
- **TUI（ターミナル UI）で全操作**：Textual ベース

---

## インストール

```bash
pip install -r requirements.txt
```

---

## 使い方

### TUI モード（推奨）

```bash
python main.py
```

起動後の操作：

| キー / ボタン | 動作 |
|---|---|
| `[解析開始]` | 指定フォルダを一括解析 |
| `r` | 選択ファイルを再解析 |
| `s` | oto.ini を保存 |
| `q` | 終了 |

### バッチモード（TUI なし）

```bash
python main.py --batch --input ./voice_bank --format cv --output ./output
```

---

## ディレクトリ構成

```
utau_oto_tool/
├── main.py                  # エントリポイント
├── requirements.txt
├── README.md
│
├── core/                    # パイプライン・結果型・形式検出
│   ├── pipeline.py          # メイン処理フロー
│   ├── result.py            # OtoParams / ProcessResult
│   └── format_detector.py   # CV/VCV 自動判定
│
├── analysis/                # 音声解析
│   ├── wav_loader.py        # WAV 読み込み
│   ├── feature_extractor.py # RMS・ZCR・MFCC・有声区間
│   └── boundary_estimator.py# 音素境界推定（信頼度付き）
│
├── formats/                 # 形式別パラメータ計算
│   ├── cv_processor.py      # CV 用パラメータ推定
│   ├── vcv_processor.py     # VCV 用パラメータ推定
│   └── oto_writer.py        # oto.ini 書き出し（Shift_JIS 対応）
│
├── tui/                     # ターミナル UI
│   ├── app.py               # Textual App 本体
│   ├── widgets.py           # ResultTable / LogPane / StatusBar
│   └── styles.tcss          # TUI スタイルシート
│
├── models/                  # 学習済みモデル統合（将来用）
│   └── base_model.py        # 抽象基底クラス + RuleBasedModel
│
└── tests/                   # ユニットテスト
    ├── test_cv_processor.py
    ├── test_vcv_processor.py
    └── test_format_detector.py
```

---

## oto.ini パラメータの推定方針

### CV（単独音）
- **offset**：子音 onset 直前（10ms パディング）
- **consonant**：子音開始〜母音開始の実測値をヒント範囲でクリップ
- **preutterance**：offset〜子音開始の距離
- **overlap**：破裂音は preutterance × 35%、鼻音・半母音は 65%（子音の立ち上がりを保護）
- **cutoff**：末尾有声区間終端 + 30ms 余白から負値で算出

### VCV（連続音）
- **offset**：先行母音の開始位置（先頭余白終端）
- **preutterance**：先行母音〜子音境界
- **overlap**：子音境界より 30ms 前（先行母音安定区間に食い込む）
- **cutoff**：CV と同様

---

## 将来の拡張ポイント

- `models/` に wav2vec2 ファインチューニングモデルを追加
- `core/pipeline.py` の `_process_one` でモデルを切り替え
- テスト合成による候補値の比較最適化（`formats/` に synthesizer.py を追加）

---

## テスト

```bash
pytest tests/ -v
```
