# otoium — UTAU 完全自動原音設定

既存の音声認識 AI（Whisper 等）には頼らず、**音素アライメントを中心にした専用エンジン**で
WAV から `oto.ini` を自動生成します。単独音(CV)・連続音(VCV)・CVVC に対応。

## 使い方
```
pip install -r requirements.txt
python run_otoium.py <原音フォルダ>                 # モード自動 (1モーラ=CV / 複数=VCV)
python run_otoium.py <フォルダ> --mode vcv           # 「あ か」「か さ」…
python run_otoium.py <フォルダ> --mode cvvc          # CV + VC(「a k」)
python run_otoium.py <フォルダ> --vcv-style vowel    # 「あ さ」(母音ベース表記)
python run_otoium.py <フォルダ> --script 台本.txt     # ファイル名が台本でない場合
python run_otoium.py                                  # GUI (tkinter)
```
* 台本ファイル: 1行1ファイル。`wav名=テキスト` / タブ区切り / 空白区切り。
* ファイル名は「解析対象の発音列の**仮説**」としてのみ使い、確定はしません（下記の独立検証）。
* 出力: 各フォルダの `oto.ini`（既定 cp932、既存は `.bak` に退避）、`oto_review.txt`（保留候補）、
  `otoium_report.json/txt`、`otoium_model.npz`（学習済みモデル）。`--labels` で Audacity ラベル。
* 終了コード: 0 正常 / 2 失敗ファイルあり / 1 実行エラー。

## 仕組み
1. **特徴抽出** (`features.py`): 5ms ホップで 有声度・ZCR・高域/低域比・重心・F1/F2近似・MFCC12・境界証拠(スペクトル変化+エネルギー谷)。
2. **音素ユニット** (`phoneset.py`/`kana.py`): かな→子音+母音。破裂音は 閉鎖+破裂、破擦音は 閉鎖+摩擦、拗音は +半母音 に分解。
3. **自己教師あり音響モデル** (`model.py`): 音響クラス事前分布から開始し、録音全体で Viterbi 再学習 (MAP) → 話者・マイクに適応。
4. **強制アライメント** (`aligner.py`): 時間長制約付きセミマルコフ Viterbi。モーラ間の任意ポーズ、境界証拠ボーナス付き。
5. **独立検証** (`verify.py`): 台本を使わないフリーデコーダで音素を認識し、アライメント結果と照合。
   母音一致率・子音の音響確認・境界確信度・母音核の数を評価。不合格のモーラ/ファイルは `oto.ini` に**入れません**。
6. **エイリアス生成** (`aliases.py`): 1ファイルから複数エイリアス。Offset/Overlap/Preutterance/Consonant/Cutoff を音素境界から算出。

## 状態
`OK` 出力 / `WARN` 出力(`--strict` で除外) / `REVIEW` 保留(`oto_review.txt`) / `ERROR`・`FAIL` 出力しない。

## テスト
`python -m unittest tests.test_otoium -v`（フォルマント合成器で正解境界つきの連続音を作り、CV/VCV/CVVC・単独音・不一致音声の検出を検証）。
