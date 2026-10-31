# otoium

WAV から oto.ini を生成するツール。単独音(CV)・連続音(VCV)・CVVC に対応。

## 使い方
```
pip install -r requirements.txt
python run_otoium.py <フォルダ>                        # 精度 high / モード auto
python run_otoium.py <フォルダ> --mode vcv --accuracy max
python run_otoium.py <フォルダ> --mode cvvc --vcv-style vowel
python run_otoium.py <フォルダ> --script 台本.txt --jobs 4 --textgrid
python run_otoium.py                                    # GUI (tkinter)
```
主なオプション: `--mode auto|cv|vcv|cvvc` `--accuracy fast|high|max` `--iters N` `--no-tune` `--no-rescue`
`--no-refine` `--jobs N` `--merge` `--prefix` `--suffix` `--head-prefix` `--tail` `--encoding` `--strict`
`--labels` `--textgrid` `--model x.npz` `--dry-run` `--no-backup` `--no-cache`

台本ファイル: 1行1ファイル、`wav名=テキスト`（タブ・空白区切りも可）。

## 処理の流れ
1. 特徴抽出（DC除去・低域カット・ノイズフロア正規化、MFCC+Δ、有声度、F1/F2近似、1ms微細包絡）。結果は `.otoium_cache` に保存。
2. 音響モデルの自己教師あり学習（Viterbi 再推定、外れファイル除外・重み付け、収束判定）。
3. 自動調整: 境界重み・時間長重み・ポーズ罰則・デコーダ罰則を、検証スコア最大化で探索。
4. 強制アライメント（時間長制約付きセミマルコフ Viterbi）→ 1ms 境界微調整。
5. 台本を使わないフリーデコーダでの独立検証（母音一致率・子音確認・母音核数・境界確信度）。
6. 不合格ファイルの救済: パラメータ変更での再解析、録音が台本より短い場合の末尾モーラ削減。救済後も不合格なら出力しない。
7. エイリアス生成と oto パラメータ算出。

## 出力
`oto.ini`（既定 cp932、既存は `.bak` に退避）, `oto_review.txt`（保留）, `otoium_report.json/txt`, `otoium_model.npz`。
状態: OK/WARN は出力（`--strict` で WARN 除外）、REVIEW は保留、ERROR/FAIL は出力しない。終了コード 0 正常 / 2 失敗あり / 1 実行エラー。

## テスト
`python -m unittest tests.test_otoium -v`
