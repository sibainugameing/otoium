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
主なオプション: `--mode auto|cv|vcv|cvvc` `--accuracy fast|high|max|ultra|custom` `--iters N` `--no-tune` `--no-rescue` `--no-net` `--no-adapt` `--no-auto-thresholds`
`--no-refine` `--jobs N` `--merge` `--prefix` `--suffix` `--head-prefix` `--tail` `--encoding` `--strict`
`--labels` `--textgrid` `--model x.npz` `--dry-run` `--no-backup` `--no-cache`

## 精度モード
| モード | ガウス学習 | 自動調整 | NN(エポック×数) | 適応 | 再学習 | 高信頼再学習 |
|---|---|---|---|---|---|---|
| fast | 3 | なし | なし | なし | 0 | 0 |
| high | 5 | 1周 | 10×1 | 1 | 2 | 0 |
| max | 8 | 2周 | 14×2 | 2 | 3 | 1 |
| ultra | 12 | 3周(細かい探索) | 20×3 | 3 | 4 | 2 |
| custom | `--iters` | `--tune-rounds` | `--net-epochs` `--net-ensemble` | `--adapt-rounds` | `--post-iters` | `--confident-rounds` |

custom 以外でもこれらを指定するとプリセットの値を上書きします（例: `--accuracy max --net-ensemble 4`）。
音素ごとの時間長分布も学習し、発話速度の違いに適応します。

台本ファイル: 1行1ファイル、`wav名=テキスト`（タブ・空白区切りも可）。

## 処理の流れ
1. 特徴抽出（DC除去・低域カット・ノイズフロア正規化、MFCC+Δ、有声度、F1/F2近似、1ms微細包絡）。結果は `.otoium_cache` に保存。
2. 音響モデルの自己教師あり学習: ガウスモデル(Viterbi 再推定、外れファイル除外・重み付け、収束判定) + numpy 製 MLP フレーム分類器(前後フレーム結合、アライメント結果を教師に学習、尤度に加算)。
3. 自動調整 (整列側のみ。検証側は自己採点になるため対象外): 境界重み・時間長重み・ポーズ罰則・デコーダ罰則を、検証スコア最大化で探索。
3b. しきい値の自動校正: 境界確信度・母音一致率・子音一致率の基準を録音データの分布から決定。
4. 録音ごとの適応(モデルを1ファイルに寄せて再整列。検証は適応前のモデルで実施) → 強制アライメント（時間長制約付きセミマルコフ Viterbi）→ 1ms 境界微調整。
5. 台本を使わないフリーデコーダでの独立検証（母音一致率・子音確認・母音核数・境界確信度）。
6. 不合格ファイルの救済: パラメータ変更での再解析、録音が台本より短い場合の末尾モーラ削減。救済後も不合格なら出力しない。
7. エイリアス生成と oto パラメータ算出。

## 学習の得点表示
学習の各段階で「検証スコア」(母音一致・境界確信度・子音確認・不合格率を合成した値、高いほど良い)を、固定のファイル10個程度で測って記録します。
* CLI/ログ: `[段階] 検証スコア 1.540 (+0.172) ▁█` の形で表示。NN の損失・精度もエポックごとに記録。
* GUI: 「学習」タブに 検証スコア / 平均対数尤度 / 境界の変化 / NN損失 / NN精度 / 自動調整スコア の曲線をリアルタイム表示。
* 得点が最高だったモデルを保持し、学習終了時に悪化していれば最高点に戻します。
* `otoium_training.csv` / `.json` に全履歴を出力。

## 失敗箇所の再検査
一通りの処理が終わった後、失敗したファイル・保留になったモーラを再検査します（最大3回、`--recheck-rounds N`、0で無効）。
1回目は成功したファイルでモデルを引き締めて再解析、2回目以降は探索範囲(パラメータ・適応回数・末尾モーラ削減の幅)を広げます。
3回でも不合格なら終了し、その箇所は oto.ini に出力せず、レポートに理由を残します。

## 安全性
* 特徴量キャッシュは npz (pickle 不使用)。壊れた/改ざんされたキャッシュは捨てて再計算します。
* oto.ini は一時ファイル経由で置換(書き込み失敗で壊れたファイルを残さない)。出力が0件のときは既存の oto.ini を変更しません。
* 自動調整は調整用ファイルと評価用ファイルを分け、評価用で悪化したら調整結果を採用しません。

## 出力
`oto.ini`（既定 cp932、既存は `.bak` に退避）, `oto_review.txt`（保留）, `otoium_report.json/txt`, `otoium_model.npz`。
状態: OK/WARN は出力（`--strict` で WARN 除外）、REVIEW は保留、ERROR/FAIL は出力しない。終了コード 0 正常 / 2 失敗あり / 1 実行エラー。

## テスト
`python -m unittest tests.test_otoium -v`

## GUI
引数なしで起動。設定 / 結果 / ログの3タブ。結果タブは波形・音素区間・モーラ状態を表示し、エイリアス行を選ぶと
Offset / Overlap / Preutterance / Consonant / Cutoff の位置を波形上に表示します。ホイールで拡大縮小、ドラッグで移動。
設定の保存・読込、実行中の中止に対応。
