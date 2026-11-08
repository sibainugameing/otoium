"""パイプライン: 読込 → 特徴 → 自己教師あり学習 → 自動調整 → 評価/救済 → 検証 → oto 生成"""
import copy
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import numpy as np
from .audio_io import read_wav
from .config import resolve
from .features import extract, SilentAudioError, Features
from .model import AcousticModel
from .aligner import align, align_region, AlignmentError
from .kana import parse_text, TextParseError
from .verify import STATUS_RANK, label_runs
from .evaluate import evaluate, evaluate_segs
from .metrics import Tracker
from .tune import autotune
from .calibrate import calibrate
from .aliases import make_entries
from .oto import write_oto, format_entry, read_lines
from .export import write_labels, write_textgrid

# モーラ長が潰れた箇所用: 時間長の重みを強め、母音・鼻音の最短長を学習済み平均長の一定割合に引き上げる
TEMPO_VARIANTS = [dict(duration_weight=3.0, min_dur_ratio=0.3), dict(duration_weight=6.0, min_dur_ratio=0.4),
                  dict(duration_weight=10.0, min_dur_ratio=0.5)]
RESCUE_EXTRA = [dict(boundary_weight=12.0, duration_weight=1.5), dict(pause_penalty=0.1, boundary_weight=2.0),
                dict(duration_weight=0.1)]
RESCUE_VARIANTS = [
    dict(pause_penalty=0.5), dict(pause_penalty=5.0, duration_weight=1.0), dict(boundary_weight=8.0),
    dict(boundary_weight=1.0, duration_weight=0.2),
]


class Cancelled(RuntimeError):
    pass


class FileJob:
    def __init__(self, path):
        self.path = Path(path)
        self.text = ""
        self.morae = []
        self.F = None
        self.X = None
        self.dur_ms = 0.0
        self.env = None
        self.status = "OK"
        self.notes = []
        self.timings = []
        self.segs = []
        self.entries = []
        self.nuclei = 0
        self.score = 0.0
        self.rechecks = 0
        self.decoded_runs = []

    def fail(self, msg):
        self.status = "FAIL"
        self.notes.append(msg)


def load_script(path):
    if not path:
        return {}
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp932"):
        try:
            txt = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("台本の文字コードを判別できません")
    d = {}
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sep in ("=", "\t", " "):
            if sep in line:
                k, v = line.split(sep, 1)
                break
        else:
            continue
        k, v = k.strip(), v.strip()
        d[k] = v
        d[Path(k).stem] = v
    return d


def _envelope(x, n=1000):
    m = max(1, len(x) // n)
    k = max(1, len(x) // m)
    a = x[:k * m].reshape(k, m)
    return np.stack([a.min(1), a.max(1)])


_FEAT_FIELDS = ["cls", "mfcc", "dmfcc", "energy_db", "f0", "bnd", "fine_e", "fine_h"]


def _cache_save(path, dur_ms, F, env):
    d = {k: getattr(F, k) for k in _FEAT_FIELDS}
    d.update(hop_s=np.array(F.hop_s), dur_s=np.array(F.dur_s), ref_db=np.array(F.ref_db),
             dur_ms=np.array(dur_ms), env=env)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **d)
    tmp.replace(path)


def _cache_load(path):
    """pickle を使わない npz キャッシュ (allow_pickle=False: 改ざんされても任意コードは実行されない)"""
    z = np.load(path, allow_pickle=False)
    F = Features(*[z[k] for k in _FEAT_FIELDS[:6]], float(z["hop_s"]), float(z["dur_s"]), float(z["ref_db"]),
                 z["fine_e"], z["fine_h"])
    return float(z["dur_ms"]), F, z["env"]


def _load_one(args):
    """ワーカー: (path, hop, win, cache_dir) → ('ok', (dur_ms, F, env)) / ('err', msg)"""
    path, hop, win, cache_dir = args
    try:
        path = Path(path)
        cp = None
        if cache_dir:
            st = path.stat()
            h = hashlib.md5(str(path.resolve()).encode()).hexdigest()[:12]
            cp = Path(cache_dir) / f"{h}.{st.st_mtime_ns}.{st.st_size}.{hop}.{win}.v3.npz"
            if cp.exists():
                try:
                    return "ok", _cache_load(cp)
                except Exception:
                    pass            # 壊れた/不正なキャッシュは捨てて再計算
        x, sr = read_wav(path)
        data = (len(x) / sr * 1000.0, extract(x, sr, hop, win), _envelope(x))
        if cp:
            cp.parent.mkdir(exist_ok=True)
            _cache_save(cp, *data)
        return "ok", data
    except (SilentAudioError, ValueError, OSError, EOFError, KeyError) as ex:
        return "err", f"{type(ex).__name__}: {ex}"
    except Exception as ex:  # wave.Error など
        return "err", f"{type(ex).__name__}: {ex}"


class Pipeline:
    def __init__(self, cfg, log=print, progress=None, on_metric=None):
        self.cfg = cfg
        self.cfg_initial = cfg
        self.logbuf, self._user_log = [], log
        self.stage_seconds, self._t_mark = {}, time.time()
        self.root = None
        self.log = self._log
        self.track = Tracker(on_metric, self._log)
        self._best_score, self._best_model, self._eval_set = -1e9, None, None
        self._best = {}          # NN 本数 -> (最高得点, モデル): 構成ごとに最良を保持する
        self.progress = progress or (lambda frac, msg="": None)
        self.jobs = []
        self.model = None
        self.report = {}
        self.P = resolve(cfg)
        self.iters = self.P["iters"]
        self.do_tune = self.P["tune"]
        self.rounds = max(1, self.P["rounds"])
        self.do_rescue = self.P["rescue"]
        self.do_net = self.P["net"]
        self.do_adapt = self.P["adapt"]
        self.cancelled = False
        self.review = {}
        self.dropped = []
        self.kept = {}
        self.forced = {}
        self._backed_up = set()

    def decision(self, j, e):
        """このエントリが今の設定で oto.ini に書かれるか: '✓' 書く / '強制' 不合格だが書く / '✗' 書かない (重複整理は含まない)"""
        limit = 0 if self.cfg.strict else 1
        over = j.status == "FAIL" or STATUS_RANK[e.status] > limit
        if e.manual is False:
            return "✗"
        if e.manual is True:
            return ("強制" if over else "✓") if e.valid else "✗"
        if j.status == "FAIL" and not self.cfg.write_failed:
            return "✗"
        if not over:
            return "✓"
        return "強制" if (self.cfg.write_failed and e.valid) else "✗"

    def recheck(self, jobs=None, rounds=None):
        """実行後の再検査 (GUI から呼ぶ)。jobs 省略で不合格箇所のある全ファイル。解決したファイル数を返す"""
        self.cancelled = False
        targets = list(jobs) if jobs is not None else [j for j in self.jobs if self._is_bad(j)]
        for j in targets:
            j.rechecks = 0
        self._recheck(jobs=targets, limit=rounds if rounds else (self.cfg.recheck_rounds or 3), retrain=False)
        return sum(1 for j in targets if not self._is_bad(j))

    def write_output(self, write_failed=None, strict=None):
        """解析結果から oto.ini を (再) 書き込みする。解析はやり直さない。dry_run でも書き込む"""
        if write_failed is not None or strict is not None:
            self.cfg = replace(self.cfg, write_failed=self.cfg.write_failed if write_failed is None else write_failed,
                               strict=self.cfg.strict if strict is None else strict)
        self._emit(self.root, write=True)
        self._report(self.root, self.report.get("seconds", 0.0) if self.report else 0.0, write=True)
        if self.cfg.result_json:
            self.export_json()
        return self.report

    def _log(self, msg):
        self.logbuf.append(msg)
        self._user_log(msg)

    def _mark(self, name):
        now = time.time()
        self.stage_seconds[name] = round(self.stage_seconds.get(name, 0.0) + now - self._t_mark, 2)
        self._t_mark = now

    def export_json(self, path=None, detail=None):
        """詳細な実行結果を JSON で書き出す。path 省略時は設定、さらに省略時は <フォルダ>/otoium_result.json"""
        from .resultjson import write_result
        detail = detail or self.cfg.result_json_detail
        path = Path(path or self.cfg.result_json_path or (self.root / "otoium_result.json"))
        write_result(self, path, detail)
        self.log(f"    詳細な実行結果 ({detail}) を JSON で出力: {path}")
        return path

    def cancel(self):
        self.cancelled = True

    def _check(self):
        if self.cancelled:
            raise Cancelled("中止しました")

    # ------------------------------------------------------------------
    def run(self, root):
        cfg = self.cfg
        t0 = time.time()
        self._t_mark = t0
        self.cfg_initial = self.cfg
        root = Path(root)
        self.root = root
        if not root.is_dir():
            raise FileNotFoundError(f"フォルダが見つかりません: {root}")
        it = root.rglob("*") if cfg.recursive else root.glob("*")
        wavs = sorted(p for p in it if p.is_file() and p.suffix.lower() == ".wav")
        if not wavs:
            raise FileNotFoundError("WAV ファイルが見つかりません")
        script = load_script(cfg.script_path)
        self.log(f"[1/6] {len(wavs)} 個の WAV を読み込み・特徴抽出中…")
        self._load(wavs, script, root)
        self._mark("load")
        ok = [j for j in self.jobs if j.status != "FAIL"]
        if not ok:
            raise RuntimeError("解析可能なファイルがありません")
        self.log("[2/6] 音響モデルを学習中…")
        if cfg.model_path:
            self.model = AcousticModel.load(cfg.model_path)
        else:
            self.model = AcousticModel(cfg.mfcc_weight, cfg.delta_weight)
            self.model.set_norm([j.F for j in ok])
        self.model.hop_ms = cfg.hop_ms
        for j in ok:
            j.X = self.model.design(j.F)
        if not cfg.model_path:
            self._train(ok, self.iters, 0.2, 0.45)
            if self.do_net:
                self._train_net(ok)
        self._mark("train")
        if self.do_tune:
            self.log("[3/6] 解析パラメータを自動調整中…")
            ok = [j for j in ok if j.status != "FAIL"]
            self.cfg = autotune(self.model, ok, self.cfg, self.log, self.rounds,
                                on_score=lambda v: self.track.add("tune_score", v, "自動調整"))
            if not cfg.model_path and self.P["post"] > 0:
                self._train(ok, self.P["post"], 0.55, 0.6, "再学習")
        else:
            self.log("[3/6] 自動調整: スキップ")
        if self.P["conf"] > 0 and not cfg.model_path:
            self._train_confident([j for j in self.jobs if j.status != "FAIL"], self.P["conf"])
        if self.cfg.track_metrics and self._best_model is not None and not cfg.model_path:
            cur = self._eval_score()
            self.track.add("score", cur, "学習終了", log=True)
            if self._best_here() > cur + 0.003 and self._restore_best():
                self.log(f"    得点が最高だったチェックポイント (スコア {self._best_here():.3f}) に戻しました")
        if self.cfg.auto_thresholds:
            ok = [j for j in self.jobs if j.status != "FAIL"]
            self.cfg = calibrate(self.model, ok, self.cfg, self.log)
        self._mark("tune_calibrate")
        self.log("[4/6] 最終アライメント・独立検証・救済…")
        fin = [j for j in self.jobs if j.status != "FAIL"]
        for i, j in enumerate(fin):
            self._check()
            self._final(j)
            self.progress(0.62 + 0.28 * (i + 1) / len(fin), j.path.name)
        self._mark("final_alignment")
        if fin:
            self.track.add("score", float(np.mean([j.score for j in fin if j.status != "FAIL"] or [0.0])), "最終解析", log=True)
        self.log("[5/6] 失敗箇所の再検査と oto.ini の生成…")
        self._recheck()
        self._mark("recheck")
        self._emit(root)
        self._mark("write")
        if not self.cfg.dry_run and self.track.history:
            self.track.save(root)
        if self.cfg.save_model and not self.cfg.dry_run and not self.cfg.model_path:
            self.model.save(root / "otoium_model.npz")
        self.log("[6/6] レポート作成中…")
        self._report(root, time.time() - t0)
        self._mark("report")
        if self.cfg.result_json:
            self.export_json()
        self.progress(1.0, "完了")
        return self.report

    # ------------------------------------------------------------------
    def _load(self, wavs, script, root):
        cfg = self.cfg
        self.jobs = []
        todo = []
        for p in wavs:
            job = FileJob(p)
            self.jobs.append(job)
            try:
                job.text = script.get(p.name) or script.get(p.stem) or p.stem
                skipped = []
                job.morae = parse_text(job.text, cfg.skip_unknown_chars, skipped)
                if skipped:
                    job.notes.append(f"未対応文字を無視しました: {''.join(skipped)}")
                todo.append(job)
            except (TextParseError, KeyError) as ex:
                job.fail(f"{type(ex).__name__}: {ex}")
        cdir = str(root / ".otoium_cache") if cfg.cache and not cfg.dry_run else None
        args = [(str(j.path), cfg.hop_ms, cfg.win_ms, cdir) for j in todo]
        if cfg.jobs > 1 and len(args) > 1:
            with ProcessPoolExecutor(cfg.jobs) as ex:
                results = ex.map(_load_one, args, chunksize=4)
                self._collect(todo, results)
        else:
            self._collect(todo, (_load_one(a) for a in args))

    def _collect(self, todo, results):
        for i, (j, (st, data)) in enumerate(zip(todo, results)):
            self._check()
            if st == "ok":
                j.dur_ms, j.F, j.env = data
            else:
                j.fail(data)
            self.progress(0.2 * (i + 1) / max(1, len(todo)), j.path.name)

    def _train(self, ok, iters, p0, p1, stage="学習"):
        cfg = self.cfg
        prev = {}
        patience, stable, prev_sc = self.P["patience"], 0, None
        for it in range(iters):
            self._check()
            items, scores, shift = [], [], []
            for j in ok:
                if j.status == "FAIL":
                    continue
                try:
                    segs, sc, _ = align(self.model, j.F, j.morae, cfg, LL=self.model.loglik(j.F, j.X))
                except AlignmentError as ex:
                    j.fail(f"AlignmentError: {ex}")
                    continue
                real = [(s.unit, s.s, s.e) for s in segs if s.e > s.s]
                b = np.array([r[1] for r in real])
                if j in prev and len(prev[j]) == len(b):
                    shift.append(float(np.mean(np.abs(prev[j] - b))))
                prev[j] = b
                items.append((j.X, real))
                scores.append(sc / j.F.T)
            if not items:
                raise RuntimeError("学習に使えるファイルがありません")
            if len(items) >= 8:
                a = np.array(scores)
                med = np.median(a)
                mad = np.median(np.abs(a - med)) * 1.4826 + 1e-9
                w = [1.0 if s >= med - 3 * mad else 0.0 for s in scores]
                w = [x * (0.5 + 0.5 * min(1.0, max(0.0, (s - (med - 3 * mad)) / (3 * mad)))) for x, s in zip(w, scores)]
                items = [(X, sg, wt) for (X, sg), wt in zip(items, w) if wt > 0]
            def update(items=items):
                self.model.fit(items)
                if self.model.net is not None:
                    self.model.fit_net(items, epochs=self.P["net_epochs"], ensemble=self.P["ens"], callback=self._nn_cb)

            accepted, sc = self._guarded_update(update, f"{stage} {it + 1}/{iters}")
            conv = float(np.mean(shift)) if shift else None
            self.track.add("ll", float(np.mean(scores)), stage)
            if conv is not None:
                self.track.add("shift", conv, stage)
            self.log(f"    {stage} {it + 1}/{iters}: {len(items)} ファイル使用" + (f", 境界変化 {conv:.2f} フレーム" if conv is not None else ""))
            self.progress(p0 + (p1 - p0) * (it + 1) / iters, "学習")
            if not accepted:
                self.log("    これ以上更新しても得点が上がらないため学習を終了")
                break
            if self.is_fixed_point(conv):
                self.log("    境界が全く動かなくなった (不動点) ため学習を終了: 以降の反復は同じ結果になります")
                break
            if patience > 0 and conv is not None:
                # 「境界がほぼ動かない」かつ「得点が伸びない」が patience 回連続し、かつ予定回数の半分は回したときだけ終了
                stable = self.next_stable(stable, conv, sc, prev_sc, cfg)
                if self.should_stop(stable, patience, it + 1, iters):
                    self.log(f"    収束 (境界変化 {conv:.3f} フレーム, 得点の伸び停止が {stable} 回連続) のため学習を終了")
                    break
            if sc is not None:
                prev_sc = sc

    @staticmethod
    def is_fixed_point(conv):
        """整列が前回と完全に同じ (境界の変化が 0): 次の反復は同じ入力から同じモデルを作るだけなので続ける意味がない"""
        return conv is not None and conv < 1e-9

    @staticmethod
    def next_stable(stable, conv, sc, prev_sc, cfg):
        """「境界がほぼ動かない」かつ「得点が伸びない」なら連続回数を +1、そうでなければ 0 に戻す"""
        flat = sc is None or prev_sc is None or (sc - prev_sc) <= cfg.converge_score
        return stable + 1 if (conv < cfg.converge_shift and flat) else 0

    @staticmethod
    def should_stop(stable, patience, done, iters):
        """patience 回連続で安定し、かつ予定回数の半分は回した (最低2回) ときだけ終了。patience 0 は終了しない"""
        return patience > 0 and stable >= patience and done >= max(2, (iters + 1) // 2)

    def _eval_score(self):
        """学習の進行を測る得点: 固定のファイル集合に対する検証スコアの平均 (高いほど良い)"""
        if self._eval_set is None:
            cand = [j for j in self.jobs if j.status != "FAIL" and j.X is not None]
            step = max(1, len(cand) // 10)
            self._eval_set = cand[::step][:10]
        sc = []
        for j in self._eval_set:
            if j.status == "FAIL":
                continue
            try:
                sc.append(evaluate(self.model, j.F, j.X, j.morae, j.dur_ms, self.cfg).score)
            except (AlignmentError, ValueError, KeyError):
                sc.append(-3.0)
        return float(np.mean(sc)) if sc else -3.0

    def _restore_best(self):
        """得点が最高だったチェックポイントに戻す。NN 構成が違うものには戻さない。
        自動調整で決めた nn_weight は、戻した後も維持する (古いチェックポイントの値で上書きしない)"""
        entry = self._best.get(len(self.model.nets))
        if entry is None:
            return False
        keep_w = self.model.nn_weight
        self.model = copy.deepcopy(entry[1])
        self.model.nn_weight = keep_w
        return True

    def _best_here(self):
        """現在のモデル構成 (NN 本数) での最高得点"""
        e = self._best.get(len(self.model.nets))
        return e[0] if e else -1e9

    def _guarded_update(self, update_fn, stage):
        """モデル更新を試し、得点が最高点より下がるなら、更新を半分 → 4分の1 に減衰して再試行、それでも下がるなら更新前に戻す。
        自己教師あり学習で「更新するほど悪化する」ことを防ぐ。(採用したか, 得点) を返す"""
        cfg = self.cfg
        if not cfg.guarded_updates or not cfg.track_metrics:
            update_fn()
            return True, self._checkpoint(stage)
        before, best_before = copy.deepcopy(self.model), self._best_here()
        update_fn()
        sc = self._checkpoint(stage)
        if sc is None or best_before < -1e8 or sc >= best_before - cfg.accept_tolerance:
            return True, sc
        new = copy.deepcopy(self.model)
        for alpha in (0.5, 0.25):
            self.model = before.blend(new, alpha)
            sc2 = self._checkpoint(f"{stage} 更新を{alpha:g}に減衰")
            if sc2 >= best_before - cfg.accept_tolerance:
                self.log(f"    更新で得点が下がるため、更新幅を {alpha:g} に減衰して採用 ({sc:.3f} → {sc2:.3f}, 最高 {best_before:.3f})")
                return True, sc2
        self.model = before
        self.log(f"    更新で得点が最高点 {best_before:.3f} を下回る ({sc:.3f}) ため、この更新は採用せず元に戻しました")
        return False, sc

    def _checkpoint(self, stage):
        if not self.cfg.track_metrics:
            return None
        sc = self._eval_score()
        self.track.add("score", sc, stage, log=True)
        key = len(self.model.nets)
        if key not in self._best or sc > self._best[key][0]:
            self._best[key] = (sc, copy.deepcopy(self.model))
        if sc > self._best_score:
            self._best_score, self._best_model = sc, self._best[key][1]
        return sc

    def _train_net(self, ok):
        """アライメント結果を教師にニューラル音響モデルを学習し、ガウスモデルと併用する"""
        self.log("    ニューラル音響モデルを学習中…")
        items = []
        for j in ok:
            self._check()
            if j.status == "FAIL":
                continue
            try:
                segs, _, _ = align(self.model, j.F, j.morae, self.cfg, LL=self.model.loglik(j.F, j.X))
            except AlignmentError:
                continue
            items.append((j.X, [(s.unit, s.s, s.e) for s in segs if s.e > s.s], 1.0))
        loss = self.model.fit_net(items, epochs=self.P["net_epochs"], ensemble=self.P["ens"], callback=self._nn_cb)
        self.model.nn_weight = self.cfg.nn_weight
        self.log(f"    ニューラルモデル学習完了 (アンサンブル {len(self.model.nets)}, loss {loss:.3f}, 重み {self.model.nn_weight})")
        sc = self._checkpoint("NN学習後")
        b0 = self._best.get(0)
        if sc is not None and self.cfg.guarded_updates and b0 and sc < b0[0] - self.cfg.accept_tolerance:
            self.model.nets, self.model.nn_weight = [], 0.0          # NN を併用すると得点が下がる → 使わない
            self.log(f"    NN の併用で得点が下がる ({sc:.3f} < {b0[0]:.3f}) ため NN は使用しません")
            return
        self._train(ok, 2, 0.45, 0.55, "NN併用学習")

    def _nn_cb(self, k, ep, loss, acc):
        self._nn_step = getattr(self, "_nn_step", 0) + 1
        self.track.add("nn_loss", loss, f"NN#{k + 1}")
        self.track.add("nn_acc", acc, f"NN#{k + 1}")
        if self._nn_step % 5 == 0:
            self.log(f"    NN#{k + 1} epoch {ep}: loss {loss:.3f} acc {acc:.3f} {self.track.spark('nn_loss')}")

    def _train_confident(self, ok, rounds):
        """検証スコアの高いファイルだけを重み付きで使い、誤整列に引きずられないよう再学習する"""
        for r in range(rounds):
            scored = []
            for j in ok:
                self._check()
                try:
                    res = evaluate(self.model, j.F, j.X, j.morae, j.dur_ms, self.cfg)
                except (AlignmentError, ValueError, KeyError):
                    continue
                if res.status == "FAIL":
                    continue
                scored.append((res.score, j, res))
            if len(scored) < 4:
                return
            sc = np.array([x[0] for x in scored])
            lo, hi = np.percentile(sc, 30), np.percentile(sc, 90) + 1e-6
            items = []
            for s_, j, res in scored:
                if s_ < lo:
                    continue
                w = 0.4 + 0.6 * float(np.clip((s_ - lo) / (hi - lo), 0, 1))
                items.append((j.X, [(g.unit, g.s, g.e) for g in res.segs if g.e > g.s], w))
            def update(items=items):
                self.model.fit(items)
                if self.model.nets:
                    self.model.fit_net(items, epochs=self.P["net_epochs"], ensemble=self.P["ens"], callback=self._nn_cb)

            accepted, _ = self._guarded_update(update, f"高信頼再学習 {r + 1}/{rounds}")
            self.log(f"    高信頼再学習 {r + 1}/{rounds}: {len(items)}/{len(scored)} ファイル使用")

    @staticmethod
    def _tail_is_silent(segs, F):
        """最後の無音区間に有声・有意なエネルギーが残っていないか"""
        real = [g for g in segs if g.e > g.s]
        vd = [g.e - g.s for g in real if g.kind == "v"]
        if len(vd) >= 3 and vd[-1] > 1.6 * float(np.median(vd[:-1])):
            return False            # 最後の母音だけ極端に長い = 余った音を引き延ばして吸収している
        lastv = [g for g in real if g.kind == "v"][-1]
        ev = F.cls[lastv.s:lastv.e, 0]
        if len(ev) >= 8 and float(np.percentile(ev, 10)) < float(np.median(ev)) - 1.0:
            return False            # 母音の途中に 10dB 以上の谷 = 別の音(子音)を含んでいる
        last = real[-1]
        if last.kind != "pau":
            return False
        e = F.cls[last.s:last.e, 0]
        return len(e) == 0 or float(np.percentile(e, 90)) < -1.8

    def _rate(self, segs):
        """この録音の発話速度比 (母音長 / 学習済み平均母音長) を中央値で推定"""
        v = []
        for g in segs:
            if g.e > g.s and g.kind == "v" and g.s_ms is not None:
                v.append(np.log(max(g.e_ms - g.s_ms, 5.0) / np.exp(self.model.dur_mu[g.unit])))
        return float(np.clip(np.exp(np.median(v)), 0.6, 1.7)) if v else 1.0

    def _need_rescue(self, r):
        if r.status != "OK":
            return True
        bad = sum(1 for t in r.timings if STATUS_RANK[t.status] >= 1)
        return bad / len(r.timings) > 0.15

    def _final(self, j, level=0):
        cfg = self.cfg
        if level:
            j.status, j.entries, j.timings, j.segs = "OK", [], [], []
        mdl = self.model
        try:
            LLb = mdl.loglik(j.F, j.X)
            LL = LLb
            best = evaluate(mdl, j.F, j.X, j.morae, j.dur_ms, cfg, LL=LL)
        except (AlignmentError, ValueError, KeyError) as ex:
            best = None
            LLb = LL = None
            j.notes.append(f"初回解析失敗: {ex}")
        rate = 1.0
        if (self.do_adapt or level) and best is not None:
            for _ in range(self.P["adapt_rounds"] + level):
                try:
                    rt = self._rate(best.segs)
                    m2 = self.model.adapt(j.X, [(g.unit, g.s, g.e) for g in best.segs if g.e > g.s])
                    LL2 = m2.loglik(j.F, j.X)
                    r2 = evaluate(m2, j.F, j.X, j.morae, j.dur_ms, cfg, LL=LL2, vmodel=self.model, vLL=LLb, rate=rt)
                except (AlignmentError, ValueError, KeyError):
                    break
                if r2.score > best.score + 0.01:
                    j.notes.append(f"録音への適応で改善 (スコア {best.score:.2f} → {r2.score:.2f}, 速度比 {rt:.2f})")
                    best, mdl, LL, rate = r2, m2, LL2, rt
                else:
                    break
        if (self.do_rescue or level) and (best is None or self._need_rescue(best)):
            base = best.score if best else -9.0
            cands = []
            for v in RESCUE_VARIANTS + (TEMPO_VARIANTS if level >= 1 else []) + (RESCUE_EXTRA if level >= 2 else []):
                cands.append((replace(cfg, **v), j.morae, f"パラメータ {v}"))
            n = len(j.morae)
            for k in range(1, min(3 + level, n - 2) + 1):
                cands.append((cfg, j.morae[:-k], f"録音が台本より末尾 {k} モーラ短いと判断"))
            for c, mo, why in cands:
                try:
                    r = evaluate(mdl, j.F, j.X, mo, j.dur_ms, c, LL=LL, vmodel=self.model, vLL=LLb, rate=rate)
                except (AlignmentError, ValueError, KeyError):
                    continue
                need = 0.15 if len(mo) != n else 0.05
                if len(mo) != n and (r.status != "OK" or any(STATUS_RANK[t.status] >= 2 for t in r.timings)):
                    continue
                if len(mo) != n and not self._tail_is_silent(r.segs, j.F):
                    continue        # 末尾を削ると、説明できない発声が残る → 採用しない
                if r.score > (best.score if best else -9.0) + need and r.status != "FAIL":
                    best = r
                    best.notes = best.notes + [f"救済: {why}"]
            if best is not None and best.score > base:
                j.notes.append(f"救済により再解析 (スコア {base:.2f} → {best.score:.2f})")
        if best is None:
            j.fail("アライメントに失敗しました")
            return
        self._apply(j, best)

    def _apply(self, j, best):
        """評価結果を FileJob に反映し、エントリを生成する (失敗でも write_failed なら生成する)"""
        cfg = self.cfg
        j.morae, j.segs, j.timings, j.nuclei, j.score = best.morae, best.segs, best.timings, best.nuclei, best.score
        j.notes = [n for n in j.notes if not n.startswith(("分割失敗", "母音核数が"))]
        for n in best.notes:
            if n not in j.notes:
                j.notes.append(n)
        j.status = best.status
        j.decoded_runs = label_runs(best.labels, cfg.hop_ms) if best.labels is not None else []
        j.entries = []
        j.entries = make_entries(j.path.name, j.timings, j.dur_ms, cfg)     # 失敗ファイルも生成 (出力するかは _emit が決める)
        for e in j.entries:
            e.alias = cfg.alias_prefix + e.alias + cfg.alias_suffix
            if best.status == "FAIL":
                if STATUS_RANK[e.status] < 2:
                    e.status = "REVIEW"
                e.notes.append("ファイル全体が検証に不合格")

    # ------------------------------------------------------------------
    @staticmethod
    def _is_bad(j):
        return j.status == "FAIL" or any(STATUS_RANK[t.status] >= 2 for t in j.timings)

    @staticmethod
    def _bad_runs(timings):
        runs = []
        for t in timings:
            if STATUS_RANK[t.status] >= 2:
                if runs and t.index == runs[-1][1] + 1:
                    runs[-1][1] = t.index
                else:
                    runs.append([t.index, t.index])
        return runs

    def _repair_run(self, j, a, b, level):
        """悪いモーラの連続区間 [a, b] だけを、前後の良いモーラを固定したまま再整列する。
        成功すれば新しい Result を返す (区間外の整列結果は変更しない)。"""
        cfg, n, segs = self.cfg, len(j.morae), j.segs
        LLb = self.model.loglik(j.F, j.X)
        good = [g for g in segs if g.e > g.s and g.mora >= 0 and not (a <= g.mora <= b)]
        try:
            mdl = self.model.adapt(j.X, [(g.unit, g.s, g.e) for g in good]) if good else self.model
            LLa = mdl.loglik(j.F, j.X)
        except (ValueError, KeyError):
            mdl, LLa = self.model, LLb
        rate = self._rate(good) if good else 1.0
        cur_bad = sum(1 for t in j.timings if STATUS_RANK[t.status] >= 2)
        variants = [None] + RESCUE_VARIANTS + TEMPO_VARIANTS + (RESCUE_EXTRA if level >= 2 else [])
        for expand in (0, 1, 2):
            aa, bb = max(0, a - expand), min(n - 1, b + expand)
            ip = max([i for i, g in enumerate(segs) if g.mora == aa - 1], default=-1) if aa > 0 else -1
            nx = [i for i, g in enumerate(segs) if g.mora == bb + 1] if bb < n - 1 else []
            ine = min(nx) if nx else len(segs)
            f0 = segs[ip].e if ip >= 0 else 0
            f1 = segs[ine].s if ine < len(segs) else j.F.T
            found = []
            for v in variants:
                c = replace(cfg, **v) if v else cfg
                try:
                    new = align_region(mdl, LLa, j.F.bnd, j.morae[aa:bb + 1], f0, f1, c, rate)
                    for g in new:
                        if g.mora >= 0:
                            g.mora += aa
                    full = [copy.copy(g) for g in segs[:ip + 1]] + new + [copy.copy(g) for g in segs[ine:]]
                    res = evaluate_segs(self.model, j.F, j.morae, full, j.dur_ms, c, LLb)
                except (AlignmentError, ValueError, KeyError):
                    continue
                bad = sum(1 for t in res.timings if STATUS_RANK[t.status] >= 2)
                if bad < cur_bad or (bad == cur_bad and res.score > j.score + 0.05):
                    found.append((bad, -res.score, res))
            if found:
                found.sort(key=lambda x: (x[0], x[1]))
                return found[0][2]
        return None

    def _repair_partial(self, j, level):
        """失敗したモーラ区間だけをやり直す。改善した区間があれば True"""
        changed = False
        tried = set()
        while True:
            runs = [r for r in self._bad_runs(j.timings) if tuple(r) not in tried]
            if not runs:
                break
            a, b = runs[0]
            tried.add((a, b))
            res = self._repair_run(j, a, b, level)
            if res is not None:
                before = sum(1 for t in j.timings if STATUS_RANK[t.status] >= 2)
                self._apply(j, res)
                after = sum(1 for t in j.timings if STATUS_RANK[t.status] >= 2)
                j.notes.append(f"部分やり直し: モーラ {a + 1}〜{b + 1} 番だけ再整列 (不合格 {before} → {after})")
                changed = True
        return changed

    def _recheck(self, jobs=None, limit=None, retrain=True):
        """一通り終わった後で失敗箇所を再検査する。まず失敗したモーラ区間だけをやり直し、
        整列自体が得られていないファイルは全体をやり直す。最大 recheck_rounds 回で打ち切る"""
        limit = self.cfg.recheck_rounds if limit is None else limit
        if limit <= 0:
            return
        pool = self.jobs if jobs is None else jobs
        for rnd in range(1, limit + 1):
            todo = [j for j in pool if j.F is not None and j.morae and j.X is not None and self._is_bad(j)]
            if not todo:
                break
            self.log(f"    再検査 {rnd}/{limit} 回目: 失敗箇所のあるファイル {len(todo)} 件")
            if rnd == 1 and retrain and not self.cfg.model_path:
                good = [j for j in self.jobs if j.status == "OK" and j.segs and not self._is_bad(j)]
                if len(good) >= 4:      # 成功したファイルだけでモデルを引き締めてから再解析
                    items = [(j.X, [(g.unit, g.s, g.e) for g in j.segs if g.e > g.s]) for j in good]
                    self.model.fit(items)
            for j in todo:
                self._check()
                j.rechecks = rnd
                j.notes.append(f"再検査 {rnd} 回目")
                if self.cfg.partial_repair and j.segs and j.timings:
                    self._repair_partial(j, rnd)
                if self._is_bad(j) and (not j.segs or not self.cfg.partial_repair or rnd >= 2):
                    self._final(j, level=rnd)       # 区間のやり直しで直らない/整列が無い → ファイル全体を再解析
            fixed = sum(1 for j in todo if not self._is_bad(j))
            self.log(f"    再検査 {rnd} 回目の結果: 解決 {fixed} / 残り {len(todo) - fixed}")
            if fixed == 0 and rnd >= 3:         # 3 回目までに使える手段は出し尽くしている。同じ試行を繰り返さない
                self.log("    これ以上の手段がないため再検査を打ち切ります")
                break
        left = [j for j in pool if j.F is not None and j.morae and self._is_bad(j) and j.rechecks >= limit]
        for j in left:
            j.notes.append(f"{limit} 回再検査しても不合格のため終了" +
                           (" (強制出力モード: 該当箇所も oto.ini に書き込みます)" if self.cfg.write_failed
                            else " (該当箇所は oto.ini に出力しません)"))
        if left:
            self.log(f"    {limit} 回再検査しても不合格: {len(left)} 件 → 終了")

    def _emit(self, root, write=None):
        cfg = self.cfg
        dry = cfg.dry_run if write is None else not write
        self.review, self.kept, self.forced, self.dropped = {}, {}, {}, []
        bydir = {}
        for j in self.jobs:
            bydir.setdefault(j.path.parent, []).append(j)
        limit = 0 if cfg.strict else 1
        for d, js in bydir.items():
            keep, review, forced = [], [], []
            for j in js:
                for e in j.entries:
                    e.forced = False
                    over = j.status == "FAIL" or STATUS_RANK[e.status] > limit
                    if e.manual is False:
                        review.append(e)
                    elif e.manual is True and e.valid:
                        e.forced = over
                        keep.append(e)
                    elif j.status == "FAIL" and not cfg.write_failed:
                        continue
                    elif not over:
                        keep.append(e)
                    elif cfg.write_failed and e.valid:
                        e.forced = True
                        keep.append(e)
                    else:
                        review.append(e)
            if cfg.dedupe:
                best = {}
                for e in keep:
                    cur = best.get(e.alias)
                    key = lambda x: (0 if x.manual is True else 1, STATUS_RANK[x.status], -x.quality)   # 手動で「書く」にしたものを最優先
                    if cur is None or key(e) < key(cur):
                        best[e.alias] = e
                self.dropped += [e for e in keep if best[e.alias] is not e]
                keep = [e for e in keep if best[e.alias] is e]
            forced = [e for e in keep if e.forced]
            self.review[d] = review
            self.kept[d] = keep
            self.forced[d] = forced
            if dry:
                continue
            if not forced:
                (d / "otoium_forced.txt").unlink(missing_ok=True)       # 前回の古い一覧を残さない
            if not review:
                (d / "oto_review.txt").unlink(missing_ok=True)
            if forced:
                (d / "otoium_forced.txt").write_text(
                    "# 検証に不合格だが write_failed により oto.ini へ書き込んだエントリ (要確認)\n" +
                    "\n".join(f"{format_entry(e)}   # {e.status}: {'; '.join(e.notes)}" for e in forced) + "\n",
                    encoding="utf-8")
            if not keep:
                self.log(f"    {d}: 出力できるエントリが 0 件のため、既存の oto.ini は変更しません")
                if review:
                    (d / "oto_review.txt").write_text(
                        "# 確信度不足・検証失敗のため oto.ini に入れなかった候補\n" +
                        "\n".join(f"{format_entry(e)}   # {e.status}: {'; '.join(e.notes)}" for e in review) + "\n",
                        encoding="utf-8")
                continue
            extra = []
            if cfg.merge:
                names = {j.path.name for j in js}
                extra = [l for l in read_lines(d / "oto.ini", cfg.encoding) if l.split("=", 1)[0] not in names]
            first = d not in self._backed_up           # バックアップは最初の書き込み時だけ (GUI での再書き込みで増やさない)
            self._backed_up.add(d)
            write_oto(d / "oto.ini", keep, cfg.encoding, cfg.backup and first, extra)
            if review:
                (d / "oto_review.txt").write_text(
                    "# 確信度不足・検証失敗のため oto.ini に入れなかった候補\n" +
                    "\n".join(f"{format_entry(e)}   # {e.status}: {'; '.join(e.notes)}" for e in review) + "\n",
                    encoding="utf-8")
            self.log(f"    {d}: {len(keep)} 件を出力 (うち強制出力 {len(forced)} / 保留 {len(review)})")
        if not dry:
            for j in self.jobs:
                if j.segs and j.status != "FAIL":
                    if cfg.export_labels:
                        write_labels(j.path.with_suffix(".labels.txt"), j)
                    if cfg.textgrid:
                        write_textgrid(j.path.with_suffix(".TextGrid"), j)

    def _report(self, root, sec, write=None):
        dry = self.cfg.dry_run if write is None else not write
        files = []
        for j in self.jobs:
            files.append({
                "file": str(j.path.relative_to(root)), "text": j.text, "status": j.status, "score": round(j.score, 3),
                "notes": j.notes, "detected_nuclei": j.nuclei, "expected_morae": len(j.morae),
                "morae": [{
                    "surface": t.mora.surface, "cons": t.mora.cons, "vowel": t.mora.vowel,
                    "cons_start_ms": round(t.cons_start, 1), "vowel_start_ms": round(t.vowel_start, 1),
                    "vowel_end_ms": round(t.vowel_end, 1), "units": [(u, round(a, 1), round(b, 1)) for u, a, b in t.segs],
                    "recognized_vowel": t.rec_vowel, "recognized_cons": t.rec_cons,
                    "vowel_agreement": round(t.agree_v, 3), "consonant_agreement": round(t.cons_agree, 3),
                    "boundary_conf": round(t.bconf, 3), "devoiced": t.devoiced, "status": t.status, "notes": t.notes} for t in j.timings],
                "aliases": [{"alias": e.alias, "kind": e.kind, "status": e.status, "confidence": round(e.quality, 3),
                             "valid": e.valid, "line": format_entry(e),
                             "notes": e.notes} for e in j.entries],
            })
        n_fail = sum(1 for j in self.jobs if j.status == "FAIL")
        n_warn = sum(1 for j in self.jobs if j.status == "WARN")
        n_ent = sum(len(v) for v in self.kept.values())
        n_rev = sum(len(v) for v in self.review.values())
        self.report = {"seconds": round(sec, 1), "files": files, "total": len(self.jobs), "failed": n_fail,
                       "warned": n_warn, "entries_written": n_ent, "entries_review": n_rev, "entries_forced": sum(len(v) for v in self.forced.values()),
                       "config": self.cfg.to_dict(), "nn_weight": self.model.nn_weight}
        if not dry:
            (root / "otoium_report.json").write_text(json.dumps(self.report, ensure_ascii=False, indent=1), encoding="utf-8")
            lines = [f"otoium レポート (処理 {sec:.1f}s)", f"ファイル {len(self.jobs)} / 失敗 {n_fail} / 警告 {n_warn}",
                     f"oto.ini 出力 {n_ent} 件 / 保留 {n_rev} 件", ""]
            for f in files:
                lines.append(f"[{f['status']}] {f['file']} ({f['text']}) 母音核 {f['detected_nuclei']}/{f['expected_morae']} score {f['score']}")
                for n in f["notes"]:
                    lines.append(f"    ! {n}")
                for m in f["morae"]:
                    if m["status"] != "OK":
                        lines.append(f"    - {m['surface']}: {m['status']} {'; '.join(m['notes'])}")
            (root / "otoium_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.log(f"完了: ファイル {len(self.jobs)} 件 (失敗 {n_fail}, 警告 {n_warn}) / oto {n_ent} 件 / 保留 {n_rev} 件")
