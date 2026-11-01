"""パイプライン: 読込 → 特徴 → 自己教師あり学習 → 自動調整 → 評価/救済 → 検証 → oto 生成"""
import hashlib
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import numpy as np
from .audio_io import read_wav
from .config import resolve
from .features import extract, SilentAudioError
from .model import AcousticModel
from .aligner import align, AlignmentError
from .kana import parse_text, TextParseError
from .verify import STATUS_RANK
from .evaluate import evaluate
from .tune import autotune
from .calibrate import calibrate
from .aliases import make_entries
from .oto import write_oto, format_entry, read_lines
from .export import write_labels, write_textgrid

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


def _load_one(args):
    """ワーカー: (path, hop, win, cache_dir) → ('ok', (dur_ms, F, env)) / ('err', msg)"""
    path, hop, win, cache_dir = args
    try:
        path = Path(path)
        cp = None
        if cache_dir:
            st = path.stat()
            h = hashlib.md5(str(path.resolve()).encode()).hexdigest()[:12]
            cp = Path(cache_dir) / f"{h}.{st.st_mtime_ns}.{st.st_size}.{hop}.{win}.v2.pkl"
            if cp.exists():
                return "ok", pickle.loads(cp.read_bytes())
        x, sr = read_wav(path)
        data = (len(x) / sr * 1000.0, extract(x, sr, hop, win), _envelope(x))
        if cp:
            cp.parent.mkdir(exist_ok=True)
            cp.write_bytes(pickle.dumps(data))
        return "ok", data
    except (SilentAudioError, ValueError, OSError, EOFError, KeyError) as ex:
        return "err", f"{type(ex).__name__}: {ex}"
    except Exception as ex:  # wave.Error など
        return "err", f"{type(ex).__name__}: {ex}"


class Pipeline:
    def __init__(self, cfg, log=print, progress=None):
        self.cfg = cfg
        self.log = log
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

    def cancel(self):
        self.cancelled = True

    def _check(self):
        if self.cancelled:
            raise Cancelled("中止しました")

    # ------------------------------------------------------------------
    def run(self, root):
        cfg = self.cfg
        t0 = time.time()
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"フォルダが見つかりません: {root}")
        it = root.rglob("*") if cfg.recursive else root.glob("*")
        wavs = sorted(p for p in it if p.is_file() and p.suffix.lower() == ".wav")
        if not wavs:
            raise FileNotFoundError("WAV ファイルが見つかりません")
        script = load_script(cfg.script_path)
        self.log(f"[1/6] {len(wavs)} 個の WAV を読み込み・特徴抽出中…")
        self._load(wavs, script, root)
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
        if self.do_tune:
            self.log("[3/6] 解析パラメータを自動調整中…")
            ok = [j for j in ok if j.status != "FAIL"]
            self.cfg = autotune(self.model, ok, self.cfg, self.log, self.rounds)
            if not cfg.model_path and self.P["post"] > 0:
                self._train(ok, self.P["post"], 0.55, 0.6)
        else:
            self.log("[3/6] 自動調整: スキップ")
        if self.P["conf"] > 0 and not cfg.model_path:
            self._train_confident([j for j in self.jobs if j.status != "FAIL"], self.P["conf"])
        if self.cfg.auto_thresholds:
            ok = [j for j in self.jobs if j.status != "FAIL"]
            self.cfg = calibrate(self.model, ok, self.cfg, self.log)
        self.log("[4/6] 最終アライメント・独立検証・救済…")
        fin = [j for j in self.jobs if j.status != "FAIL"]
        for i, j in enumerate(fin):
            self._check()
            self._final(j)
            self.progress(0.62 + 0.28 * (i + 1) / len(fin), j.path.name)
        self.log("[5/6] oto.ini を生成中…")
        self._emit(root)
        if self.cfg.save_model and not self.cfg.dry_run and not self.cfg.model_path:
            self.model.save(root / "otoium_model.npz")
        self.log("[6/6] レポート作成中…")
        self._report(root, time.time() - t0)
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
                job.morae = parse_text(job.text)
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

    def _train(self, ok, iters, p0, p1):
        cfg = self.cfg
        prev = {}
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
            self.model.fit(items)
            if self.model.net is not None:
                self.model.fit_net(items, epochs=self.P["net_epochs"], ensemble=self.P["ens"])
            conv = float(np.mean(shift)) if shift else None
            self.log(f"    学習 {it + 1}/{iters}: {len(items)} ファイル使用" + (f", 境界変化 {conv:.2f} フレーム" if conv is not None else ""))
            self.progress(p0 + (p1 - p0) * (it + 1) / iters, "学習")
            if conv is not None and conv < 0.1:
                self.log("    収束したため学習を終了")
                break

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
        loss = self.model.fit_net(items, epochs=self.P["net_epochs"], ensemble=self.P["ens"])
        self.model.nn_weight = self.cfg.nn_weight
        self.log(f"    ニューラルモデル学習完了 (アンサンブル {len(self.model.nets)}, loss {loss:.3f}, 重み {self.model.nn_weight})")
        self._train(ok, 2, 0.45, 0.55)

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
            self.model.fit(items)
            if self.model.nets:
                self.model.fit_net(items, epochs=self.P["net_epochs"], ensemble=self.P["ens"])
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

    def _final(self, j):
        cfg = self.cfg
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
        if self.do_adapt and best is not None:
            for _ in range(self.P["adapt_rounds"]):
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
        if self.do_rescue and (best is None or self._need_rescue(best)):
            base = best.score if best else -9.0
            cands = []
            for v in RESCUE_VARIANTS:
                cands.append((replace(cfg, **v), j.morae, f"パラメータ {v}"))
            n = len(j.morae)
            for k in range(1, min(3, n - 2) + 1):
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
        j.morae, j.segs, j.timings, j.nuclei, j.score = best.morae, best.segs, best.timings, best.nuclei, best.score
        j.notes += best.notes
        if best.status == "FAIL":
            j.status = "FAIL"
            return
        j.status = best.status
        j.entries = make_entries(j.path.name, j.timings, j.dur_ms, cfg)
        for e in j.entries:
            e.alias = cfg.alias_prefix + e.alias + cfg.alias_suffix

    # ------------------------------------------------------------------
    def _emit(self, root):
        cfg = self.cfg
        bydir = {}
        for j in self.jobs:
            bydir.setdefault(j.path.parent, []).append(j)
        limit = 0 if cfg.strict else 1
        for d, js in bydir.items():
            keep, review = [], []
            for j in js:
                if j.status == "FAIL":
                    continue
                for e in j.entries:
                    (keep if STATUS_RANK[e.status] <= limit else review).append(e)
            if cfg.dedupe:
                best = {}
                for e in keep:
                    cur = best.get(e.alias)
                    if cur is None or (STATUS_RANK[e.status], -e.quality) < (STATUS_RANK[cur.status], -cur.quality):
                        best[e.alias] = e
                self.dropped += [e for e in keep if best[e.alias] is not e]
                keep = [e for e in keep if best[e.alias] is e]
            self.review[d] = review
            self.kept[d] = keep
            if cfg.dry_run:
                continue
            extra = []
            if cfg.merge:
                names = {j.path.name for j in js}
                extra = [l for l in read_lines(d / "oto.ini", cfg.encoding) if l.split("=", 1)[0] not in names]
            write_oto(d / "oto.ini", keep, cfg.encoding, cfg.backup, extra)
            if review:
                (d / "oto_review.txt").write_text(
                    "# 確信度不足・検証失敗のため oto.ini に入れなかった候補\n" +
                    "\n".join(f"{format_entry(e)}   # {e.status}: {'; '.join(e.notes)}" for e in review) + "\n",
                    encoding="utf-8")
            self.log(f"    {d}: {len(keep)} 件を出力 (保留 {len(review)})")
        if not cfg.dry_run:
            for j in self.jobs:
                if j.segs and j.status != "FAIL":
                    if cfg.export_labels:
                        write_labels(j.path.with_suffix(".labels.txt"), j)
                    if cfg.textgrid:
                        write_textgrid(j.path.with_suffix(".TextGrid"), j)

    def _report(self, root, sec):
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
                    "boundary_conf": round(t.bconf, 3), "status": t.status, "notes": t.notes} for t in j.timings],
                "aliases": [{"alias": e.alias, "kind": e.kind, "status": e.status, "line": format_entry(e),
                             "notes": e.notes} for e in j.entries],
            })
        n_fail = sum(1 for j in self.jobs if j.status == "FAIL")
        n_warn = sum(1 for j in self.jobs if j.status == "WARN")
        n_ent = sum(len(v) for v in self.kept.values())
        n_rev = sum(len(v) for v in self.review.values())
        self.report = {"seconds": round(sec, 1), "files": files, "total": len(self.jobs), "failed": n_fail,
                       "warned": n_warn, "entries_written": n_ent, "entries_review": n_rev,
                       "config": self.cfg.to_dict(), "nn_weight": self.model.nn_weight}
        if not self.cfg.dry_run:
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
