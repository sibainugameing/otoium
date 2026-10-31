"""パイプライン本体: 読込 → 特徴 → 自己教師あり学習 → アライメント → 検証 → oto 生成"""
import json
import time
from pathlib import Path
import numpy as np
from .audio_io import read_wav
from .features import extract, SilentAudioError
from .model import AcousticModel
from .aligner import align, AlignmentError
from .kana import parse_text, TextParseError
from .verify import free_decode, build_timings, assess, STATUS_RANK
from .aliases import make_entries
from .oto import write_oto, format_entry


class FileJob:
    def __init__(self, path):
        self.path = Path(path)
        self.text = ""
        self.morae = []
        self.F = None
        self.X = None
        self.dur_ms = 0.0
        self.env = None
        self.status = "OK"          # OK / WARN / FAIL
        self.notes = []
        self.timings = []
        self.segs = []
        self.entries = []
        self.nuclei = 0

    def fail(self, msg):
        self.status = "FAIL"
        self.notes.append(msg)


def load_script(path):
    """『wav名=テキスト』(または タブ/空白区切り) の台本を読む"""
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
    if len(x) == 0:
        return np.zeros((2, 1))
    m = max(1, len(x) // n)
    k = len(x) // m
    a = x[:k * m].reshape(k, m)
    return np.stack([a.min(1), a.max(1)])


class Pipeline:
    def __init__(self, cfg, log=print, progress=None):
        self.cfg = cfg
        self.log = log
        self.progress = progress or (lambda frac, msg="": None)
        self.jobs = []
        self.model = None
        self.report = {}

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
        self.log(f"[1/5] {len(wavs)} 個の WAV を読み込み・特徴抽出中…")
        self.jobs = []
        for i, p in enumerate(wavs):
            job = FileJob(p)
            self.jobs.append(job)
            try:
                job.text = script.get(p.name) or script.get(p.stem) or p.stem
                job.morae = parse_text(job.text)
                x, sr = read_wav(p)
                job.dur_ms = len(x) / sr * 1000.0
                job.F = extract(x, sr, cfg.hop_ms, cfg.win_ms)
                job.env = _envelope(x)
            except (TextParseError, SilentAudioError, KeyError, ValueError, OSError, EOFError) as ex:
                job.fail(f"{type(ex).__name__}: {ex}")
            self.progress(0.2 * (i + 1) / len(wavs), p.name)
        ok = [j for j in self.jobs if j.status != "FAIL"]
        if not ok:
            raise RuntimeError("解析可能なファイルがありません")
        self.log("[2/5] 音響モデルを準備中…")
        if cfg.model_path:
            self.model = AcousticModel.load(cfg.model_path)
        else:
            self.model = AcousticModel(cfg.mfcc_weight)
            self.model.set_norm([j.F for j in ok])
        for j in ok:
            j.X = self.model.design(j.F)
        if not cfg.model_path:
            self._train(ok)
        self.log("[3/5] 最終アライメントと独立検証…")
        for i, j in enumerate(ok):
            self._final(j)
            self.progress(0.6 + 0.3 * (i + 1) / len(ok), j.path.name)
        self.log("[4/5] oto.ini を生成中…")
        self._emit(root)
        if cfg.save_model and not cfg.dry_run and not cfg.model_path:
            self.model.save(root / "otoium_model.npz")
        self.log("[5/5] レポート作成中…")
        self._report(root, time.time() - t0)
        self.progress(1.0, "完了")
        return self.report

    # ------------------------------------------------------------------
    def _train(self, ok):
        cfg = self.cfg
        for it in range(cfg.train_iters):
            items, scores = [], []
            for j in ok:
                if j.status == "FAIL":
                    continue
                try:
                    segs, sc, _ = align(self.model, j.F, j.morae, cfg, LL=self.model.loglik(j.F, j.X))
                except AlignmentError as ex:
                    j.fail(f"AlignmentError: {ex}")
                    continue
                real = [(s.unit, s.s, s.e) for s in segs if s.e > s.s]
                items.append((j.X, real))
                scores.append(sc / j.F.T)
            if not items:
                raise RuntimeError("学習に使えるファイルがありません")
            if len(items) >= 8:
                a = np.array(scores)
                med = np.median(a)
                mad = np.median(np.abs(a - med)) * 1.4826 + 1e-9
                items = [it_ for it_, s in zip(items, scores) if s >= med - 3 * mad]
            self.model.fit(items)
            self.log(f"    学習 {it + 1}/{cfg.train_iters}: {len(items)} ファイル使用")
            self.progress(0.2 + 0.4 * (it + 1) / cfg.train_iters, "学習")

    def _final(self, j):
        cfg = self.cfg
        try:
            LL = self.model.loglik(j.F, j.X)
            segs, _, _ = align(self.model, j.F, j.morae, cfg, LL=LL)
            j.segs = segs
            j.timings = build_timings(segs, j.morae, cfg.hop_ms, j.dur_ms)
            labels = free_decode(self.model, LL, cfg.free_switch_penalty)
            fstat, notes, j.nuclei = assess(j.timings, segs, LL, j.F, self.model, labels, cfg)
            j.notes += notes
            if fstat == "FAIL":
                j.status = "FAIL"
            else:
                j.status = fstat if STATUS_RANK[fstat] > STATUS_RANK[j.status] else j.status
                j.entries = make_entries(j.path.name, j.timings, j.dur_ms, cfg)
        except (AlignmentError, ValueError, KeyError) as ex:
            j.fail(f"{type(ex).__name__}: {ex}")

    def _emit(self, root):
        cfg = self.cfg
        bydir = {}
        for j in self.jobs:
            bydir.setdefault(j.path.parent, []).append(j)
        self.review = {}
        self.dropped = []
        for d, js in bydir.items():
            cand = [(e, j) for j in js if j.status != "FAIL" for e in j.entries]
            limit = 0 if cfg.strict else 1
            keep, review = [], []
            for e, j in cand:
                (keep if STATUS_RANK[e.status] <= limit else review).append(e)
            if cfg.dedupe:
                best = {}
                for e in keep:
                    cur = best.get(e.alias)
                    if cur is None or (STATUS_RANK[e.status], -e.quality) < (STATUS_RANK[cur.status], -cur.quality):
                        best[e.alias] = e
                uniq = [e for e in keep if best[e.alias] is e]
                self.dropped += [e for e in keep if best[e.alias] is not e]
                keep = uniq
            self.review[d] = review
            if cfg.dry_run:
                continue
            write_oto(d / "oto.ini", keep, cfg.encoding, cfg.backup)
            if review:
                (d / "oto_review.txt").write_text(
                    "# otoium: 確信度不足・検証失敗のため oto.ini に入れなかった候補\n" +
                    "\n".join(f"{format_entry(e)}   # {e.status}: {'; '.join(e.notes)}" for e in review) + "\n",
                    encoding="utf-8")
            self.log(f"    {d}: {len(keep)} 件を oto.ini に出力 (保留 {len(review)} / 重複除外 "
                     f"{sum(1 for e in self.dropped if e.wav in {x.path.name for x in js})})")
            self._kept = getattr(self, "_kept", {})
            self._kept[d] = keep
        if cfg.export_labels and not cfg.dry_run:
            for j in self.jobs:
                if j.segs:
                    lines = [f"{(s.s - .5) * j.F.hop_s:.4f}\t{(s.e - .5) * j.F.hop_s:.4f}\t{s.unit}"
                             for s in j.segs if s.e > s.s and s.unit != "pau"]
                    j.path.with_suffix(".labels.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _report(self, root, sec):
        files = []
        for j in self.jobs:
            files.append({
                "file": str(j.path.relative_to(root)), "text": j.text, "status": j.status, "notes": j.notes,
                "detected_nuclei": j.nuclei, "expected_morae": len(j.morae),
                "morae": [{
                    "surface": t.mora.surface, "cons": t.mora.cons, "vowel": t.mora.vowel,
                    "cons_start_ms": round(t.cons_start, 1), "vowel_start_ms": round(t.vowel_start, 1),
                    "vowel_end_ms": round(t.vowel_end, 1), "units": [(u, round(a, 1), round(b, 1)) for u, a, b in t.segs],
                    "recognized_vowel": t.rec_vowel, "recognized_cons": t.rec_cons,
                    "vowel_agreement": round(t.agree_v, 3), "boundary_conf": round(t.bconf, 3),
                    "status": t.status, "notes": t.notes} for t in j.timings],
                "aliases": [{"alias": e.alias, "kind": e.kind, "status": e.status, "line": format_entry(e),
                             "notes": e.notes} for e in j.entries],
            })
        n_fail = sum(1 for j in self.jobs if j.status == "FAIL")
        n_warn = sum(1 for j in self.jobs if j.status == "WARN")
        n_ent = sum(len(v) for v in getattr(self, "_kept", {}).values())
        n_rev = sum(len(v) for v in self.review.values())
        self.report = {"seconds": round(sec, 1), "files": files, "total": len(self.jobs), "failed": n_fail,
                       "warned": n_warn, "entries_written": n_ent, "entries_review": n_rev,
                       "config": self.cfg.to_dict()}
        if not self.cfg.dry_run:
            (root / "otoium_report.json").write_text(json.dumps(self.report, ensure_ascii=False, indent=1), encoding="utf-8")
            lines = [f"otoium レポート  (処理 {sec:.1f}s)", f"ファイル {len(self.jobs)} / 失敗 {n_fail} / 警告 {n_warn}",
                     f"oto.ini 出力 {n_ent} 件 / 保留 {n_rev} 件", ""]
            for f in files:
                lines.append(f"[{f['status']}] {f['file']}  ({f['text']})  核 {f['detected_nuclei']}/{f['expected_morae']}")
                for n in f["notes"]:
                    lines.append(f"    ! {n}")
                for m in f["morae"]:
                    if m["status"] != "OK":
                        lines.append(f"    - {m['surface']}: {m['status']} {'; '.join(m['notes'])}")
            (root / "otoium_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.log(f"完了: ファイル {len(self.jobs)} 件 (失敗 {n_fail}, 警告 {n_warn}) / oto {n_ent} 件 / 保留 {n_rev} 件")
