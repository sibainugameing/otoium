"""実行結果の詳細 JSON (summary / standard / full)"""
import datetime
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
import numpy as np
from . import __version__
from .oto import format_entry
from .verify import STATUS_RANK

DETAILS = ("summary", "standard", "full")
SCHEMA = "otoium-result/1"


def _clean(o):
    """numpy 型・Path・NaN/Inf を JSON 化できる値へ (NaN/Inf は null)"""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, Path):
        return str(o)
    return o


def _sha256(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _entry(p, j, e):
    return {"alias": e.alias, "kind": e.kind, "moras": [m + 1 for m in e.moras],
            "offset": e.offset, "consonant": e.consonant, "cutoff": e.cutoff, "preutterance": e.pre,
            "overlap": e.overlap, "used_length_ms": e.cut_len, "status": e.status,
            "confidence": round(e.quality, 4), "valid": e.valid, "forced": e.forced, "manual": e.manual,
            "decision": p.decision(j, e), "line": format_entry(e), "notes": e.notes}


def _mora(t):
    return {"index": t.index + 1, "surface": t.mora.surface, "kana": t.mora.kana, "consonant": t.mora.cons,
            "vowel": t.mora.vowel, "status": t.status, "notes": t.notes,
            "cons_start_ms": t.cons_start, "vowel_start_ms": t.vowel_start, "vowel_end_ms": t.vowel_end,
            "cons_end_for_vc_ms": t.cons_end_vc, "gap_before_ms": t.gap_before, "gap_absorbed_into_consonant_ms": t.gap_absorbed,
            "energy_dip_db": t.energy_dip_db, "silent_onset_consonant": t.mora.silent_onset,
            "units": [{"unit": u, "start_ms": a, "end_ms": b} for u, a, b in t.segs],
            "recognized_vowel": t.rec_vowel, "recognized_consonant": t.rec_cons,
            "vowel_agreement": t.agree_v, "consonant_agreement": t.cons_agree,
            "boundary_confidence": t.bconf, "devoiced_vowel_suspected": t.devoiced, "quality": t.quality}


def _file(p, j, detail):
    d = {"file": str(j.path.relative_to(p.root)), "text": j.text, "status": j.status, "score": j.score,
         "notes": j.notes, "rechecks": j.rechecks}
    if detail == "summary":
        return d
    d.update({"expected_morae": len(j.morae), "detected_vowel_nuclei": j.nuclei,
              "morae": [_mora(t) for t in j.timings], "entries": [_entry(p, j, e) for e in j.entries]})
    if detail != "full":
        return d
    d["wav_sha256"] = _sha256(j.path)
    d["wav_duration_ms"] = j.dur_ms
    if j.F is not None:
        F = j.F
        voiced = F.cls[:, 1] > 0.5
        d["features"] = {"frames": int(F.T), "hop_ms": F.hop_s * 1000, "reference_level_db": F.ref_db,
                         "f0_median_hz": float(np.median(F.f0[voiced])) if voiced.any() else None,
                         "voiced_frame_ratio": float(voiced.mean())}
    d["segments"] = [{"unit": g.unit, "mora": (g.mora + 1) if g.mora >= 0 else None, "kind": g.kind,
                      "start_frame": g.s, "end_frame": g.e, "start_ms": g.s_ms, "end_ms": g.e_ms,
                      "boundary_confidence": g.bconf} for g in j.segs if g.e > g.s]
    d["independent_recognition_runs"] = j.decoded_runs
    return d


def build_result(p, detail="full"):
    if detail not in DETAILS:
        raise ValueError(f"detail は {DETAILS} のいずれか: {detail}")
    jobs = p.jobs
    count = lambda st: sum(1 for j in jobs if j.status == st)
    ents = [(j, e) for j in jobs for e in j.entries]
    by_status, by_kind = {}, {}
    for _, e in ents:
        by_status[e.status] = by_status.get(e.status, 0) + 1
        by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
    mora_status = {}
    for j in jobs:
        for t in j.timings:
            mora_status[t.status] = mora_status.get(t.status, 0) + 1
    out = {
        "schema": SCHEMA, "detail": detail, "otoium_version": __version__,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "root": str(p.root),
        "summary": {
            "files": len(jobs), "ok": count("OK"), "warn": count("WARN"), "fail": count("FAIL"),
            "rechecked_files": sum(1 for j in jobs if j.rechecks),
            "entries_total": len(ents), "entries_by_status": by_status, "entries_by_kind": by_kind,
            "entries_written": sum(len(v) for v in p.kept.values()),
            "entries_forced": sum(len(v) for v in p.forced.values()),
            "entries_held_back": sum(len(v) for v in p.review.values()),
            "morae_by_status": mora_status, "seconds": (p.report or {}).get("seconds"),
        },
        "files": [_file(p, j, detail) for j in jobs],
    }
    if detail == "full":
        out["environment"] = {"python": sys.version.split()[0], "platform": platform.platform(),
                              "numpy": np.__version__, "scipy": __import__("scipy").__version__}
        out["config_initial"] = p.cfg_initial.to_dict()
        out["config_final"] = p.cfg.to_dict()
        out["resolved_plan"] = p.P
        out["stage_seconds"] = p.stage_seconds
        m = p.model
        if m is not None:
            out["model"] = {"trained": m.trained, "nn_weight": m.nn_weight, "nn_ensemble": len(m.nets),
                            "durations": {u: {"mean_ms": float(np.exp(m.dur_mu[u])), "sigma": m.dur_sd[u],
                                              "max_ms": m.dur_max[u]} for u in m.units}}
        out["training_history"] = {k: [{"step": s, "stage": g, "value": v} for s, v, g in rows]
                                   for k, rows in p.track.history.items()}
        out["best_checkpoint_score"] = p._best_score if p._best_model is not None else None
        out["duplicate_aliases_dropped"] = [{"wav": e.wav, "alias": e.alias, "status": e.status,
                                             "confidence": round(e.quality, 4)} for e in p.dropped]
        out["held_back"] = {str(d): [format_entry(e) + "   # " + e.status + ": " + "; ".join(e.notes) for e in v]
                            for d, v in p.review.items()}
        out["log"] = list(p.logbuf)
    return _clean(out)


def write_result(p, path, detail="full"):
    data = build_result(p, detail)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)
    return path
