"""1ファイルの アライメント→微調整→独立検証 をまとめて評価し、スコア化する"""
from dataclasses import dataclass
import numpy as np
from .aligner import align
from .refine import refine
from .verify import free_decode, build_timings, assess, STATUS_RANK


@dataclass
class Result:
    segs: list
    timings: list
    status: str
    notes: list
    nuclei: int
    score: float
    morae: list


def file_score(timings, status):
    n = len(timings)
    agree = np.mean([t.agree_v for t in timings])
    bc = np.mean([min(t.bconf, 1.5) for t in timings])
    ca = np.mean([t.cons_agree for t in timings])
    bad = sum(1 for t in timings if STATUS_RANK[t.status] >= 2) / n
    return float(agree + 0.25 * bc + 0.25 * ca - 1.0 * bad - (1.0 if status == "FAIL" else 0.0))


def evaluate(model, F, X, morae, dur_ms, cfg, LL=None):
    if LL is None:
        LL = model.loglik(F, X)
    segs, _, _ = align(model, F, morae, cfg, LL=LL)
    refine(segs, F, cfg)
    timings = build_timings(segs, morae, cfg.hop_ms, dur_ms)
    labels = free_decode(model, LL, cfg.free_switch_penalty)
    status, notes, nuc = assess(timings, segs, LL, F, model, labels, cfg)
    return Result(segs, timings, status, notes, nuc, file_score(timings, status), morae)
