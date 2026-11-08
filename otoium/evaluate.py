"""1ファイルの アライメント→微調整→独立検証 をまとめて評価し、スコア化する"""
from dataclasses import dataclass
import numpy as np
from .aligner import align
from .refine import refine
from .verify import free_decode, build_timings, assess, STATUS_RANK, tempo_deviation


@dataclass
class Result:
    segs: list
    timings: list
    status: str
    notes: list
    nuclei: int
    score: float
    morae: list
    labels: object = None


def file_score(timings, status):
    n = len(timings)
    agree = np.mean([1.0 if t.devoiced else t.agree_v for t in timings])
    bc = np.mean([min(t.bconf, 1.5) for t in timings])
    ca = np.mean([t.cons_agree for t in timings])
    bad = sum(1 for t in timings if STATUS_RANK[t.status] >= 2) / n
    return float(agree + 0.25 * bc + 0.25 * ca - 1.0 * bad - 0.3 * tempo_deviation(timings) - (1.0 if status == "FAIL" else 0.0))


def evaluate_segs(vmodel, F, morae, segs, dur_ms, cfg, vLL):
    """整列済みの区間列を 微調整 → 独立検証 → 採点 する (部分やり直しで使う)"""
    refine(segs, F, cfg)
    timings = build_timings(segs, morae, cfg.hop_ms, dur_ms)
    labels = free_decode(vmodel, vLL, cfg.free_switch_penalty, cfg.dv_penalty)
    status, notes, nuc = assess(timings, segs, vLL, F, vmodel, labels, cfg)
    return Result(segs, timings, status, notes, nuc, file_score(timings, status), morae, labels)


def evaluate(model, F, X, morae, dur_ms, cfg, LL=None, vmodel=None, vLL=None, rate=1.0):
    """整列には model/LL、独立検証には vmodel/vLL (既定は同じ)。適応モデルで整列しても検証は適応前で行う。"""
    if LL is None:
        LL = model.loglik(F, X)
    segs, _, _ = align(model, F, morae, cfg, LL=LL, rate=rate)
    vmodel = vmodel or model
    vLL = LL if vLL is None else vLL
    return evaluate_segs(vmodel, F, morae, segs, dur_ms, cfg, vLL)
