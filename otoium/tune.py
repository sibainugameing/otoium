"""自動調整: 正解なしで検証スコアを最大化するように解析パラメータを座標降下で探索"""
from dataclasses import replace
import numpy as np
from .aligner import AlignmentError
from .evaluate import evaluate

GRID = {
    "boundary_weight": [2.0, 4.0, 8.0],
    "duration_weight": [0.25, 0.5, 1.0],
    "pause_penalty": [1.0, 2.0, 4.0],
    "free_switch_penalty": [6.0, 10.0, 16.0],
    "mfcc_weight": None,   # 音響モデル側の重み (後述)
}


def _objective(model, subset, cfg):
    sc = []
    for j, LL in subset:
        try:
            sc.append(evaluate(model, j.F, j.X, j.morae, j.dur_ms, cfg, LL=LL).score)
        except (AlignmentError, ValueError):
            sc.append(-3.0)
    return float(np.mean(sc))


def autotune(model, jobs, cfg, log, rounds=1, max_files=16):
    if not jobs:
        return cfg
    step = max(1, len(jobs) // max_files)
    sel = jobs[::step][:max_files]
    subset = [(j, model.loglik(j.F, j.X)) for j in sel]
    best, best_s = cfg, _objective(model, subset, cfg)
    log(f"    自動調整: 初期スコア {best_s:.3f} ({len(sel)} ファイルで評価)")
    for r in range(rounds):
        changed = False
        for key, vals in GRID.items():
            if vals is None:
                continue
            for v in vals:
                if v == getattr(best, key):
                    continue
                cand = replace(best, **{key: v})
                s = _objective(model, subset, cand)
                if s > best_s + 1e-4:
                    best, best_s, changed = cand, s, True
                    log(f"    自動調整: {key}={v} → {s:.3f}")
        if not changed:
            break
    return best
