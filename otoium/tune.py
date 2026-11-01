"""自動調整: 検証スコアを最大化するように「整列側」のパラメータを座標降下で探索。
検証側 (free_switch_penalty など) は自己採点になるため調整対象にしない。"""
from dataclasses import replace
import numpy as np
from .aligner import AlignmentError
from .evaluate import evaluate

GRID = {
    "boundary_weight": [2.0, 4.0, 8.0],
    "duration_weight": [0.25, 0.5, 1.0],
    "pause_penalty": [1.0, 2.0, 4.0],
}
NN_GRID = [0.0, 1.0, 2.0, 4.0]
FINE_GRID = {
    "boundary_weight": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0],
    "duration_weight": [0.15, 0.25, 0.4, 0.5, 0.7, 1.0, 1.5],
    "pause_penalty": [0.5, 1.0, 2.0, 3.0, 4.0, 6.0],
}
FINE_NN = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _objective(model, subset, cfg):
    sc = []
    for j, LL in subset:
        try:
            sc.append(evaluate(model, j.F, j.X, j.morae, j.dur_ms, cfg, LL=LL).score)
        except (AlignmentError, ValueError):
            sc.append(-3.0)
    return float(np.mean(sc))


def autotune(model, jobs, cfg, log, rounds=1, max_files=16, on_score=None):
    """調整用ファイルで探索し、別のホールドアウトファイルで悪化していれば調整結果を採用しない (過適合の防止)"""
    if not jobs:
        return cfg
    step = max(1, len(jobs) // max_files)
    sel = jobs[::step][:max_files]
    if len(sel) >= 6:
        tune_set, hold_set = sel[0::2], sel[1::2]
    else:
        tune_set, hold_set = sel, []
    orig_cfg, orig_nn = cfg, model.nn_weight
    hold_base = None
    if hold_set:
        hold_base = _objective(model, [(j, model.loglik(j.F, j.X)) for j in hold_set], cfg)
    subset = [(j, model.loglik(j.F, j.X)) for j in tune_set]
    best, best_s = cfg, _objective(model, subset, cfg)
    log(f"    自動調整: 初期スコア {best_s:.3f} (調整用 {len(tune_set)} / 評価用 {len(hold_set)} ファイル)")
    if on_score:
        on_score(best_s)
    grid, nn_grid = (FINE_GRID, FINE_NN) if rounds >= 3 else (GRID, NN_GRID)
    for r in range(rounds):
        changed = False
        if model.net is not None:
            for v in nn_grid:
                if v == model.nn_weight:
                    continue
                old = model.nn_weight
                model.nn_weight = v
                sub2 = [(j, model.loglik(j.F, j.X)) for j in tune_set]
                s = _objective(model, sub2, best)
                if s > best_s + 1e-4:
                    best_s, subset, changed = s, sub2, True
                    log(f"    自動調整: nn_weight={v} → {s:.3f}")
                    if on_score:
                        on_score(s)
                else:
                    model.nn_weight = old
        for key, vals in grid.items():
            for v in vals:
                if v == getattr(best, key):
                    continue
                cand = replace(best, **{key: v})
                s = _objective(model, subset, cand)
                if s > best_s + 1e-4:
                    best, best_s, changed = cand, s, True
                    log(f"    自動調整: {key}={v} → {s:.3f}")
                    if on_score:
                        on_score(s)
        if not changed:
            break
    if hold_set:
        hold_best = _objective(model, [(j, model.loglik(j.F, j.X)) for j in hold_set], best)
        log(f"    評価用ファイルでの確認: 調整前 {hold_base:.3f} → 調整後 {hold_best:.3f}")
        if hold_best < hold_base - 0.01:
            model.nn_weight = orig_nn
            log("    評価用で悪化したため、調整結果を採用せず元の設定を維持")
            return orig_cfg
    return best
