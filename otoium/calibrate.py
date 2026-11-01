"""しきい値の自動校正: 録音データ全体の分布から、境界確信度・母音一致率・子音一致率の基準を決める"""
from dataclasses import replace
import numpy as np
from .aligner import AlignmentError
from .evaluate import evaluate


def _robust_low(a, lo, hi):
    a = np.asarray(a, dtype=float)
    if len(a) < 5:
        return None
    med = np.median(a)
    mad = 1.4826 * np.median(np.abs(a - med)) + 0.03
    return float(np.clip(med - 3 * mad, lo, hi))


def calibrate(model, jobs, cfg, log, max_files=80):
    step = max(1, len(jobs) // max_files)
    conf, agree, cagree = [], [], []
    for j in jobs[::step][:max_files]:
        try:
            r = evaluate(model, j.F, j.X, j.morae, j.dur_ms, cfg)
        except (AlignmentError, ValueError):
            continue
        for t in r.timings:
            conf.append(t.bconf)
            agree.append(t.agree_v)
            if t.mora.has_cons:
                cagree.append(t.cons_agree)
    if len(conf) < 5:
        return cfg
    f = float(np.clip(np.median(conf) / 2.5, 0.3, 1.0))
    ok = _robust_low(agree, 0.25, 0.55)
    cr = _robust_low(cagree, 0.10, 0.34)
    cw = _robust_low(cagree, 0.25, 0.60)
    new = replace(cfg, thr_contrast=cfg.thr_contrast * f, thr_bnd=cfg.thr_bnd * f,
                  ok_agree=ok if ok is not None else cfg.ok_agree,
                  fail_agree=float(np.clip(0.5 * (ok if ok is not None else cfg.ok_agree), 0.1, 0.25)),
                  cons_review=cr if cr is not None else cfg.cons_review,
                  cons_warn=cw if cw is not None else cfg.cons_warn)
    log(f"    しきい値を自動校正: 境界×{f:.2f} (thr_contrast={new.thr_contrast:.2f}, thr_bnd={new.thr_bnd:.2f}), "
        f"ok_agree={new.ok_agree:.2f}, fail_agree={new.fail_agree:.2f}, cons_review={new.cons_review:.2f}, cons_warn={new.cons_warn:.2f}")
    return new
