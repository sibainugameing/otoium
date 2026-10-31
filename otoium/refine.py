"""1ms 精度の境界微調整 (エネルギー段差・高域比の変化・同母音間のエネルギー谷)"""
import numpy as np
from .verify import unit_group

W = 8      # 左右の平均窓(ms)


def _mean(a, i0, i1):
    i0, i1 = max(0, i0), min(len(a), i1)
    return float(np.mean(a[i0:i1])) if i1 > i0 else 0.0


def refine(segs, F, cfg):
    real = [s for s in segs if s.e > s.s]
    hop = cfg.hop_ms
    for s in real:
        s.s_ms = max(0.0, (s.s - 0.5) * hop)
        s.e_ms = max(0.0, (s.e - 0.5) * hop)
    if not getattr(cfg, "refine", True) or F.fine_e is None or len(real) < 2:
        return
    fe, fh = F.fine_e, F.fine_h
    off = 2.0   # 微細フレーム i の中心時刻 = i + 2 ms
    for i in range(1, len(real)):
        a, b = real[i - 1], real[i]
        t0 = b.s_ms
        R = int(min(8, 0.4 * (a.e_ms - a.s_ms), 0.4 * (b.e_ms - b.s_ms)))
        if R < 1:
            continue
        same = unit_group(a.unit) == unit_group(b.unit)
        best, bs = t0, -1e9
        for d in range(-R, R + 1):
            t = t0 + d
            k = int(round(t - off))
            if same:
                sc = -_mean(fe, k - 2, k + 3) / 6.0
            else:
                de = _mean(fe, k, k + W) - _mean(fe, k - W, k)
                dh = _mean(fh, k, k + W) - _mean(fh, k - W, k)
                sc = abs(de) / 6.0 + abs(dh) / 4.0
            sc -= 0.03 * abs(d)
            if sc > bs:
                bs, best = sc, t
        b.s_ms = best
    for i in range(1, len(real)):
        real[i].s_ms = max(real[i].s_ms, real[i - 1].s_ms + 1.0)
    for i in range(len(real) - 1):
        real[i].e_ms = real[i + 1].s_ms
