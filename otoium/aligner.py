"""強制アライメント: 時間長制約付きセミマルコフ Viterbi + 境界証拠ボーナス"""
import math
from dataclasses import dataclass
import numpy as np
from .phoneset import UNIT_CLASS, CLASS_PRIOR, DUR_SIGMA

NEG = -1e18


class AlignmentError(RuntimeError):
    pass


@dataclass
class Token:
    unit: str
    mora: int          # -1: ポーズ
    kind: str          # 'c' 子音ユニット / 'v' 母音(核) / 'pau'
    dmin: int
    dmax: int
    dur_lp: np.ndarray


@dataclass
class Seg:
    unit: str
    mora: int
    kind: str
    s: int
    e: int
    bconf: float = 1.0


def build_tokens(morae, T, hop_ms, cfg):
    toks = []

    def mk(unit, mora, kind, edge=False):
        c = UNIT_CLASS[unit]
        _, _, dmin_ms, dmax_ms, dmean_ms = CLASS_PRIOR[c]
        if unit == "pau":
            dmin = 0
            dmax = T if edge else min(T, int(math.ceil(1500 / hop_ms)))
            lp = np.zeros(dmax + 1)
            if not edge:
                lp[1:] = -cfg.pause_penalty
        else:
            dmin = max(1, int(math.ceil(dmin_ms / hop_ms)))
            dmax = min(T, int(math.ceil(dmax_ms / hop_ms)))
            d = np.arange(dmax + 1, dtype=float)
            d[0] = 1
            lp = -0.5 * cfg.duration_weight * (np.log(d * hop_ms / dmean_ms) / DUR_SIGMA.get(c, 0.7)) ** 2
        return Token(unit, mora, kind, dmin, dmax, lp)

    toks.append(mk("pau", -1, "pau", edge=True))
    for i, m in enumerate(morae):
        us = m.units
        for j, u in enumerate(us):
            toks.append(mk(u, i, "v" if j == len(us) - 1 else "c"))
        if i < len(morae) - 1:
            toks.append(mk("pau", -1, "pau"))
    toks.append(mk("pau", -1, "pau", edge=True))
    if sum(t.dmin for t in toks) > T:
        raise AlignmentError("録音が短すぎて指定された音素列を割り当てられません")
    return toks


def viterbi(LL, idx, toks, bnd, boundary_weight):
    U, T = LL.shape
    N = len(toks)
    cum = {}
    bonus = np.zeros(T + 1)
    bonus[1:T] = boundary_weight * bnd[1:T]
    dp = np.full((N + 1, T + 1), NEG)
    back = np.zeros((N + 1, T + 1), dtype=np.int32)
    dp[0, 0] = 0.0
    for i, tok in enumerate(toks, start=1):
        if tok.unit not in cum:
            cum[tok.unit] = np.concatenate(([0.0], np.cumsum(LL[idx[tok.unit]])))
        c = cum[tok.unit]
        prev = dp[i - 1]
        prevb = prev + bonus
        cur, bk = dp[i], back[i]
        for d in range(tok.dmin, tok.dmax + 1):
            if d == 0:
                cand = prev
            else:
                cand = np.full(T + 1, NEG)
                cand[d:] = prevb[:T + 1 - d] + (c[d:] - c[:T + 1 - d]) + tok.dur_lp[d]
            better = cand > cur
            if better.any():
                cur[better] = cand[better]
                bk[better] = d
    if dp[N, T] < NEG / 2:
        raise AlignmentError("アライメント経路が存在しません")
    segs = [None] * N
    e = T
    for i in range(N, 0, -1):
        d = int(back[i, e])
        t = toks[i - 1]
        segs[i - 1] = Seg(t.unit, t.mora, t.kind, e - d, e)
        e -= d
    return segs, float(dp[N, T])


def align(model, F, morae, cfg, LL=None):
    if LL is None:
        LL = model.loglik(F)
    toks = build_tokens(morae, F.T, cfg.hop_ms, cfg)
    segs, score = viterbi(LL, model.idx, toks, F.bnd, cfg.boundary_weight)
    return segs, score, LL
