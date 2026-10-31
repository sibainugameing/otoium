"""独立検証: 台本を使わないフリーデコーダによる音素認識と、境界の確からしさ評価"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from .phoneset import UNIT_CLASS, is_burst
from .kana import Mora

GROUPS = ["a", "i", "u", "e", "o", "N", "FR", "CL", "BR", "SM", "SIL"]
GIDX = {g: i for i, g in enumerate(GROUPS)}
STATUS_RANK = {"OK": 0, "WARN": 1, "REVIEW": 2, "ERROR": 3}


def unit_group(u):
    if u in "aiueo" and len(u) == 1:
        return u
    if u == "NN" or u.startswith("nas"):
        return "N"
    if u.startswith("fr"):
        return "FR"
    if u in ("cl", "clv"):
        return "CL"
    if u.startswith("br"):
        return "BR"
    if u in ("rf", "gl_y", "gl_w"):
        return "SM"
    return "SIL"


def accepted_groups(u):
    g = unit_group(u)
    extra = {
        "N": {"N", "u"}, "FR": {"FR"}, "CL": {"CL", "SIL"}, "BR": {"BR", "FR", "CL"}, "SM": {"SM", "N", "CL", "u", "i"},
    }
    acc = set(extra.get(g, {g}))
    if u in ("frH", "frHY", "frF"):
        acc |= {"SIL"}
    if u in ("frZ", "frJ", "clv"):
        acc |= {"N", "CL"}
    return acc


def free_decode(model, LL, switch):
    """台本なしのフレーム単位 Viterbi。ユニットをグループ化して最尤系列を求める"""
    G, T = len(GROUPS), LL.shape[1]
    em = np.full((G, T), -1e9)
    for u, k in model.idx.items():
        gi = GIDX[unit_group(u)]
        em[gi] = np.maximum(em[gi], LL[k])
    dp = np.zeros((T, G))
    bp = np.zeros((T, G), dtype=np.int32)
    dp[0] = em[:, 0]
    ar = np.arange(G)
    for t in range(1, T):
        prev = dp[t - 1]
        b = int(prev.argmax())
        sw = prev[b] - switch
        use = sw > prev
        dp[t] = np.where(use, sw, prev) + em[:, t]
        bp[t] = np.where(use, b, ar)
    lab = np.zeros(T, dtype=np.int32)
    lab[-1] = int(dp[-1].argmax())
    for t in range(T - 1, 0, -1):
        lab[t - 1] = bp[t, lab[t]]
    return lab


@dataclass
class MoraTiming:
    index: int
    mora: Mora
    segs: list                       # [(unit, s_ms, e_ms)]
    cons_start: float
    vowel_start: float
    vowel_end: float
    cons_end_vc: Optional[float]
    gap_before: float
    v_frames: tuple = (0, 0)
    c_frames: tuple = (0, 0)
    agree_v: float = 1.0
    cons_agree: float = 1.0
    rec_vowel: str = ""
    rec_cons: str = ""
    bconf: float = 1.0
    status: str = "OK"
    notes: list = field(default_factory=list)

    @property
    def quality(self):
        return float(min(1.0, self.bconf) * (0.5 + 0.5 * self.agree_v))


def boundary_conf(LL, bnd, idx, prev, seg, cfg):
    s = seg.s
    w = max(1, min(6, prev.e - prev.s, seg.e - seg.s))
    up, un = idx[prev.unit], idx[seg.unit]
    contrast = float(np.mean(LL[up, s - w:s] - LL[un, s - w:s]) + np.mean(LL[un, s:s + w] - LL[up, s:s + w]))
    peak = float(np.max(bnd[max(0, s - 3):s + 3])) if len(bnd) else 0.0
    return max(contrast / cfg.thr_contrast, peak / cfg.thr_bnd)


def build_timings(segs, morae, hop_ms, dur_ms):
    """Seg 列 → MoraTiming 列 (ms)"""
    def ms(f):
        return float(min(max(0.0, (f - 0.5) * hop_ms), dur_ms))

    real = [s for s in segs if s.e > s.s]
    out = []
    last_end = 0.0
    for mi, m in enumerate(morae):
        ss = [s for s in real if s.mora == mi]
        if not ss:
            raise ValueError(f"モーラ {mi} が空区間になりました")
        st = lambda g: ms(g.s) if g.s_ms is None else float(min(max(0.0, g.s_ms), dur_ms))
        en = lambda g: ms(g.e) if g.e_ms is None else float(min(max(0.0, g.e_ms), dur_ms))
        cs, vs, ve = st(ss[0]), st(ss[-1]), en(ss[-1])
        cons_end_vc = None
        cons = [s for s in ss if s.kind == "c"]
        if cons:
            bursts = [s for s in cons if is_burst(s.unit)]
            cons_end_vc = st(bursts[0]) if bursts else cs + 0.8 * (vs - cs)
            cons_end_vc = max(cons_end_vc, cs + 15.0)
        t = MoraTiming(mi, m, [(s.unit, st(s), en(s)) for s in ss], cs, vs, ve, cons_end_vc,
                       max(0.0, cs - last_end),
                       v_frames=(ss[-1].s, ss[-1].e), c_frames=(ss[0].s, ss[-1].s))
        out.append(t)
        last_end = ve
    return out


def _mode(labels):
    return int(np.bincount(labels, minlength=len(GROUPS)).argmax()) if len(labels) else 0


def assess(timings, segs, LL, F, model, labels, cfg):
    """各モーラの母音一致率・子音一致率・境界確信度を計算し、状態を決める。ファイル単位の所見を返す"""
    real = [s for s in segs if s.e > s.s]
    for i, s in enumerate(real):
        s.bconf = boundary_conf(LL, F.bnd, model.idx, real[i - 1], s, cfg) if i > 0 else 1.0
    notes = []
    for t, m in zip(timings, [None] * len(timings)):
        ss = [s for s in real if s.mora == t.index]
        vseg = ss[-1]
        L = vseg.e - vseg.s
        a, b = (vseg.s + int(0.2 * L), vseg.e - int(0.2 * L)) if L >= 4 else (vseg.s, vseg.e)
        lv = labels[a:b] if b > a else labels[vseg.s:vseg.e]
        eg = GIDX[unit_group(vseg.unit)]
        t.agree_v = float(np.mean(lv == eg)) if len(lv) else 0.0
        t.rec_vowel = GROUPS[_mode(lv)]
        cons = [s for s in ss if s.kind == "c"]
        if cons:
            acc = set()
            for s in cons:
                acc |= accepted_groups(s.unit)
            lc = labels[cons[0].s:cons[-1].e]
            t.cons_agree = float(np.mean([GROUPS[x] in acc for x in lc])) if len(lc) else 0.0
            t.rec_cons = GROUPS[_mode(lc)]
        else:
            t.cons_agree, t.rec_cons = 1.0, ""
        confs = [ss[0].bconf, vseg.bconf]
        t.bconf = float(min(confs))
        t.notes = []
        if t.agree_v < cfg.fail_agree:
            t.status = "ERROR"
            t.notes.append(f"母音の独立認識が不一致 (期待 {vseg.unit} / 認識 {t.rec_vowel}, 一致率 {t.agree_v:.2f})")
        elif t.agree_v < cfg.ok_agree:
            t.status = "REVIEW"
            t.notes.append(f"母音の一致率が低い ({t.agree_v:.2f}, 認識 {t.rec_vowel})")
        elif t.cons_agree < 0.34:
            t.status = "REVIEW"
            t.notes.append(f"子音が音響的に確認できない (期待 {t.mora.cons} / 認識 {t.rec_cons or '-'}, 一致率 {t.cons_agree:.2f})")
        elif t.bconf < 0.5:
            t.status = "REVIEW"
            t.notes.append(f"音素境界が不明瞭 (確信度 {t.bconf:.2f})")
        elif t.bconf < 1.0 or t.cons_agree < 0.6:
            t.status = "WARN"
            t.notes.append(f"境界確信度 {t.bconf:.2f} / 子音一致率 {t.cons_agree:.2f}")
        else:
            t.status = "OK"
    # ファイル単位: 独立に検出した母音/鼻音核の数 と 期待モーラ数
    nuclei = 0
    run, cur = 0, -1
    vn = {GIDX[g] for g in "aiueo"}
    for x in list(labels) + [-1]:
        if x == cur:
            run += 1
        else:
            if cur in vn and run >= 4:
                nuclei += 1
            cur, run = x, 1
    exp = max(1, sum(1 for t in timings if t.mora.vowel != "N"))
    ratio = nuclei / exp
    file_status = "OK"
    mean_agree = float(np.mean([t.agree_v for t in timings]))
    n_bad = sum(1 for t in timings if STATUS_RANK[t.status] >= 2)
    if n_bad / len(timings) > 0.5:
        file_status = "FAIL"
        notes.append(f"分割失敗: {len(timings)} モーラ中 {n_bad} モーラが音響検証に不合格 (台本と音声が一致しない可能性)")
    elif ratio < 0.5 or mean_agree < 0.35:
        file_status = "FAIL"
        notes.append(f"分割失敗: 独立検出した母音核 {nuclei} / 期待 {exp}, 平均母音一致率 {mean_agree:.2f}")
    if file_status == "OK" and (ratio < 0.75 or ratio > 1.8):
        file_status = "WARN"
        notes.append(f"母音核数が期待と乖離: 検出 {nuclei} / 期待 {exp}")
    return file_status, notes, nuclei
