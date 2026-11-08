"""独立検証: 台本を使わないフリーデコーダによる音素認識と、境界の確からしさ評価"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from .phoneset import UNIT_CLASS, is_burst
from .kana import Mora

GROUPS = ["a", "i", "u", "e", "o", "N", "FR", "CL", "BR", "SM", "SIL", "DV"]
GIDX = {g: i for i, g in enumerate(GROUPS)}
STATUS_RANK = {"OK": 0, "WARN": 1, "REVIEW": 2, "ERROR": 3}


def unit_group(u):
    if u.endswith("_dv"):
        return "DV"
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
    if u.startswith("nas"):
        acc |= {"u", "o"}       # 残響下の子音の鼻音は低い母音と区別しにくい
    if u in ("frH", "frHY", "frF"):
        acc |= {"SIL", "DV"}    # h は後続母音の無声版 (息の音) なので、無声化母音とも区別しにくい
    if u in ("cl", "clv") or u.startswith("br"):
        acc |= {"DV"}           # 破裂の気息・閉鎖区間も同様
    if u in ("clv", "brG", "brD", "brB"):
        acc |= {"u", "N"}       # 有声閉鎖音の声帯振動 (ボイスバー) は低い u や鼻音に聞こえる
    if u in ("frZ", "frJ", "frV", "clv"):
        acc |= {"N", "CL"}
    return acc


def free_decode(model, LL, switch, dv_penalty=3.0):
    """台本なしのフレーム単位 Viterbi。ユニットをグループ化して最尤系列を求める"""
    G, T = len(GROUPS), LL.shape[1]
    em = np.full((G, T), -1e9)
    for u, k in model.idx.items():
        gi = GIDX[unit_group(u)]
        em[gi] = np.maximum(em[gi], LL[k])
    em[GIDX["DV"]] -= dv_penalty          # 文脈なしの認識では、息の混じる音を無声化母音と取り違えやすい
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
    devoiced: bool = False
    gap_absorbed: float = 0.0     # 無声子音の前の短い無音を子音に含めた長さ (ms)
    energy_dip_db: float = 0.0    # 母音が連続する箇所のエネルギーの谷の深さ (dB)
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
    T = LL.shape[1]
    s = int(min(max(seg.s, 1), T - 1))
    w = max(1, min(6, prev.e - prev.s, seg.e - seg.s, s, T - s))
    up, un = idx[prev.unit], idx[seg.unit]
    a, b = LL[up, s - w:s] - LL[un, s - w:s], LL[un, s:s + w] - LL[up, s:s + w]
    contrast = float(a.mean() + b.mean()) if a.size and b.size else 0.0
    pk = bnd[max(0, s - 3):s + 3]
    peak = float(pk.max()) if pk.size else 0.0
    conf = max(contrast / cfg.thr_contrast, peak / cfg.thr_bnd)
    return float(np.nan_to_num(conf, nan=0.0, posinf=1e3, neginf=-1e3))


def check_timings(timings, dur_ms):
    """時間情報の整合性: 有限・録音長内・単調増加・モーラ間で逆転しない。不正なら ValueError"""
    prev_end = 0.0
    for t in timings:
        pts = [t.cons_start, t.vowel_start, t.vowel_end]
        if not all(np.isfinite(pts)):
            raise ValueError(f"モーラ {t.index + 1} の時刻が数値として不正です")
        if not (0.0 <= t.cons_start <= t.vowel_start <= t.vowel_end <= dur_ms + 1e-6):
            raise ValueError(f"モーラ {t.index + 1} の境界が不整合です (子音開始 {t.cons_start:.1f} / 母音開始 "
                             f"{t.vowel_start:.1f} / 母音終了 {t.vowel_end:.1f} / 録音長 {dur_ms:.1f} ms)")
        if t.cons_start < prev_end - 1.0:
            raise ValueError(f"モーラ {t.index + 1} が前のモーラと重なっています")
        for u, a, b in t.segs:
            if not (np.isfinite(a) and np.isfinite(b)) or b < a - 1e-6:
                raise ValueError(f"モーラ {t.index + 1} の音素 {u} の区間が不正です")
        prev_end = t.vowel_end


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
        gap, absorbed = max(0.0, cs - last_end), 0.0
        if mi > 0 and m.silent_onset and 0.0 < gap < 100.0:
            # 無声子音の前の短い無音 (声が止まってから摩擦・破裂が始まるまで) は子音の一部。正常なので子音に含める
            absorbed, cs, gap = gap, last_end, 0.0
            if cons_end_vc is not None:
                cons_end_vc = max(cons_end_vc, cs + 15.0)
        t = MoraTiming(mi, m, [(s.unit, st(s), en(s)) for s in ss], cs, vs, ve, cons_end_vc, gap,
                       v_frames=(ss[-1].s, ss[-1].e), c_frames=(ss[0].s, ss[-1].s))
        t.gap_absorbed = absorbed
        out.append(t)
        last_end = ve
    check_timings(out, dur_ms)
    return out


from .phoneset import UNVOICED_CONS_UNITS as _UNVOICED_C


def _maybe_devoiced(F, vseg, cons_units):
    """無声化母音 (し・す・き・く 等で i/u が声帯振動なしになる) の可能性: 直前が無声子音、有声度が低いが音は存在する"""
    if vseg.unit not in ("i", "u") or not cons_units or cons_units[-1] not in _UNVOICED_C:
        return False
    seg = F.cls[vseg.s:vseg.e]
    if len(seg) < 3:
        return False
    return float(np.mean(seg[:, 1])) < 0.35 and float(np.median(seg[:, 0])) > -3.5


def energy_dip_db(F, boundary_ms, left_limit_ms, right_limit_ms):
    """境界付近 (±30ms) のエネルギーの最小値が、両側の安定部 (境界から 40〜100ms) の低い方より何 dB 低いか。
    連続して発音された母音の遷移では小さく、母音を 1 つずつ切って発音すると大きくなる"""
    hop = F.hop_s * 1000.0

    def seg(a, b):
        i0, i1 = max(0, int(round(a / hop))), min(F.T, int(round(b / hop)) + 1)
        return F.energy_db[i0:i1]

    L = seg(max(left_limit_ms, boundary_ms - 100.0), boundary_ms - 40.0)
    R = seg(boundary_ms + 40.0, min(right_limit_ms, boundary_ms + 100.0))
    W = seg(boundary_ms - 30.0, boundary_ms + 30.0)
    if L.size < 3 or R.size < 3 or W.size < 3:
        return 0.0
    return float(max(0.0, min(np.median(L), np.median(R)) - W.min()))


def devoiced_verdict(lv, base_gi):
    """無声化母音として整列されたフレーム列 lv の独立認識による判定。
    元の有声母音と認識されたフレームが 6 割以上なら "voiced" (整列の選び間違い) / そうでなければ "devoiced"。
    (一致率, 判定) を返す。無声化母音の認識は 息の音 (DV / FR) も許容する"""
    if len(lv) == 0:
        return 0.0, "devoiced"
    frac_base = float(np.mean(lv == base_gi))
    agree = float(np.mean(np.isin(lv, [GIDX["DV"], GIDX["FR"], base_gi])))
    return agree, ("voiced" if frac_base >= 0.6 else "devoiced")


def tempo_flags(timings, min_morae=4):
    """モーラ長の一貫性検査。同じファイル内のモーラ長の中央値 (最後のモーラは余韻で長いので除外) と比べ、
    極端に短い/長いモーラは整列が崩れている可能性が高い。[(status|None, note|None)] を返す"""
    out = [(None, None)] * len(timings)
    body = timings[:-1]
    if len(body) < min_morae - 1:
        return out
    durs = [t.vowel_end - t.cons_start for t in body]
    med = float(np.median(durs))
    if med <= 0:
        return out
    for i, dd in enumerate(durs):
        r = dd / med
        if r < 0.25:
            out[i] = ("REVIEW", f"モーラ長が極端に短い ({dd:.0f}ms / 他の中央値 {med:.0f}ms の {r:.2f} 倍): 整列が崩れている可能性")
        elif r < 0.4:
            out[i] = ("WARN", f"モーラ長が短い ({dd:.0f}ms / 他の中央値 {med:.0f}ms の {r:.2f} 倍)")
        elif r > 3.0:
            out[i] = ("WARN", f"モーラ長が長い ({dd:.0f}ms / 他の中央値 {med:.0f}ms の {r:.2f} 倍)")
    return out


def tempo_deviation(timings, min_morae=4):
    """モーラ長の中央値からのずれ (対数, 上限2) の平均。0 に近いほど一貫している"""
    body = timings[:-1]
    if len(body) < min_morae - 1:
        return 0.0
    durs = np.array([max(t.vowel_end - t.cons_start, 1.0) for t in body])
    med = float(np.median(durs))
    return float(np.mean(np.minimum(np.abs(np.log(durs / med)), 2.0)))


def label_runs(labels, hop_ms):
    """フレームごとの認識ラベル列 → 連続区間 [{group, start_ms, end_ms}]"""
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append({"group": GROUPS[int(labels[start])], "start_ms": round(max(0.0, (start - 0.5) * hop_ms), 1),
                        "end_ms": round(max(0.0, (i - 0.5) * hop_ms), 1)})
            start = i
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
        aligned_dv = vseg.unit.endswith("_dv")
        if aligned_dv:      # 無声化母音として整列された: 独立認識が有声母音なら整列の選び間違いとして有声母音に戻す
            t.agree_v, verdict = devoiced_verdict(lv, GIDX[vseg.unit[0]])
            if verdict == "voiced":
                base = vseg.unit[0]
                vseg.unit = base
                t.segs[-1] = (base, t.segs[-1][1], t.segs[-1][2])
                aligned_dv = False
        else:
            eg = [GIDX[unit_group(vseg.unit)]]
            if t.mora.palatal:        # 拗音の母音は立ち上がりが半母音/i に聞こえる
                eg += [GIDX["SM"], GIDX["i"]]
            t.agree_v = float(np.mean(np.isin(lv, eg))) if len(lv) else 0.0
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
        dev = t.agree_v < cfg.ok_agree and _maybe_devoiced(F, vseg, [x.unit for x in cons])
        t.devoiced = bool(dev) or aligned_dv
        i0 = real.index(ss[0])
        if i0 > 0 and real[i0 - 1].unit == ss[0].unit == "NN":
            t.bconf = max(t.bconf, 1.0)
            t.notes.append("ん の連続は音響的に境界を特定できないため推定位置")
        if dev:
            t.status = "WARN"
            t.notes.append(f"無声化母音の可能性 (有声度 {float(np.mean(F.cls[vseg.s:vseg.e, 1])):.2f}): 母音の有声検証を免除")
            if t.cons_agree < cfg.cons_review:
                t.status = "REVIEW"
                t.notes.append(f"子音が音響的に確認できない (一致率 {t.cons_agree:.2f})")
        elif t.agree_v < cfg.fail_agree:
            t.status = "ERROR"
            t.notes.append(f"母音の独立認識が不一致 (期待 {vseg.unit} / 認識 {t.rec_vowel}, 一致率 {t.agree_v:.2f})")
        elif t.agree_v < cfg.ok_agree:
            t.status = "REVIEW"
            t.notes.append(f"母音の一致率が低い ({t.agree_v:.2f}, 認識 {t.rec_vowel})")
        elif t.cons_agree < cfg.cons_review:
            t.status = "REVIEW"
            t.notes.append(f"子音が音響的に確認できない (期待 {t.mora.cons} / 認識 {t.rec_cons or '-'}, 一致率 {t.cons_agree:.2f})")
        elif t.bconf < 0.5:
            t.status = "REVIEW"
            t.notes.append(f"音素境界が不明瞭 (確信度 {t.bconf:.2f})")
        elif t.bconf < 1.0 or t.cons_agree < cfg.cons_warn:
            t.status = "WARN"
            t.notes.append(f"境界確信度 {t.bconf:.2f} / 子音一致率 {t.cons_agree:.2f}")
        else:
            t.status = "OK"
    if cfg.cut_dip_db > 0:
        for k in range(1, len(timings)):
            t, pv = timings[k], timings[k - 1]
            if t.mora.cons == "" and t.mora.vowel != "N" and pv.mora.vowel != "N":      # 母音 → 母音 (連続して発音するはず)
                t.energy_dip_db = energy_dip_db(F, t.vowel_start, pv.vowel_start, t.vowel_end)
                if t.energy_dip_db >= cfg.cut_dip_db:
                    if t.status == "OK":
                        t.status = "WARN"
                    t.notes.append(f"前の母音との間で声が途切れています (エネルギーの谷 {t.energy_dip_db:.0f}dB): "
                                   "連続音は母音を切らずに繋げて発音して収録します")
    for t, (st, note) in zip(timings, tempo_flags(timings)):
        if st and STATUS_RANK[st] > STATUS_RANK[t.status]:
            t.status = st
        if note:
            t.notes.append(note)
    for t in timings:
        if t.status == "OK" and any(u.endswith("_dv") for u, _, _ in t.segs):
            t.status = "WARN"
            t.notes.append("無声化母音として整列 (有声の母音ではありません。--strict では除外)")
    # ファイル単位: 独立に検出した母音/鼻音核の数 と 期待モーラ数
    nuclei = 0
    run, cur = 0, -1
    only_n = all(t.mora.vowel == "N" for t in timings)
    vn = {GIDX["N"]} if only_n else {GIDX[g] for g in "aiueo"}
    min_run = 6 if only_n else 4
    for x in list(labels) + [-1]:
        if x == cur:
            run += 1
        else:
            if cur in vn and run >= min_run:
                nuclei += 1
            cur, run = x, 1
    exp = 1 if only_n else sum(1 for t in timings if t.mora.vowel != "N" and not t.devoiced)
    ratio = nuclei / exp if exp else 1.0          # 有声の母音が期待されない (全て無声化母音/ん) なら核の数は検査しない
    file_status = "OK"
    mean_agree = float(np.mean([1.0 if t.devoiced else t.agree_v for t in timings]))
    n_bad = sum(1 for t in timings if STATUS_RANK[t.status] >= 2)
    if n_bad / len(timings) >= 0.4:
        file_status = "FAIL"
        notes.append(f"分割失敗: {len(timings)} モーラ中 {n_bad} モーラが音響検証に不合格 (台本と音声が一致しない可能性)")
    elif (exp and ratio < 0.5) or mean_agree < 0.35:
        file_status = "FAIL"
        notes.append(f"分割失敗: 独立検出した母音核 {nuclei} / 期待 {exp}, 平均母音一致率 {mean_agree:.2f}")
    if file_status == "OK" and not only_n and exp and (ratio < 0.75 or ratio > 1.8):
        file_status = "WARN"
        notes.append(f"母音核数が期待と乖離: 検出 {nuclei} / 期待 {exp}")
    return file_status, notes, nuclei
