"""エイリアス生成と oto パラメータ算出 (CV / VCV / CVVC)"""
from dataclasses import dataclass, field
from .kana import VOWEL_KANA, VOWEL_ROMAJI
from .verify import STATUS_RANK


@dataclass
class AliasEntry:
    wav: str
    alias: str
    offset: float
    consonant: float
    cutoff: float          # 負値: offset からの使用長
    pre: float
    overlap: float
    kind: str
    status: str = "OK"
    notes: list = field(default_factory=list)
    moras: list = field(default_factory=list)
    quality: float = 1.0
    cut_len: float = 0.0


def _status(moras_t, extra=None):
    st = max((t.status for t in moras_t), key=lambda s: STATUS_RANK[s])
    notes = [n for t in moras_t for n in t.notes]
    return st, notes


def _finalize(e, wav_ms):
    e.offset = max(0.0, e.offset)
    e.pre = max(1.0, e.pre)
    e.consonant = max(e.consonant, e.pre)
    e.overlap = min(max(0.0, e.overlap), e.pre)
    cut_len = e.cut_len
    if cut_len < e.consonant + 5.0:
        cut_len = e.consonant + 5.0
    if e.offset + cut_len > wav_ms:
        cut_len = wav_ms - e.offset
        if cut_len < e.consonant + 1.0:
            e.status = "ERROR"
            e.notes.append("使用区間が録音長を超えるため生成不可")
    e.cut_len = cut_len
    e.cutoff = -cut_len
    return e


def _vend(m):
    d = m.vowel_end - m.vowel_start
    cut = m.vowel_end - 0.1 * d
    return max(cut, m.vowel_start + min(d, 30.0))


def _cv(wav, m, prev, wav_ms, cfg, label, kind):
    cs, vs = m.cons_start, m.vowel_start
    if prev is None or m.gap_before >= 5.0:
        lead = min(cfg.iso_lead_ms, cs)
    else:
        lead = min(cfg.cv_lead_ms, cs)
    offset = cs - lead
    pre = vs - offset
    d = m.vowel_end - vs
    e = AliasEntry(wav, label, offset, pre + min(cfg.fixed_extra_ms, 0.35 * d), 0.0, pre, min(pre / 3.0, 30.0), kind,
                   moras=[m.index])
    e.cut_len = _vend(m) - offset
    e.status, e.notes = _status([m])
    e.quality = m.quality
    return _finalize(e, wav_ms)


def _vsym(m, style):
    v = m.mora.vowel
    return VOWEL_KANA[v] if style == "kana" else VOWEL_ROMAJI[v]


def _vcv(wav, m, prev, wav_ms, cfg):
    pv = prev.vowel_end - prev.vowel_start
    tail = max(5.0, min(cfg.vcv_tail_ms, 0.7 * pv))
    offset = max(m.cons_start - tail, prev.vowel_start)
    pre = m.vowel_start - offset
    d = m.vowel_end - m.vowel_start
    p = prev.mora.surface if cfg.vcv_style == "mora" else VOWEL_KANA[prev.mora.vowel]
    e = AliasEntry(wav, f"{p} {m.mora.surface}", offset, pre + min(cfg.fixed_extra_ms, 0.35 * d), 0.0, pre,
                   0.5 * (m.cons_start - offset), "VCV", moras=[prev.index, m.index])
    e.cut_len = _vend(m) - offset
    e.status, e.notes = _status([prev, m])
    if m.gap_before >= 40.0:
        if STATUS_RANK[e.status] < 1:
            e.status = "WARN"
        e.notes.append(f"モーラ間に {m.gap_before:.0f}ms の無音があり、純粋なVCVではありません")
    e.quality = min(prev.quality, m.quality)
    return _finalize(e, wav_ms)


def _vc(wav, m, prev, wav_ms, cfg):
    pv = prev.vowel_end - prev.vowel_start
    tail = max(5.0, min(cfg.vc_tail_ms, 0.7 * pv))
    offset = max(m.cons_start - tail, prev.vowel_start)
    pre = m.cons_start - offset
    clen = max(5.0, m.cons_end_vc - m.cons_start)
    csym = m.mora.cons
    alias = f"{_vsym(prev, cfg.vc_style)} {csym}"
    e = AliasEntry(wav, alias, offset, pre + 0.5 * clen, 0.0, pre, 0.5 * pre, "VC", moras=[prev.index, m.index])
    e.cut_len = m.cons_end_vc - offset
    e.status, e.notes = _status([prev, m])
    e.quality = min(prev.quality, m.quality)
    return _finalize(e, wav_ms)


def _tail(wav, m, wav_ms, cfg, trailing_ms):
    d = m.vowel_end - m.vowel_start
    tl = max(10.0, min(100.0, 0.7 * d))
    offset = m.vowel_end - tl
    e = AliasEntry(wav, f"{VOWEL_KANA[m.mora.vowel] if cfg.vc_style == 'kana' else VOWEL_ROMAJI[m.mora.vowel]} {cfg.tail_alias}",
                   offset, 0.5 * tl, 0.0, 0.5 * tl, 0.25 * tl, "TAIL", moras=[m.index])
    e.cut_len = tl + min(30.0, trailing_ms)
    e.status, e.notes = _status([m])
    e.quality = m.quality
    return _finalize(e, wav_ms)


def make_entries(wav, timings, wav_ms, cfg):
    mode = cfg.mode
    if mode == "auto":
        mode = "cv" if len(timings) == 1 else "vcv"
    out = []
    for k, m in enumerate(timings):
        prev = timings[k - 1] if k else None
        if mode == "cv":
            out.append(_cv(wav, m, prev, wav_ms, cfg, m.mora.surface, "CV"))
        elif mode == "vcv":
            if prev is None:
                out.append(_cv(wav, m, None, wav_ms, cfg, cfg.head_prefix + m.mora.surface, "HEAD"))
            else:
                out.append(_vcv(wav, m, prev, wav_ms, cfg))
        elif mode == "cvvc":
            out.append(_cv(wav, m, prev, wav_ms, cfg, m.mora.surface, "CV"))
            if prev is not None and m.mora.has_cons:
                out.append(_vc(wav, m, prev, wav_ms, cfg))
        else:
            raise ValueError(f"未知のモード: {mode}")
    if cfg.add_tail and mode in ("vcv", "cvvc") and timings:
        last = timings[-1]
        out.append(_tail(wav, last, wav_ms, cfg, wav_ms - last.vowel_end))
    return out
