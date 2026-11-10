"""エイリアス生成と oto パラメータ算出 (CV / VCV / CVVC)"""
from dataclasses import dataclass, field
from .kana import VOWEL_KANA, VOWEL_ROMAJI
from .phoneset import cons_units
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
    valid: bool = True       # パラメータが物理的に成立しているか (False は強制出力でも書かない)
    forced: bool = False     # 失敗・保留を承知で出力するエントリ
    manual: object = None    # GUI での手動指定: True=必ず書く / False=書かない / None=自動判定


def _status(moras_t, extra=None):
    st = max((t.status for t in moras_t), key=lambda s: STATUS_RANK[s])
    notes = [n for t in moras_t for n in t.notes]
    return st, notes


def format_problem(e):
    """oto.ini の行形式 (ファイル名=エイリアス,数値×5) を壊す内容。問題があれば理由を返す"""
    if "=" in e.wav or "\n" in e.wav:
        return "WAV ファイル名に '=' や改行があり oto.ini の行形式を壊します (ファイル名を変更してください)"
    if "," in e.alias or "=" in e.alias or "\n" in e.alias or "\r" in e.alias:
        return "エイリアスに ',' '=' や改行が含まれ oto.ini の行形式を壊します"
    return None


def encoding_problem(e, encoding):
    """指定の文字コードで書けない文字を含むか (書き込み全体の失敗を防ぐため、エントリ単位で除外する)"""
    for field in (e.wav, e.alias):
        try:
            field.encode(encoding)
        except UnicodeEncodeError as ex:
            return (f"文字 {field[ex.start:ex.end]!r} は文字コード {encoding} で表せません "
                    f"(--encoding utf-8 を使うか、別の表記にしてください)")
    return None


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
            e.valid = False
            e.notes.append("使用区間が録音長を超えるため生成不可")
    e.cut_len = cut_len
    e.cutoff = -cut_len
    problem = format_problem(e)
    if problem:
        e.valid = False
        e.status = "ERROR"
        e.notes.append(problem)
    vals = (e.offset, e.consonant, e.cutoff, e.pre, e.overlap)
    if not all(v == v and abs(v) != float("inf") for v in vals):
        e.valid = False
        e.status = "ERROR"
        e.notes.append("パラメータが数値として不正")
    return e


_SONORANT_UNITS = {"nas", "nas_m", "nas_ny", "rf", "gl_y", "gl_w"}


def _first_cons_unit(mora):
    cu = cons_units(mora.cons) if mora.cons else []
    return cu[0] if cu else None


def _fixed_extra(m, cfg, d):
    """固定範囲 (consonant) の、先行発声より後ろへの延長。な行・ま行・や行・ら行・わ行は
    「音の遷移が完全に終わった所」まで長めに取らないと発音が間延びする (UTAU wiki)"""
    base = cfg.fixed_extra_sonorant_ms if _first_cons_unit(m.mora) in _SONORANT_UNITS else cfg.fixed_extra_ms
    return min(base, 0.5 * d)


def _cv_overlap(m, pre, cfg):
    """CV (単独音・先頭) のオーバーラップ。UTAU wiki の行ごとの目安に従う:
    あ行/ん: なし or 先行発声と同じ / か行・た行・ぱ行・ち・つ: なし / それ以外 (さ行・は行・な行・ま行・や行・ら行・わ行・濁音):
    先行発声の 20〜50ms 手前"""
    first = _first_cons_unit(m.mora)
    if first is None:
        return pre
    if first == "cl":                      # 無声閉鎖・破擦 (k t p ch ts)
        return 0.0
    return min(pre, max(0.0, pre - cfg.cv_overlap_margin_ms))


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
    e = AliasEntry(wav, label, offset, pre + _fixed_extra(m, cfg, 0.7 * d), 0.0, pre, _cv_overlap(m, pre, cfg), kind,
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
    p = {"mora": prev.mora.surface, "vowel": VOWEL_KANA[prev.mora.vowel],
         "romaji": VOWEL_ROMAJI[prev.mora.vowel]}[cfg.vcv_style]
    # オーバーラップは先行発声の 1/3 (VCV の定石)。ただし前の母音の尾 (子音の開始まで) の内側に収める
    overlap = min(pre / 3.0, max(0.0, m.cons_start - offset))
    e = AliasEntry(wav, f"{p} {m.mora.surface}", offset, pre + _fixed_extra(m, cfg, 0.7 * d), 0.0, pre,
                   overlap, "VCV", moras=[prev.index, m.index])
    e.cut_len = _vend(m) - offset
    e.status, e.notes = _status([prev, m])
    if m.gap_before >= 40.0 and not m.mora.silent_onset:
        # 連続音は声を繋げて収録するのが正しい。か行・た行・ぱ行・さ行 など無声子音の前の無音は正常なので警告しない
        if STATUS_RANK[e.status] < 1:
            e.status = "WARN"
        e.notes.append(f"モーラ間に {m.gap_before:.0f}ms の無音があり、声が途切れています (連続音は繋げて収録します)")
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
