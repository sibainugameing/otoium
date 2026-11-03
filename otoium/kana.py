"""かな / ローマ字 → モーラ列"""
import unicodedata
from dataclasses import dataclass
from .phoneset import mora_units, cons_units

V = "aiueo"
_ROWS = [
    ("あいうえお", ["", "", "", "", ""]), ("かきくけこ", ["k"] * 5), ("がぎぐげご", ["g"] * 5),
    ("さしすせそ", ["s", "sh", "s", "s", "s"]), ("ざじずぜぞ", ["z", "j", "z", "z", "z"]),
    ("たちつてと", ["t", "ch", "ts", "t", "t"]), ("だぢづでど", ["d", "j", "z", "d", "d"]),
    ("なにぬねの", ["n", "ny", "n", "n", "n"]), ("はひふへほ", ["h", "hy", "f", "h", "h"]),
    ("ばびぶべぼ", ["b"] * 5), ("ぱぴぷぺぽ", ["p"] * 5), ("まみむめも", ["m"] * 5),
    ("らりるれろ", ["r"] * 5),
]
BASE = {}
for _chars, _cs in _ROWS:
    for _ch, _c, _v in zip(_chars, _cs, V):
        BASE[_ch] = (_c, _v)
BASE.update({"や": ("y", "a"), "ゆ": ("y", "u"), "よ": ("y", "o"), "わ": ("w", "a"),
             "を": ("", "o"), "ん": ("", "N"), "ゐ": ("", "i"), "ゑ": ("", "e"),
             "ゔ": ("v", "u"), "ゎ": ("w", "a"), "ゕ": ("k", "a"), "ゖ": ("k", "e"),
             "ぁ": ("", "a"), "ぃ": ("", "i"), "ぅ": ("", "u"), "ぇ": ("", "e"), "ぉ": ("", "o")})
SMALL_Y = {"ゃ": "a", "ゅ": "u", "ょ": "o"}
SMALL_V = {"ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o", "ゎ": "a"}
PAL_SELF = {"sh", "j", "ch", "ny", "hy"}
SKIP = set("_ -・.,、。()（）〜~ー　0123456789'’") | set("っッ")


class TextParseError(ValueError):
    pass


def to_hira(ch):
    o = ord(ch)
    return chr(o - 0x60) if 0x30A1 <= o <= 0x30F6 else ch


def _combine(base, small):
    c, v = BASE[base]
    if small in SMALL_Y:
        nv = SMALL_Y[small]
        if c in PAL_SELF:
            return c, nv
        return (c + "y" if c else "y"), nv
    nv = SMALL_V[small]
    if v == "i" and c not in PAL_SELF:
        return (c + "y" if c else "y"), nv
    if c in ("k", "g"):
        return c + "w", nv
    if c == "" and v == "u":
        return "w", nv
    return c, nv


@dataclass(frozen=True)
class Mora:
    surface: str
    kana: str
    cons: str
    vowel: str          # a i u e o / N(ん)

    @property
    def units(self):
        return mora_units(self.cons, self.vowel)

    @property
    def has_cons(self):
        return bool(self.cons)


def _romaji_table():
    t = {}
    for ch, (c, v) in BASE.items():
        if v != "N" and ch not in SMALL_V and ch not in "ゎゕゖ":
            t.setdefault(c + v, (c, v))
    for base in "かきくぐがぎげごさしすせそざじずぜぞたちつてとだぢづでどなにぬねのはひふへほばびぶべぼぱぴぷぺぽまみむめもらりるれろゔうい":
        for sm in "ゃゅょぁぃぅぇぉ":
            try:
                c, v = _combine(base, sm)
            except KeyError:
                continue
            t.setdefault(c + v, (c, v))
    t.update({"n": ("", "N"), "hi": ("hy", "i"), "ni": ("ny", "i"), "wo": ("", "o"), "nn": ("", "N")})
    # 訓令式・日本式・その他の入力表記 (し/ち/つ/ふ/じ は一般的な読みに揃える)
    alt = {"si": ("sh", "i"), "ti": ("ch", "i"), "tu": ("ts", "u"), "hu": ("f", "u"), "zi": ("j", "i"),
           "di": ("j", "i"), "du": ("z", "u"), "thi": ("t", "i"), "dhi": ("d", "i"), "twu": ("t", "u"),
           "dwu": ("d", "u"), "wi": ("w", "i"), "we": ("w", "e"), "ye": ("y", "e"), "kwa": ("kw", "a"),
           "gwa": ("gw", "a"), "tsa": ("ts", "a"), "tsi": ("ts", "i"), "tse": ("ts", "e"), "tso": ("ts", "o"),
           "fa": ("f", "a"), "fi": ("f", "i"), "fe": ("f", "e"), "fo": ("f", "o"),
           "va": ("v", "a"), "vi": ("v", "i"), "vu": ("v", "u"), "ve": ("v", "e"), "vo": ("v", "o")}
    for pre, (c, v) in {"sy": ("sh", None), "ty": ("ch", None), "cy": ("ch", None), "zy": ("j", None),
                        "jy": ("j", None), "dy": ("j", None), "hy": ("hy", None), "ny": ("ny", None)}.items():
        for vv in "aueo":
            alt[pre + vv] = (c, vv)
    t.update(alt)
    return t


ROMAJI = _romaji_table()


def _parse_romaji(run, out):
    s = run.lower()
    i = 0
    while i < len(s):
        # 促音 (kka, tta, ssha ...): 同じ子音の連続は前の子音を無視。ただし nn は ん
        if i + 1 < len(s) and s[i] == s[i + 1] and s[i] not in "aiueon" and s[i].isalpha():
            i += 1
            continue
        for L in (4, 3, 2, 1):
            k = s[i:i + L]
            if len(k) == L and k in ROMAJI:
                c, v = ROMAJI[k]
                out.append(Mora(k, k, c, v))
                i += L
                break
        else:
            raise TextParseError(f"ローマ字を解釈できません: {run!r} の {i} 文字目")


def parse_text(text, skip_unknown=False, skipped=None):
    """かな/ローマ字 → モーラ列。skip_unknown=True なら未対応文字を無視し、無視した文字を skipped に追加する"""
    s = unicodedata.normalize("NFKC", text)
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in SKIP:
            i += 1
            continue
        h = to_hira(ch)
        if h in BASE:
            if i + 1 < len(s):
                nx = to_hira(s[i + 1])
                if nx in SMALL_Y or nx in SMALL_V:
                    c, v = _combine(h, nx)
                    out.append(Mora(ch + s[i + 1], h + nx, c, v))
                    i += 2
                    continue
            c, v = BASE[h]
            out.append(Mora(ch, h, c, v))
            i += 1
            continue
        if ch.isascii() and ch.isalpha():
            j = i
            while j < len(s) and s[j].isascii() and s[j].isalpha():
                j += 1
            try:
                _parse_romaji(s[i:j], out)
            except TextParseError:
                if not skip_unknown:
                    raise
                if skipped is not None:
                    skipped.append(s[i:j])
            i = j
            continue
        if skip_unknown:
            if skipped is not None:
                skipped.append(ch)
            i += 1
            continue
        raise TextParseError(f"解釈できない文字 {ch!r} (テキスト: {text!r})。"
                             "未対応文字を無視するには設定で「未対応文字を無視」を有効にしてください")
    if not out:
        raise TextParseError(f"モーラが得られません: {text!r}")
    for m in out:
        cons_units(m.cons)
    return out


VOWEL_KANA = {"a": "あ", "i": "い", "u": "う", "e": "え", "o": "お", "N": "ん"}
VOWEL_ROMAJI = {"a": "a", "i": "i", "u": "u", "e": "e", "o": "o", "N": "n"}
