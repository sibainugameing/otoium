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
             "を": ("", "o"), "ん": ("", "N"), "ゐ": ("", "i"), "ゑ": ("", "e")})
SMALL_Y = {"ゃ": "a", "ゅ": "u", "ょ": "o"}
SMALL_V = {"ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o"}
PAL_SELF = {"sh", "j", "ch", "ny", "hy"}
SKIP = set("_ -・.,、。()（）〜~ー　0123456789") | set("っッ")


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
        if v != "N":
            t.setdefault(c + v, (c, v))
    for base in "きぎにひみりびぴしじちぢて":
        for sm in "ゃゅょ":
            c, v = _combine(base, sm)
            t.setdefault(c + v, (c, v))
    for base, sm in [("つ", "ぁ"), ("つ", "ぃ"), ("つ", "ぇ"), ("つ", "ぉ"), ("ふ", "ぁ"), ("ふ", "ぃ"), ("ふ", "ぇ"),
                     ("ふ", "ぉ"), ("う", "ぃ"), ("う", "ぇ"), ("う", "ぉ"), ("い", "ぇ"), ("く", "ぁ"), ("く", "ぃ"),
                     ("く", "ぇ"), ("く", "ぉ"), ("ぐ", "ぁ"), ("ぐ", "ぃ"), ("ぐ", "ぇ"), ("ぐ", "ぉ")]:
        c, v = _combine(base, sm)
        t.setdefault(c + v, (c, v))
    t.update({"hi": ("hy", "i"), "ni": ("ny", "i"), "n": ("", "N")})
    return t


ROMAJI = _romaji_table()


def _parse_romaji(run, out):
    s = run.lower()
    i = 0
    while i < len(s):
        for L in (4, 3, 2, 1):
            k = s[i:i + L]
            if len(k) == L and k in ROMAJI:
                c, v = ROMAJI[k]
                out.append(Mora(k, k, c, v))
                i += L
                break
        else:
            raise TextParseError(f"ローマ字を解釈できません: {run!r} の {i} 文字目")


def parse_text(text):
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
            _parse_romaji(s[i:j], out)
            i = j
            continue
        raise TextParseError(f"解釈できない文字 {ch!r} (テキスト: {text!r})")
    if not out:
        raise TextParseError(f"モーラが得られません: {text!r}")
    for m in out:
        cons_units(m.cons)
    return out


VOWEL_KANA = {"a": "あ", "i": "い", "u": "う", "e": "え", "o": "お", "N": "ん"}
VOWEL_ROMAJI = {"a": "a", "i": "i", "u": "u", "e": "e", "o": "o", "N": "n"}
