"""oto.ini の書き出し"""
import shutil
from pathlib import Path


def _n(v):
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def format_entry(e):
    return f"{e.wav}={e.alias},{_n(e.offset)},{_n(e.consonant)},{_n(e.cutoff)},{_n(e.pre)},{_n(e.overlap)}"


def read_lines(path, encoding="cp932"):
    path = Path(path)
    if not path.exists():
        return []
    for enc in (encoding, "utf-8", "cp932"):
        try:
            return [l for l in path.read_bytes().decode(enc).splitlines() if "=" in l]
        except UnicodeDecodeError:
            continue
    return []


def write_oto(path, entries, encoding="cp932", backup=True, extra_lines=()):
    path = Path(path)
    if backup and path.exists():
        bak = path.with_name(path.name + ".bak")
        i = 1
        while bak.exists():
            bak = path.with_name(f"{path.name}.bak{i}")
            i += 1
        shutil.copy2(path, bak)
    lines = [format_entry(e) for e in entries] + list(extra_lines)
    text = "\n".join(lines) + ("\n" if lines else "")
    try:
        data = text.encode(encoding)
    except UnicodeEncodeError as ex:
        raise ValueError(f"文字コード {encoding} で表現できない文字があります: {ex}")
    path.write_bytes(data)
