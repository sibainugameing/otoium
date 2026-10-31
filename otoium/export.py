"""ラベル / TextGrid 出力"""


def write_labels(path, j):
    lines = [f"{s.s_ms / 1000:.4f}\t{s.e_ms / 1000:.4f}\t{s.unit}" for s in j.segs
             if s.e > s.s and s.unit != "pau" and s.s_ms is not None]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_textgrid(path, j):
    real = [s for s in j.segs if s.e > s.s and s.s_ms is not None]
    dur = j.dur_ms / 1000.0
    ph = [(s.s_ms / 1000, s.e_ms / 1000, "" if s.unit == "pau" else s.unit) for s in real]
    mo = [(t.cons_start / 1000, t.vowel_end / 1000, t.mora.surface) for t in j.timings]

    def tier(name, items):
        out = [f'    class = "IntervalTier"', f'    name = "{name}"', "    xmin = 0", f"    xmax = {dur:.4f}",
               f"    intervals: size = {len(items)}"]
        for i, (a, b, l) in enumerate(items, 1):
            out += [f"    intervals [{i}]:", f"        xmin = {a:.4f}", f"        xmax = {b:.4f}", f'        text = "{l}"']
        return out

    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", f"xmax = {dur:.4f}",
             "tiers? <exists>", "size = 2", "item []:", "  item [1]:"] + tier("phones", ph) + ["  item [2]:"] + tier("morae", mo)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
