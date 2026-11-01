"""学習の進行を得点として記録する (報酬曲線・学習曲線に相当)"""
import csv
import json

INFO = {
    "score": ("検証スコア", "up"),
    "ll": ("平均対数尤度/フレーム", "up"),
    "shift": ("境界の変化 (フレーム)", "down"),
    "nn_loss": ("NN 損失", "down"),
    "nn_acc": ("NN フレーム精度", "up"),
    "tune_score": ("自動調整スコア", "up"),
}
_BARS = "▁▂▃▄▅▆▇█"


class Tracker:
    def __init__(self, cb=None, log=None):
        self.cb = cb
        self.log = log
        self.history = {}      # name -> [(step, value, stage)]

    def add(self, name, value, stage="", log=False):
        h = self.history.setdefault(name, [])
        prev = h[-1][1] if h else None
        h.append((len(h) + 1, float(value), stage))
        if self.cb:
            self.cb(name, float(value), stage)
        if log and self.log:
            d = f" ({value - prev:+.3f})" if prev is not None else ""
            self.log(f"    [{stage}] {INFO.get(name, (name,))[0]} {value:.3f}{d} {self.spark(name)}")

    def values(self, name):
        return [v for _, v, _ in self.history.get(name, [])]

    def spark(self, name, n=16):
        v = self.values(name)[-n:]
        if not v:
            return ""
        lo, hi = min(v), max(v)
        if hi - lo < 1e-12:
            return _BARS[3] * len(v)
        return "".join(_BARS[int((x - lo) / (hi - lo) * (len(_BARS) - 1))] for x in v)

    def save(self, folder):
        with open(folder / "otoium_training.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "step", "stage", "value"])
            for name, rows in self.history.items():
                for step, val, stage in rows:
                    w.writerow([name, step, stage, f"{val:.6f}"])
        (folder / "otoium_training.json").write_text(
            json.dumps({k: [{"step": s, "stage": g, "value": v} for s, v, g in rows] for k, rows in self.history.items()},
                       ensure_ascii=False, indent=1), encoding="utf-8")
