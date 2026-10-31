"""テスト用の簡易フォルマント合成器。正解の音素境界付きで連続音を生成する。"""
import numpy as np
from scipy.signal import lfilter, butter, sosfilt
from otoium.kana import parse_text

SR = 44100
FORM = {"a": (800, 1300, 2500), "i": (300, 2300, 3000), "u": (350, 1250, 2400),
        "e": (500, 1900, 2600), "o": (500, 900, 2400)}
BW = (80, 110, 160)
DUR = {"a": .22, "i": .22, "u": .22, "e": .22, "o": .22, "NN": .18, "nas": .07, "nas_m": .07, "nas_ny": .08,
       "frS": .10, "frSH": .11, "frTS": .08, "frCH": .09, "frH": .08, "frHY": .08, "frF": .08, "frZ": .08,
       "frJ": .07, "cl": .05, "clv": .035, "brK": .03, "brT": .025, "brP": .02, "brG": .015, "brD": .015,
       "brB": .015, "rf": .02, "gl_y": .04, "gl_w": .04}


def _res(x, f, bw):
    r = np.exp(-np.pi * bw / SR)
    th = 2 * np.pi * f / SR
    return lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], x)


def _voiced(dur, f0, formants, amp, rng):
    n = int(dur * SR)
    p = np.zeros(n)
    p[np.arange(0, n, SR / f0).astype(int)] = 1.0
    y = lfilter([1], [1, -0.95], p)
    for f, bw in zip(formants, BW):
        y = _res(y, f, bw)
    y = y / (np.max(np.abs(y)) + 1e-9)
    return amp * y + 0.003 * rng.standard_normal(n)


def _noise(dur, lo, hi, amp, rng):
    n = int(dur * SR)
    sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    y = sosfilt(sos, rng.standard_normal(n))
    return amp * y / (np.max(np.abs(y)) + 1e-9)


def _unit(u, dur, nxt_v, f0, rng):
    n = int(dur * SR)
    fnext = FORM.get(nxt_v, FORM["a"])
    if u in FORM:
        return _voiced(dur, f0, FORM[u], 1.0, rng)
    if u == "NN":
        return _voiced(dur, f0, (260, 1100, 2400), 0.45, rng)
    if u in ("nas", "nas_m", "nas_ny"):
        return _voiced(dur, f0, (260, 1100, 2400), 0.30, rng)
    if u == "frS":
        return _noise(dur, 4500, 7800, 0.22, rng)
    if u in ("frSH", "frTS", "frCH"):
        return _noise(dur, 2500, 7000, 0.25, rng)
    if u in ("frH", "frHY", "frF"):
        y = _noise(dur, 300, 6000, 0.07, rng)
        return y
    if u in ("frZ", "frJ"):
        return _voiced(dur, f0, (300, 1500, 2600), 0.25, rng) + _noise(dur, 4500, 7800, 0.08, rng)
    if u == "cl":
        return 0.002 * rng.standard_normal(n)
    if u == "clv":
        t = np.arange(n) / SR
        return 0.07 * np.sin(2 * np.pi * f0 * t) + 0.002 * rng.standard_normal(n)
    if u in ("brK", "brT", "brP"):
        band = {"brK": (1500, 3500), "brT": (3000, 7500), "brP": (400, 1800)}[u]
        b = _noise(dur, band[0], band[1], 0.20, rng)
        env = np.exp(-np.arange(n) / (0.35 * n + 1))
        return b * env
    if u in ("brG", "brD", "brB"):
        return _voiced(dur, f0, (400, 1500, 2500), 0.2, rng)
    if u == "rf":
        return _voiced(dur, f0, (400, 1400, 2500), 0.35, rng)
    if u == "gl_y":
        return _voiced(dur, f0, FORM["i"], 0.6, rng)
    if u == "gl_w":
        return _voiced(dur, f0, FORM["u"], 0.6, rng)
    raise KeyError(u)


def render(text, seed=0, f0=125.0, lead=0.3, trail=0.3, sr_out=SR):
    """returns (audio float32, truth) truth=[(mora_idx, unit, start_s, end_s)]"""
    rng = np.random.default_rng(seed)
    morae = parse_text(text)
    parts = [0.001 * rng.standard_normal(int(lead * SR))]
    t = lead
    truth = []
    for mi, m in enumerate(morae):
        us = m.units
        nxt_v = m.vowel
        for u in us:
            dur = DUR[u] * (1 + 0.15 * rng.standard_normal() * 0.5)
            y = _unit(u, dur, nxt_v, f0 * (1 + 0.04 * rng.standard_normal()), rng)
            k = min(int(0.004 * SR), len(y) // 3)
            if k > 0:
                ramp = np.linspace(0, 1, k)
                y = y.copy()
                y[:k] *= ramp
                y[-k:] *= ramp[::-1]
            parts.append(y)
            truth.append((mi, u, t, t + len(y) / SR))
            t += len(y) / SR
    parts.append(0.001 * rng.standard_normal(int(trail * SR)))
    x = np.concatenate(parts)
    x = 0.6 * x / np.max(np.abs(x))
    return x.astype(np.float32), truth


def write_wav(path, x):
    import wave
    a = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(a.tobytes())
