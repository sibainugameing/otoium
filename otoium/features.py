"""音響特徴抽出: 8次元クラス特徴 + MFCC12 + Δ12 + 境界証拠 + 1ms 微細包絡"""
from dataclasses import dataclass
from math import gcd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import dct
from scipy.ndimage import uniform_filter1d
from scipy.signal import resample_poly, butter, sosfiltfilt

SR = 16000
NFFT = 512
_FB = None


class SilentAudioError(ValueError):
    pass


def _mel_fb(n_mel=26, fmin=40.0, fmax=7800.0):
    global _FB
    if _FB is not None:
        return _FB
    hz2mel = lambda f: 2595.0 * np.log10(1 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mel + 2))
    freqs = np.fft.rfftfreq(NFFT, 1.0 / SR)
    fb = np.zeros((n_mel, len(freqs)))
    for i in range(n_mel):
        l, c, r = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.maximum(0, np.minimum((freqs - l) / (c - l), (r - freqs) / (r - c)))
    _FB = fb
    return fb


@dataclass
class Features:
    cls: np.ndarray
    mfcc: np.ndarray
    dmfcc: np.ndarray
    energy_db: np.ndarray
    f0: np.ndarray
    bnd: np.ndarray
    hop_s: float
    dur_s: float
    ref_db: float
    fine_e: np.ndarray = None   # 1ms 刻みのエネルギー(dB), 4ms 窓
    fine_h: np.ndarray = None   # 1ms 刻みの高域強調比(dB)

    @property
    def T(self):
        return self.cls.shape[0]


def extract(x, sr, hop_ms=5.0, win_ms=25.0):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()                                   # DC 除去
    if sr != SR:
        g = gcd(int(sr), SR)
        x16 = resample_poly(x, SR // g, int(sr) // g)
    else:
        x16 = x
    dur_s = len(x) / float(sr)
    if len(x16) < 400:
        x16 = np.pad(x16, (0, 400 - len(x16)))
    if np.max(np.abs(x16)) < 1e-7:
        raise SilentAudioError("無音のファイルです")
    x16 = sosfiltfilt(butter(2, 40.0, "highpass", fs=SR, output="sos"), x16)   # 低域ノイズ除去
    hop = int(round(hop_ms * SR / 1000))
    win = int(round(win_ms * SR / 1000))
    win += win % 2
    xp = np.pad(x16, (win // 2, win // 2))
    fr = sliding_window_view(xp, win)[::hop]
    T = fr.shape[0]
    w = np.hanning(win)
    fw = fr * w
    spec = np.fft.rfft(fw, NFFT, axis=1)
    P = (spec.real ** 2 + spec.imag ** 2) / win
    freqs = np.fft.rfftfreq(NFFT, 1.0 / SR)
    energy_db = 10 * np.log10(np.mean(fw ** 2, axis=1) + 1e-12)
    tot = P.sum(1) + 1e-12
    hf = P[:, freqs >= 4000].sum(1) / tot
    lf = P[:, freqs < 600].sum(1) / tot
    cent = (P * freqs).sum(1) / tot / 8000.0
    zc = np.abs(np.diff(np.signbit(fr).astype(np.int8), axis=1)).mean(1)
    n2 = 1024
    S = np.fft.rfft(fw, n2, axis=1)
    R = np.fft.irfft(S.real ** 2 + S.imag ** 2, n2, axis=1)[:, :260]
    rw = np.fft.irfft(np.abs(np.fft.rfft(w, n2)) ** 2, n2)[:260]
    Rn = R / (R[:, :1] + 1e-12) / np.maximum(rw / rw[0], 0.5)
    lo, hi = SR // 400, SR // 70
    seg = Rn[:, lo:hi]
    k = np.argmax(seg, axis=1)
    voi = np.clip(seg[np.arange(T), k], 0, 1)
    f0 = SR / (k + lo)
    voi[energy_db < -90] = 0.0
    Ps = uniform_filter1d(P, 5, axis=1)
    m1 = (freqs >= 200) & (freqs <= 1000)
    m2 = (freqs > 1000) & (freqs <= 3200)
    w1, w2 = Ps[:, m1] ** 2, Ps[:, m2] ** 2
    f1 = (w1 * freqs[m1]).sum(1) / (w1.sum(1) + 1e-12) / 1000.0
    f2 = (w2 * freqs[m2]).sum(1) / (w2.sum(1) + 1e-12) / 3200.0
    # 音量・ノイズフロアに適応した相対エネルギー
    ref = float(np.percentile(energy_db, 95))
    floor = float(np.percentile(energy_db, 5))
    rng = max(ref - floor, 30.0)
    e = energy_db - ref
    low = e < -20.0
    e[low] = -20.0 + (e[low] + 20.0) * (35.0 / (rng - 20.0))   # 下端(無音)が常に約 -55dB になるよう正規化
    e_rel = np.clip(e, -80, 5) / 10.0
    cls = np.stack([e_rel, voi, zc, hf, lf, cent, f1, f2], axis=1).astype(np.float32)
    fb = _mel_fb()
    mel = np.log(P @ fb.T + 1e-10)
    mfcc = dct(mel, type=2, axis=1, norm="ortho")[:, 1:13].astype(np.float32)
    idx = np.arange(T)
    dm = ((mfcc[np.minimum(idx + 2, T - 1)] - mfcc[np.maximum(idx - 2, 0)]) / 4.0).astype(np.float32)
    bnd = _boundary_evidence(mfcc, energy_db)
    # 1ms 微細包絡
    xf = np.pad(x16, (0, max(0, 80 - len(x16))))
    fv = sliding_window_view(xf, 64)[::16]
    fe = 10 * np.log10(np.mean(fv ** 2, axis=1) + 1e-12)
    dv = np.diff(fv, axis=1)
    fh = 10 * np.log10(np.mean(dv ** 2, axis=1) + 1e-14) - fe
    return Features(cls, mfcc, dm, energy_db.astype(np.float32), f0.astype(np.float32), bnd,
                    hop / SR, dur_s, ref, fe.astype(np.float32), fh.astype(np.float32))


def _boundary_evidence(mfcc, energy_db):
    T = len(energy_db)
    m = (mfcc - mfcc.mean(0)) / (mfcc.std(0) + 1e-6)
    idx = np.arange(T)
    flux = np.linalg.norm(m[np.minimum(idx + 2, T - 1)] - m[np.maximum(idx - 2, 0)], axis=1)
    flux = np.convolve(np.pad(flux, 1, mode="edge"), [0.25, 0.5, 0.25], "valid")
    flux = np.clip(flux / (np.percentile(flux, 90) + 1e-6), 0, 2)
    es = uniform_filter1d(energy_db, 3, mode="nearest")
    L = 10
    pad = np.pad(es, (L, L), mode="constant", constant_values=es.min())
    left = np.max(np.stack([pad[L - k: L - k + T] for k in range(1, L + 1)]), axis=0)
    right = np.max(np.stack([pad[L + k: L + k + T] for k in range(1, L + 1)]), axis=0)
    dip = np.clip((np.minimum(left, right) - es) / 10.0, 0, 2)
    return (0.6 * flux + 1.0 * dip).astype(np.float32)
