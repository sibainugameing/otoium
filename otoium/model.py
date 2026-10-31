"""自己教師あり音響モデル: 音素ユニットごとの対角ガウス (クラス事前分布 + MAP 再推定)"""
import numpy as np
from .phoneset import UNITS, UNIT_CLASS, CLASS_PRIOR, VOWEL_F, N_CLS, N_MFCC, N_DELTA, D

LL_FLOOR = 12.0


class AcousticModel:
    def __init__(self, mfcc_weight=0.35, delta_weight=0.25):
        self.units = list(UNITS)
        self.idx = {u: i for i, u in enumerate(self.units)}
        U = len(self.units)
        self.prior_mu = np.zeros((U, D))
        self.prior_var = np.ones((U, D)) * 1.5
        for u, i in self.idx.items():
            mean, std = CLASS_PRIOR[UNIT_CLASS[u]][:2]
            self.prior_mu[i, :N_CLS] = mean
            self.prior_var[i, :N_CLS] = np.square(std)
            if u in VOWEL_F:
                self.prior_mu[i, 6:8] = VOWEL_F[u]
                self.prior_var[i, 6:8] = 0.12 ** 2
        self.mu = self.prior_mu.copy()
        self.var = self.prior_var.copy()
        self.cmean = np.zeros(N_MFCC)
        self.cstd = np.ones(N_MFCC)
        self.dstd = np.ones(N_DELTA)
        self.w = np.r_[np.ones(N_CLS), np.full(N_MFCC, mfcc_weight), np.full(N_DELTA, delta_weight)]
        self.trained = False

    def set_norm(self, feats):
        rows = [(f.mfcc[f.cls[:, 0] > -3.5], f.dmfcc[f.cls[:, 0] > -3.5]) for f in feats if f is not None]
        rows = [r for r in rows if len(r[0])]
        if not rows:
            return
        a = np.concatenate([r[0] for r in rows])
        d = np.concatenate([r[1] for r in rows])
        self.cmean, self.cstd = a.mean(0), a.std(0) + 1e-6
        self.dstd = d.std(0) + 1e-6

    def design(self, F):
        return np.hstack([F.cls, (F.mfcc - self.cmean) / self.cstd, F.dmfcc / self.dstd]).astype(np.float64)

    def loglik(self, F, X=None):
        if X is None:
            X = self.design(F)
        U, T = len(self.units), X.shape[0]
        LL = np.empty((U, T))
        for k in range(U):
            z = (X - self.mu[k]) / np.sqrt(self.var[k])
            lp = -0.5 * z * z - 0.5 * np.log(self.var[k]) - 0.9189
            np.maximum(lp, -LL_FLOOR, out=lp)
            LL[k] = lp @ self.w
        return LL

    def fit(self, items, tau_cls=20.0, tau_mfcc=10.0):
        """items: [(X, [(unit, s, e), ...], weight)] または [(X, segs)]"""
        U = len(self.units)
        n = np.zeros(U)
        s1 = np.zeros((U, D))
        s2 = np.zeros((U, D))
        for it in items:
            X, segs = it[0], it[1]
            wt = it[2] if len(it) > 2 else 1.0
            for u, s, e in segs:
                L = e - s
                if L <= 0:
                    continue
                a, b = (s + int(0.2 * L), e - int(0.2 * L)) if L >= 5 else (s, e)
                if b <= a:
                    a, b = s, e
                x = X[a:b]
                k = self.idx[u]
                n[k] += wt * len(x)
                s1[k] += wt * x.sum(0)
                s2[k] += wt * (x * x).sum(0)
        pool = {}
        for c in sorted(set(UNIT_CLASS.values())):
            ks = [self.idx[u] for u in self.units if UNIT_CLASS[u] == c]
            nn = n[ks].sum()
            if nn >= 30:
                m = s1[ks].sum(0) / nn
                pool[c] = (m, np.maximum(s2[ks].sum(0) / nn - m * m, 0.05))
        tau = np.r_[np.full(N_CLS, tau_cls), np.full(N_MFCC + N_DELTA, tau_mfcc)]
        for u, k in self.idx.items():
            if n[k] < 3:
                continue
            m = s1[k] / n[k]
            v = np.maximum(s2[k] / n[k] - m * m, 1e-6)
            pm, pv = self.prior_mu[k].copy(), self.prior_var[k].copy()
            c = UNIT_CLASS[u]
            if c in pool:
                pm[N_CLS:], pv[N_CLS:] = pool[c][0][N_CLS:], pool[c][1][N_CLS:]
            nm = (n[k] * m + tau * pm) / (n[k] + tau)
            nv = (n[k] * (v + (m - nm) ** 2) + tau * (pv + (pm - nm) ** 2)) / (n[k] + tau)
            floor = np.r_[(0.15 * np.sqrt(self.prior_var[k, :N_CLS])) ** 2, np.full(N_MFCC + N_DELTA, 0.15)]
            self.mu[k], self.var[k] = nm, np.maximum(nv, floor)
        self.trained = True

    def save(self, path):
        np.savez(path, mu=self.mu, var=self.var, cmean=self.cmean, cstd=self.cstd, dstd=self.dstd,
                 units=np.array(self.units), w=self.w)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        m = cls()
        if list(z["units"]) != m.units or z["mu"].shape != m.mu.shape:
            raise ValueError("モデルの構成が現在のバージョンと一致しません")
        m.mu, m.var, m.cmean, m.cstd, m.dstd, m.w = z["mu"], z["var"], z["cmean"], z["cstd"], z["dstd"], z["w"]
        m.trained = True
        return m
