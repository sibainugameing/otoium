"""自己教師あり音響モデル: 音素ユニットごとの対角ガウス (クラス事前分布 + MAP 再推定)"""
import numpy as np
from .phoneset import (UNITS, UNIT_CLASS, CLASS_PRIOR, VOWEL_F, N_CLS, N_MFCC, D)

LL_FLOOR = 12.0


class AcousticModel:
    def __init__(self, mfcc_weight=0.35):
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
        self.w = np.r_[np.ones(N_CLS), np.full(N_MFCC, mfcc_weight)]
        self.trained = False

    # ---- 正規化 ----
    def set_norm(self, feats):
        rows = [f.mfcc[f.cls[:, 0] > -3.5] for f in feats if f is not None]
        rows = [r for r in rows if len(r)]
        if not rows:
            return
        a = np.concatenate(rows)
        self.cmean = a.mean(0)
        self.cstd = a.std(0) + 1e-6

    def design(self, F):
        return np.hstack([F.cls, (F.mfcc - self.cmean) / self.cstd]).astype(np.float64)

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

    # ---- 再推定 ----
    def fit(self, items, tau_cls=20.0, tau_mfcc=10.0):
        """items: [(X, [(unit, s, e), ...])]"""
        U = len(self.units)
        n = np.zeros(U)
        s1 = np.zeros((U, D))
        s2 = np.zeros((U, D))
        for X, segs in items:
            for u, s, e in segs:
                L = e - s
                if L <= 0:
                    continue
                a, b = (s + int(0.2 * L), e - int(0.2 * L)) if L >= 5 else (s, e)
                if b <= a:
                    a, b = s, e
                x = X[a:b]
                k = self.idx[u]
                n[k] += len(x)
                s1[k] += x.sum(0)
                s2[k] += (x * x).sum(0)
        # クラスプール
        cls_names = sorted(set(UNIT_CLASS.values()))
        pool = {}
        for c in cls_names:
            ks = [self.idx[u] for u in self.units if UNIT_CLASS[u] == c]
            nn = n[ks].sum()
            if nn >= 30:
                m = s1[ks].sum(0) / nn
                v = s2[ks].sum(0) / nn - m * m
                pool[c] = (m, np.maximum(v, 0.05))
        for u, k in self.idx.items():
            if n[k] < 3:
                continue
            m = s1[k] / n[k]
            v = np.maximum(s2[k] / n[k] - m * m, 1e-6)
            pm = self.prior_mu[k].copy()
            pv = self.prior_var[k].copy()
            c = UNIT_CLASS[u]
            if c in pool:
                pm[N_CLS:] = pool[c][0][N_CLS:]
                pv[N_CLS:] = pool[c][1][N_CLS:]
            tau = np.r_[np.full(N_CLS, tau_cls), np.full(N_MFCC, tau_mfcc)]
            nm = (n[k] * m + tau * pm) / (n[k] + tau)
            nv = (n[k] * (v + (m - nm) ** 2) + tau * (pv + (pm - nm) ** 2)) / (n[k] + tau)
            floor = np.r_[(0.15 * np.sqrt(self.prior_var[k, :N_CLS])) ** 2, np.full(N_MFCC, 0.15)]
            self.mu[k] = nm
            self.var[k] = np.maximum(nv, floor)
        self.trained = True

    # ---- 保存 ----
    def save(self, path):
        np.savez(path, mu=self.mu, var=self.var, cmean=self.cmean, cstd=self.cstd,
                 units=np.array(self.units), w=self.w)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        m = cls()
        if list(z["units"]) != m.units:
            raise ValueError("モデルのユニット構成が一致しません")
        m.mu, m.var, m.cmean, m.cstd, m.w = z["mu"], z["var"], z["cmean"], z["cstd"], z["w"]
        m.trained = True
        return m
