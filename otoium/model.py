"""自己教師あり音響モデル: 音素ユニットごとの対角ガウス (クラス事前分布 + MAP 再推定)"""
import copy
import numpy as np
from .mlp import FrameNet
import math
from .phoneset import UNITS, UNIT_CLASS, CLASS_PRIOR, VOWEL_F, N_CLS, N_MFCC, N_DELTA, D, DUR_SIGMA

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
        self.nets = []
        self.nn_weight = 0.0
        self.hop_ms = 5.0
        # 音素ごとの時間長分布 (対数ms の平均/標準偏差, 最大ms): 事前分布から開始し学習で話者の速度に適応
        self.dur_mu, self.dur_sd, self.dur_max = {}, {}, {}
        for u in self.units:
            c = UNIT_CLASS[u]
            _, _, _, dmax, dmean = CLASS_PRIOR[c]
            self.dur_mu[u] = math.log(dmean)
            self.dur_sd[u] = DUR_SIGMA.get(c, 0.7)
            self.dur_max[u] = float(dmax)

    @property
    def net(self):
        return self.nets[0] if self.nets else None

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
        if self.nets and self.nn_weight > 0:
            S = self.splice(X)
            lp = np.mean([n.logpost(S) - n.logprior for n in self.nets], axis=0)
            LL += self.nn_weight * np.clip(lp, -12.0, 12.0).T
        return LL

    @staticmethod
    def splice(X, k=3):
        T = len(X)
        i = np.arange(T)
        return np.hstack([X[np.maximum(i - k, 0)], X, X[np.minimum(i + k, T - 1)]])

    def fit_net(self, items, epochs=6, max_frames=250000, seed=0, ensemble=1):
        """items: [(X, segs, w)] の区間をラベルに、フレーム分類器 (アンサンブル) を学習"""
        xs, ys, ws = [], [], []
        for it in items:
            X, segs = it[0], it[1]
            wt = it[2] if len(it) > 2 else 1.0
            S = self.splice(X)
            for u, s, e in segs:
                L = e - s
                if L <= 0:
                    continue
                a, b = (s + int(0.15 * L), e - int(0.15 * L)) if L >= 5 else (s, e)
                if b <= a:
                    a, b = s, e
                xs.append(S[a:b])
                ys.append(np.full(b - a, self.idx[u]))
                ws.append(np.full(b - a, wt))
        if not xs:
            return None
        X, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
        fresh = not self.nets
        if fresh:
            self.nets = [FrameNet(X.shape[1], len(self.units), seed=seed + k) for k in range(max(1, ensemble))]
        losses = []
        for k, net in enumerate(self.nets):
            if len(X) > max_frames:
                ix = np.random.default_rng(seed + k).choice(len(X), max_frames, replace=False)
                Xk, yk, wk = X[ix], y[ix], w[ix]
            else:
                Xk, yk, wk = X, y, w
            losses.append(net.fit(Xk, yk, wk, epochs=epochs if fresh else max(2, epochs // 3),
                                  lr=2e-3 if fresh else 5e-4, reset_norm=fresh, seed=seed + k))
        return float(np.mean(losses))

    def update_durations(self, items, tau=5.0):
        """整列結果から音素ごとの時間長分布を MAP 推定 (ms)。話者の発話速度に適応する"""
        acc = {}
        for it in items:
            for u, s, e in it[1]:
                if u != "pau" and e > s:
                    acc.setdefault(u, []).append(math.log(max((e - s) * self.hop_ms, 1.0)))
        for u, v in acc.items():
            n = len(v)
            if n < 4:
                continue
            m, var = float(np.mean(v)), float(np.var(v))
            pm, ps = self.dur_mu[u], self.dur_sd[u]
            nm = (n * m + tau * pm) / (n + tau)
            nv = (n * (var + (m - nm) ** 2) + tau * (ps ** 2 + (pm - nm) ** 2)) / (n + tau)
            self.dur_mu[u] = nm
            self.dur_sd[u] = float(np.clip(math.sqrt(nv), 0.25, 0.9))
            self.dur_max[u] = max(self.dur_max[u], float(np.exp(np.max(v))) * 1.3)

    def adapt(self, X, segs, tau=40.0):
        """1ファイルの整列結果に平均・分散を寄せた複製を返す (話者/マイク/部屋への適応)"""
        m = copy.copy(self)
        m.mu, m.var = self.mu.copy(), self.var.copy()
        acc = {}
        for u, s, e in segs:
            L = e - s
            if L <= 0:
                continue
            a, b = (s + int(0.2 * L), e - int(0.2 * L)) if L >= 5 else (s, e)
            acc.setdefault(u, []).append(X[a:max(b, a + 1)])
        for u, xs in acc.items():
            x = np.concatenate(xs)
            if len(x) < 3:
                continue
            k = self.idx[u]
            mm, vv = x.mean(0), np.maximum(x.var(0), 1e-6)
            n = len(x)
            nm = (n * mm + tau * self.mu[k]) / (n + tau)
            nv = (n * (vv + (mm - nm) ** 2) + tau * (self.var[k] + (self.mu[k] - nm) ** 2)) / (n + tau)
            m.mu[k], m.var[k] = nm, np.maximum(nv, 0.5 * self.var[k])
        return m

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
        self.update_durations(items)
        self.trained = True

    def save(self, path):
        d = dict(mu=self.mu, var=self.var, cmean=self.cmean, cstd=self.cstd, dstd=self.dstd,
                 units=np.array(self.units), w=self.w, nn_weight=np.array(self.nn_weight),
                 dur=np.array([[self.dur_mu[u], self.dur_sd[u], self.dur_max[u]] for u in self.units]))
        for k, net in enumerate(self.nets):
            d.update({f"net{k}_" + key: v for key, v in net.state().items()})
        np.savez(path, **d)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        m = cls()
        if list(z["units"]) != m.units or z["mu"].shape != m.mu.shape:
            raise ValueError("モデルの構成が現在のバージョンと一致しません")
        m.mu, m.var, m.cmean, m.cstd, m.dstd, m.w = z["mu"], z["var"], z["cmean"], z["cstd"], z["dstd"], z["w"]
        m.trained = True
        if "dur" in z.files:
            for u, (a, b, c) in zip(m.units, z["dur"]):
                m.dur_mu[u], m.dur_sd[u], m.dur_max[u] = float(a), float(b), float(c)
        k = 0
        while f"net{k}_nl" in z.files:
            pre = f"net{k}_"
            m.nets.append(FrameNet.from_state({key[len(pre):]: z[key] for key in z.files if key.startswith(pre)}))
            k += 1
        if m.nets:
            m.nn_weight = float(z["nn_weight"])
        return m
