"""numpy だけのフレーム分類ニューラルネット (ReLU MLP, Adam)。アライメント結果で自己教師あり学習する"""
import numpy as np


class FrameNet:
    def __init__(self, n_in, n_out, hidden=(128, 64), seed=0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, n_out]
        self.W = [(rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, dtype=np.float32) for b in sizes[1:]]
        self.mean = np.zeros(n_in)
        self.std = np.ones(n_in)
        self.logprior = np.full(n_out, -np.log(n_out))

    def _forward(self, X):
        acts = [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0.0)
            acts.append(h)
        return acts

    def logpost(self, X):
        Z = self._forward((X - self.mean) / self.std)[-1]
        Z = Z - Z.max(1, keepdims=True)
        return Z - np.log(np.exp(Z).sum(1, keepdims=True))

    def fit(self, X, y, w, epochs=6, lr=2e-3, batch=512, smooth=0.05, seed=0, reset_norm=True, callback=None):
        n_out = self.W[-1].shape[1]
        if reset_norm:
            self.mean = X.mean(0)
            self.std = X.std(0) + 1e-6
            cnt = np.bincount(y, weights=w, minlength=n_out) + 1.0
            self.logprior = np.log(cnt / cnt.sum())
        Xn = ((X - self.mean) / self.std).astype(np.float32)
        w = np.asarray(w, dtype=np.float32)
        rng = np.random.default_rng(seed)
        m = [np.zeros_like(p) for p in self.W + self.b]
        v = [np.zeros_like(p) for p in self.W + self.b]
        t = 0
        N = len(Xn)
        loss = 0.0
        for ep in range(epochs):
            order = rng.permutation(N)
            loss, cnt_b, correct, wsum = 0.0, 0, 0.0, 0.0
            for s in range(0, N, batch):
                ix = order[s:s + batch]
                xb, yb, wb = Xn[ix], y[ix], w[ix]
                acts = self._forward(xb)
                Z = acts[-1] - acts[-1].max(1, keepdims=True)
                P = np.exp(Z)
                P /= P.sum(1, keepdims=True)
                T = np.full_like(P, smooth / n_out, dtype=np.float32)
                T[np.arange(len(yb)), yb] += 1.0 - smooth
                loss += float(-(wb * (T * np.log(P + 1e-12)).sum(1)).mean())
                cnt_b += 1
                correct += float((wb * (P.argmax(1) == yb)).sum())
                wsum += float(wb.sum())
                g = (P - T) * (wb[:, None] / len(yb))
                gW, gb = [None] * len(self.W), [None] * len(self.W)
                for i in range(len(self.W) - 1, -1, -1):
                    gW[i] = acts[i].T @ g + 1e-5 * self.W[i]
                    gb[i] = g.sum(0)
                    if i > 0:
                        g = (g @ self.W[i].T) * (acts[i] > 0)
                t += 1
                params = self.W + self.b
                grads = gW + gb
                for k, (p_, g_) in enumerate(zip(params, grads)):
                    m[k] = 0.9 * m[k] + 0.1 * g_
                    v[k] = 0.999 * v[k] + 0.001 * g_ * g_
                    p_ -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            lr *= 0.8
            loss /= max(1, cnt_b)
            if callback:
                callback(ep + 1, loss, correct / max(wsum, 1e-9))
        return loss

    def state(self):
        d = {"mean": self.mean, "std": self.std, "logprior": self.logprior, "nl": np.array(len(self.W))}
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            d[f"W{i}"], d[f"b{i}"] = W, b
        return d

    @classmethod
    def from_state(cls, z):
        nl = int(z["nl"])
        net = cls.__new__(cls)
        net.W = [np.array(z[f"W{i}"], dtype=np.float32) for i in range(nl)]
        net.b = [np.array(z[f"b{i}"], dtype=np.float32) for i in range(nl)]
        net.mean, net.std, net.logprior = np.array(z["mean"]), np.array(z["std"]), np.array(z["logprior"])
        return net
