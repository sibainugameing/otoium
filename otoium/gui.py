"""tkinter GUI: 設定 / 結果 (波形・音素境界・oto パラメータの可視化) / ログ"""
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from .config import Config, ACCURACY_MODES, resolve
from .pipeline import Pipeline, Cancelled
from .phoneset import UNIT_CLASS
from .metrics import INFO

STATUS_FG = {"OK": "#1b7f3b", "WARN": "#b7791f", "REVIEW": "#c05621", "ERROR": "#c53030", "FAIL": "#c53030"}
CLASS_BG = {"V": "#9ec5f4", "NN": "#b4bdf2", "N": "#cdb4f2", "S": "#f4cf9e", "H": "#f4e29e", "Z": "#f4b99e",
            "CL": "#d0d0d0", "CLV": "#d8c4c4", "BR": "#f49eb4", "BRV": "#e6a8c8", "RF": "#9ef4dc", "GL": "#9ef4be",
            "PAU": "#eeeeee"}
MARKS = [("offset", "#c53030"), ("overlap", "#dd8a00"), ("pre", "#1b7f3b"), ("consonant", "#2b6cb0"), ("cutoff", "#7b3fb0")]


def _nice_step(span_ms, target=8):
    raw = max(span_ms / target, 1e-3)
    for s in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
        if s >= raw:
            return s
    return 10000


class MetricChart(tk.Canvas):
    """学習曲線 (得点の推移)。段階が変わる位置に点線とラベル、最良点に緑の丸を描く"""

    def __init__(self, master, title, better="up", **kw):
        super().__init__(master, bg="#ffffff", highlightthickness=1, highlightbackground="#cccccc", **kw)
        self.title, self.better = title, better
        self.vals, self.stages = [], []
        self.bind("<Configure>", lambda e: self.redraw())

    def clear(self):
        self.vals, self.stages = [], []
        self.redraw()

    def add(self, v, stage=""):
        self.vals.append(float(v))
        self.stages.append(stage)
        self.redraw()

    def best(self):
        if not self.vals:
            return None
        return max(self.vals) if self.better == "up" else min(self.vals)

    def redraw(self):
        self.delete("all")
        W, H = max(self.winfo_width(), 120), max(self.winfo_height(), 90)
        L, R, T, B = 48, 10, 24, 16
        v = self.vals
        head = self.title + (f"   現在 {v[-1]:.3f}   最良 {self.best():.3f}" if v else "   (待機中)")
        self.create_text(8, 11, text=head, anchor="w", font=("", 9, "bold"), fill="#222222")
        if not v:
            return
        lo, hi = min(v), max(v)
        if hi - lo < 1e-9:
            lo, hi = lo - 0.5, hi + 0.5
        pad = (hi - lo) * 0.1
        lo, hi = lo - pad, hi + pad
        pw, ph = W - L - R, H - T - B
        n = len(v)
        X = lambda i: L + (pw / 2 if n == 1 else i * pw / (n - 1))
        Y = lambda val: T + (hi - val) / (hi - lo) * ph
        for k in range(4):
            val = lo + (hi - lo) * k / 3
            y = Y(val)
            self.create_line(L, y, W - R, y, fill="#eeeeee")
            self.create_text(L - 4, y, text=f"{val:.2f}", anchor="e", font=("", 7), fill="#777777")
        last = None
        for i, st in enumerate(self.stages):
            g = st.split()[0] if st else ""
            if g != last and i > 0:
                self.create_line(X(i), T, X(i), H - B, fill="#cccccc", dash=(3, 3))
                self.create_text(X(i) + 2, H - 7, text=g, anchor="w", font=("", 7), fill="#999999")
            last = g
        if n >= 2:
            self.create_line(*[c for i in range(n) for c in (X(i), Y(v[i]))], fill="#2b6cb0", width=2)
        if n <= 80:
            for i in range(n):
                self.create_oval(X(i) - 2, Y(v[i]) - 2, X(i) + 2, Y(v[i]) + 2, fill="#2b6cb0", outline="")
        b = v.index(self.best())
        self.create_oval(X(b) - 4, Y(v[b]) - 4, X(b) + 4, Y(v[b]) + 4, fill="#1b7f3b", outline="white")


class WaveView(tk.Canvas):
    """波形 + 音素区間 + モーラ + 選択エイリアスの oto マーカー。ホイールで拡大縮小、ドラッグで移動。"""

    def __init__(self, master, **kw):
        super().__init__(master, bg="#ffffff", highlightthickness=1, highlightbackground="#bbbbbb", **kw)
        self.job = None
        self.entry = None
        self.t0, self.t1 = 0.0, 1.0
        self._drag = None
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Button-4>", lambda e: self._zoom(e.x, 0.8))
        self.bind("<Button-5>", lambda e: self._zoom(e.x, 1.25))
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._move)
        self.bind("<Double-Button-1>", lambda e: self.reset())

    def set_job(self, job):
        self.job, self.entry = job, None
        self.reset()

    def set_entry(self, e):
        self.entry = e
        self.redraw()

    def reset(self):
        self.t0, self.t1 = 0.0, (self.job.dur_ms if self.job else 1.0)
        self.redraw()

    def _x(self, ms):
        w = max(self.winfo_width(), 50)
        return (ms - self.t0) / max(self.t1 - self.t0, 1e-6) * w

    def _zoom(self, x, f):
        if not self.job:
            return
        w = max(self.winfo_width(), 50)
        c = self.t0 + x / w * (self.t1 - self.t0)
        span = min(max((self.t1 - self.t0) * f, 20.0), self.job.dur_ms)
        self.t0 = max(0.0, c - (c - self.t0) * span / (self.t1 - self.t0))
        self.t1 = min(self.job.dur_ms, self.t0 + span)
        self.t0 = max(0.0, self.t1 - span)
        self.redraw()

    def _wheel(self, e):
        self._zoom(e.x, 0.8 if e.delta > 0 else 1.25)

    def _press(self, e):
        self._drag = (e.x, self.t0, self.t1)

    def _move(self, e):
        if not self._drag or not self.job:
            return
        x0, a, b = self._drag
        w = max(self.winfo_width(), 50)
        d = (x0 - e.x) / w * (b - a)
        d = max(-a, min(d, self.job.dur_ms - b))
        self.t0, self.t1 = a + d, b + d
        self.redraw()

    def redraw(self):
        self.delete("all")
        j = self.job
        W, H = max(self.winfo_width(), 50), max(self.winfo_height(), 120)
        if j is None or j.env is None:
            self.create_text(W / 2, H / 2, text="ファイルを選択してください", fill="#888888")
            return
        top, bot = 24, H - 38
        mid = (top + bot) / 2
        amp = (bot - top) / 2 * 0.95
        env = j.env
        n = env.shape[1]
        per = j.dur_ms / n
        for i in range(n):
            x = self._x((i + 0.5) * per)
            if 0 <= x <= W:
                self.create_line(x, mid - env[1, i] * amp, x, mid - env[0, i] * amp - 1, fill="#6b7280")
        # 時間軸
        step = _nice_step(self.t1 - self.t0)
        t = int(self.t0 // step) * step
        while t <= self.t1:
            x = self._x(t)
            self.create_line(x, H - 14, x, H - 8, fill="#888888")
            self.create_text(x + 2, H - 4, text=f"{t:g}", anchor="sw", font=("", 7), fill="#666666")
            t += step
        # 音素区間
        for s in j.segs:
            if s.e <= s.s or s.s_ms is None:
                continue
            x0, x1 = self._x(s.s_ms), self._x(s.e_ms)
            if x1 < 0 or x0 > W:
                continue
            col = CLASS_BG.get(UNIT_CLASS[s.unit], "#cccccc")
            self.create_rectangle(x0, bot, x1, bot + 18, fill=col, outline="#ffffff")
            if s.unit != "pau" and x1 - x0 > 12:
                self.create_text((x0 + x1) / 2, bot + 9, text=s.unit, font=("", 8))
            self.create_line(x0, top, x0, bot, fill="#d4d4d8", dash=(2, 3))
        # モーラ
        for t_ in j.timings:
            x0, x1 = self._x(t_.cons_start), self._x(t_.vowel_end)
            if x1 < 0 or x0 > W:
                continue
            self.create_rectangle(x0, 2, x1, 20, outline=STATUS_FG.get(t_.status, "#888"), width=2)
            self.create_text((x0 + x1) / 2, 11, text=t_.mora.surface, font=("", 10, "bold"),
                             fill=STATUS_FG.get(t_.status, "#333"))
        # oto マーカー
        e = self.entry
        if e is not None:
            pos = {"offset": e.offset, "overlap": e.offset + e.overlap, "pre": e.offset + e.pre,
                   "consonant": e.offset + e.consonant, "cutoff": e.offset + e.cut_len}
            for k, (name, col) in enumerate(MARKS):
                x = self._x(pos[name])
                self.create_line(x, top, x, bot, fill=col, width=2)
                self.create_text(x + 3, top + 10 + 12 * k, text=name, anchor="w", fill=col, font=("", 8))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("otoium")
        self.geometry("1280x820")
        self.minsize(980, 640)
        try:
            ttk.Style(self).theme_use("clam")
        except tk.TclError:
            pass
        st = ttk.Style(self)
        st.configure("Treeview", rowheight=22)
        st.configure("Accent.TButton", font=("", 10, "bold"))
        self.q = queue.Queue()
        self.pipe = None
        self.thread = None
        self.V = {}
        self._build_vars()
        self._build_top()
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._build_settings()
        self._build_training()
        self._build_results()
        self._build_log()
        self.after(100, self._poll)

    # ---------------- 変数 ----------------
    def _build_vars(self):
        d = Config()
        str_vars = dict(folder="", script="", mode=d.mode, vcv_style=d.vcv_style, vc_style=d.vc_style,
                        encoding=d.encoding, accuracy=d.accuracy, head_prefix="", alias_prefix="", alias_suffix="")
        for k, v in str_vars.items():
            self.V[k] = tk.StringVar(value=v)
        self.V["jobs"] = tk.IntVar(value=1)
        for k in ("add_tail", "strict", "dry_run", "merge", "textgrid", "export_labels"):
            self.V[k] = tk.BooleanVar(value=False)
        for k in ("refine", "tune", "rescue", "net", "adapt", "auto_thresholds", "recursive", "backup", "dedupe", "cache"):
            self.V[k] = tk.BooleanVar(value=True)
        for k in ("thr_contrast", "thr_bnd", "ok_agree", "fail_agree", "cons_review", "cons_warn"):
            self.V[k] = tk.DoubleVar(value=getattr(d, k))
        for k, val in dict(c_iters=5, c_rounds=1, c_epochs=10, c_ens=1, c_adapt=1, c_post=2, c_conf=0).items():
            self.V[k] = tk.IntVar(value=val)
        self.V["mode_info"] = tk.StringVar(value="")
        self.V["recheck"] = tk.IntVar(value=3)
        self.V["filter"] = tk.StringVar(value="全て")
        self.V["stage"] = tk.StringVar(value="待機中")

    def collect_cfg(self):
        v = {k: x.get() for k, x in self.V.items()}
        custom = {}
        if v["accuracy"] == "custom":
            custom = dict(train_iters=int(v["c_iters"]), tune_rounds=int(v["c_rounds"]), net_epochs=int(v["c_epochs"]),
                          net_ensemble=int(v["c_ens"]), adapt_rounds=int(v["c_adapt"]), post_iters=int(v["c_post"]),
                          confident_rounds=int(v["c_conf"]))
        return Config(
            **custom, mode=v["mode"], vcv_style=v["vcv_style"], vc_style=v["vc_style"], head_prefix=v["head_prefix"],
            add_tail=v["add_tail"], encoding=v["encoding"], accuracy=v["accuracy"], script_path=v["script"],
            auto_tune=None if v["tune"] else False, rescue=None if v["rescue"] else False,
            use_net=None if v["net"] else False, adapt=None if v["adapt"] else False, refine=v["refine"],
            auto_thresholds=v["auto_thresholds"], thr_contrast=v["thr_contrast"], thr_bnd=v["thr_bnd"],
            ok_agree=v["ok_agree"], fail_agree=v["fail_agree"], cons_review=v["cons_review"], cons_warn=v["cons_warn"],
            strict=v["strict"], dry_run=v["dry_run"], merge=v["merge"], textgrid=v["textgrid"],
            export_labels=v["export_labels"], alias_prefix=v["alias_prefix"], alias_suffix=v["alias_suffix"],
            jobs=max(1, int(v["jobs"] or 1)), recursive=v["recursive"], backup=v["backup"], dedupe=v["dedupe"],
            cache=v["cache"], recheck_rounds=int(v["recheck"]))

    # ---------------- 画面 ----------------
    def _build_top(self):
        f = ttk.Frame(self, padding=8)
        f.pack(fill="x")
        ttk.Label(f, text="原音フォルダ").grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.V["folder"]).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(f, text="参照…", command=self.pick_dir).grid(row=0, column=2)
        self.btn_run = ttk.Button(f, text="実行", style="Accent.TButton", command=self.start)
        self.btn_run.grid(row=0, column=3, padx=(10, 4))
        self.btn_stop = ttk.Button(f, text="中止", command=self.stop, state="disabled")
        self.btn_stop.grid(row=0, column=4)
        self.btn_open = ttk.Button(f, text="フォルダを開く", command=self.open_dir)
        self.btn_open.grid(row=0, column=5, padx=4)
        f.columnconfigure(1, weight=1)
        self.bar = ttk.Progressbar(f, maximum=1.0)
        self.bar.grid(row=1, column=0, columnspan=5, sticky="we", pady=(8, 0))
        ttk.Label(f, textvariable=self.V["stage"], width=34, anchor="w").grid(row=1, column=5, sticky="w", padx=6, pady=(8, 0))

    def _combo(self, parent, label, key, values, row, col, width=10):
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 6), pady=3)
        ttk.Combobox(parent, textvariable=self.V[key], values=values, width=width, state="readonly").grid(
            row=row, column=col + 1, sticky="w", pady=3)

    def _check(self, parent, label, key, row, col, span=1):
        ttk.Checkbutton(parent, text=label, variable=self.V[key]).grid(row=row, column=col, columnspan=span, sticky="w", pady=2)

    def _build_settings(self):
        tab = ttk.Frame(self.nb, padding=10)
        self.nb.add(tab, text="設定")
        g1 = ttk.LabelFrame(tab, text="入出力", padding=8)
        g1.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        ttk.Label(g1, text="台本ファイル").grid(row=0, column=0, sticky="w")
        ttk.Entry(g1, textvariable=self.V["script"], width=34).grid(row=0, column=1, columnspan=2, sticky="we", pady=3)
        ttk.Button(g1, text="参照…", command=self.pick_script).grid(row=0, column=3, padx=4)
        self._combo(g1, "モード", "mode", ["auto", "cv", "vcv", "cvvc"], 1, 0)
        self._combo(g1, "VCV表記", "vcv_style", ["mora", "vowel"], 1, 2)
        self._combo(g1, "VC表記", "vc_style", ["romaji", "kana"], 2, 0)
        self._combo(g1, "文字コード", "encoding", ["cp932", "utf-8"], 2, 2)
        for i, (lab, key) in enumerate([("先頭の接頭辞", "head_prefix"), ("接頭辞", "alias_prefix"), ("接尾辞", "alias_suffix")]):
            ttk.Label(g1, text=lab).grid(row=3 + i, column=0, sticky="w")
            ttk.Entry(g1, textvariable=self.V[key], width=12).grid(row=3 + i, column=1, sticky="w", pady=3)
        self._check(g1, "末尾エイリアス「V -」", "add_tail", 6, 0, 2)
        self._check(g1, "既存 oto.ini とマージ", "merge", 6, 2, 2)
        self._check(g1, "サブフォルダも処理", "recursive", 7, 0, 2)
        self._check(g1, "既存を .bak に退避", "backup", 7, 2, 2)
        self._check(g1, "重複エイリアスを整理", "dedupe", 8, 0, 2)
        g2 = ttk.LabelFrame(tab, text="精度・自動調整", padding=8)
        g2.grid(row=0, column=1, sticky="nsew", pady=(0, 8))
        self._combo(g2, "精度", "accuracy", ACCURACY_MODES, 0, 0)
        ttk.Label(g2, text="並列数").grid(row=0, column=2, sticky="w", padx=(12, 6))
        ttk.Spinbox(g2, from_=1, to=32, textvariable=self.V["jobs"], width=5).grid(row=0, column=3, sticky="w")
        self._check(g2, "解析パラメータの自動調整", "tune", 1, 0, 4)
        self._check(g2, "ニューラル音響モデル", "net", 2, 0, 4)
        self._check(g2, "録音ごとの適応", "adapt", 3, 0, 4)
        self._check(g2, "失敗ファイルの救済", "rescue", 4, 0, 4)
        self._check(g2, "1ms 境界微調整", "refine", 5, 0, 4)
        self._check(g2, "しきい値の自動校正", "auto_thresholds", 6, 0, 4)
        self._check(g2, "特徴量キャッシュ", "cache", 7, 0, 4)
        ttk.Label(g2, text="(チェックを外すとその処理を無効化。精度プリセットで無効なものは有効になりません)",
                  foreground="#666666", wraplength=360).grid(row=8, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Label(g2, textvariable=self.V["mode_info"], foreground="#2b6cb0", wraplength=360, justify="left").grid(
            row=9, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Label(g2, text="失敗箇所の再検査 (最大回数)").grid(row=10, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Spinbox(g2, from_=0, to=10, textvariable=self.V["recheck"], width=5).grid(row=10, column=2, sticky="w", pady=(6, 0))
        g5 = ttk.LabelFrame(tab, text="カスタム (精度で custom を選んだときに有効)", padding=8)
        g5.grid(row=2, column=0, columnspan=2, sticky="we", pady=(8, 0))
        self.custom_widgets = []
        spec = [("学習回数", "c_iters", 1, 50), ("自動調整の周回", "c_rounds", 0, 10), ("NNエポック", "c_epochs", 1, 100),
                ("NNアンサンブル数", "c_ens", 1, 8), ("適応の反復", "c_adapt", 0, 10), ("調整後の再学習", "c_post", 0, 20),
                ("高信頼再学習の周回", "c_conf", 0, 10)]
        for i, (lab, key, lo, hi) in enumerate(spec):
            ttk.Label(g5, text=lab).grid(row=i // 4, column=(i % 4) * 2, sticky="w", padx=(0, 6), pady=3)
            sp = ttk.Spinbox(g5, from_=lo, to=hi, textvariable=self.V[key], width=6)
            sp.grid(row=i // 4, column=(i % 4) * 2 + 1, sticky="w", padx=(0, 16))
            self.custom_widgets.append(sp)
        for k in ["accuracy", "tune", "rescue", "net", "adapt"] + [x[1] for x in spec]:
            self.V[k].trace_add("write", lambda *a: self._update_mode_info())
        self._update_mode_info()
        g3 = ttk.LabelFrame(tab, text="検証しきい値 (自動校正が有効なときは初期値として使用)", padding=8)
        g3.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        for i, (lab, key) in enumerate([("境界コントラスト", "thr_contrast"), ("境界証拠", "thr_bnd"), ("母音一致 OK", "ok_agree"),
                                        ("母音一致 失敗", "fail_agree"), ("子音 保留", "cons_review"), ("子音 警告", "cons_warn")]):
            ttk.Label(g3, text=lab).grid(row=i // 2, column=(i % 2) * 2, sticky="w", padx=(0, 6), pady=3)
            ttk.Spinbox(g3, from_=0.0, to=10.0, increment=0.05, textvariable=self.V[key], width=7).grid(
                row=i // 2, column=(i % 2) * 2 + 1, sticky="w", padx=(0, 14))
        g4 = ttk.LabelFrame(tab, text="出力オプション", padding=8)
        g4.grid(row=1, column=1, sticky="nsew")
        self._check(g4, "厳格 (WARN も oto.ini から除外)", "strict", 0, 0)
        self._check(g4, "ドライラン (ファイルを書かない)", "dry_run", 1, 0)
        self._check(g4, "TextGrid を出力", "textgrid", 2, 0)
        self._check(g4, "Audacity ラベルを出力", "export_labels", 3, 0)
        bar = ttk.Frame(tab)
        bar.grid(row=3, column=0, columnspan=2, sticky="w", pady=10)
        ttk.Button(bar, text="設定を保存…", command=self.save_settings).pack(side="left")
        ttk.Button(bar, text="設定を読込…", command=self.load_settings).pack(side="left", padx=6)
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)

    def _build_training(self):
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text="学習")
        head = ttk.Frame(tab)
        head.pack(fill="x")
        self.V["score_text"] = tk.StringVar(value="検証スコア: -")
        ttk.Label(head, textvariable=self.V["score_text"], font=("", 16, "bold")).pack(side="left")
        ttk.Label(head, text="   学習の各段階で、固定のファイル集合に対する検証スコア(母音一致・境界確信度・子音確認・不合格率)を測った値。"
                             "高いほど良い。最高点のモデルを保存し、悪化したら戻します。",
                  foreground="#666666", wraplength=720).pack(side="left", padx=8)
        grid = ttk.Frame(tab)
        grid.pack(fill="both", expand=True, pady=(8, 0))
        self.charts = {}
        for i, (key, (title, better)) in enumerate(INFO.items()):
            c = MetricChart(grid, title + (" ↑" if better == "up" else " ↓"), better, height=170)
            c.grid(row=i // 2, column=i % 2, sticky="nsew", padx=4, pady=4)
            self.charts[key] = c
        for r in range(3):
            grid.rowconfigure(r, weight=1)
        for c in range(2):
            grid.columnconfigure(c, weight=1)

    def _on_metric(self, name, value, stage):
        ch = self.charts.get(name)
        if ch is not None:
            ch.add(value, stage)
        if name == "score":
            self.V["score_text"].set(f"検証スコア: {value:.3f}   (最高 {ch.best():.3f} / {stage})")

    def _update_mode_info(self):
        try:
            cfg = self.collect_cfg()
            p = resolve(cfg)
        except (tk.TclError, ValueError):
            return
        acc = self.V["accuracy"].get()
        for w in self.custom_widgets:
            w.state(["!disabled"] if acc == "custom" else ["disabled"])
        speed = {"fast": "短時間", "high": "標準", "max": "長時間", "ultra": "非常に長時間", "custom": "設定による"}[acc]
        nn = f"NN {p['net_epochs']}エポック×{p['ens']}" if p["net"] else "NN なし"
        self.V["mode_info"].set(
            f"[{speed}] 学習 {p['iters']} 回 / 自動調整 {p['rounds'] if p['tune'] else 0} 周 / {nn} / "
            f"適応 {p['adapt_rounds'] if p['adapt'] else 0} 回 / 再学習 {p['post']} 回 / 高信頼再学習 {p['conf']} 周 / "
            f"救済 {'あり' if p['rescue'] else 'なし'}")

    def _build_results(self):
        tab = ttk.Frame(self.nb, padding=6)
        self.nb.add(tab, text="結果")
        pw = ttk.PanedWindow(tab, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw)
        top = ttk.Frame(left)
        top.pack(fill="x", pady=(0, 4))
        ttk.Label(top, text="表示").pack(side="left")
        cb = ttk.Combobox(top, textvariable=self.V["filter"], values=["全て", "OK", "WARN", "FAIL"], width=8, state="readonly")
        cb.pack(side="left", padx=6)
        cb.bind("<<ComboboxSelected>>", lambda e: self.populate())
        self.lbl_sum = ttk.Label(top, text="")
        self.lbl_sum.pack(side="right")
        self.tree = ttk.Treeview(left, columns=("st", "file", "score"), show="headings", selectmode="browse")
        for c, t, w in (("st", "状態", 56), ("file", "ファイル", 230), ("score", "スコア", 56)):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="w")
        sb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        for k, c in STATUS_FG.items():
            self.tree.tag_configure(k, foreground=c)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        pw.add(left, weight=1)
        right = ttk.Frame(pw)
        self.lbl_file = ttk.Label(right, text="", font=("", 10, "bold"))
        self.lbl_file.pack(fill="x")
        self.lbl_notes = ttk.Label(right, text="", wraplength=760, foreground="#c05621", justify="left")
        self.lbl_notes.pack(fill="x", pady=(0, 4))
        self.wave = WaveView(right, height=260)
        self.wave.pack(fill="x")
        ttk.Label(right, text="ホイール: 拡大縮小 / ドラッグ: 移動 / ダブルクリック: 全体表示 / エイリアス行を選ぶと oto マーカーを表示",
                  foreground="#777777").pack(anchor="w", pady=(2, 4))
        nb2 = ttk.Notebook(right)
        nb2.pack(fill="both", expand=True)
        f_al = ttk.Frame(nb2)
        nb2.add(f_al, text="エイリアス")
        cols = ("alias", "kind", "off", "cons", "cut", "pre", "ovl", "st")
        self.t_al = ttk.Treeview(f_al, columns=cols, show="headings", selectmode="browse", height=8)
        for c, t, w in zip(cols, ("エイリアス", "種別", "Offset", "Consonant", "Cutoff", "Preutterance", "Overlap", "状態"),
                           (110, 55, 70, 80, 70, 90, 70, 70)):
            self.t_al.heading(c, text=t)
            self.t_al.column(c, width=w, anchor="w")
        self.t_al.pack(fill="both", expand=True)
        self.t_al.bind("<<TreeviewSelect>>", self.on_alias)
        for k, c in STATUS_FG.items():
            self.t_al.tag_configure(k, foreground=c)
        f_mo = ttk.Frame(nb2)
        nb2.add(f_mo, text="モーラ検証")
        cols = ("mora", "rec", "agree", "cagree", "bconf", "st", "memo")
        self.t_mo = ttk.Treeview(f_mo, columns=cols, show="headings", selectmode="browse", height=8)
        for c, t, w in zip(cols, ("モーラ", "認識母音", "母音一致", "子音一致", "境界確信", "状態", "メモ"),
                           (60, 70, 70, 70, 70, 70, 400)):
            self.t_mo.heading(c, text=t)
            self.t_mo.column(c, width=w, anchor="w")
        self.t_mo.pack(fill="both", expand=True)
        for k, c in STATUS_FG.items():
            self.t_mo.tag_configure(k, foreground=c)
        pw.add(right, weight=4)
        self.shown = []

    def _build_log(self):
        tab = ttk.Frame(self.nb, padding=6)
        self.nb.add(tab, text="ログ")
        self.log = tk.Text(tab, wrap="word", state="disabled")
        sb = ttk.Scrollbar(tab, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")

    # ---------------- 操作 ----------------
    def pick_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.V["folder"].set(d)

    def pick_script(self):
        f = filedialog.askopenfilename()
        if f:
            self.V["script"].set(f)

    def open_dir(self):
        d = self.V["folder"].get()
        if not d or not os.path.isdir(d):
            return
        if sys.platform.startswith("win"):
            os.startfile(d)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", d])
        else:
            subprocess.Popen(["xdg-open", d])

    def save_settings(self):
        f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if f:
            with open(f, "w", encoding="utf-8") as fp:
                json.dump({k: v.get() for k, v in self.V.items() if k not in ("stage", "filter", "score_text", "mode_info")}, fp, ensure_ascii=False, indent=1)

    def load_settings(self):
        f = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not f:
            return
        try:
            with open(f, encoding="utf-8") as fp:
                d = json.load(fp)
            for k, val in d.items():
                if k in self.V:
                    self.V[k].set(val)
        except (OSError, ValueError, tk.TclError) as ex:
            messagebox.showerror("otoium", f"設定を読み込めません: {ex}")

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        folder = self.V["folder"].get()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("otoium", "原音フォルダを選択してください")
            return
        try:
            cfg = self.collect_cfg()
        except (tk.TclError, ValueError) as ex:
            messagebox.showerror("otoium", f"設定値が不正です: {ex}")
            return
        self.tree.delete(*self.tree.get_children())
        self.t_al.delete(*self.t_al.get_children())
        self.t_mo.delete(*self.t_mo.get_children())
        self.wave.set_job(None)
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.bar["value"] = 0
        self.btn_run.state(["disabled"])
        self.btn_stop.state(["!disabled"])
        for c in self.charts.values():
            c.clear()
        self.V["score_text"].set("検証スコア: -")
        self.pipe = Pipeline(cfg, log=lambda m: self.q.put(("log", m)), progress=lambda f, m="": self.q.put(("prog", (f, m))),
                             on_metric=lambda n, v, st: self.q.put(("metric", (n, v, st))))

        def work():
            try:
                self.pipe.run(folder)
                self.q.put(("done", None))
            except Cancelled:
                self.q.put(("cancel", None))
            except Exception as ex:
                self.q.put(("err", f"{type(ex).__name__}: {ex}"))

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()
        self.nb.select(1)

    def stop(self):
        if self.pipe:
            self.pipe.cancel()
            self.V["stage"].set("中止しています…")

    def _poll(self):
        try:
            while True:
                k, v = self.q.get_nowait()
                if k == "log":
                    self.log.configure(state="normal")
                    self.log.insert("end", v + "\n")
                    self.log.see("end")
                    self.log.configure(state="disabled")
                    self.V["stage"].set(v.strip()[:40])
                elif k == "prog":
                    self.bar["value"] = v[0]
                elif k == "metric":
                    self._on_metric(*v)
                elif k in ("done", "cancel", "err"):
                    self.btn_run.state(["!disabled"])
                    self.btn_stop.state(["disabled"])
                    if k == "done":
                        self.V["stage"].set("完了")
                        self.populate()
                        self.nb.select(2)
                    elif k == "cancel":
                        self.V["stage"].set("中止しました")
                    else:
                        self.V["stage"].set("エラー")
                        messagebox.showerror("otoium", v)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    # ---------------- 結果表示 ----------------
    def populate(self):
        self.tree.delete(*self.tree.get_children())
        if not self.pipe:
            return
        flt = self.V["filter"].get()
        self.shown = []
        for j in self.pipe.jobs:
            if flt != "全て" and j.status != flt:
                continue
            self.shown.append(j)
            self.tree.insert("", "end", iid=str(len(self.shown) - 1), values=(j.status, j.path.name, f"{j.score:.2f}"),
                             tags=(j.status,))
        r = self.pipe.report
        if r:
            self.lbl_sum.configure(text=f"{r['total']} 件 / 失敗 {r['failed']} / 警告 {r['warned']} / oto {r['entries_written']} 件")

    def on_select(self, _e=None):
        sel = self.tree.selection()
        if not sel:
            return
        j = self.shown[int(sel[0])]
        self.lbl_file.configure(text=f"{j.path.name}   台本: {j.text}   状態: {j.status}")
        self.lbl_notes.configure(text="\n".join(j.notes))
        self.wave.set_job(j if j.segs else None)
        self.t_al.delete(*self.t_al.get_children())
        self.t_mo.delete(*self.t_mo.get_children())
        self._entries = j.entries
        for i, e in enumerate(j.entries):
            self.t_al.insert("", "end", iid=str(i), tags=(e.status,),
                             values=(e.alias, e.kind, f"{e.offset:.1f}", f"{e.consonant:.1f}", f"{e.cutoff:.1f}",
                                     f"{e.pre:.1f}", f"{e.overlap:.1f}", e.status))
        for t in j.timings:
            self.t_mo.insert("", "end", tags=(t.status,),
                             values=(t.mora.surface, t.rec_vowel, f"{t.agree_v:.2f}", f"{t.cons_agree:.2f}",
                                     f"{t.bconf:.2f}", t.status, " / ".join(t.notes)))

    def on_alias(self, _e=None):
        sel = self.t_al.selection()
        if sel:
            self.wave.set_entry(self._entries[int(sel[0])])


def run_gui():
    App().mainloop()
