"""tkinter GUI: 設定・実行・結果確認(波形と音素境界の表示)"""
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from .config import Config
from .pipeline import Pipeline
from .phoneset import UNIT_CLASS

COLORS = {"V": "#4c9be8", "NN": "#6f7fe8", "N": "#9b6fe8", "S": "#e8a24c", "H": "#e8c84c", "Z": "#e87f4c",
          "CL": "#999999", "CLV": "#aa8888", "BR": "#e84c6f", "BRV": "#d46fa0", "RF": "#4ce8c2", "GL": "#4ce88a",
          "PAU": "#dddddd"}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("otoium - 完全自動原音設定")
        self.geometry("1100x700")
        self.q = queue.Queue()
        self.pipe = None
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)
        self.var_dir = tk.StringVar()
        self.var_mode = tk.StringVar(value="auto")
        self.var_style = tk.StringVar(value="mora")
        self.var_enc = tk.StringVar(value="cp932")
        self.var_script = tk.StringVar()
        self.var_iter = tk.IntVar(value=3)
        self.var_strict = tk.BooleanVar(value=False)
        self.var_dry = tk.BooleanVar(value=False)
        ttk.Label(top, text="原音フォルダ").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.var_dir, width=60).grid(row=0, column=1, columnspan=5, sticky="we")
        ttk.Button(top, text="参照", command=self.pick_dir).grid(row=0, column=6)
        ttk.Label(top, text="台本(任意)").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.var_script, width=60).grid(row=1, column=1, columnspan=5, sticky="we")
        ttk.Button(top, text="参照", command=self.pick_script).grid(row=1, column=6)
        ttk.Label(top, text="モード").grid(row=2, column=0, sticky="w")
        ttk.Combobox(top, textvariable=self.var_mode, values=["auto", "cv", "vcv", "cvvc"], width=8,
                     state="readonly").grid(row=2, column=1, sticky="w")
        ttk.Label(top, text="VCV表記").grid(row=2, column=2)
        ttk.Combobox(top, textvariable=self.var_style, values=["mora", "vowel"], width=8,
                     state="readonly").grid(row=2, column=3, sticky="w")
        ttk.Label(top, text="文字コード").grid(row=2, column=4)
        ttk.Combobox(top, textvariable=self.var_enc, values=["cp932", "utf-8"], width=8,
                     state="readonly").grid(row=2, column=5, sticky="w")
        ttk.Label(top, text="学習回数").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(top, from_=1, to=10, textvariable=self.var_iter, width=5).grid(row=3, column=1, sticky="w")
        ttk.Checkbutton(top, text="厳格(WARNも除外)", variable=self.var_strict).grid(row=3, column=2, columnspan=2)
        ttk.Checkbutton(top, text="ドライラン", variable=self.var_dry).grid(row=3, column=4)
        self.btn = ttk.Button(top, text="自動原音設定を実行", command=self.start)
        self.btn.grid(row=3, column=5, columnspan=2, sticky="e")
        self.bar = ttk.Progressbar(self, maximum=1.0)
        self.bar.pack(fill="x", padx=8)
        mid = ttk.PanedWindow(self, orient="horizontal")
        mid.pack(fill="both", expand=True, padx=8, pady=6)
        left = ttk.Frame(mid)
        self.lst = tk.Listbox(left, width=38)
        self.lst.pack(fill="both", expand=True)
        self.lst.bind("<<ListboxSelect>>", self.show_sel)
        mid.add(left, weight=1)
        right = ttk.Frame(mid)
        self.canvas = tk.Canvas(right, bg="white", height=260)
        self.canvas.pack(fill="x")
        self.log = tk.Text(right, height=18)
        self.log.pack(fill="both", expand=True)
        mid.add(right, weight=4)
        self.after(100, self.poll)

    def pick_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.var_dir.set(d)

    def pick_script(self):
        f = filedialog.askopenfilename()
        if f:
            self.var_script.set(f)

    def start(self):
        if not self.var_dir.get():
            messagebox.showwarning("otoium", "フォルダを選択してください")
            return
        cfg = Config(mode=self.var_mode.get(), vcv_style=self.var_style.get(), encoding=self.var_enc.get(),
                     script_path=self.var_script.get(), train_iters=self.var_iter.get(),
                     strict=self.var_strict.get(), dry_run=self.var_dry.get())
        self.btn.state(["disabled"])
        self.lst.delete(0, "end")
        self.log.delete("1.0", "end")
        self.pipe = Pipeline(cfg, log=lambda m: self.q.put(("log", m)),
                             progress=lambda f, m="": self.q.put(("prog", f)))

        def work():
            try:
                self.pipe.run(self.var_dir.get())
                self.q.put(("done", None))
            except Exception as ex:
                self.q.put(("err", str(ex)))
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        try:
            while True:
                k, v = self.q.get_nowait()
                if k == "log":
                    self.log.insert("end", v + "\n")
                    self.log.see("end")
                elif k == "prog":
                    self.bar["value"] = v
                elif k == "done":
                    self.btn.state(["!disabled"])
                    for j in self.pipe.jobs:
                        self.lst.insert("end", f"[{j.status}] {j.path.name}")
                elif k == "err":
                    self.btn.state(["!disabled"])
                    messagebox.showerror("otoium", v)
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def show_sel(self, _e=None):
        sel = self.lst.curselection()
        if not sel or not self.pipe:
            return
        j = self.pipe.jobs[sel[0]]
        c = self.canvas
        c.delete("all")
        W, H = max(c.winfo_width(), 600), 260
        self.log.delete("1.0", "end")
        self.log.insert("end", f"{j.path.name}  台本: {j.text}  状態: {j.status}\n")
        for n in j.notes:
            self.log.insert("end", f"! {n}\n")
        if j.env is None or not j.segs:
            return
        env = j.env
        n = env.shape[1]
        mid = 120
        for i in range(n):
            x = i * W / n
            c.create_line(x, mid - env[1, i] * 100, x, mid - env[0, i] * 100, fill="#555")
        total = j.F.T
        for s in j.segs:
            if s.e <= s.s:
                continue
            x0, x1 = s.s * W / total, s.e * W / total
            col = COLORS.get(UNIT_CLASS[s.unit], "#ccc")
            c.create_rectangle(x0, 225, x1, 250, fill=col, outline="white")
            c.create_text((x0 + x1) / 2, 238, text=s.unit if s.unit != "pau" else "", font=("", 8))
            c.create_line(x0, 20, x0, 225, fill=col, dash=(2, 3))
        for t in j.timings:
            self.log.insert("end", f"{t.mora.surface}: 母音認識={t.rec_vowel} 一致率={t.agree_v:.2f} "
                                   f"境界確信度={t.bconf:.2f} [{t.status}]\n")
        for e in j.entries:
            self.log.insert("end", f"{e.alias}  off={e.offset:.0f} cons={e.consonant:.0f} cut={e.cutoff:.0f} "
                                   f"pre={e.pre:.0f} ovl={e.overlap:.0f} [{e.status}]\n")


def run_gui():
    App().mainloop()
