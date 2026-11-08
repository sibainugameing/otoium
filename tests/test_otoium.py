import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tests.synth import render, write_wav
from otoium.config import Config
from otoium.pipeline import Pipeline
from otoium.kana import parse_text

BANK = ["_あかさたな", "_あんああいあうあ", "_かんかかきかくか", "_さんささしさすさ", "_なんななになぬな",
        "_はんははひはふは", "_まんままみまむま", "_らんららりらるら", "_たんたたてぃたとぅた", "_きゃんきゃきゃききゃきゅきゃ"]


def make_bank(d, texts=BANK, extra=None):
    truth = {}
    for i, t in enumerate(texts):
        x, tr = render(t, seed=i)
        write_wav(Path(d) / f"{t}.wav", x)
        truth[f"{t}.wav"] = tr
    for name, x in (extra or {}).items():
        write_wav(Path(d) / name, x)
    return truth


def run(d, **kw):
    cfg = Config(**kw)
    p = Pipeline(cfg, log=lambda m: None)
    rep = p.run(d)
    return p, rep


def read_oto(d):
    out = {}
    for line in (Path(d) / "oto.ini").read_bytes().decode("cp932").splitlines():
        wav, rest = line.split("=", 1)
        f = rest.split(",")
        out.setdefault(wav, []).append((f[0], *[float(v) for v in f[1:]]))
    return out


class T(unittest.TestCase):
    def test_vcv_multiple_aliases_and_boundaries(self):
        with tempfile.TemporaryDirectory() as d:
            truth = make_bank(d)
            p, rep = run(d, mode="vcv")
            oto = read_oto(d)
            aliases = [a[0] for a in oto["_あかさたな.wav"]]
            self.assertEqual(aliases, ["あ", "あ か", "か さ", "さ た", "た な"])
            # 全ファイルで複数エイリアス (1ファイル=1音素ではない)
            for w, rows in oto.items():
                self.assertGreater(len(rows), 1, w)
            # 境界精度: 母音開始位置 (preutterance 位置) が正解±35ms
            tr = truth["_あかさたな.wav"]
            vstarts = [a for (mi, u, a, b) in tr if u == "a"]
            for k, row in enumerate(oto["_あかさたな.wav"]):
                _, off, cons, cut, pre, ovl = row
                self.assertAlmostEqual((off + pre) / 1000, vstarts[k], delta=0.035, msg=str(row))
                self.assertLess(cut, 0)
                self.assertGreaterEqual(cons, pre)
                self.assertLessEqual(ovl, pre)

    def test_cv_and_cvvc(self):
        with tempfile.TemporaryDirectory() as d:
            make_bank(d)
            run(d, mode="cv")
            cv = read_oto(d)["_あかさたな.wav"]
            self.assertEqual([a[0] for a in cv], ["あ", "か", "さ", "た", "な"])
        with tempfile.TemporaryDirectory() as d:
            make_bank(d)
            run(d, mode="cvvc")
            r = [a[0] for a in read_oto(d)["_あかさたな.wav"]]
            self.assertEqual(r, ["あ", "か", "a k", "さ", "a s", "た", "a t", "な", "a n"])

    def test_single_isolated_cv(self):
        with tempfile.TemporaryDirectory() as d:
            texts = ["か", "さ", "な", "あ", "ま", "ら", "は", "た"]
            make_bank(d, texts)
            run(d, mode="auto")
            oto = read_oto(d)
            for t in texts:
                self.assertEqual([a[0] for a in oto[f"{t}.wav"]], [t])

    def test_failure_is_not_written(self):
        with tempfile.TemporaryDirectory() as d:
            make_bank(d)
            x_a, _ = render("_あいうえお", seed=77)
            x_one, _ = render("_あ", seed=78)
            noise = (0.2 * np.random.default_rng(0).standard_normal(44100 * 3)).astype("float32")
            # 名前は「_かきくけこ」だが中身は別の母音列 / 1音のみ / ノイズ
            extra = {"_かきくけこ.wav": x_a, "_さしすせそ.wav": x_one, "_なにぬねの.wav": noise}
            for n_, x_ in extra.items():
                write_wav(Path(d) / n_, x_)
            p, rep = run(d, mode="vcv")
            st = {f["file"]: f["status"] for f in rep["files"]}
            self.assertEqual(st["_さしすせそ.wav"], "FAIL")
            self.assertEqual(st["_なにぬねの.wav"], "FAIL")
            oto = read_oto(d)
            self.assertNotIn("_さしすせそ.wav", oto)
            self.assertNotIn("_なにぬねの.wav", oto)
            # 内容が別の音(中身 あいうえお)を「かきくけこ」として扱っていないこと
            self.assertEqual(st["_かきくけこ.wav"], "FAIL")
            self.assertNotIn("_かきくけこ.wav", oto)
            # 正常ファイルは影響を受けない
            self.assertEqual(st["_あかさたな.wav"], "OK")
            self.assertIn("_あかさたな.wav", oto)

    def test_script_vs_audio_vowel_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            make_bank(d)
            x, _ = render("_いえあおう", seed=61)                    # 名前は「あいうえお」だが中身は母音列が全部違う
            write_wav(Path(d) / "_あいうえお.wav", x)
            p, rep = run(d, mode="vcv", accuracy="fast")
            st = {f["file"]: f["status"] for f in rep["files"]}
            self.assertEqual(st["_あいうえお.wav"], "FAIL")
            self.assertNotIn("_あいうえお.wav", read_oto(d))

    def test_script_parsing(self):
        names = [m.surface for m in parse_text("_ガんガガギガグガ")]
        self.assertEqual(names, ["ガ", "ん", "ガ", "ガ", "ギ", "ガ", "グ", "ガ"])
        full = """_あんああいあうあ _きゃんきゃきゃききゃきゅきゃ _てゅんてゅてゅえてゅおてゅ _つぁんつぁつぁつぃつぁつつぁ
_ふぇんふぇふぇふぉふぇふぁふぇ _うぃんうぃうぃううぃうぇうぃ _ううぇううぉいゆいいぇ _くぉんくぉくぉくぁくぉくぃくぉ""".split()
        for t in full:
            parse_text(t)

    def test_noisy_and_pitch_conditions(self):
        for kw in (dict(f0=220.0), dict(f0=90.0)):
            with tempfile.TemporaryDirectory() as d:
                for i, t in enumerate(BANK):
                    x, _ = render(t, seed=i, **kw)
                    write_wav(Path(d) / f"{t}.wav", x)
                p, rep = run(d, mode="vcv")
                self.assertEqual(rep["failed"], 0, kw)
                self.assertEqual([a[0] for a in read_oto(d)["_あかさたな.wav"]], ["あ", "あ か", "か さ", "さ た", "た な"])

    def test_truncated_recording_rescue(self):
        with tempfile.TemporaryDirectory() as d:
            make_bank(d)
            x, _ = render("_あかさ", seed=91)            # 名前は5モーラだが録音は3モーラ
            write_wav(Path(d) / "_あかさたな2.wav", x)
            p, rep = run(d, mode="vcv")
            f = [f for f in rep["files"] if f["file"] == "_あかさたな2.wav"][0]
            self.assertNotEqual(f["status"], "FAIL")
            self.assertEqual(len(f["morae"]), 3)

    def test_fast_preset_and_options(self):
        with tempfile.TemporaryDirectory() as d:
            make_bank(d)
            (Path(d) / "oto.ini").write_bytes("other.wav=あ,1,2,-3,4,5\n".encode("cp932"))
            run(d, mode="cvvc", accuracy="fast", alias_prefix="x_", alias_suffix="_y", merge=True, textgrid=True,
                export_labels=True)
            self.assertTrue((Path(d) / "_あかさたな.TextGrid").exists())
            self.assertTrue((Path(d) / "_あかさたな.labels.txt").exists())
            self.assertTrue((Path(d) / "otoium_model.npz").exists())
            oto = read_oto(d)
            for wav, rows in oto.items():
                if wav == "other.wav":
                    self.assertEqual(rows[0][0], "あ")              # merge: 今回処理しないファイルの行はそのまま
                    continue
                for row in rows:
                    self.assertTrue(row[0].startswith("x_") and row[0].endswith("_y"), row[0])
            self.assertIn("other.wav", oto)

    def test_moraic_nasal_only_files(self):
        with tempfile.TemporaryDirectory() as d:
            make_bank(d, BANK + ["_んん", "_んんん"])
            p, rep = run(d, mode="vcv")
            st = {f["file"]: f["status"] for f in rep["files"]}
            self.assertEqual(st["_んん.wav"], "OK")
            self.assertEqual(st["_んんん.wav"], "OK")
            self.assertEqual([a[0] for a in read_oto(d)["_んん.wav"]], ["ん", "ん ん"])

    def test_net_calibration_and_model_roundtrip(self):
        from otoium.model import AcousticModel
        with tempfile.TemporaryDirectory() as d:
            make_bank(d)
            lines = []
            p = Pipeline(Config(mode="vcv", accuracy="high"), log=lines.append)
            rep = p.run(d)
            self.assertTrue(any("しきい値" in l for l in lines))           # 自動校正が実行された
            self.assertIsNotNone(p.model.net)
            m = AcousticModel.load(Path(d) / "otoium_model.npz")
            self.assertIsNotNone(m.net)
            self.assertEqual(rep["failed"], 0)
            # 保存モデルの再利用
            p2, rep2 = run(d, mode="vcv", model_path=str(Path(d) / "otoium_model.npz"))
            self.assertEqual(rep2["failed"], 0)

    def test_custom_mode_ensemble_and_confident_retrain(self):
        from otoium.model import AcousticModel
        with tempfile.TemporaryDirectory() as d:
            for i, t in enumerate(BANK):
                x, _ = render(t, seed=i, tempo=0.8 + 0.07 * i, blur=0.03, reverb=0.5, noise=0.02, jitter=0.05)
                write_wav(Path(d) / f"{t}.wav", x)
            p, rep = run(d, mode="vcv", accuracy="custom", train_iters=3, tune_rounds=1, net_epochs=4,
                         net_ensemble=2, adapt_rounds=2, post_iters=1, confident_rounds=1,
                         guarded_updates=False)          # ガード有効だと難条件では NN が不採用になりうる。ここは設定の反映だけを見る
            self.assertEqual(len(p.model.nets), 2)
            self.assertEqual(rep["failed"], 0)
            # 難条件 (残響+ノイズ+重なり) では短い鼻音の確認で最後の1件が保留になりうる。
            # ここでは「期待外のエイリアスを出さない」「大半は出る」ことだけを見る (精度ではなくカスタム設定の動作確認)
            got = [a[0] for a in read_oto(d)["_あかさたな.wav"]]
            expected = ["あ", "あ か", "か さ", "さ た", "た な"]
            self.assertTrue(set(got) <= set(expected), got)
            self.assertGreaterEqual(len(got), 4, got)
            self.assertEqual(len(AcousticModel.load(Path(d) / "otoium_model.npz").nets), 2)
            self.assertTrue(any(abs(v - 0.0) > 1e-9 for v in p.model.dur_mu.values()))

    def test_training_metrics_recorded(self):
        events = []
        with tempfile.TemporaryDirectory() as d:
            make_bank(d)
            p = Pipeline(Config(mode="vcv", accuracy="high"), log=lambda m: None,
                         on_metric=lambda n, v, st: events.append((n, v, st)))
            p.run(d)
            h = p.track.history
            self.assertGreaterEqual(len(h["score"]), 4)
            self.assertGreaterEqual(max(v for _, v, _ in h["score"]), h["score"][0][1])
            nl = [v for _, v, _ in h["nn_loss"]]
            self.assertLess(nl[-1], nl[0])
            self.assertTrue(any(e[0] == "nn_acc" for e in events))
            self.assertTrue((Path(d) / "otoium_training.csv").exists())
            self.assertTrue((Path(d) / "otoium_training.json").exists())

    def test_empty_result_keeps_existing_oto_and_atomic_write(self):
        with tempfile.TemporaryDirectory() as d:
            noise = (0.2 * np.random.default_rng(0).standard_normal(44100 * 3)).astype("float32")
            write_wav(Path(d) / "_あかさ.wav", noise)
            old = "keep.wav=あ,1,2,-3,4,5\n".encode("cp932")
            (Path(d) / "oto.ini").write_bytes(old)
            run(d, mode="vcv", backup=False)
            self.assertEqual((Path(d) / "oto.ini").read_bytes(), old)
            self.assertEqual(list(Path(d).glob("*.tmp")), [])

    def test_cache_is_npz_not_pickle_and_survives_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            make_bank(d)
            run(d, mode="vcv", accuracy="fast")
            files = list((Path(d) / ".otoium_cache").glob("*"))
            self.assertTrue(files and all(f.suffix == ".npz" for f in files))
            files[0].write_bytes(b"corrupted")
            p, rep = run(d, mode="vcv", accuracy="fast")
            self.assertEqual(rep["failed"], 0)

    def test_recheck_gives_up_after_three(self):
        with tempfile.TemporaryDirectory() as d:
            make_bank(d)
            noise = (0.2 * np.random.default_rng(0).standard_normal(44100 * 3)).astype("float32")
            write_wav(Path(d) / "_なにぬねの.wav", noise)
            lines = []
            p = Pipeline(Config(mode="vcv", accuracy="fast"), log=lines.append)
            rep = p.run(d)
            j = [j for j in p.jobs if j.path.name == "_なにぬねの.wav"][0]
            self.assertEqual(j.status, "FAIL")
            self.assertEqual(j.rechecks, 3)
            self.assertTrue(any("3 回再検査しても不合格" in n for n in j.notes))
            self.assertEqual(sum(1 for l in lines if "再検査" in l and "回目:" in l), 3)
            self.assertNotIn("_なにぬねの.wav", read_oto(d))

    def test_recheck_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            make_bank(d)
            noise = (0.2 * np.random.default_rng(0).standard_normal(44100 * 3)).astype("float32")
            write_wav(Path(d) / "_なにぬねの.wav", noise)
            p, rep = run(d, mode="vcv", accuracy="fast", recheck_rounds=0)
            self.assertEqual([j for j in p.jobs if j.path.name == "_なにぬねの.wav"][0].rechecks, 0)

    def test_tune_holdout_does_not_crash_small_sets(self):
        with tempfile.TemporaryDirectory() as d:
            make_bank(d, BANK[:4])
            p, rep = run(d, mode="vcv", accuracy="high")
            self.assertEqual(rep["failed"], 0)

    def test_patience_zero_runs_all_iterations(self):
        lines = []
        with tempfile.TemporaryDirectory() as d:
            make_bank(d)
            p = Pipeline(Config(mode="vcv", accuracy="custom", train_iters=6, patience=0, use_net=False,
                                tune_rounds=0, adapt_rounds=0, post_iters=0, confident_rounds=0), log=lines.append)
            p.run(d)
        n = sum(1 for l in lines if l.strip().startswith("学習 ") and "/6:" in l)
        self.assertTrue(n == 6 or (n < 6 and any("不動点" in l for l in lines)), n)     # 全回数、または不動点に達した場合のみ早く終わる
        self.assertFalse(any("収束" in l for l in lines))

    def test_convergence_rule(self):
        cfg = Config()                                    # converge_shift=0.03, converge_score=0.002
        ns, stop = Pipeline.next_stable, Pipeline.should_stop
        self.assertEqual(ns(0, 0.01, 1.000, 0.999, cfg), 1)       # 境界が動かず得点も伸びない → +1
        self.assertEqual(ns(1, 0.01, 1.000, 0.999, cfg), 2)
        self.assertEqual(ns(2, 0.50, 1.000, 1.000, cfg), 0)       # 境界がまだ動く → リセット
        self.assertEqual(ns(2, 0.01, 1.100, 1.000, cfg), 0)       # 得点がまだ伸びる → リセット
        self.assertEqual(ns(0, 0.01, None, None, cfg), 1)         # 得点なしなら境界だけで判断
        self.assertFalse(stop(2, 2, 3, 12))                       # 予定12回の半分(6回)未満では止めない
        self.assertTrue(stop(2, 2, 6, 12))
        self.assertFalse(stop(1, 2, 8, 12))                       # 連続回数が足りない
        self.assertFalse(stop(5, 0, 12, 12))                      # patience 0 は早期終了しない
        self.assertTrue(stop(1, 1, 2, 3))                         # 最低2回

    def test_net_is_dropped_when_it_lowers_the_score(self):
        """NN を併用すると得点が下がる場合は NN を使わない (ガード有効時)"""
        from otoium.model import AcousticModel
        from otoium.mlp import FrameNet
        lines = []
        p = Pipeline(Config(mode="vcv"), log=lines.append)
        p.model = AcousticModel()
        p._best = {0: (1.5, p.model)}                         # NN なしの最高点 1.5
        p.model.fit_net = lambda *a, **k: 0.1                 # NN 学習は即終了 (ダミー)
        p.model.nets = [FrameNet(4, 3)]
        p.model.nn_weight = 2.0
        p._checkpoint = lambda stage: 1.2                     # NN 併用の得点 1.2 < 1.5
        p._train = lambda *a, **k: self.fail("NN が不採用なら併用学習は行わない")
        p.jobs = []
        p._train_net([])
        self.assertEqual((p.model.nets, p.model.nn_weight), ([], 0.0))
        self.assertTrue(any("NN は使用しません" in l for l in lines))

    def test_fixed_point_detection(self):
        self.assertTrue(Pipeline.is_fixed_point(0.0))
        self.assertFalse(Pipeline.is_fixed_point(0.01))
        self.assertFalse(Pipeline.is_fixed_point(None))                        # 初回 (比較対象なし) は不動点とみなさない

    def test_patience_zero_never_logs_convergence(self):
        lines = []
        with tempfile.TemporaryDirectory() as d:
            make_bank(d, BANK[:4])
            Pipeline(Config(mode="vcv", accuracy="custom", train_iters=5, patience=0, use_net=False, tune_rounds=0,
                            adapt_rounds=0, post_iters=0, confident_rounds=0, track_metrics=False),
                     log=lines.append).run(d)
        self.assertFalse(any("収束" in l for l in lines))

    def test_partial_repair_only_touches_failed_region(self):
        import copy
        from otoium.evaluate import evaluate_segs
        with tempfile.TemporaryDirectory() as d:
            truth = make_bank(d)
            p, rep = run(d, mode="vcv", accuracy="fast", recheck_rounds=0)
            j = [j for j in p.jobs if j.path.name == "_あかさたな.wav"][0]
            segs = [copy.copy(g) for g in j.segs]
            m3 = [g for g in segs if g.mora == 3]
            m2v = [g for g in segs if g.mora == 2][-1]
            B, E = m3[0].s, m3[-1].e
            Bn = E - 12
            m2v.e = Bn
            for g, (a, b) in zip(m3, [(Bn, Bn + 3), (Bn + 3, Bn + 6), (Bn + 6, E)]):
                g.s, g.e = a, b
            LLb = p.model.loglik(j.F, j.X)
            bad = evaluate_segs(p.model, j.F, j.morae, segs, j.dur_ms, p.cfg, LLb)
            p._apply(j, bad)
            runs = p._bad_runs(j.timings)
            self.assertTrue(runs, "壊した区間が不合格にならない")
            orig0 = [(g.unit, g.s, g.e) for g in j.segs if g.mora == 0]
            n_bad = sum(1 for t in j.timings if t.status in ("REVIEW", "ERROR"))
            ok = p._repair_partial(j, 1)
            self.assertTrue(ok)
            self.assertLess(sum(1 for t in j.timings if t.status in ("REVIEW", "ERROR")), n_bad)
            self.assertEqual(orig0, [(g.unit, g.s, g.e) for g in j.segs if g.mora == 0])   # 区間外は不変
            tv = [a for (mi, u, a, b) in truth["_あかさたな.wav"] if mi == 3 and u == "a"][0]
            self.assertAlmostEqual(j.timings[3].vowel_start / 1000, tv, delta=0.03)
            self.assertTrue(any("部分やり直し" in n for n in j.notes))

    def test_write_failed_forces_entries_into_oto(self):
        with tempfile.TemporaryDirectory() as d:
            make_bank(d)
            noise = (0.2 * np.random.default_rng(0).standard_normal(44100 * 3)).astype("float32")
            write_wav(Path(d) / "_なにぬねの.wav", noise)
            p, rep = run(d, mode="vcv", accuracy="fast", write_failed=True)
            j = [j for j in p.jobs if j.path.name == "_なにぬねの.wav"][0]
            self.assertEqual(j.status, "FAIL")
            oto = read_oto(d)
            self.assertIn("_なにぬねの.wav", oto)
            self.assertTrue((Path(d) / "otoium_forced.txt").exists())
            self.assertGreater(rep["entries_forced"], 0)
            self.assertIn("_あかさたな.wav", oto)

    def test_post_run_write_recheck_and_manual_override(self):
        with tempfile.TemporaryDirectory() as d:
            make_bank(d)
            noise = (0.2 * np.random.default_rng(0).standard_normal(44100 * 3)).astype("float32")
            write_wav(Path(d) / "_なにぬねの.wav", noise)
            (Path(d) / "oto.ini").write_bytes("old.wav=あ,1,2,-3,4,5\n".encode("cp932"))
            p, rep = run(d, mode="vcv", accuracy="fast", dry_run=True, recheck_rounds=1)
            self.assertEqual((Path(d) / "oto.ini").read_bytes(), "old.wav=あ,1,2,-3,4,5\n".encode("cp932"))  # dry-run は書かない
            bad = [j for j in p.jobs if j.path.name == "_なにぬねの.wav"][0]
            # 実行後の再検査 (回数を指定。解決はしない)
            self.assertEqual(p.recheck([bad], 2), 0)
            self.assertEqual(bad.rechecks, 2)
            # 書き込み: 失敗ファイルは出ない
            p.write_output(write_failed=False)
            oto = read_oto(d)
            self.assertNotIn("_なにぬねの.wav", oto)
            self.assertIn("_あかさたな.wav", oto)
            self.assertEqual(len(list(Path(d).glob("oto.ini.bak*"))), 1)
            # 失敗も書き込む
            p.write_output(write_failed=True)
            self.assertIn("_なにぬねの.wav", read_oto(d))
            self.assertTrue((Path(d) / "otoium_forced.txt").exists())
            # 元に戻すと、強制出力の一覧も消える
            p.write_output(write_failed=False)
            self.assertNotIn("_なにぬねの.wav", read_oto(d))
            self.assertFalse((Path(d) / "otoium_forced.txt").exists())
            self.assertEqual(len(list(Path(d).glob("oto.ini.bak*"))), 1)   # バックアップは増えない
            # 個別指定: 失敗ファイルの1件だけ書く / 合格エイリアス1件を書かない
            e_bad = bad.entries[0]
            self.assertEqual(p.decision(bad, e_bad), "✗")
            e_bad.manual = True
            self.assertEqual(p.decision(bad, e_bad), "強制")
            good = [j for j in p.jobs if j.path.name == "_あかさたな.wav"][0]
            e_good = good.entries[1]
            e_good.manual = False
            p.write_output()
            oto = read_oto(d)
            self.assertEqual([a[0] for a in oto["_なにぬねの.wav"]], [e_bad.alias])
            self.assertNotIn(e_good.alias, [a[0] for a in oto["_あかさたな.wav"]])

    def test_kana_extended_and_romaji_variants(self):
        g = lambda t: [(m.cons, m.vowel) for m in parse_text(t)]
        self.assertEqual(g("_んふぉんヴぃヴぃり"), [("", "N"), ("f", "o"), ("", "N"), ("v", "i"), ("v", "i"), ("r", "i")])
        self.assertEqual(g("ヴ_ヴぁヴぇヴぉヴゃ"), [("v", "u"), ("v", "a"), ("v", "e"), ("v", "o"), ("vy", "a")])
        self.assertEqual(g("sya tya kka si ti tu hu zi"), [("sh", "a"), ("ch", "a"), ("k", "a"), ("sh", "i"), ("ch", "i"),
                                                           ("ts", "u"), ("f", "u"), ("j", "i")])
        self.assertEqual(g("ka-n'a"), [("k", "a"), ("", "N"), ("", "a")])
        self.assertEqual(g("くゎ"), [("kw", "a")])
        self.assertEqual(g("va vi ve vo vu"), [("v", v) for v in "aieou"])
        sk = []
        self.assertEqual(len(parse_text("_あ☆か?", True, sk)), 2)
        self.assertEqual(sk, ["☆", "?"])
        with self.assertRaises(Exception):
            parse_text("_あ☆か")

    def test_v_sound_file_is_aligned(self):
        with tempfile.TemporaryDirectory() as d:
            make_bank(d, BANK + ["_んふぉんヴぃヴぃり"])
            p, rep = run(d, mode="vcv")
            f = [f for f in rep["files"] if f["file"].startswith("_んふぉん")][0]
            self.assertNotEqual(f["status"], "FAIL", f["notes"])
            self.assertEqual([a[0] for a in read_oto(d)["_んふぉんヴぃヴぃり.wav"]],
                             ["ん", "ん ふぉ", "ふぉ ん", "ん ヴぃ", "ヴぃ ヴぃ", "ヴぃ り"])

    def test_devoiced_vowels_are_not_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            make_bank(d)
            for i, t in enumerate(["_きしすくち", "_ひしふきち"]):
                x, _ = render(t, seed=40 + i, devoice=True)
                write_wav(Path(d) / f"{t}.wav", x)
            p, rep = run(d, mode="vcv")
            for t in ["_きしすくち", "_ひしふきち"]:
                f = [f for f in rep["files"] if f["file"] == f"{t}.wav"][0]
                self.assertNotEqual(f["status"], "FAIL", f["notes"])
                self.assertGreaterEqual(len(read_oto(d)[f"{t}.wav"]), 4)
                self.assertTrue(all(m["devoiced"] for m in f["morae"]), f["morae"])     # 無声化母音として認識
            # 有声で録音した通常のファイルを、無声化母音と取り違えない
            for j in p.jobs:
                if j.path.stem in BANK:
                    self.assertFalse(any(g.unit.endswith("_dv") for g in j.segs), j.path.name)
                    self.assertEqual(j.status, "OK", j.path.name)

    def test_devoiced_vowel_detector(self):
        from types import SimpleNamespace as NS
        from otoium.verify import _maybe_devoiced
        mk = lambda voi, e: NS(cls=np.tile(np.array([[e, voi, 0.1, 0.1, 0.3, 0.2, 0.5, 0.5]]), (10, 1)))
        seg = lambda u: NS(unit=u, s=0, e=10)
        self.assertTrue(_maybe_devoiced(mk(0.1, -1.0), seg("i"), ["cl", "brK"]))       # 無声子音の後・有声度低・音あり
        self.assertFalse(_maybe_devoiced(mk(0.9, -1.0), seg("i"), ["cl", "brK"]))      # 有声 → 通常の母音
        self.assertFalse(_maybe_devoiced(mk(0.1, -5.0), seg("i"), ["cl", "brK"]))      # 無音 → 母音がない
        self.assertFalse(_maybe_devoiced(mk(0.1, -1.0), seg("a"), ["cl", "brK"]))      # a は対象外
        self.assertFalse(_maybe_devoiced(mk(0.1, -1.0), seg("i"), ["clv", "brG"]))     # 有声子音の後は対象外

    def test_timing_consistency_checks(self):
        from otoium.verify import check_timings, MoraTiming
        from otoium.kana import parse_text as pt
        m = pt("か")[0]
        mk = lambda i, a, b, c: MoraTiming(i, m, [("a", b, c)], a, b, c, None, 0.0)
        check_timings([mk(0, 10, 20, 30), mk(1, 30, 40, 50)], 100)
        for bad in ([mk(0, 10, 5, 30)], [mk(0, 10, 20, 300)], [mk(0, 10, 20, 30), mk(1, 5, 40, 50)],
                    [mk(0, float("nan"), 20, 30)]):
            with self.assertRaises(ValueError):
                check_timings(bad, 100)

    def test_skip_unknown_chars_option(self):
        with tempfile.TemporaryDirectory() as d:
            make_bank(d)
            x, _ = render("_あかさ", seed=5)
            write_wav(Path(d) / "_あか☆さ.wav", x)
            p, rep = run(d, mode="vcv", accuracy="fast")
            self.assertEqual([f for f in rep["files"] if "☆" in f["file"]][0]["status"], "FAIL")
            p, rep = run(d, mode="vcv", accuracy="fast", skip_unknown_chars=True)
            f = [f for f in rep["files"] if "☆" in f["file"]][0]
            self.assertNotEqual(f["status"], "FAIL")
            self.assertTrue(any("☆" in n for n in f["notes"]))

    def _strict_json(self, path):
        def bad(c):
            raise ValueError(f"JSON に {c} が含まれています")
        return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=bad)

    def test_result_json_full_summary_and_standard(self):
        with tempfile.TemporaryDirectory() as d:
            make_bank(d, BANK[:5])
            p, rep = run(d, mode="vcv", accuracy="fast", result_json=True, result_json_detail="full")
            full = self._strict_json(Path(d) / "otoium_result.json")
            self.assertEqual(full["schema"], "otoium-result/1")
            self.assertEqual(full["summary"]["files"], 5)
            for k in ("environment", "config_initial", "config_final", "resolved_plan", "stage_seconds", "model",
                      "training_history", "log"):
                self.assertIn(k, full)
            f = [f for f in full["files"] if f["file"] == "_あかさたな.wav"][0]
            self.assertEqual([e["alias"] for e in f["entries"]], ["あ", "あ か", "か さ", "さ た", "た な"])
            self.assertEqual(len(f["morae"]), 5)
            self.assertTrue(f["segments"] and f["independent_recognition_runs"] and len(f["wav_sha256"]) == 64)
            oto_lines = [l for l in (Path(d) / "oto.ini").read_bytes().decode("cp932").splitlines()
                         if l.startswith("_あかさたな.wav=")]
            self.assertEqual(sorted(e["line"] for e in f["entries"]), sorted(oto_lines))      # JSON と oto.ini が一致
            # 詳細度
            p.export_json(Path(d) / "std.json", "standard")
            p.export_json(Path(d) / "sum.json", "summary")
            std, summ = self._strict_json(Path(d) / "std.json"), self._strict_json(Path(d) / "sum.json")
            self.assertIn("entries", std["files"][0])
            self.assertNotIn("segments", std["files"][0])
            self.assertNotIn("entries", summ["files"][0])
            self.assertNotIn("log", summ)
            with self.assertRaises(ValueError):
                p.export_json(Path(d) / "x.json", "bogus")

    def test_result_json_reflects_post_run_changes_and_cli(self):
        from otoium.cli import main
        with tempfile.TemporaryDirectory() as d:
            make_bank(d, BANK[:4])
            out = Path(d) / "sub" / "r.json"
            rc = main([d, "--accuracy", "fast", "--mode", "vcv", "--json", str(out), "--json-detail", "standard"])
            self.assertEqual(rc, 0)
            data = self._strict_json(out)
            self.assertEqual(data["detail"], "standard")
            self.assertNotIn("log", data)
            rc = main([d, "--accuracy", "fast", "--mode", "vcv", "--json"])          # パス省略 → <フォルダ>/otoium_result.json
            self.assertEqual(rc, 0)
            self.assertTrue((Path(d) / "otoium_result.json").exists())
            # 実行後の手動指定が JSON に反映される
            p, rep = run(d, mode="vcv", accuracy="fast", result_json=True)
            e = p.jobs[0].entries[1]
            e.manual = False
            p.write_output()
            full = self._strict_json(Path(d) / "otoium_result.json")
            ent = [x for x in full["files"][0]["entries"] if x["alias"] == e.alias][0]
            self.assertEqual((ent["manual"], ent["decision"]), (False, "✗"))

    # ---- 実録音の結果 (otoium_result_1.json) から見つかった問題の回帰テスト ----
    def test_breathy_and_voiced_stop_consonants_accept_confusable_groups(self):
        from otoium.verify import accepted_groups as ag
        for u in ("frH", "frHY", "frF"):                 # h/f は「無声化母音」と認識されても良い (実録音で13件が誤って不合格)
            self.assertIn("DV", ag(u), u)
        for u in ("brK", "brT", "cl"):
            self.assertIn("DV", ag(u), u)
        for u in ("clv", "brD", "brG", "brB"):           # 有声閉鎖音のボイスバーは u / N と認識されうる
            self.assertTrue({"u", "N"} <= ag(u), u)
        self.assertNotIn("DV", ag("frS"))                # 強い摩擦音まで緩めない

    def test_palatal_morae_detected(self):
        pal = lambda t: parse_text(t)[0].palatal
        for t in ("しゅ", "ちゅ", "じゅ", "きゃ", "ひょ", "にゃ", "てゅ", "sya"):
            self.assertTrue(pal(t), t)
        for t in ("か", "し", "あ", "ふ", "へ", "な"):
            self.assertFalse(pal(t) and t != "し", t)
        self.assertTrue(pal("し"))                       # sh は口蓋化子音

    def test_tempo_flags_catch_squeezed_and_stretched_morae(self):
        from otoium.verify import tempo_flags, tempo_deviation, MoraTiming
        m = parse_text("か")[0]
        mk = lambda i, a, b: MoraTiming(i, m, [("a", a, b)], a, a + 10, b, None, 0.0)
        # 通常 250ms のモーラ列。3番目だけ 40ms (実録音で見つかった ぬ=40ms の潰れ)、5番目は 900ms、最後は余韻で長い
        ds = [250, 250, 40, 250, 900, 250, 1000]
        ts, t0 = [], 0.0
        for i, dd in enumerate(ds):
            ts.append(mk(i, t0, t0 + dd))
            t0 += dd
        fl = tempo_flags(ts)
        self.assertEqual(fl[2][0], "REVIEW")
        self.assertEqual(fl[4][0], "WARN")
        self.assertEqual(fl[0][0], None)
        self.assertEqual(fl[6][0], None)                  # 最後のモーラは検査しない
        self.assertGreater(tempo_deviation(ts), 0.3)
        even = [mk(i, 250.0 * i, 250.0 * (i + 1)) for i in range(7)]
        self.assertTrue(all(x[0] is None for x in tempo_flags(even)))
        self.assertLess(tempo_deviation(even), 1e-9)
        self.assertEqual(tempo_flags(even[:3]), [(None, None)] * 3)    # 少数モーラでは検査しない

    def test_min_dur_ratio_prevents_squeezed_vowels(self):
        from otoium.aligner import build_tokens
        from otoium.model import AcousticModel
        m = AcousticModel()
        morae = parse_text("ぬぬ")
        base = build_tokens(morae, 1000, 5.0, Config(), m)
        forced = build_tokens(morae, 1000, 5.0, Config(min_dur_ratio=0.4), m)
        vb = [t for t in base if t.kind == "v"][0]
        vf = [t for t in forced if t.kind == "v"][0]
        self.assertGreater(vf.dmin, vb.dmin)
        self.assertGreaterEqual(vf.dmin * 5.0, 0.4 * 150 * 0.99)          # 学習前の母音平均 150ms の 4 割以上

    def test_restore_best_keeps_tuned_nn_weight(self):
        import copy as cp
        from otoium.model import AcousticModel
        from otoium.mlp import FrameNet
        p = Pipeline(Config(), log=lambda m: None)
        p.model = AcousticModel()
        p.model.nn_weight = 1.0                                           # 自動調整で決めた値
        old = cp.deepcopy(p.model)
        old.nn_weight = 2.0                                               # 古いチェックポイントの値
        p._best = {0: (1.46, old)}
        self.assertEqual(p._best_here(), 1.46)
        self.assertTrue(p._restore_best())
        self.assertEqual(p.model.nn_weight, 1.0)                          # 戻しても上書きされない
        self.assertIsNot(p.model, old)                                    # チェックポイント自体は保持 (参照を共有しない)
        p.model.nets = [FrameNet(4, 3)]                                   # NN 構成が違うチェックポイントには戻さない
        self.assertFalse(p._restore_best())
        self.assertLess(p._best_here(), -1e8)

    def test_guarded_update_accepts_damps_or_rolls_back(self):
        """学習の更新で得点が下がるなら、減衰 → 更新なし の順に試す (更新するほど悪化することがない)"""
        from otoium.model import AcousticModel

        def make(scores, **kw):
            p = Pipeline(Config(**kw), log=lambda m: None)
            p.model = AcousticModel()
            seq = list(scores)
            p._eval_score = lambda: seq.pop(0)
            p._checkpoint("base")                                     # 最高点 = scores[0]
            return p, p.model.mu.copy()

        bump = lambda p: setattr(p.model, "mu", p.model.mu + 1.0)    # 「更新」: 平均を +1 ずらす
        p, base = make([1.0, 1.2])                                    # (1) 改善 → そのまま採用
        ok, sc = p._guarded_update(lambda: bump(p), "u")
        self.assertTrue(ok)
        self.assertAlmostEqual(sc, 1.2)
        self.assertTrue(np.allclose(p.model.mu, base + 1.0))
        p, base = make([1.0, 0.5, 1.05])                              # (2) 悪化 → 0.5 に減衰すると改善 → 採用
        ok, sc = p._guarded_update(lambda: bump(p), "u")
        self.assertTrue(ok)
        self.assertTrue(np.allclose(p.model.mu, base + 0.5))
        p, base = make([1.0, 0.5, 0.6, 0.7])                          # (3) どれも悪化 → 更新前に戻す
        ok, sc = p._guarded_update(lambda: bump(p), "u")
        self.assertFalse(ok)
        self.assertTrue(np.allclose(p.model.mu, base))
        self.assertAlmostEqual(p._best_here(), 1.0)
        p, base = make([1.0, 0.999])                                  # (4) 許容幅 (0.003) 内の低下は採用
        ok, _ = p._guarded_update(lambda: bump(p), "u")
        self.assertTrue(ok)
        p = Pipeline(Config(guarded_updates=False), log=lambda m: None)      # ガード無効なら悪化しても採用
        p.model = AcousticModel()
        seq = [1.0, 0.1]
        p._eval_score = lambda: seq.pop(0)
        p._checkpoint("base")
        ok, _ = p._guarded_update(lambda: bump(p), "u")
        self.assertTrue(ok)

    def test_model_blend_interpolates_parameters_and_nets(self):
        from otoium.model import AcousticModel
        from otoium.mlp import FrameNet
        a, b = AcousticModel(), AcousticModel()
        b.mu = a.mu + 2.0
        b.dur_mu["a"] = a.dur_mu["a"] + 1.0
        a.nets, b.nets = [FrameNet(6, 3, seed=1)], [FrameNet(6, 3, seed=2)]
        m = a.blend(b, 0.25)
        self.assertTrue(np.allclose(m.mu, a.mu + 0.5))
        self.assertAlmostEqual(m.dur_mu["a"], a.dur_mu["a"] + 0.25)
        for wa, wb, wm in zip(a.nets[0].W, b.nets[0].W, m.nets[0].W):
            self.assertTrue(np.allclose(wm, 0.75 * wa + 0.25 * wb, atol=1e-6))
        self.assertTrue(np.allclose(a.mu, a.prior_mu))                    # 元のモデルは変更しない
        b.nets = []                                                       # NN 構成が違えば NN は混ぜない
        self.assertTrue(np.allclose(a.blend(b, 0.5).nets[0].W[0], a.nets[0].W[0]))

    def test_recheck_stops_when_all_strategies_are_exhausted(self):
        lines = []
        with tempfile.TemporaryDirectory() as d:
            make_bank(d, BANK[:4])
            noise = (0.2 * np.random.default_rng(0).standard_normal(44100 * 3)).astype("float32")
            write_wav(Path(d) / "_なにぬねの.wav", noise)
            Pipeline(Config(mode="vcv", accuracy="fast", recheck_rounds=5), log=lines.append).run(d)
        self.assertEqual(sum(1 for l in lines if "再検査" in l and "回目:" in l), 3)      # 5 回指定でも 3 回で打ち切り
        self.assertTrue(any("これ以上の手段がない" in l for l in lines))

    # ---- 連続音の収録方針 (声を繋げる。か行・た行・ぱ行などの無声子音の無音は正常) ----
    def test_silent_onset_consonants(self):
        so = lambda t: parse_text(t)[0].silent_onset
        for t in ("か", "た", "ぱ", "さ", "し", "は", "ひ", "ふ", "ち", "つ", "きゃ", "ぴょ", "てぃ"):
            self.assertTrue(so(t), t)                       # 無声子音: 発音上の無音/息の区間が入るのが正常
        for t in ("あ", "な", "ま", "ら", "や", "わ", "が", "ざ", "だ", "ば", "じ", "ん", "ぎゃ"):
            self.assertFalse(so(t), t)                      # 有声: 声が繋がるのが正しい

    def _gap_timings(self, text, gap_frames):
        from otoium.aligner import Seg
        from otoium.verify import build_timings
        mo = parse_text(text)
        g = gap_frames
        c1 = [Seg("frS" if mo[1].cons == "s" else "nas", 1, "c", 100 + g, 120 + g)]
        segs = [Seg("pau", -1, "pau", 0, 60), Seg("a", 0, "v", 60, 100), Seg("pau", -1, "pau", 100, 100 + g)] + c1 + \
               [Seg(mo[1].vowel, 1, "v", 120 + g, 160 + g), Seg("pau", -1, "pau", 160 + g, 200 + g)]
        return build_timings(segs, mo, 5.0, 2000.0)

    def test_gap_before_unvoiced_consonant_is_absorbed_and_not_warned(self):
        from otoium.aliases import make_entries
        t = self._gap_timings("あさ", 10)                   # 50ms の無音 + さ (無声摩擦音)
        self.assertAlmostEqual(t[1].gap_before, 0.0)
        self.assertAlmostEqual(t[1].gap_absorbed, 50.0, delta=1.0)
        self.assertAlmostEqual(t[1].cons_start, t[0].vowel_end, delta=0.01)   # 子音の開始 = 前の母音の終わり
        e = [x for x in make_entries("x.wav", t, 2000.0, Config(mode="vcv")) if x.kind == "VCV"][0]
        self.assertFalse(any("途切れ" in n for n in e.notes), e.notes)
        t = self._gap_timings("あさ", 30)                   # 150ms は長すぎるので吸収しない
        self.assertGreater(t[1].gap_before, 100.0)

    def test_gap_before_voiced_consonant_is_warned_as_broken_continuity(self):
        from otoium.aliases import make_entries
        t = self._gap_timings("あな", 10)                   # 有声 (鼻音) の前の 50ms の無音 = 声が途切れている
        self.assertAlmostEqual(t[1].gap_before, 50.0, delta=1.0)
        self.assertEqual(t[1].gap_absorbed, 0.0)
        e = [x for x in make_entries("x.wav", t, 2000.0, Config(mode="vcv")) if x.kind == "VCV"][0]
        self.assertEqual(e.status, "WARN")
        self.assertTrue(any("声が途切れています" in n for n in e.notes), e.notes)

    def test_energy_dip_measures_cut_vowels(self):
        from types import SimpleNamespace as NS
        from otoium.verify import energy_dip_db
        T = 400                                              # 5ms 刻み = 2000ms
        F = NS(hop_s=0.005, T=T, energy_db=np.full(T, 60.0))
        self.assertAlmostEqual(energy_dip_db(F, 1000.0, 500.0, 1500.0), 0.0)             # 繋がっている
        F.energy_db[190:210] = 25.0                                                      # 境界付近に 35dB の谷 (母音を切っている)
        self.assertGreater(energy_dip_db(F, 1000.0, 500.0, 1500.0), 30.0)
        F2 = NS(hop_s=0.005, T=T, energy_db=np.full(T, 60.0))
        F2.energy_db[195:205] = 55.0                                                     # 5dB の浅い谷 (自然な遷移)
        self.assertLess(energy_dip_db(F2, 1000.0, 500.0, 1500.0), 6.0)
        self.assertEqual(energy_dip_db(F, 20.0, 0.0, 40.0), 0.0)                          # 範囲が足りなければ 0

    def test_devoiced_verdict_prefers_independent_recognition(self):
        from otoium.verify import devoiced_verdict, GIDX
        i = GIDX["i"]
        lv = np.array([i] * 8 + [GIDX["DV"]] * 2)
        agree, v = devoiced_verdict(lv, i)                   # 独立認識が有声の i → 整列の選び間違い (実録音で13件)
        self.assertEqual(v, "voiced")
        self.assertAlmostEqual(agree, 1.0)
        agree, v = devoiced_verdict(np.array([GIDX["DV"]] * 6 + [GIDX["FR"]] * 4), i)    # 息の音 → 無声化母音
        self.assertEqual(v, "devoiced")
        self.assertAlmostEqual(agree, 1.0)
        agree, v = devoiced_verdict(np.array([GIDX["a"]] * 10), i)                       # 別の母音 → 不一致
        self.assertEqual((v, agree), ("devoiced", 0.0))

    def test_cut_vowel_recording_is_flagged_and_connected_one_is_not(self):
        from tests.synth import SR
        with tempfile.TemporaryDirectory() as d:
            make_bank(d)

            def vowels(gap_ms, seed0):
                parts = []
                for i, v in enumerate("あいうえお"):          # 母音を 1 つずつ発音し、間に gap_ms の (ほぼ) 無音を入れる
                    x, _ = render("_" + v, seed=seed0 + i, lead=0.0, trail=0.0)
                    parts += [x, np.zeros(int(gap_ms / 1000 * SR), dtype="float32") + 0.0005]
                z = np.zeros(int(0.3 * SR), dtype="float32") + 0.0005
                return np.concatenate([z] + parts + [z])

            connected, _ = render("_あいうえお", seed=71)
            write_wav(Path(d) / "_あいうえお1.wav", connected)          # 繋げて発音
            write_wav(Path(d) / "_あいうえお2.wav", vowels(90, 80))     # 90ms の無音で切る (整列上のモーラ間ギャップ)
            write_wav(Path(d) / "_あいうえお3.wav", vowels(25, 90))     # 25ms の切れ目 (ギャップにならない短い谷)
            p, rep = run(d, mode="vcv", accuracy="fast", recheck_rounds=0)

            def broken(name):
                f = [f for f in rep["files"] if f["file"] == name][0]
                a = [n for e in f["aliases"] for n in e["notes"] if "途切れ" in n]
                m = [n for mo in f["morae"] for n in mo["notes"] if "途切れ" in n]
                return a, m
            self.assertEqual(broken("_あいうえお1.wav"), ([], []))                  # 繋がった収録には出ない
            a, m = broken("_あいうえお2.wav")
            self.assertGreaterEqual(len(a) + len(m), 3)                             # 無音で切った収録は検出される
            a, m = broken("_あいうえお3.wav")
            self.assertGreaterEqual(len(a) + len(m), 3)                             # 短い切れ目もエネルギーの谷で検出される
            for f in rep["files"]:                                                  # 通常の連続音 (バンク) には誤って出ない
                if f["file"][:-4] in BANK:
                    self.assertEqual(broken(f["file"]), ([], []), f["file"])

    def test_resolve_presets(self):
        from otoium.config import resolve
        self.assertFalse(resolve(Config(accuracy="fast"))["net"])
        u, m = resolve(Config(accuracy="ultra")), resolve(Config(accuracy="max"))
        self.assertGreater(u["iters"], m["iters"])
        self.assertGreater(u["ens"], m["ens"])
        c = resolve(Config(accuracy="custom", train_iters=20, tune_rounds=0, adapt_rounds=0))
        self.assertEqual((c["iters"], c["tune"], c["adapt"]), (20, False, False))

    def test_whole_8mora_script_parses(self):
        path = Path(__file__).resolve().parents[1] / "scripts" / "8mora.txt"
        for line in path.read_text(encoding="utf-8").split():
            parse_text(line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
