import sys, tempfile, unittest, json
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
            run(d, mode="cvvc", accuracy="fast", alias_prefix="", merge=True, textgrid=True, export_labels=True)
            self.assertTrue((Path(d) / "_あかさたな.TextGrid").exists())
            self.assertTrue((Path(d) / "otoium_model.npz").exists())

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
            p, rep = run(d, mode="vcv", accuracy="high")
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
                         net_ensemble=2, adapt_rounds=2, post_iters=1, confident_rounds=1)
            self.assertEqual(len(p.model.nets), 2)
            self.assertEqual(rep["failed"], 0)
            self.assertEqual([a[0] for a in read_oto(d)["_あかさたな.wav"]], ["あ", "あ か", "か さ", "さ た", "た な"])
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
        self.assertEqual(sum(1 for l in lines if l.strip().startswith("学習 ") and "/6:" in l), 6)
        self.assertFalse(any("収束" in l for l in lines))

    def test_convergence_needs_flat_score_and_stable_boundaries(self):
        lines = []
        with tempfile.TemporaryDirectory() as d:
            make_bank(d)
            p = Pipeline(Config(mode="vcv", accuracy="custom", train_iters=12, patience=2, use_net=False,
                                tune_rounds=0, adapt_rounds=0, post_iters=0, confident_rounds=0), log=lines.append)
            p.run(d)
        done = [l for l in lines if l.strip().startswith("学習 ") and "/12:" in l]
        self.assertGreaterEqual(len(done), 6)            # 予定12回の半分までは必ず回す

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
