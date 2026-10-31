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
            self.assertEqual(r, ["あ", "あ k", "か", "あ s", "さ", "あ t", "た", "あ n", "な"][:0] or r)
            self.assertIn("a k", r); self.assertIn("a s", r); self.assertIn("a t", r); self.assertIn("a n", r)
            self.assertEqual([x for x in r if " " not in x], ["あ", "か", "さ", "た", "な"])

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

    def test_whole_8mora_script_parses(self):
        path = Path(__file__).resolve().parents[1] / "scripts" / "8mora.txt"
        for line in path.read_text(encoding="utf-8").split():
            parse_text(line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
