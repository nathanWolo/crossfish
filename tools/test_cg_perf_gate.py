"""Unit tests for the CodinGame performance gate's pure helpers."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cg_perf_gate as gate  # noqa: E402


class RatioTest(unittest.TestCase):
    def test_identical_builds_give_ratio_one(self):
        r, lo, hi = gate.ratio_ci([(700_000, 700_000)] * 8)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(lo, 1.0)
        self.assertAlmostEqual(hi, 1.0)

    def test_uniform_slowdown_is_recovered(self):
        r, lo, hi = gate.ratio_ci([(a, a * 0.9) for a in (600_000, 700_000, 800_000, 650_000)])
        self.assertAlmostEqual(r, 0.9)
        self.assertLessEqual(lo, r)
        self.assertGreaterEqual(hi, r)

    def test_interval_contains_geometric_mean(self):
        pairs = [(100, 90), (100, 110), (100, 95), (100, 105), (100, 100)]
        r, lo, hi = gate.ratio_ci(pairs)
        self.assertTrue(lo < r < hi)
        self.assertAlmostEqual(r, math.exp(sum(math.log(b / a) for a, b in pairs) / len(pairs)))


class SizeTest(unittest.TestCase):
    def test_counts_utf16_units_like_codingame(self):
        # CJK14 payload characters are one UTF-16 unit each, astral ones two.
        self.assertEqual(gate.utf16_len("abc"), 3)
        self.assertEqual(gate.utf16_len("一跿"), 2)
        self.assertEqual(gate.utf16_len("\U0001f600"), 2)


class EvalChangeTest(unittest.TestCase):
    BASE = "a" * 64

    def write(self, **decl):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        path = Path(d.name) / "eval_change.json"
        body = dict(change="a new net", reason="slower per node, stronger", base_cg_input_sha256=self.BASE,
                    speed_ratio=[0.45, 0.7])
        body.update(decl)
        path.write_text(json.dumps(body), encoding="utf-8")
        return path

    def test_no_file_means_no_declaration(self):
        self.assertEqual(gate.read_eval_change(Path("does/not/exist.json"), self.BASE), (None, ""))

    def test_applies_only_to_the_base_it_names(self):
        path = self.write()
        decl, note = gate.read_eval_change(path, self.BASE)
        self.assertEqual(decl["speed_ratio"], (0.45, 0.7))
        self.assertIn("applies", note)
        stale, note = gate.read_eval_change(path, "b" * 64)
        self.assertIsNone(stale)
        self.assertIn("ignored", note)

    def test_malformed_declarations_are_errors(self):
        for bad in (dict(speed_ratio=[0.7, 0.45]), dict(speed_ratio=[0, 0.5]), dict(speed_ratio=[0.5]),
                    dict(base_cg_input_sha256="abc"), dict(reason=" "), dict(change="")):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                gate.read_eval_change(self.write(**bad), self.BASE)

    def test_speed_verdict(self):
        # usual check: a material and significant slowdown fails
        self.assertTrue(gate.speed_passes(0.97, 0.99, 0.05))
        self.assertTrue(gate.speed_passes(0.90, 1.01, 0.05))
        self.assertFalse(gate.speed_passes(0.90, 0.95, 0.05))
        # declared: the point ratio must lie in the range, at both ends
        decl = dict(speed_ratio=(0.45, 0.7))
        self.assertTrue(gate.speed_passes(0.55, 0.6, 0.05, decl))
        self.assertFalse(gate.speed_passes(0.40, 0.45, 0.05, decl))
        self.assertFalse(gate.speed_passes(0.98, 1.02, 0.05, decl))

    def test_sha_ignores_line_endings(self):
        self.assertEqual(gate.text_sha256("a\r\nb\n"), gate.text_sha256("a\nb\n"))

    def test_committed_declaration_is_well_formed(self):
        path = gate.EVAL_CHANGE
        if not path.exists():
            self.skipTest("no eval-change declaration in this tree")
        base = json.loads(path.read_text(encoding="utf-8"))["base_cg_input_sha256"]
        decl, _ = gate.read_eval_change(path, base)
        lo, hi = decl["speed_ratio"]
        self.assertTrue(0 < lo < hi < 1.0)  # an eval change that is not slower needs no declaration


if __name__ == "__main__":
    unittest.main()
