"""Unit tests for the CodinGame performance gate's pure helpers."""
import math
import sys
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


if __name__ == "__main__":
    unittest.main()
