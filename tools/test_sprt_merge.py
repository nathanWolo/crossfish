"""sprt_merge must reproduce test_bots's pentanomial LLR and Elo, or pooled shards would be judged differently."""
import unittest

from sprt_merge import llr, pentanomial_elo


class TestSprtMerge(unittest.TestCase):
    # Lines printed by test_bots (90 ms, H0 0 / H1 5) during this week's SPRTs.
    CASES = [
        # (pentanomial counts, Elo, CI, LLR)
        ([144, 622, 1046, 615, 183], 4.72595, 6.58057, 0.988274),
        ([697, 2749, 4475, 2843, 726], 2.29813, 3.14772, -0.39131),
    ]

    def test_matches_test_bots(self):
        for penta, elo, ci, value in self.CASES:
            got_elo, got_ci = pentanomial_elo(penta)
            self.assertAlmostEqual(got_elo, elo, places=3)
            self.assertAlmostEqual(got_ci, ci, places=3)
            self.assertAlmostEqual(llr(penta, 0.0, 5.0), value, places=4)

    def test_pooling_is_summing(self):
        a, b = [10, 40, 60, 45, 12], [5, 30, 50, 33, 9]
        pooled = [x + y for x, y in zip(a, b)]
        self.assertGreater(abs(llr(pooled, 0.0, 5.0)), 0.0)
        self.assertEqual(llr([0, 0, 0, 0, 0], 0.0, 5.0), 0.0)


if __name__ == "__main__":
    unittest.main()
