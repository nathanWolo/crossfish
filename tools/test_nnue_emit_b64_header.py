"""The committed NNUE payload (cpp_impl/nnue_b64_net.hpp) against the verified CodinGame build.

The constants are the verified build's (net r16_x128_l2400_s1601_rs, W1, improvement log section 69; the
checkpoint and export are in datasets/nnue2/, not in the repository): the sha256 of its payload, its widths
(encoder 27 -> 128 -> 128 -> 32 on the B-64 head) and the hashes of the 16 integer tables its loader bakes, as
printed by `python tools/nnue_emit_b64_header.py --check`.
unit_tests.cpp checks the same table hashes on the C++ loader, so the two mirrors of load() agree.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nnue_emit_b64_header as emit  # noqa: E402

PAYLOAD_SHA256 = "c6d0c3ada487e8292312240aaba2624b0de796852908f2af2faded6372d261a2"
SCALES = (9, 12, 13, 13, 10)
TABLE_HASHES = dict(
    T=0x1508b99e11f9398a, TP=0x6477c257efc89302, F=0x1719ae9206389376, FP=0xba92cd6962ac309f,
    DEC=0x4766c17357e5054e, DECP=0xa44ed68d4c3606d6, CON=0xd0dbaf4efd5111e4, CONP=0x349a2b848d3f5ba0,
    BIAS=0x90c2f9f247591608, BIASP=0x9a691300c548b8fb, W1p=0xc4e128ceb1f4faf5, B1=0xf755960c4bce70ef,
    W2p=0xb00a6ec10065dc86, B2=0x4bf8e6cb92fa4fb9, WO=0x640607e5087c6bca, BO=0x8464289de03c56a4)


class CommittedHeaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload, cls.qexp = emit.parse_header(emit.HEADER)
        cls.params = emit.generator_params(emit.unpack(cls.payload))
        cls.T, cls.F = emit.bake_f32(cls.params)

    def test_payload_is_the_verified_one(self):
        import hashlib
        self.assertEqual(len(self.payload), 80027)
        self.assertEqual(hashlib.sha256(self.payload).hexdigest(), PAYLOAD_SHA256)
        self.assertEqual(self.qexp, SCALES)

    def test_widths_are_the_verified_ones(self):
        self.assertEqual((emit.A, emit.L1, emit.L2), (64, 16, 32))
        self.assertEqual(tuple(emit.ENC), (128, 128, 32))
        self.assertEqual(emit.n_params(), 51435)

    def test_scales_fit_the_net(self):
        got, bounds = emit.scales(self.params, self.T, self.F)
        self.assertEqual(tuple(got), SCALES)
        lo, hi = bounds["with_con_f"]
        self.assertTrue(-32768 <= lo and hi <= 32767)

    def test_baked_tables_are_the_verified_ones(self):
        hashes = emit.table_hashes(emit.int_tables(self.params, self.T, self.F, self.qexp))
        for name, want in TABLE_HASHES.items():
            self.assertEqual(hashes[name], want, name)


class PayloadCodingTest(unittest.TestCase):
    def test_pack_unpack_round_trip(self):
        rng = np.random.default_rng(7)
        Q, SC = {}, {}
        for name in emit.MATS:
            rows, cols = emit.SHAPES[name]
            Q[name] = rng.integers(-300, 300, size=(rows, cols))
            n_scales = emit.scale_index(name, rows).max() + 1
            SC[name] = emit.bf16_up(rng.uniform(1e-4, 1e-2, n_scales))
        back = emit.unpack(emit.pack(Q, SC))
        for name in emit.MATS:
            steps = SC[name][emit.scale_index(name, Q[name].shape[0])]
            self.assertTrue(np.array_equal(back[name], emit.deq(Q[name], steps)), name)

    def test_bf16_up_is_the_smallest_bf16_not_below(self):
        x = np.array([0.0, 1.0, 1.00001, 3.14159, 1e-3], np.float32)
        b = emit.bf16_up(x)
        self.assertTrue(np.all(b >= x))
        self.assertTrue(np.all((b.view(np.uint32) & 0xFFFF) == 0))
        self.assertEqual(float(b[1]), 1.0)


if __name__ == "__main__":
    unittest.main()
