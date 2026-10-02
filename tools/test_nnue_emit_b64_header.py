"""The committed NNUE payload (cpp_impl/nnue_b64_net.hpp) against the verified CodinGame build.

The constants are the verified build's (net r13w_20, improvement log section 64; the checkpoint and export are in
datasets/nnue2/, not in the repository): the sha256 of its payload and the hashes of the 16 integer tables its
loader bakes, as printed by `python tools/nnue_emit_b64_header.py --check`.
unit_tests.cpp checks the same table hashes on the C++ loader, so the two mirrors of load() agree.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nnue_emit_b64_header as emit  # noqa: E402

PAYLOAD_SHA256 = "492a36eaf2f1c1011fdd5fb2533a4ee2416574dfb73346252511ef2accd11916"
SCALES = (9, 12, 13, 13, 10)
TABLE_HASHES = dict(
    T=0xf623c7a433e9a061, TP=0x9f6946f7fa2959aa, F=0xafded9ee1ef97baf, FP=0xa2b2b1f01b573ae4,
    DEC=0xe1d9159b20fda09d, DECP=0xc3f79c86aa8e1c19, CON=0xbbfe1b08d4e93216, CONP=0x28e3600224ecb9d8,
    BIAS=0xf0e9125df319e7bb, BIASP=0x9a691300c548b8fb, W1p=0x1523376b7f54dcf9, B1=0x1d236f5c045033a8,
    W2p=0xad0edf378bbe6acb, B2=0x9a2c5c8b9668e27b, WO=0x7e46fe2cf04bc9df, BO=0x064ec39d99e87f9d)


class CommittedHeaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload, cls.qexp = emit.parse_header(emit.HEADER)
        cls.params = emit.generator_params(emit.unpack(cls.payload))
        cls.T, cls.F = emit.bake_f32(cls.params)

    def test_payload_is_the_verified_one(self):
        import hashlib
        self.assertEqual(len(self.payload), 53834)
        self.assertEqual(hashlib.sha256(self.payload).hexdigest(), PAYLOAD_SHA256)
        self.assertEqual(self.qexp, SCALES)

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
