"""The committed NNUE payload (cpp_impl/nnue_b64_net.hpp) against the verified CodinGame build.

The constants are the verified build's (net r14_d5_final_s2_rs, improvement log section 65; the checkpoint and
export are in datasets/nnue2/, not in the repository): the sha256 of its payload and the hashes of the 16 integer
tables its loader bakes, as printed by `python tools/nnue_emit_b64_header.py --check`.
unit_tests.cpp checks the same table hashes on the C++ loader, so the two mirrors of load() agree.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nnue_emit_b64_header as emit  # noqa: E402

PAYLOAD_SHA256 = "cde8c6109b36689ecf43faf5a049c8225a54c364077aad8290832eb416221834"
SCALES = (9, 12, 13, 13, 10)
TABLE_HASHES = dict(
    T=0x9bbaa082f8554047, TP=0xdd257686df3e3139, F=0x08b77ff215924031, FP=0xa8fb18727ed0c670,
    DEC=0x38f22f9a6c383aeb, DECP=0xd64aeb301bcf3bab, CON=0x3debb6217730b511, CONP=0x888f865e5c67d11f,
    BIAS=0x10cc379b6ed5fc3d, BIASP=0x9a691300c548b8fb, W1p=0x0b0d9ecc90d342cb, B1=0x00c61dcbf7904c1a,
    W2p=0x83a5f9151fe78cc5, B2=0x6544804488cf5d81, WO=0xb5d9f93d417af25e, BO=0x5360ca77c125e377)


class CommittedHeaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload, cls.qexp = emit.parse_header(emit.HEADER)
        cls.params = emit.generator_params(emit.unpack(cls.payload))
        cls.T, cls.F = emit.bake_f32(cls.params)

    def test_payload_is_the_verified_one(self):
        import hashlib
        self.assertEqual(len(self.payload), 53865)
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
