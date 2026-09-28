"""The committed NNUE payload (cpp_impl/nnue_b64_net.hpp) against the verified CodinGame build.

The constants are the verified CodinGame build's (net r12_M2, datasets/nnue2/cg/r12, not in the repository): the sha256
of its payload and the hashes of the 16 integer tables its loader bakes, as printed by
`python tools/nnue_emit_b64_header.py --check`.
unit_tests.cpp checks the same table hashes on the C++ loader, so the two mirrors of load() agree.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nnue_emit_b64_header as emit  # noqa: E402

PAYLOAD_SHA256 = "4ba93b1a422c480c9876e1a826f91b813d024af2541aac7b1b951e6f76e43a2b"
SCALES = (9, 12, 13, 13, 11)
TABLE_HASHES = dict(
    T=0xfaccfc87cfd0b6d0, TP=0x9e3ec366d8c3fb4a, F=0xf456392f8ba17975, FP=0xd331d7412ca0185a,
    DEC=0xf696cbe1be77ca93, DECP=0xae23933a779e37ad, CON=0x3157b6212f30ac80, CONP=0x99bd53d0b46749e3,
    BIAS=0x02e8408a9380a607, BIASP=0x9a691300c548b8fb, W1p=0x7cf86c63bbd44d5e, B1=0xeddd686a2a1ccfe3,
    W2p=0x70f17bbda9b46047, B2=0x4db449aabff4541e, WO=0x7662257aebc69d1f, BO=0xcafde29f2adc4a1f)


class CommittedHeaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload, cls.qexp = emit.parse_header(emit.HEADER)
        cls.params = emit.generator_params(emit.unpack(cls.payload))
        cls.T, cls.F = emit.bake_f32(cls.params)

    def test_payload_is_the_verified_one(self):
        import hashlib
        self.assertEqual(len(self.payload), 54159)
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
