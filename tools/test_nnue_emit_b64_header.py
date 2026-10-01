"""The committed NNUE payload (cpp_impl/nnue_b64_net.hpp) against the verified CodinGame build.

The constants are the verified build's (net r13w_11, improvement log section 64; the checkpoint and export are in
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

PAYLOAD_SHA256 = "712039a021b912bc92503b7f66c70cd6079a798f7f1734e72be6108f45342047"
SCALES = (9, 12, 13, 13, 10)
TABLE_HASHES = dict(
    T=0x8467a8b43b1781ab, TP=0xd115d9d2344a015c, F=0xcdae0bbc33217c1b, FP=0x0827752cb06824cb,
    DEC=0x10b25653b45b4759, DECP=0x25e158f8276b3d58, CON=0x3538d69f7a3a74e7, CONP=0x71329dd7cab3cabd,
    BIAS=0x27b847b7560d6d92, BIASP=0x9a691300c548b8fb, W1p=0x4ffb93373ab96169, B1=0x9650ef9b37e383d3,
    W2p=0x7155982c2fbf7f0f, B2=0x5435d739cc7b6f4b, WO=0x084874555f3d0ae2, BO=0x64be4e773b169f15)


class CommittedHeaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload, cls.qexp = emit.parse_header(emit.HEADER)
        cls.params = emit.generator_params(emit.unpack(cls.payload))
        cls.T, cls.F = emit.bake_f32(cls.params)

    def test_payload_is_the_verified_one(self):
        import hashlib
        self.assertEqual(len(self.payload), 53927)
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
