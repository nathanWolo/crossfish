import random
import unittest

from tools.nnue_cjk14 import (
    CJK14_BASE,
    decode_cjk14,
    encode_cjk14,
    wrap_cjk14,
)


def utf16_units(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


class TestCjk14Encoding(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(encode_cjk14(b""), "")
        self.assertEqual(decode_cjk14(""), b"")

    def test_known_vector(self):
        # Mirrored by the C++ test_cjk14_decoder vector.
        encoded = encode_cjk14(bytes(range(10)))
        self.assertEqual(
            [ord(c) for c in encoded],
            [0x4E00, 0x5E20, 0x5A10, 0x5306, 0x4FC2, 0x4E90],
        )
        self.assertEqual(
            [ord(c) for c in encode_cjk14(b"\xff" * 7)], [0x8DFF] * 4
        )

    def test_roundtrip_all_tail_lengths(self):
        rng = random.Random(7)
        for length in range(1, 200):
            payload = bytes(rng.randrange(256) for _ in range(length))
            encoded = encode_cjk14(payload)
            self.assertEqual(len(encoded), -(-8 * length // 14))
            self.assertTrue(
                all(CJK14_BASE <= ord(c) < CJK14_BASE + (1 << 14) for c in encoded)
            )
            decoded = decode_cjk14(encoded)
            # At most one trailing zero byte comes from final-group padding.
            self.assertEqual(decoded[:length], payload)
            self.assertLessEqual(len(decoded) - length, 1)
            self.assertFalse(any(decoded[length:]))

    def test_every_character_is_one_utf16_unit(self):
        encoded = encode_cjk14(bytes(range(256)) * 4)
        self.assertEqual(utf16_units(encoded), len(encoded))

    def test_density(self):
        payload = bytes(range(256)) * 64
        self.assertLessEqual(len(encode_cjk14(payload)) * 14, len(payload) * 8 + 13)

    def test_wrap_and_foreign_characters_are_skipped(self):
        payload = bytes(range(100))
        encoded = encode_cjk14(payload)
        wrapped = wrap_cjk14(encoded, width=13)
        self.assertEqual(wrapped.replace("\n", ""), encoded)
        self.assertTrue(all(len(line) <= 13 for line in wrapped.splitlines()))
        self.assertEqual(decode_cjk14(" \n" + wrapped + "\r\n")[:100], payload)


if __name__ == "__main__":
    unittest.main()
