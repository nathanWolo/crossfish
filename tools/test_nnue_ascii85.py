import base64
import random
import unittest

from tools.nnue_ascii85 import encode_ascii85, wrap_ascii85


class TestAscii85Encoding(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(encode_ascii85(b""), "")

    def test_zero_group_is_not_abbreviated(self):
        self.assertEqual(encode_ascii85(b"\0\0\0\0"), "!!!!!")

    def test_roundtrip_all_tail_lengths(self):
        rng = random.Random(7)
        for length in range(1, 130):
            payload = bytes(rng.randrange(256) for _ in range(length))
            encoded = encode_ascii85(payload)
            self.assertTrue(all(33 <= ord(char) <= 117 for char in encoded))
            self.assertEqual(base64.a85decode(encoded), payload)

    def test_wrap_preserves_payload(self):
        encoded = encode_ascii85(bytes(range(64)))
        wrapped = wrap_ascii85(encoded, width=13)
        self.assertEqual(wrapped.replace("\n", ""), encoded)
        self.assertTrue(all(len(line) <= 13 for line in wrapped.splitlines()))


if __name__ == "__main__":
    unittest.main()
