"""Checks of the native CodinGame submission cpp_impl/cg_input_native.py.

These run anywhere (no clang, no Linux): the committed launcher must be valid
Python 3, fit CodinGame's 100,000-character cap with no surrogates, and carry
exactly the binary that tools/cg_native/manifest.json records. The binary is
built from the bot's sources, so the manifest also records their hashes: a
change to the bot that is not followed by `make cg-native` (on Linux) fails
here, because the live submission would no longer be the bot in the repository.
See documentation/native_build.md.
"""
from __future__ import annotations

import ast
import hashlib
import json
import lzma
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "cg_native"))

import pack  # noqa: E402

MANIFEST = json.loads((ROOT / "tools" / "cg_native" / "manifest.json").read_text(encoding="utf-8"))
SUBMISSION = ROOT / MANIFEST["submission"]


def read_submission() -> str:
    # The file is checked out with LF everywhere (.gitattributes); normalize anyway,
    # so a CRLF checkout fails only the byte-identity test.
    return SUBMISSION.read_bytes().decode("utf-8")


class TestNativeSubmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_text = read_submission()
        cls.text = cls.raw_text.replace("\r\n", "\n")
        tree = ast.parse(cls.text)
        cls.payload = next(n.value.value for n in tree.body
                           if isinstance(n, ast.Assign) and n.targets[0].id == "D")
        m = re.search(r"to_bytes\((\d+),", cls.text)
        cls.xz = pack.u15_decode(cls.payload, int(m.group(1)))
        cls.binary = lzma.decompress(cls.xz)

    def test_byte_identical_to_manifest(self):
        self.assertEqual(hashlib.sha256(self.raw_text.encode()).hexdigest(),
                         MANIFEST["submission_sha256"],
                         "cg_input_native.py is not the file the last `make cg-native` wrote")

    def test_fits_codingame(self):
        units = len(self.text.encode("utf-16-le")) // 2
        self.assertEqual(units, MANIFEST["submission_utf16_units"])
        self.assertLess(units, pack.CAP)
        self.assertFalse(any(0xD800 <= ord(c) <= 0xDFFF for c in self.text))
        self.assertTrue(self.text.splitlines()[1].startswith('D="'))  # the payload is one line
        compile(self.text, "cg_input_native.py", "exec")

    def test_payload_is_the_recorded_binary(self):
        self.assertEqual(hashlib.sha256(self.xz).hexdigest(), MANIFEST["xz_sha256"])
        self.assertEqual(len(self.binary), MANIFEST["binary_bytes"])
        self.assertEqual(hashlib.sha256(self.binary).hexdigest(), MANIFEST["binary_sha256"])
        # A stripped x86-64 ELF executable.
        self.assertEqual(self.binary[:4], b"\x7fELF")
        self.assertEqual(self.binary[4], 2)  # 64-bit
        self.assertEqual(int.from_bytes(self.binary[18:20], "little"), 62)  # EM_X86_64

    def test_launcher_matches_the_packer(self):
        # The packer's template and encoder reproduce the file from its own xz
        # stream (no recompression: liblzma versions may encode differently).
        self.assertEqual(pack.u15(self.xz), self.payload)
        self.assertEqual(pack.render(self.binary, self.xz), self.text)
        self.assertIn('"cf_%s"' % MANIFEST["binary_sha256"][:12], self.text)

    def test_sources_unchanged_since_the_build(self):
        stale = [p for p, h in MANIFEST["sources"].items() if pack.source_sha256(ROOT / p) != h]
        self.assertEqual(stale, [], "the native submission was built from other sources; run "
                         "`make cg-native` and `make cg-native-check` on Linux "
                         "(documentation/native_build.md)")


if __name__ == "__main__":
    unittest.main()
