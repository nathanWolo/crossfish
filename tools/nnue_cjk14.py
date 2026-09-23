"""Dense, deterministic text encoding for generated NNUE payloads.

CodinGame caps a submission at 100,000 UTF-16 code units, not bytes. Every
character in U+4E00..U+8DFF (CJK Unified Ideographs) is one UTF-16 unit, so
mapping 14-bit groups onto that block carries 14 payload bits per counted
character, against 6.4 for ASCII85.

The block is deliberately plain ideographs: no combining marks, line or
paragraph separators, bidi controls, invisible characters or characters with
normalization decompositions, so an editor or paste box has nothing to
rewrite. Each character is three UTF-8 bytes, which the C++ decoder reads
straight out of an ordinary narrow string literal.
"""

from __future__ import annotations

CJK14_BASE = 0x4E00
CJK14_BITS = 14
_MASK = (1 << CJK14_BITS) - 1


def encode_cjk14(data: bytes) -> str:
    """Pack bytes MSB-first into 14-bit groups; zero-pad the final group.

    Decoding yields floor(14 * chars / 8) bytes, which is the input length
    plus one trailing zero byte when the final group carries 8 or more padding
    bits. Loaders must size their reads from the known layout, not the count.
    """
    out: list[str] = []
    acc = 0
    bits = 0
    for byte in data:
        acc = (acc << 8) | byte
        bits += 8
        while bits >= CJK14_BITS:
            bits -= CJK14_BITS
            out.append(chr(CJK14_BASE + ((acc >> bits) & _MASK)))
        acc &= (1 << bits) - 1
    if bits:
        out.append(chr(CJK14_BASE + ((acc << (CJK14_BITS - bits)) & _MASK)))
    return "".join(out)


def decode_cjk14(text: str) -> bytes:
    """Mirror of the C++ decoder: skips characters outside the alphabet."""
    out = bytearray()
    acc = 0
    bits = 0
    for char in text:
        value = ord(char) - CJK14_BASE
        if not 0 <= value <= _MASK:
            continue
        acc = (acc << CJK14_BITS) | value
        bits += CJK14_BITS
        while bits >= 8:
            bits -= 8
            out.append((acc >> bits) & 0xFF)
        acc &= (1 << bits) - 1
    return bytes(out)


def wrap_cjk14(encoded: str, width: int = 64) -> str:
    return "\n".join(
        encoded[offset : offset + width]
        for offset in range(0, len(encoded), width)
    )


# Emitted verbatim into generated headers; `mini` is renamed per symbol tag.
CJK14_DECODER = r'''static int mini_cjk_decode(
    const char *s, unsigned char *out, int out_max) {
    int n = 0;
    int bits = 0;
    uint32_t acc = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if ((*p & 0xF0) != 0xE0) continue;
        uint32_t code = ((p[0] & 15u) << 12) | ((p[1] & 63u) << 6)
                      | (p[2] & 63u);
        p += 2;
        acc = (acc << 14) | (code - 0x4E00u);
        bits += 14;
        while (bits >= 8) {
            if (n >= out_max) return -1;
            bits -= 8;
            out[n++] = (unsigned char)(acc >> bits);
        }
    }
    return n;
}
'''
