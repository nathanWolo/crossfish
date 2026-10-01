"""Dense, deterministic text encoding for generated payloads.

CodinGame caps a submission at 100,000 UTF-16 code units, not bytes. Every
character below U+10000 is one unit, so mapping 15-bit groups onto an
alphabet of 2^15 such characters carries 15 payload bits per counted
character, against 6.4 for ASCII85.

The U15 alphabet is U+3400..U+9FFF (CJK Unified Ideographs Extension A, the
64 Yijing hexagram symbols, CJK Unified Ideographs: 27,648 characters) followed
by U+E000..U+F3FF (5,120 Private Use characters). None of them is a combining
mark, a line or paragraph separator, a bidi control, an invisible character or
a character with a Unicode normalization decomposition, so an editor or paste
box has nothing to rewrite (the private-use characters merely render as boxes).
Each is three UTF-8 bytes, which the C++ decoders read straight out of an
ordinary narrow string literal.

CJK14 (14 bits per character, U+4E00..U+8DFF) is the previous alphabet, kept
to read headers generated before the switch.
"""

from __future__ import annotations

CJK14_BASE = 0x4E00
CJK14_BITS = 14
_MASK = (1 << CJK14_BITS) - 1

U15_BITS = 15
U15_SPLIT = 0xA000 - 0x3400  # 27,648 characters in the first range


def u15_char(value: int) -> str:
    return chr(0x3400 + value) if value < U15_SPLIT else chr(0xE000 + value - U15_SPLIT)


def u15_value(char: str) -> int:
    """The 15-bit value of a U15 character, or -1 for any other character."""
    cp = ord(char)
    if 0x3400 <= cp < 0xA000:
        return cp - 0x3400
    if 0xE000 <= cp < 0xE000 + (1 << U15_BITS) - U15_SPLIT:
        return cp - 0xE000 + U15_SPLIT
    return -1


def encode_u15(data: bytes) -> str:
    """Pack bytes MSB-first into 15-bit groups; zero-pad the final group.

    Decoding yields floor(15 * chars / 8) bytes: the input plus one trailing
    zero byte when the final group carries 8 or more padding bits. Loaders size
    their reads from the known layout, not the count.
    """
    out: list[str] = []
    acc = 0
    bits = 0
    for byte in data:
        acc = (acc << 8) | byte
        bits += 8
        while bits >= U15_BITS:
            bits -= U15_BITS
            out.append(u15_char((acc >> bits) & ((1 << U15_BITS) - 1)))
        acc &= (1 << bits) - 1
    if bits:
        out.append(u15_char((acc << (U15_BITS - bits)) & ((1 << U15_BITS) - 1)))
    return "".join(out)


def decode_u15(text: str) -> bytes:
    """Mirror of the C++ decoders: skips characters outside the alphabet."""
    out = bytearray()
    acc = 0
    bits = 0
    for char in text:
        value = u15_value(char)
        if value < 0:
            continue
        acc = (acc << U15_BITS) | value
        bits += U15_BITS
        while bits >= 8:
            bits -= 8
            out.append((acc >> bits) & 0xFF)
        acc &= (1 << bits) - 1
    return bytes(out)


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
        acc = (acc << 15) | (code >= 0xE000u ? code - 0xE000u + 27648u : code - 0x3400u);
        bits += 15;
        while (bits >= 8) {
            if (n >= out_max) return -1;
            bits -= 8;
            out[n++] = (unsigned char)(acc >> bits);
        }
    }
    return n;
}
'''
