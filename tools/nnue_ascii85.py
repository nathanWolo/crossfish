"""Compact, deterministic ASCII85 encoding for generated NNUE headers."""

from __future__ import annotations


def encode_ascii85(data: bytes) -> str:
    """Encode without the optional ``z`` shorthand.

    The emitted alphabet is the contiguous range ``!`` through ``u``. This
    keeps the C++ decoder small and makes every four payload bytes occupy five
    source characters.
    """
    encoded: list[str] = []
    for offset in range(0, len(data), 4):
        chunk = data[offset : offset + 4]
        value = int.from_bytes(chunk.ljust(4, b"\0"), "big")
        digits = [""] * 5
        for index in range(4, -1, -1):
            value, digit = divmod(value, 85)
            digits[index] = chr(digit + 33)
        encoded.extend(digits[: len(chunk) + 1])
    return "".join(encoded)


def wrap_ascii85(encoded: str, width: int = 100) -> str:
    return "\n".join(
        encoded[offset : offset + width]
        for offset in range(0, len(encoded), width)
    )
