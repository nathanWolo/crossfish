#!/usr/bin/env python3
"""Aggressive C++ minifier for CodinGame's 100k-character cap.

Inspired by MinusKelvin/ice4 (TCEC 4K chess engine): tokenize, rename
identifiers to the shortest names that do not collide with kept symbols,
then pack tokens with a space only when two alphanumerics would glue.

ice4 also parses to an AST (decl/expr merging, scope coloring) and xz-compresses
the result into a self-extracting script. CodinGame needs compilable source, so
we skip xz. We also skip a full C++ parser — AVX, templates, and raw strings in
this bot are outside ice4's subset — and do global rename with a reserved set
instead of ice4's scope graph.

Preserves:
  - preprocessor lines (#include, #pragma) as their own lines
  - string / char / raw-string literals, including R"MNUE(...)MNUE"
  - C++ keywords, std names, intrinsics (__m256i, _mm256_*, __builtin_*)
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter

# ice4/minifier/renamer.rs IDENT_CHARACTERS; first character skips '_'.
_IDENT_CHARS = "_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

KEYWORDS = frozenset(
    """
    alignas alignof and and_eq asm atomic_cancel atomic_commit atomic_noexcept
    auto bitand bitor bool break case catch char char8_t char16_t char32_t
    class compl concept const consteval constexpr constinit const_cast continue
    co_await co_return co_yield decltype default delete do double dynamic_cast
    else enum explicit export extern false float for friend goto if inline int
    long mutable namespace new noexcept not not_eq nullptr operator or or_eq
    private protected public reflexpr register reinterpret_cast requires return
    short signed sizeof static static_assert static_cast struct switch
    synchronized template this thread_local throw true try typedef typeid
    typename union unsigned using virtual void volatile wchar_t while xor xor_eq
    override final consteval constinit
    """.split()
)

# Global names from C headers / cstdint (not written as std::T).
C_NAMES = frozenset(
    """
    std memcpy memset memmove malloc free strlen strcmp strncmp
    printf fprintf sprintf snprintf scanf sscanf
    stdin stdout stderr NULL EOF CLOCKS_PER_SEC
    int8_t int16_t int32_t int64_t uint8_t uint16_t uint32_t uint64_t
    size_t ptrdiff_t intptr_t uintptr_t
    UINT8_MAX UINT16_MAX UINT32_MAX UINT64_MAX INT8_MAX INT16_MAX
    INT32_MAX INT64_MAX SIZE_MAX EXIT_SUCCESS EXIT_FAILURE
    main argc argv
    abs labs llabs fabs fabsf fabsl exit atoi atol atof
    sqrt pow sin cos tan atan atan2 log exp floor ceil round
    lround llround
    """.split()
)

# std::vector/chrono/iostream methods used with . or -> (not after std::).
STD_METHODS = frozenset(
    """
    data size empty top push pop front back at begin end
    ignore flush get count now c_str length substr
    insert erase clear resize reserve emplace emplace_back
    push_back pop_back fill peek put write read
    good eof fail rdstate
    """.split()
)

RESERVED = KEYWORDS | C_NAMES

# Concatenating these adjacent single-char tokens would change the program.
_GLUE_PAIRS = frozenset(
    [
        ("+", "+"),
        ("+", "="),
        ("-", "-"),
        ("-", "="),
        ("-", ">"),
        ("&", "&"),
        ("&", "="),
        ("|", "|"),
        ("|", "="),
        ("<", "<"),
        ("<", "="),
        (">", ">"),
        (">", "="),
        ("=", "="),
        ("!", "="),
        ("*", "="),
        ("/", "="),
        ("%", "="),
        ("^", "="),
        ("/", "*"),
        ("/", "/"),
        (":", ":"),
        ("#", "#"),
        (".", "."),
        ("+", "-"),  # 1+-2 is fine, but keep 1e + -2 spacing via word rule
    ]
)


def _is_word_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def _is_ident(tok: str) -> bool:
    return bool(tok) and (tok[0].isalpha() or tok[0] == "_") and tok.replace("_", "a").isalnum()


def _raw_string_end(src: str, i: int) -> tuple[str, int]:
    n = len(src)
    j = i + 1
    if j >= n or src[j] != '"':
        return src[i], i + 1
    j += 1
    d0 = j
    while j < n and src[j] != "(":
        j += 1
    if j >= n:
        return src[i:], n
    delim = src[d0:j]
    j += 1
    close = ")" + delim + '"'
    k = src.find(close, j)
    if k < 0:
        return src[i:], n
    return src[i : k + len(close)], k + len(close)


def tokenize(src: str) -> list[str]:
    n = len(src)
    i = 0
    tokens: list[str] = []
    while i < n:
        c = src[i]
        if c in " \t\r\n":
            i += 1
            continue
        if c == "#":
            j = i + 1
            while j < n and src[j] != "\n":
                if src[j] == "\\" and j + 1 < n and src[j + 1] == "\n":
                    j += 2
                    continue
                j += 1
            tokens.append(src[i:j].rstrip() + "\n")
            i = j + 1 if j < n and src[j] == "\n" else j
            continue
        if c == "R" and i + 1 < n and src[i + 1] == '"':
            lex, i = _raw_string_end(src, i)
            if lex.startswith('R"') and "(" in lex:
                lp = lex.find("(")
                rp = lex.rfind(")")
                inner = lex[lp + 1 : rp].replace("\n", "").replace("\r", "")
                lex = lex[: lp + 1] + inner + lex[rp:]
            tokens.append(lex)
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            i += 2
            while i + 1 < n and not (src[i] == "*" and src[i + 1] == "/"):
                i += 1
            i = min(n, i + 2)
            continue
        if c == '"':
            j = i + 1
            while j < n:
                if src[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if src[j] == '"':
                    j += 1
                    break
                j += 1
            tokens.append(src[i:j])
            i = j
            continue
        if c == "'":
            j = i + 1
            while j < n:
                if src[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if src[j] == "'":
                    j += 1
                    break
                j += 1
            tokens.append(src[i:j])
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i + 1
            while j < n and _is_word_char(src[j]):
                j += 1
            tokens.append(src[i:j])
            i = j
            continue
        if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            j = i + 1
            while j < n and (_is_word_char(src[j]) or src[j] == "."):
                j += 1
            tokens.append(src[i:j])
            i = j
            continue
        three = src[i : i + 3]
        if three in ("<<=", ">>=", "..."):
            tokens.append(three)
            i += 3
            continue
        two = src[i : i + 2]
        if two in (
            "==", "!=", "<=", ">=", "&&", "||", "<<", ">>",
            "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
            "::", "->", "++", "--", "##",
        ):
            tokens.append(two)
            i += 2
            continue
        tokens.append(c)
        i += 1
    return tokens


def _keep_ident(name: str) -> bool:
    if name in RESERVED:
        return True
    if name.startswith("__") or name.startswith("_mm") or name.startswith("_MM"):
        return True
    return False


def _follows_std_scope(tokens: list[str], i: int) -> bool:
    """True if tokens[i] is inside a std::... nested-name specifier."""
    j = i
    while j >= 2 and tokens[j - 1] == "::" and _is_ident(tokens[j - 2]):
        if tokens[j - 2] == "std":
            return True
        j -= 2
    return False


def _after_member_op(tokens: list[str], i: int) -> bool:
    return i > 0 and tokens[i - 1] in (".", "->")


def generate_variable_names(count: int) -> list[str]:
    """Same generator as ice4/minifier/renamer.rs generate_variable_names."""
    idents: list[str] = []
    ident_state = [1]
    while len(idents) < count:
        ident = "".join(_IDENT_CHARS[i] for i in ident_state)
        idents.append(ident)
        for k in range(len(ident_state) - 1, -1, -1):
            ident_state[k] += 1
            if ident_state[k] == len(_IDENT_CHARS):
                ident_state[k] = 0
            else:
                break
        if ident_state[0] == 0:
            ident_state[0] = 1
            ident_state.append(0)
    return idents


def _user_ident_at(tokens: list[str], i: int) -> bool:
    t = tokens[i]
    if not _is_ident(t) or _keep_ident(t):
        return False
    if _follows_std_scope(tokens, i):
        return False
    if _after_member_op(tokens, i) and t in STD_METHODS:
        return False
    return True


def rename_identifiers(tokens: list[str]) -> list[str]:
    """Global rename: most expensive names get the shortest ice4-style ids.

    ice4 colors names that never share a scope so they can reuse `a`. Without
    an AST we map each distinct user identifier to its own short name, which
    is slightly worse but safe for this file.
    """
    counts: Counter[str] = Counter()
    for i, t in enumerate(tokens):
        if _user_ident_at(tokens, i):
            counts[t] += 1
    if not counts:
        return tokens

    # Every spelling already in the file is taken. ice4 reuses short names
    # across disjoint scopes via graph coloring; without an AST we do not.
    kept = set(RESERVED) | set(STD_METHODS)
    for t in tokens:
        if _is_ident(t):
            kept.add(t)

    # Highest (len-1)*freq first so mini_board_states beats a rare long name.
    ranked = sorted(counts, key=lambda n: (len(n) - 1) * counts[n], reverse=True)
    pool = generate_variable_names(len(ranked) + 128)
    mapping: dict[str, str] = {}
    pi = 0
    for name in ranked:
        if len(name) <= 1:
            continue
        while pi < len(pool) and pool[pi] in kept:
            pi += 1
        if pi >= len(pool):
            break
        short = pool[pi]
        pi += 1
        if len(short) < len(name):
            mapping[name] = short
            kept.add(short)

    out: list[str] = []
    for i, t in enumerate(tokens):
        if _user_ident_at(tokens, i):
            out.append(mapping.get(t, t))
        else:
            out.append(t)
    return out


def _compact_pp(tok: str) -> str:
    line = tok.strip()
    if line.startswith("#include"):
        line = re.sub(r"#include\s+<", "#include<", line)
        line = re.sub(r'#include\s+"', '#include"', line)
    return line + "\n"


def stringify(tokens: list[str]) -> str:
    """ice4 parse::stringify: space only when two word-chars would merge."""
    if not tokens:
        return ""
    out: list[str] = []
    prev = ""
    for t in tokens:
        if t.startswith("#") and t.endswith("\n"):
            if out and not str(out[-1]).endswith("\n"):
                out.append("\n")
            out.append(_compact_pp(t))
            prev = "\n"
            continue
        if prev:
            a = prev[-1]
            b = t[0]
            need_space = _is_word_char(a) and _is_word_char(b)
            if (a, b) in _GLUE_PAIRS:
                need_space = True
            if need_space:
                out.append(" ")
        out.append(t)
        prev = t
    text = "".join(out)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    if not text.endswith("\n"):
        text += "\n"
    return text


def minify_cpp(src: str, rename: bool = True) -> str:
    tokens = tokenize(src)
    if rename:
        tokens = rename_identifiers(tokens)
    return stringify(tokens)


def main() -> None:
    ap = argparse.ArgumentParser(description="Minify a C++ CodinGame submission")
    ap.add_argument("src")
    ap.add_argument("-o", "--out", default="")
    ap.add_argument("--no-rename", action="store_true", help="whitespace-only (old conservative mode)")
    args = ap.parse_args()
    with open(args.src, encoding="utf-8") as inf:
        src = inf.read()
    out = minify_cpp(src, rename=not args.no_rename)
    dest = args.out or args.src
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    print(
        f"{args.src} {len(src)} -> {dest} {len(out)}  saved {len(src) - len(out)}  "
        f"cap {100000 - len(out)} left"
    )
    if len(out) >= 100000:
        print("WARNING: still over 100k", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
