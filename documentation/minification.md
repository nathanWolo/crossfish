# CodinGame submission generation and minification

This document describes how Crossfish turns the readable local engine into the
single C++ source file pasted into CodinGame. It covers the generated neural
evaluation data and opening-book payload, their textual encoding and runtime
reconstruction, local-header bundling, the tokenizer and identifier renamer,
output packing, how CodinGame compiles the result, and the checks required
before shipping a regenerated submission.

The short version is:

```text
accepted NNUE export (BGN1)          gameplay text book
        |                                    |
        v                                    v
tools/nnue_emit_b64_header.py        play_book_pack
  nnue_b64_net.hpp (+ nnue_b64.hpp)    play_book_data.hpp (+ play_book.hpp)
  macro_eval.hpp, d16_helpers.hpp            |
        |                                    |
        +---------- included by -------------+
                            |
codingame_nnue.cpp ---------+
        |
        v
inline repository-local headers
        |
        v
tokenize -> strip comments -> rename identifiers -> pack tokens
        |
        v
cg_input.cpp
```

`cpp_impl/codingame_nnue.cpp` is the readable source of truth for the
CodinGame bot. `cpp_impl/cg_input.cpp` is generated output. Do not hand-edit
the minified file: make the change in the readable source or generated
evaluator inputs, then regenerate it.

> **Live submission.** Since 5 October 2026 the file on CodinGame is
> `cpp_impl/cg_input_native.py`. It is the same `codingame_nnue.cpp` compiled
> by clang and carried in a Python 3 launcher, and it is about 7.5% faster
> than CodinGame's g++ build of `cg_input.cpp`. It reuses this document's
> U15 alphabet (section 4) for its payload. `cg_input.cpp` stays the
> reference and the fallback: everything below still applies, CI still gates
> it, and the native build must match its fingerprint. See
> [native_build.md](native_build.md).

## 1. Files and responsibilities

| File | Responsibility |
| --- | --- |
| `cpp_impl/codingame_nnue.cpp` | Readable, standalone CodinGame engine logic. |
| `cpp_impl/nnue_b64.hpp` | The evaluation's runtime: payload reader, start-up bake and quantization, lazy accumulator stack, AVX2 kernels. Shared with the local engines. |
| `cpp_impl/nnue_b64_net.hpp` | Generated NNUE payload (the pattern generator, CJK14) and its integer scales. |
| `tools/nnue_emit_b64_header.py` | Converts an accepted net's lane-paired BGN1 export into `nnue_b64_net.hpp`; `--check` verifies any header. |
| `cpp_impl/macro_eval.hpp` | Generated macro-context net and exact lookup table builder; the bot uses it as the macro correction history's prior. |
| `cpp_impl/d16_helpers.hpp` | The CJK14 byte decoder, the constraint helper and the horizontal sum the bot keeps from `mini_eval_d16.hpp`. |
| `cpp_impl/mini_eval_d16.hpp` | The retired D16/H8 MiniNet evaluator (Dev and Prev still include it; the paste file does not). |
| `tools/nnue_emit_macro_header.py` | Converts an accepted macro checkpoint into `macro_eval.hpp`. |
| `tools/nnue_emit_mininet_header.py` | Converts a D16/H8 checkpoint into `mini_eval_d16.hpp` (no longer shipped). |
| `tools/nnue_cjk14.py` | Deterministic 14-bits-per-character payload encoder and decoder shared by the emitters. |
| `cpp_impl/play_book.hpp` / `play_book_data.hpp` | Gameplay opening book runtime and its generated CJK14 payload ([play_book.md](play_book.md)). |
| `tools/cg_minify.py` | Bundles local headers, tokenizes C++, shortens identifiers, and emits one compact source file. |
| `cpp_impl/cg_input.cpp` | Final generated file to paste into CodinGame. |
| `tools/cg_perf_gate.py` | CI gate: checks the committed paste file is fresh, under the cap, and fast when built CodinGame's way (section 13). |
| `tools/cg_native/` | The native submission's clang build, packer and checks (`make cg-native`, `make cg-native-check`; [native_build.md](native_build.md)). |
| `cpp_impl/cg_input_native.py` | The live submission: the clang binary of the same bot, xz + U15, in a Python 3 launcher. Generated; `tools/test_cg_native.py` checks it in CI. |

The normal build entry point is:

```bash
make -C cpp_impl cg-input
```

It is equivalent to:

```bash
python3 tools/cg_minify.py cpp_impl/codingame_nnue.cpp \
  -o cpp_impl/cg_input.cpp --inline-local
```

Always provide `-o` when invoking the script manually. Without it, the CLI
overwrites its input file.

## 2. Why generated evaluator headers are separate

The neural evaluators contain tens of thousands of binary parameter bytes plus
the code that reconstructs fast runtime tables. Keeping this material in
generated headers has three advantages:

1. The main engine remains readable and reviewable.
2. Training/export code owns the exact binary layout.
3. Local builds and tests use the same data that is eventually bundled into
   the CodinGame submission.

CodinGame accepts only one pasted source file, so the minifier's
`--inline-local` mode flattens these headers into `codingame_nnue.cpp`
immediately before token minification.

## 3. Neural evaluation included in the submission

The evaluation is the pattern-generator NNUE r16_x128_l2400_s1601_rs (W1:
51,435 parameters, encoder 27-128-128-32; improvement log section 69;
r14_d5_final_s2_rs from 2026-10-04 to 2026-10-09, section 65; r13w_20 from
2026-10-01, section 64; r12_M2 from 2026-09-28, section 57; B64_d5M_57ep
before that). The paste file carries two networks:

- the NNUE's generator (section 3.1), from which the bot bakes its 25.6 MB of
  integer tables at start-up;
- the compact macro-context net (section 3.2), which the NNUE bot uses only
  as the prior of its macro correction history.

The D16/H8 MiniNet payload of the earlier hybrid evaluator (section 3.3) is no
longer in the paste file.

The minification process does not retrain, quantize, or otherwise alter any
network (the NNUE is quantized by its emitter, before the header exists). It
transports the exact accepted payload bytes into a single source file.

For the full model and training rationale, see the
[NNUE training and implementation guide](nnue_training_and_implementation.md).

### 3.1 NNUE generator payload

The baked tables would be 9 x 3^9 + 3^9 rows of 65 int16 values; the payload
is the generator that makes them. `tools/nnue_emit_b64_header.py` writes it
as one MSB-first bit stream of 11 matrices, each row one output unit with its
bias in column 0:

| Matrix | Rows x columns | Content |
| --- | --- | --- |
| `enc0`, `enc1`, `enc2` | ENC0 x 28, ENC1 x (ENC0 + 1), 32 x (ENC1 + 1): 128 x 28, 128 x 129, 32 x 129 for W1 (64 x 28, 64 x 65, 32 x 65 before it) | the encoder, one-hot 27 -> ENC0 -> ENC1 -> 32; the header's widths line (`B64_ENC0`, `B64_ENC1`) gives the widths |
| `proj` | 585 x 33 | row 65m + j: lane j of location m's projection (lane 64 is the PSQT lane) |
| `fwd` | 65 x 33 | the forced-board projection, one row per lane |
| `dec`, `con` | 65 x 27, 65 x 20 | decided and constraint rows, transposed to one row per lane |
| `bias` | 1 x 65 | the accumulator bias |
| `d1`, `d2`, `do` | 16 x 129, 32 x 17, 1 x 33 | the head, 128 -> 16 -> 32 -> 1 |

Each matrix starts with its bf16 scales (one per row; `proj`'s nine locations
share one per lane, because the accumulator needs the same absolute step for
every location), then two 4-bit Rice parameters (normal rows, PSQT rows),
then every value as Rice(zigzag(q)). A value is `float32(q) * scale`, the same
IEEE single arithmetic in numpy and in C++.

- **Bits.** enc 14, proj 12, fwd 12, dec 14, con 13, dense 14; the PSQT rows
  and the bias 14. The PSQT rows get their own scale and width because an
  error there moves the eval 500 times as far.
- **Rounding.** `proj`, `fwd` and the head are first refit by least squares
  to the float net's outputs given the already-quantized inputs; then GPTQ
  rounds column by column, feeding each rounding error back through the
  inverse input Hessian. The encoder and projections are calibrated on the
  11,093 live patterns weighted by their frequency in real positions, the
  head on positions passed through the quantized first layer; `dec`, `con`
  and `bias` round to nearest. The encoder is GPTQ-rounded but not refit: on
  this net its refit moved rare patterns' embeddings (max error 116 against 11
  at 16 bits everywhere).
- **Size.** W1: 80,027 bytes = **42,682 U15 characters** (payload sha256
  `c6d0c3ada487e829...`, pinned by `tools/test_nnue_emit_b64_header.py`).
  The wider encoder is the difference: r14_d5_final_s2_rs's payload was
  53,865 bytes = 28,728 characters (`cde8c610...`), r13w_20's 53,834 = 28,712,
  r13w_11's 53,927 = 28,762, r12_M2's 54,159 = 28,885.
- **Error.** For W1 the dequantized generator alone, in float, is 1.75 mean /
  65.1 max eval units from the float net on the 20,000 parity positions, and
  the bot's integer eval is 10.1 / 40 from the float net on the 16 positions
  `unit_tests.cpp` pins (equal to `--int-eval`'s integer reference, which
  also matches the runtime on 40,000 positions). For r14_d5_final_s2_rs the dequantized generator alone, in
  float, is 1.73 mean / 74.4 max eval units from the float net on the 20,000
  parity positions, and the bot's integer eval is 8.8 / 48 from the PyTorch
  net on the 16 positions `unit_tests.cpp` pins (r13w_20: 1.97 / 66.7 and
  6.1 / 30; r13w_11: 1.69 / 41.8 and 8.9 / 31; the 20,000-position integer
  parity was not re-measured for any of them). For r12_M2 on 20,000 positions, the bot's integer eval is 5.80
  mean / 166 max eval units from the float net (the unquantized export in the
  same integer engine: 5.63 / 211; the dequantized generator alone, in float:
  1.08 / 45). B64_d5M_57ep's were 4.72 / 163 and 0.91 / 61. A
  configuration enc 14, proj 11, fwd 12, dec 14, con 12, dense 14, psqt 14
  would free about 1,350 characters for about 0.1 more mean error, if the
  headroom is ever needed. Lossless recoding has nothing left to take: the
  quantized weights use 12-13 significant bits each (median |q| 300-1,600 on
  a 14-bit grid), and per-row Rice, Laplacian, Gaussian and adaptive
  bit-length models all land within 1% of the shipped Rice code.

At start-up `b64::load()` reads the bits straight from the U15 characters
(`b64::Bits`, no byte buffer), then bakes and quantizes the tables, inside
the 1,000 ms first turn: about 50 ms for a 64-wide encoder and about 130 ms
for W1's at -O3 on the desktop (190 ms on the Dell), about 260 ms cold for
W1 built with CodinGame's flags on the Dell (the
encoder runs only here, so its width costs nothing per node)
([nnue_training_and_implementation.md](nnue_training_and_implementation.md)
section 6). The bake runs in a fixed float order without FMA, so every
compiler bakes the same tables; the unit tests pin their hashes.

### 3.2 Macro residual payload

The macro network sees the state of all nine won/drawn miniboard cells and the
current forced-board constraint. Its exporter preprojects categorical
embeddings through a 16-unit hidden layer before packing.

Its binary layout is:

| Field | Count | Encoding | Bytes |
| --- | ---: | --- | ---: |
| Hidden bias/base | 16 | little-endian `float32` | 64 |
| Constraint projections | 10 × 16 | little-endian `float32` | 640 |
| Super-board projections | 9 × 4 × 16 | little-endian `float32` | 2,304 |
| Output weights | 16 | little-endian `float32` | 64 |
| Output bias | 1 | little-endian `float32` | 4 |
| **Total** |  |  | **3,076** |

The generated header reuses the D16 CJK14 decoder. On first load it expands
the network into:

```text
MACRO_SCORE[10][1 << 18]
```

Each 18-bit key stores two bits for each of nine super-board cells. The table
contains the exact clipped network result for every constraint/key pair. It
occupies roughly 5 MiB at runtime, but only the 3,076-byte model payload and
table-building code appear in the source.

### 3.3 The retired D16/H8 local-pattern payload

This payload shipped from round seven until the NNUE replaced it on
2026-09-27. It is no longer in the paste file; `mini_eval_d16.hpp` still
carries it for the local engines' tools and unit tests.

Every 3x3 local board has nine cells with three states: empty, mine, or
opponent. That gives:

```text
3^9 = 19,683
```

possible local patterns. The accepted network uses 16-dimensional embeddings
and an 8-unit hidden layer. To fit the source cap, the emitter clusters the
19,683 embeddings into 256 centroids in first-layer projection space. The
payload stores one centroid code per local pattern and the 256 centroid
vectors, rather than all 19,683 full embeddings.

The exact binary layout is:

| Field | Count | Encoding | Bytes |
| --- | ---: | --- | ---: |
| Ternary-pattern centroid codes | 19,683 | `uint8_t` | 19,683 |
| Centroids | 256 × 16 | little-endian `float32` | 16,384 |
| Super-board class embeddings | 4 × 16 | little-endian `float32` | 256 |
| Miniboard location embeddings | 9 × 16 | little-endian `float32` | 576 |
| Forced-board embeddings | 10 × 16 | little-endian `float32` | 640 |
| Active-board embeddings | 2 × 16 | little-endian `float32` | 128 |
| First-layer weights | 8 × 160 | little-endian `float32` | 5,120 |
| First-layer bias | 8 | little-endian `float32` | 32 |
| Output weights | 8 | little-endian `float32` | 32 |
| Output bias | 1 | little-endian `float32` | 4 |
| **Total** |  |  | **42,855** |

`tools/nnue_emit_mininet_header.py` performs the clustering, preserves the
empty-board behavior with an output-bias adjustment, packs this layout, and
emits `D16_MINI_PACK_CJK`.

At startup, `d16_mini_load_packed()`:

1. decodes the CJK14 text into a temporary byte buffer;
2. copies each field into its typed static array;
3. builds the 18-bit mask-to-centroid-code table;
4. preprojects constant first-layer terms;
5. builds the integer factor tables used by the hot evaluation path.

The textual payload is compact, while the larger speed-oriented tables exist
only in process memory and therefore do not consume source characters.

## 4. U15 payload encoding

The network payloads (the NNUE generator, the macro net and the retired D16
MiniNet) use the deterministic encoder in `tools/nnue_cjk14.py` (`encode_u15`).
The gameplay opening book packs its coded stream into the same alphabet and
decodes with the same decoder ([play_book.md](play_book.md)).

### 4.1 Why not ASCII

CodinGame measures the 100,000 cap in UTF-16 code units (a Java string
length), not bytes. A CodinGame forum user established this by testing,
reporting a consistent limit only in UTF-16 units, and the official
documentation says only "100k characters".
Every character from U+0000 through U+FFFF except surrogates is one unit. An
alphabet of 2^15 such characters therefore carries 15 payload bits per counted
character:

```text
ASCII85  8 bits per 1.25 characters  = 6.4 bits per character
Base64   8 bits per 1.33 characters  = 6.0 bits per character
CJK14                                = 14 bits per character (until 2026-09-30)
U15                                  = 15 bits per character
```

The U15 alphabet is U+3400 through U+9FFF (CJK Unified Ideographs Extension
A, the 64 Yijing hexagram symbols and the CJK Unified Ideographs: 27,648
characters) followed by U+E000 through U+F3FF (5,120 Private Use characters);
a value below 27,648 maps into the first range, the rest into the second.
None of these characters is a combining mark, a line or paragraph separator,
a bidi control, an invisible character or a character with a Unicode
normalization decomposition (`tools/test_nnue_cjk14.py` walks the whole
alphabet with `unicodedata`), so an editor or paste box has nothing to
rewrite; the private-use characters merely render as boxes. No single block
of 2^15 such characters exists, which is why the previous alphabet, CJK14
(U+4E00 through U+8DFF), stopped at 14 bits. Wider alphabets are possible:
one top CodinGame bot packs 15.875 bits per character by treating every
non-surrogate, non-separator code point as a base digit, which takes in
blocks with canonical decompositions (Hangul syllables) and a bignum decode;
that is another 4% on the payloads if it is ever needed.

`decode_cjk14` stays in `tools/nnue_cjk14.py` to read headers generated
before the switch.

The file is UTF-8 on disk. Each payload character is three UTF-8 bytes, so the
submission is larger in bytes than in counted characters.

### 4.2 Encoding algorithm

1. Treat the payload as one big-endian bit stream.
2. Emit each 15-bit group as `U+3400 + group` below 27,648, else
   `U+E000 + group - 27648`.
3. Zero-pad the final group.

Decoding yields `floor(15 * characters / 8)` bytes. When the final group
carries eight or more padding bits that is one zero byte more than the input.
Every loader sizes its reads from the known layout (`count < need` fails,
extra bytes are ignored; the NNUE reader stops at the end of its last
matrix), so the padding is harmless.

### 4.3 Raw-string delimiter safety

The generated arrays use a C++ raw string with `~` as the delimiter:

```cpp
static const char D16_MINI_PACK_CJK[] = R"~(
...payload...
)~";
```

Payload characters are all non-ASCII, so the payload can never contain the
terminating sequence `)~"`. This is a structural guarantee.

### 4.4 Decoder behavior

`d16_mini_cjk_decode()` (in the bot, from `d16_helpers.hpp`) reads the UTF-8
bytes of the ordinary narrow literal:

- skips every byte that does not start a three-byte UTF-8 sequence, including
  formatting newlines;
- reassembles each code point and subtracts U+4E00;
- shifts 14 bits into an accumulator and emits a byte whenever eight are
  available;
- checks the destination capacity and returns `-1` on overflow.

The encoding is transport-only. Decoding restores the exact original byte
sequence, including the little-endian `float32` representation expected by the
AVX2 x86 runtime. `tools/nnue_cjk14.py` also holds the decoder source the
emitters write into generated headers, and a Python mirror used by its tests.

The NNUE payload is read differently: `b64::Bits` pulls 14-bit groups
straight from the characters (skipping the formatting newlines the same way)
and hands the Rice decoder one bit field at a time, so no byte buffer exists.

The committed tests pin the payloads to:

| Payload | Bytes | Characters | Pinned hash |
| --- | ---: | ---: | --- |
| NNUE generator (W1) | 80,027 | 42,682 | sha256 `c6d0c3ada487e829...` (`tools/test_nnue_emit_b64_header.py`), plus the 16 baked tables' hashes (`unit_tests.cpp`) |
| Macro residual | 3,076 | 1,641 | FNV-1a 64 `626e29f3a8d65679` |
| D16 local evaluator (retired) | 42,855 | 24,489 | FNV-1a 64 `e35e987c17a453cf` |

The macro and D16 bytes are the same the ASCII85 encoding decoded to.

## 5. Local-header bundling

`inline_local_includes()` runs before tokenization when `--inline-local` is
enabled.

For every quoted include:

```cpp
#include "some_header.hpp"
```

the bundler:

1. resolves the path relative to the including file;
2. leaves the include unchanged if the file cannot be resolved;
3. reads and recursively processes resolved local files;
4. removes the first `#pragma once` from an inlined header;
5. tracks resolved paths in a `seen` set, preventing duplicate expansion and
   include cycles;
6. inserts the resulting header text at the include location.

Angle-bracket system includes are never inlined:

```cpp
#include <immintrin.h>
```

This produces one translation unit containing the readable engine, the NNUE
runtime and payload, the macro net, the MiniNet helpers and the opening book.
For the current submission:

```text
readable codingame_nnue.cpp: 108,020 characters
after local-header bundling: 196,506 characters
```

The bundled form is intentionally larger than the readable source. Its purpose
is completeness; token minification and compact payload encoding bring the
final file back below the cap.

## 6. C++ tokenizer

`tools/cg_minify.py` uses a purpose-built lexer rather than regular-expression
whitespace deletion. Removing characters without token awareness can silently
turn valid C++ into a different program, for example by creating `++`, `->`,
`/*`, or another multi-character token.

The tokenizer recognizes:

- whitespace;
- preprocessor lines, including backslash continuations;
- line and block comments;
- normal string literals with escapes;
- character literals with escapes;
- raw strings with arbitrary delimiters;
- identifiers;
- numeric tokens and suffixes;
- three-character operators such as `<<=`, `>>=`, and `...`;
- two-character operators such as `::`, `->`, `&&`, and `+=`;
- remaining single-character punctuation.

Whitespace and comments are discarded. Literal contents are preserved.
Preprocessor directives remain whole tokens because their line boundaries can
be semantically significant.

### 6.1 Raw payload compaction

Generated headers wrap the payload at 64 characters for readable diffs. When the
tokenizer encounters a raw string, it locates the raw delimiter and closing
sequence, treats the entire literal as one token, and removes carriage returns
and newlines from the literal body.

The readable header can therefore remain line-wrapped while `cg_input.cpp`
contains one uninterrupted payload string.

## 7. Identifier shortening

After tokenization, `rename_identifiers()` globally maps eligible user-defined
identifiers to short names.

### 7.1 Names that are preserved

The renamer does not touch:

- C++ keywords;
- common C and fixed-width integer names;
- `main`, `argc`, and `argv`;
- names inside a `std::...` nested-name specifier;
- known standard/container method names such as `data`, `size`, `push`, and
  `pop`;
- compiler builtins beginning with `__`, and `__attribute__` / `always_inline`
  explicitly (renaming the attribute to a short name makes GCC ignore it with
  only a warning, which silently undoes section 13);
- AVX intrinsic names beginning with `_mm` or `_MM`;
- `find`, `first` and `second`, which the code calls on standard containers
  and pairs;
- `j0`, `j1`, `jn`, `y0`, `y1` and `yn`, which glibc's `<math.h>` declares at
  global scope as Bessel functions. A type renamed to one of them is hidden by
  the function and the file stops compiling.

Keeping standard method spellings globally also protects user-defined
stack-like classes that expose methods such as `top()`, `push()`, and `pop()`.

### 7.2 Ranking names by expected savings

Eligible identifiers are counted and ranked by:

```text
(original_length - 1) × occurrence_count
```

This gives the shortest names to identifiers that offer the largest source
savings. A frequently used name such as `mini_board_states` is more valuable
than a longer spelling that appears once.

Candidate names follow the ice4-style sequence:

```text
a, b, ..., z, A, ..., Z, a_, aa, ab, ...
```

The first character never starts with `_`. Every identifier spelling already
present in the translation unit, and every reserved name above, is treated as
occupied, preventing collisions.
A mapping is applied only when its replacement is shorter.

### 7.3 Why the mapping is global

The minifier deliberately does not contain a complete C++ parser. It cannot
prove that two names live in disjoint scopes, so each distinct source spelling
receives at most one distinct short spelling throughout the file.

This is less compact than AST-based scope coloring, but it is much safer for
the engine's templates, AVX intrinsics, preprocessor directives, raw strings,
and C++ constructs. It also makes the transformation deterministic.

A small source edit can change identifier frequencies and therefore cascade
into many different short names in `cg_input.cpp`. Such a generated diff does
not imply a large semantic change.

### 7.4 The `#define` pass

Keywords, attributes and intrinsics survive renaming, and the bot repeats
them thousands of times (`int` 647 times, `__attribute__((always_inline))`
100 times). After renaming, `define_macros()` greedily picks the run of 1-8
tokens whose replacement saves the most characters (occurrences times the
run's length, less the define line), replaces it with a placeholder, and
repeats until no run saves eight characters. Bodies must be bracket-balanced:
the AVX intrinsics are function-like macros when CodinGame compiles without
`-O`, and an expansion that opened or closed a parenthesis across their
argument lists broke them. Every `#include` moves into the leading
preprocessor block so no header ever sees a macro, and the defines follow that
block. Finally the renamed identifiers and the macros share one ranking by use
count, so the most used tokens get the one-letter names. `stringify()`
separates a literal from a following macro name, which would otherwise lex as
a user-defined-literal suffix, and a name that is also an encoding prefix
(`u`, `U`, `L`, `R`, `u8`, `uR`, `LR`, `UR`, `u8R`) from a following literal,
which would otherwise lex as one prefixed literal. Equal gains go to the
longer run, so the choice does not depend on dictionary order. An `#include`
inside an `#if` block cannot be hoisted and stops the minifier with an error
(the bundle has none). `--no-macros` skips the pass; it costs about 15 s
against 0.05 s for renaming alone. The pass took `cg_input.cpp` from 95,927
to 80,883 characters; `tools/test_cg_minify.py` covers it.

## 8. Token reconstruction

`stringify()` joins the renamed token stream with the minimum required
separation.

A space is inserted when:

- the previous token ends with a word character and the next begins with one;
- concatenating adjacent punctuation would form a different token or a
  comment opener.

The protected punctuation pairs include:

```text
++  +=  --  -=  ->  &&  &=  ||  |=
<<   <=  >>  >=  ==  !=  *=  /=  %=
^=   /*  //  ::  ##  ..  +-
```

Preprocessor directives are placed on their own lines. Include whitespace is
compacted:

```cpp
#include <vector>
```

becomes:

```cpp
#include<vector>
```

Repeated blank lines are removed and the file ends with one newline.

## 9. What the minifier intentionally does not do

The implementation is inspired by ice4, but it is not a full optimizing C++
compiler. It does not:

- parse the source into an AST;
- eliminate dead code;
- merge declarations or expressions;
- color identifiers by scope;
- expand or rewrite the source's own macros;
- alter constants, control flow, search, or evaluation;
- compress the program into a self-extracting binary/text wrapper.

These constraints keep the output directly compilable by CodinGame and make
the transformation easier to validate.

The reserved-name lists are necessarily pragmatic rather than a complete C++
standard-library model. If new external APIs, unusual member names, or macro
patterns are introduced, compile both readable and minified sources and add a
focused minifier test where appropriate.

## 10. Current size accounting

The current generation command reports:

```text
cpp_impl/codingame_nnue.cpp 118238 (bundled 216073)
-> cpp_impl/cg_input.cpp 82191
saved 133882
cap 17809 left
```

The `saved` value compares the minified result with the fully bundled
translation unit, not with the readable top-level source.

Sizes are UTF-16 code units, which is what CodinGame counts. Everything
outside the three payload literals is ASCII, and every payload character is
one UTF-16 unit, so the unit count equals Python's `len`. It does not equal
`wc -c`: each payload character is three UTF-8 bytes, and the file is 170,837
bytes. The CLI exits with failure when output is 100,000 units or larger.

| Part of `cg_input.cpp` | UTF-16 units |
| --- | ---: |
| code (minified engine, NNUE runtime, book decoder) | 37,868 |
| NNUE generator payload (W1) | 42,682 |
| gameplay opening book payload (no book) | 0 |
| macro net payload | 1,641 |
| **total** | **82,191** (17,809 left) |

The ASCII85 conversion originally reduced the accepted 96,674-character
submission to 92,759 characters. Round nine brought it to 96,887, leaving
3,113. Replacing ASCII85 with CJK14 cut the two payloads from 57,414 to
26,247 characters, bringing the submission to 65,731 with 34,269 left. The
first (full-coverage) opening book added 8,312, for 74,043. The CodinGame
compiler work (`always_inline` attributes and the `cf_*` helpers, section 13)
and speed rounds ten and eleven brought it to 81,317, and the larger uttt.ai
opening book (13,456 payload characters against 4,656) to 90,095, with
9,905 left. The NNUE (improvement log section 56) replaced the MiniNet's
24,489 payload characters with the generator's 30,923 and took the HCE and
MiniNet code out of the bot (50,392 code characters down to 48,760): **94,897,
with 5,103 left**. The r12_M2 payload (section 57) is 25 characters longer:
**94,922, with 5,078 left**; the futility and IIR rounds brought it to 95,927.
On 2026-09-30 the `#define` pass (section 7.4) took it to 80,883, the
arithmetic-coded opening book ([play_book.md](play_book.md)) to 74,853, the
U15 alphabet (section 4) to 72,317 and dropping `evaluate_macro_fast` from
the shipped macro header to 72,105; with improvement log sections 61 and 62
and the book's net fingerprint it is **73,088, with 26,912 left**; round
thirteen's net r13w_20 (section 64) makes it **72,803, with 27,197 left**, and
round fourteen's r14_d5_final_s2_rs (section 65) **72,914, with 27,086 left**.
ProbCut (improvement log section 67) added 596 characters of code, for 73,510.
The deep-search second-player book (section 68) replaced uttt.ai's
second-player half: its payload is 1,895 characters instead of 5,778, and the
multi-move runtime (payload format 2) costs 234 characters of code, for
**69,861, with 30,139 left**. Round sixteen's W1 (section 69) has a 128-wide
encoder, a 42,682-character payload (+13,954), and ships without a book
(-1,895; the runtime's width-generic bake and the no-book guard add about 300
characters of code): **82,191, with 17,809 left**. A smaller NNUE payload
configuration would still free about 1,350 (section 3.1).

## 11. Reproducible generation procedure

### 11.1 When only engine code changed

Edit `cpp_impl/codingame_nnue.cpp`, then run:

```bash
make -C cpp_impl cg-input
```

Do not regenerate evaluator headers unless the accepted network or exporter
changed.

### 11.2 When an evaluator changed

For a new NNUE, emit its payload from the accepted net's lane-paired BGN1
export ([nnue_training_and_implementation.md](nnue_training_and_implementation.md)
section 10), then take the new table hashes and scales for the tests from
`--check`:

```bash
python3 tools/nnue_emit_b64_header.py datasets/nnue2/fast/NAME_perm.bin --label NAME
python3 tools/nnue_emit_b64_header.py --check
```

The shipped header came from
`python tools/nnue_emit_b64_header.py datasets/nnue2/r16/build/r16_x128_l2400_s1601_rs_rtB84db7aa2/r16_x128_l2400_s1601_rs_perm.bin --label r16_x128_l2400_s1601_rs`
with every other option at its default; the emitter reads the encoder widths from the net file and writes them
into the header. The same command on `datasets/nnue2/fast/r14_d5_final_s2_rs_perm.bin`, `r13w_20_perm.bin`,
`r13w_11_perm.bin`, `r12_M2_perm.bin` or `B64_d5M_57ep_perm.bin` rebuilds the previous headers' payloads byte for
byte (r14's header gains only the widths line and its four comment lines); the emitter's GPTQ calibration reads
`datasets/nnue2/d8_a.cfdg`, which is not in the repository. A net with another encoder width needs no code
change; the integer parity check is `make -C cpp_impl nnue-parity` and `bin/nnue_parity POS.cfdg OUT.txt
[scratch|incremental]` against `tools/nnue_emit_b64_header.py --int-eval POS.cfdg REF.txt` (identical files).

The macro net has its own emitter (and so does the retired MiniNet,
`tools/nnue_emit_mininet_header.py`):

```bash
python3 tools/nnue_emit_macro_header.py \
  artifacts/macro_d8h16.pt \
  -o cpp_impl/macro_eval.hpp \
  --scale 1.25 \
  --clip 2000
```

Use the exact exporter options recorded for the accepted experiment. Then
regenerate the single-file submission:

```bash
make -C cpp_impl cg-input
```

The generated headers and `cg_input.cpp` are committed so the reviewed payload
is exactly the one pasted into CodinGame.

### 11.3 Diagnostic conservative mode

To isolate identifier-renaming issues:

```bash
python3 tools/cg_minify.py cpp_impl/codingame_nnue.cpp \
  -o /tmp/cg_input.no_rename.cpp --inline-local --no-rename
```

This still bundles headers, removes comments, and packs whitespace, but keeps
user identifier spellings.

## 12. Required validation

Regeneration is not complete merely because the script reports fewer than
100,000 characters.

### 12.1 Reproducibility and size

The committed `cg_input.cpp` must be exactly the generator's output for the
committed sources:

```bash
make -C cpp_impl cg-input
git diff --exit-code cpp_impl/cg_input.cpp
```

The CodinGame performance gate's `fresh` check enforces this on every pull
request, and its `size` check the 100,000-unit cap, so no hash is pinned here.
Generation is deterministic: two runs on the same sources produce the same
bytes.

### 12.2 Compile both forms

The readable and bundled/minified sources must both compile, and the paste
file must also compile with CodinGame's own command line:

```bash
make -C cpp_impl compile-cg     # includes bin/cg_input_cgflags
```

Compiling only `codingame_nnue.cpp` does not validate include bundling,
identifier shortening, raw-string handling, or the final pasted source.

### 12.3 Unit and equivalence tests

Run:

```bash
make -j16 test
make -C cpp_impl verify
```

Coverage relevant to this pipeline includes:

- tokenizer and raw-string edge cases;
- local-include recursion and `#pragma once` removal;
- reserved-name and identifier-renaming behavior;
- punctuation spacing and preprocessor reconstruction;
- CLI output and the 100,000-character failure gate;
- CJK14 randomized round trips, all final-group lengths, and the one-unit-per-character property;
- known C++ CJK14 decoder vectors and the overflow return;
- UTF-16 unit counting for the cap, including astral characters counting double;
- exact D16 and macro payload lengths and hashes;
- the NNUE payload's sha256 and the 16 baked tables' hashes, in C++ and in the
  emitter's Python mirror of `load()`;
- NNUE evals at 16 fixed positions, and incremental == from scratch == a
  scalar reference along a search-like walk;
- scalar versus optimized evaluator paths.

### 12.4 Minifier identity

`make -C cpp_impl cg-min-check` minifies `cg_selfcheck.cpp` itself (it bundles
the CodinGame file, so every rename and macro applies to the whole program),
builds it at `-O3` and with CodinGame's flags (no `-O`, where the AVX
intrinsics are function-like macros), and requires the fixed-depth
checksums, node counts and book checksum of both at depths 5, 7 and 9 to
equal the readable build's. Run it after any minifier change; it is the gate
the `#define` pass was accepted through.

### 12.5 Behavioral and timing checks

For a representation-only payload change, verify:

1. old and new payloads decode to identical bytes;
2. fixed-depth searches produce identical scores and node counts;
3. first-move initialization timing does not regress;
4. the official 90 ms SPRT/referee path records no timeout losses.

For any change that alters model values, runtime arithmetic, or search
behavior, payload identity is no longer applicable and a normal strength SPRT
is required.

The persistent-process match protocol acknowledges `NEW` and `SYNC` before a
move timer starts. This is important because table decoding and engine
construction belong to CodinGame's 1000 ms first-turn allowance, not the
100 ms later-turn deadline. External-match workers are also pinned one per
physical core so process scheduling does not manufacture timeout regressions.

## 13. How CodinGame compiles the paste file

CodinGame builds C++ with g++ 11.2 and
`-std=gnu++17 -Werror=return-type -g -pthread`: no `-O` flag at all. The only
optimization the submission gets is what its own source asks for, which is why
`codingame_nnue.cpp` starts with `#pragma GCC optimize("O3")` *above* its
`#include`s. At a global `-O0` that pragma optimizes each function body, but
GCC then inlines only functions marked `always_inline`, so:

- every hot-path helper in `codingame_nnue.cpp` and in the eval headers
  carries `__attribute__((always_inline))` (`nnue_b64.hpp`'s per-move and
  per-evaluation helpers, and what the macro emitter writes). The NNUE's
  `eval_avx` and `Stack::sync` stay real calls, one per uncached evaluation,
  as in the build that was verified with CodinGame's compiler;
- the hot path uses `cf_array`, `cf_heap_array`, `cf_min` and `cf_max` instead
  of `std::array`, `std::vector`, `std::min` and `std::max`;
- the `#pragma GCC target("avx2,...")` line stays *after* the includes (GCC 13
  rejects it in front of libstdc++).

Before these rules the shipped bot searched about a fifth of the nodes every
local test measured (improvement log sections 47 and 52). The README's
**Compiler and local builds** section lists the checks (`cg-flags`,
`cg-speed`, the objdump call listing), and CI runs `tools/cg_perf_gate.py` in a
`gcc:11.2` container on every pull request.

The native submission ([native_build.md](native_build.md)) does not depend
on any of these rules. clang compiles the readable source at `-O3`, and the
pragmas are only hints there. Its gain over this build (+7.5% cycles) is
clang's code generation. A g++ `-O3 -march=haswell` build gains about 1%,
which shows the rules above already give the paste file -O3 code.

## 14. Troubleshooting

### Output is unexpectedly larger

- Check whether a new local header was included.
- Inspect the reported bundled size separately from the top-level source size.
- Look for large new literals; identifier renaming cannot offset arbitrary
  data growth.
- Confirm generated binary data is encoded rather than emitted as decimal
  float initializers.
- Compare `--no-rename` output to determine how much savings comes from the
  renamer.

### The readable source compiles but `cg_input.cpp` does not

- Compile a `--no-rename` output. If it works, inspect newly introduced
  library methods, compiler builtins, macros, and externally required names.
- Add the required spelling to the appropriate reserved set.
- Add a focused case to `tools/test_cg_minify.py`.
- Check raw-string delimiters and local-header resolution.
- Never repair only `cg_input.cpp`; fix the generator or readable source.

### The network fails to initialize

- For the NNUE: run `python tools/nnue_emit_b64_header.py --check` on the
  header (it decodes, bakes and prints the table hashes without C++) and the
  `nnue_*` unit tests, which compare the C++ bake with the same hashes. A
  different hash with an unchanged payload points at the bake's float order
  (an FMA, a reordered sum) rather than the payload.
- Run the payload length/hash unit test.
- Check the emitter's field order against the loader's `memcpy` order.
- Confirm all floating-point arrays are emitted as little-endian `float32`.
- Confirm the destination buffer is large enough and the decoder did not
  return `-1`.
- Verify the raw-string payload contains no hand edits.

### CodinGame shows about a fifth of the usual nodes per move

The bot prints `N<nodes>` after each searched move. About a fifth of the usual
count (with the MiniNet evaluator: around 130k at 90 ms against 600k-900k) is
the signature of a build without the optimize pragma or the `always_inline`
attributes (section 13). The NNUE bot searches about 55% of the MiniNet bot's
nodes per millisecond by design (about 1.1M nodes per move at 90 ms on the
laptop); compare with that, not with the old numbers. The usual cause
is pasting an old or wrong file: the first line of the paste must be
`#pragma GCC optimize("O3")`. If the paste is right, build it with
`make -C cpp_impl cg-flags` and look for new out-of-line calls.

### CodinGame times out on the first move

Minification reduces source size, not runtime initialization work. Profile:

- CJK14 decoding;
- the NNUE bake and quantization (about 50 ms);
- macro lookup-table construction;
- the opening book decode;
- opening warm-up search.

On the laptop with CodinGame's flags the whole first turn takes about 175-220
ms, and 474 ms at worst with two bots starting on one core at once.

The submitted engine must keep this combined initialization and first response
inside CodinGame's first-turn allowance, while later searches remain below the
external 100 ms move deadline.
