# CodinGame submission generation and minification

This document describes how Crossfish turns the readable local engine into the
single C++ source file pasted into CodinGame. It covers the generated neural
evaluation data, its textual encoding and runtime reconstruction, local-header
bundling, the tokenizer and identifier renamer, output packing, and the checks
required before shipping a regenerated submission.

The short version is:

```text
accepted NN checkpoints
        |
        v
generated evaluator headers
  mini_eval_d16.hpp + macro_eval.hpp
        |
        +------ included by ------+
                                 |
codingame_nnue.cpp --------------+
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

## 1. Files and responsibilities

| File | Responsibility |
| --- | --- |
| `cpp_impl/codingame_nnue.cpp` | Readable, standalone CodinGame engine logic. |
| `cpp_impl/mini_eval_d16.hpp` | Generated D16/H8 local-pattern evaluator, packed payload, decoder, and runtime tables. |
| `cpp_impl/macro_eval.hpp` | Generated macro-context residual and exact runtime lookup table builder. |
| `tools/nnue_emit_mininet_header.py` | Converts an accepted D16/H8 checkpoint into `mini_eval_d16.hpp`. |
| `tools/nnue_emit_macro_header.py` | Converts an accepted macro checkpoint into `macro_eval.hpp`. |
| `tools/nnue_cjk14.py` | Deterministic 14-bits-per-character payload encoder and decoder shared by the emitters. |
| `tools/cg_minify.py` | Bundles local headers, tokenizes C++, shortens identifiers, and emits one compact source file. |
| `cpp_impl/cg_input.cpp` | Final generated file to paste into CodinGame. |

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

The shipped evaluation is hybrid:

- a fast handcrafted evaluation supplies the baseline;
- a D16/H8 local-pattern network adds learned miniboard information at selected
  search nodes;
- a compact macro residual adds learned super-board and forced-board context.

The minification process does not retrain, quantize, or otherwise alter either
network. It transports the exact accepted payload bytes into a single source
file.

For the full model and training rationale, see the
[NNUE training and implementation guide](nnue_training_and_implementation.md).

### 3.1 D16/H8 local-pattern payload

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

## 4. CJK14 payload encoding

Both evaluator payloads use the deterministic encoder in
`tools/nnue_cjk14.py`.

### 4.1 Why not ASCII

CodinGame measures the 100,000 cap in UTF-16 code units (a Java string
length), not bytes. A CodinGame forum user established this by testing,
reporting a consistent limit only in UTF-16 units, and the official
documentation says only "100k characters".
Every character from U+0000 through U+FFFF except surrogates is one unit. An
alphabet of 2^14 such characters therefore carries 14 payload bits per counted
character:

```text
ASCII85  8 bits per 1.25 characters  = 6.4 bits per character
Base64   8 bits per 1.33 characters  = 6.0 bits per character
CJK14                                = 14 bits per character
```

The alphabet is U+4E00 through U+8DFF, the first 16,384 CJK Unified
Ideographs. The block contains no combining marks, line or paragraph
separators, bidi controls, invisible characters, or characters with Unicode
normalization decompositions, so an editor or paste box has nothing to
rewrite. A 15-bit alphabet would have to span other blocks with combining
marks, which is why the encoding stops at 14 bits.

The file is UTF-8 on disk. Each payload character is three UTF-8 bytes, so the
submission is larger in bytes than in counted characters.

### 4.2 Encoding algorithm

1. Treat the payload as one big-endian bit stream.
2. Emit each 14-bit group as the character `U+4E00 + group`.
3. Zero-pad the final group.

Decoding yields `floor(14 * characters / 8)` bytes. When the final group
carries eight or more padding bits that is one zero byte more than the input.
Both loaders size their reads from the known layout (`count < need` fails,
extra bytes are ignored), so the padding is harmless. The current payloads pad
by fewer than eight bits and decode to exactly their input length.

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

`d16_mini_cjk_decode()` reads the UTF-8 bytes of the ordinary narrow literal:

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

The committed tests pin the current payloads to:

| Payload | Bytes | Characters | FNV-1a 64 |
| --- | ---: | ---: | --- |
| D16 local evaluator | 42,855 | 24,489 | `e35e987c17a453cf` |
| Macro residual | 3,076 | 1,758 | `626e29f3a8d65679` |

These are the same bytes and hashes the ASCII85 encoding decoded to.

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

This produces one translation unit containing the readable engine and both
generated evaluator implementations. For the current submission:

```text
readable codingame_nnue.cpp: 102,956 characters
after local-header bundling: 176,729 characters
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
- compiler builtins beginning with `__`;
- AVX intrinsic names beginning with `_mm` or `_MM`.

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
present in the translation unit is treated as occupied, preventing collisions.
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
- expand or rewrite macros;
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
cpp_impl/codingame_nnue.cpp 120661 (bundled 176508)
-> cpp_impl/cg_input.cpp 74043
saved 102465
cap 25957 left
```

The `saved` value compares the minified result with the fully bundled
translation unit, not with the readable top-level source.

Sizes are UTF-16 code units, which is what CodinGame counts. Everything
outside the two payload literals is ASCII, and every payload character is one
UTF-16 unit, so the unit count equals Python's `len`. It does not equal
`wc -c`: each payload character is three UTF-8 bytes, and the file is 118,225
bytes. The CLI exits with failure when output is 100,000 units or larger.

The ASCII85 conversion originally reduced the accepted 96,674-character
submission to 92,759 characters. Round nine brought it to 96,887, leaving
3,113. Replacing ASCII85 with CJK14 cut the two payloads from 57,414 to
26,247 characters, bringing the submission to 65,731 with 34,269 left. The
gameplay opening book ([play_book.md](play_book.md)) then added 8,312,
for 74,043 with 25,957 left.

## 11. Reproducible generation procedure

### 11.1 When only engine code changed

Edit `cpp_impl/codingame_nnue.cpp`, then run:

```bash
make -C cpp_impl cg-input
```

Do not regenerate evaluator headers unless the accepted network or exporter
changed.

### 11.2 When an evaluator changed

First export the accepted checkpoints:

```bash
python3 tools/nnue_emit_mininet_header.py \
  artifacts/mininet_d16h8.bin \
  -o cpp_impl/mini_eval_d16.hpp

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

Run the generator twice and confirm the output hash does not change:

```bash
make -C cpp_impl cg-input
sha256sum cpp_impl/cg_input.cpp
wc -c cpp_impl/cg_input.cpp
```

The current expected values are:

```text
SHA-256  9b2a077f3675eb2627d5a1ba946e50cbe178aec74938b3958d50a4810c993070
size     96,887 bytes
```

These values must be updated intentionally whenever the readable engine or
generated headers change.

### 12.2 Compile both forms

The readable and bundled/minified sources must both compile:

```bash
make -C cpp_impl compile-cg
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
- scalar versus optimized evaluator paths;
- NNUE incremental-state refresh consistency.

### 12.4 Behavioral and timing checks

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

## 13. Troubleshooting

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

- Run the payload length/hash unit test.
- Check the emitter's field order against the loader's `memcpy` order.
- Confirm all floating-point arrays are emitted as little-endian `float32`.
- Confirm the destination buffer is large enough and the decoder did not
  return `-1`.
- Verify the raw-string payload contains no hand edits.

### CodinGame times out on the first move

Minification reduces source size, not runtime initialization work. Profile:

- CJK14 decoding;
- D16 fast-table construction;
- macro lookup-table construction;
- opening warm-up search.

The submitted engine must keep this combined initialization and first response
inside CodinGame's first-turn allowance, while later searches remain below the
external 100 ms move deadline.
