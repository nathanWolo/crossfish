# Native CodinGame submission (clang build in a Python 3 launcher)

**The live CodinGame submission is `cpp_impl/cg_input_native.py`** (from
5 October 2026, ladder submissions 41456153 and 41456253). It is a Python 3
file that carries the bot as a Linux x86-64 executable compiled by clang, and
runs it. The bot inside is the same `codingame_nnue.cpp` that
`cpp_impl/cg_input.cpp` holds, and it searches exactly the same tree: identical
node counts and checksums at depths 5, 7 and 9. clang's code generation makes
it about **7.5% faster** than CodinGame's own g++ build of the paste file. At
62 ms per move, a GSPRT [0, 5] accepted H1 after 5,400 games, which is about
**+6 to +8 Elo**.

`cpp_impl/cg_input.cpp` stays as the reference and the fallback. Its pipeline,
tests and CI gate are unchanged, and the native build must keep matching its
fingerprint. Switching back means pasting `cg_input.cpp` into the IDE with the
language set to C++ (see **Switching back**).

| File | |
| --- | --- |
| `cpp_impl/cg_input_native.py` | Generated: the submission (72,056 UTF-16 units, 27,944 under the cap). Committed, like `cg_input.cpp`. Never hand-edit it. |
| `tools/cg_native/manifest.json` | Generated: the build record. Holds the binary's and the file's sha256, the toolchain, the flags, the symbol versions, and the hashes of the sources it was built from. |
| `tools/cg_native/build.sh` | Builds the binary, checks it against CodinGame's runtime, then packs it (`make cg-native`). |
| `tools/cg_native/pack.py` | xz + U15 packer and the launcher template. |
| `tools/cg_native/native_main.cpp` | The binary's `main`: the bot, plus `selfcheck` and `match` modes. |
| `tools/cg_native/verify.sh` | Fingerprint, book and protocol checks (`make cg-native-check`). |
| `tools/test_cg_native.py` | Static checks that run in CI without clang. |

## 1. How it works

**The binary.** `native_main.cpp` includes `cg_selfcheck.cpp`, with its `main`
renamed by `build.sh`. `cg_selfcheck.cpp` includes `codingame_nnue.cpp`, with
the bot's `main` renamed. So the binary is the shipped bot plus two extra
modes:

```text
<bin>                       the bot (CodinGame protocol)
<bin> match                 the repository's NEW/APPLY/GO match protocol (roundrobin.py, SPRT workers)
<bin> selfcheck <pos> <d>   cg_selfcheck's fixed-depth fingerprint and book check
```

The launcher passes its arguments through. So
`python3 cpp_impl/cg_input_native.py selfcheck 120 9` checks the submission
itself, through every step it takes on CodinGame.

**The launcher** has 1,058 characters of stdlib-only Python 3, plus the payload:

1. **Preflight.** It checks glibc >= 2.34 (`os.confstr`) and loads
   `libstdc++.so.6` with ctypes. A failure prints a `crossfish launcher:`
   line on stderr, which shows in the IDE's error output.
2. **Decode.** The payload is one string literal in the U15 alphabet (15 bits
   per character, U+3400..U+9FFF then U+E000..U+F3FF, no surrogates;
   [minification.md](minification.md) section 4). It decodes to an xz stream
   (x86 BCJ filter + LZMA2 preset 9e), and `lzma` decompresses that.
3. **Write.** It writes the binary to an anonymous `memfd`. The fallbacks are
   `/tmp/cf_<sha12>`, then the working directory, then the script's directory.
   It writes atomically and reuses the file if it is already there.
4. **`os.execv`.** The engine replaces Python and reads CodinGame's stdin
   directly. Every failure goes to stderr.

Startup costs about **45 ms** over the binary itself: Python start, decode,
unxz and exec. That is far inside the 1,000 ms first turn. Measured on a
throttled laptop core, the first P1 turn (spawn to first output, including the
90 ms search) took 408 ms through the launcher, 325 ms for the binary directly,
and 361 ms for the CG-flags build.

**Linking is dynamic**, because nothing else fits under 100,000 characters:

| Linkage (g++ build) | Stripped | xz | U15 characters |
| --- | --- | --- | --- |
| fully static | 1,840 KB | 636 KB | ~340k |
| `-static-libstdc++ -static-libgcc` | 822 KB | 287 KB | ~153k |
| **dynamic (shipped)** | 260 KB | 142 KB | 75.6k |

The clang binary needs `libstdc++.so.6`, `libm.so.6`, `libgcc_s.so.1` and
`libc.so.6`. Its newest symbol versions are GLIBC_2.34, GLIBCXX_3.4.29 and
CXXABI_1.3.11. `build.sh` refuses any binary that needs more than CodinGame
has, or that links any other library.

## 2. CodinGame's runtime

The user measured this on 5 October 2026 with a probe bot that prints its environment from the IDE:

| | |
| --- | --- |
| Python | 3.11.5 |
| glibc | 2.36 (Debian 12 packages) |
| libstdc++ | `libstdc++.so.6.0.30` (gcc 12): up to GLIBCXX_3.4.30, CXXABI_1.3.13 |
| memfd, `/tmp` | `memfd_create` works; `/tmp` is writable |
| CPU | "Intel Core Processor (Haswell, no TSX)": AVX2, FMA, BMI2, ABM, POPCNT; `nproc` 8 |
| kernel | 5.4 |

If CodinGame moves to an older image, the launcher's preflight line says so,
and `cg_input.cpp` is the fallback. A newer image is no problem.

## 3. Building

### Toolchain

The build host must be Linux x86-64, because the output is a Linux ELF
executable. Windows cannot produce it, so build on the ThinkPad or another
Linux machine. WSL with Ubuntu 22.04 should also work, but it is untested. The host's glibc decides the binary's symbol
versions, so use **Ubuntu 22.04 / Debian 12 or older** (the shipped binary came
from Pop!_OS 22.04, glibc 2.35). A newer distribution may need GLIBC_2.38 or
later, and `build.sh` would refuse that binary.

- **clang/lld: LLVM 23.1.2**, the official release tarball
  `LLVM-23.1.2-Linux-X64.tar.zst` from
  https://github.com/llvm/llvm-project/releases/tag/llvmorg-23.1.2. Its
  sha256 is `6382de1c1a210ce5a5cc49d18bc8444d137742e7cbf9b19f4ae602bb1ab52534`.
  The GPG signature published with it verifies against LLVM's release keys
  (https://releases.llvm.org/release-keys.asc; the tarball was signed by key
  `FFB3368980F3E6BB5737145A316C56D064CACBA5`, Douglas Yung, an LLVM release
  manager). No root is needed: unpack it anywhere. The build only uses
  `clang`, `clang++`, `lld`/`ld.lld`, `llvm-objdump` and the clang resource
  directory (`lib/clang`).
- **libstdc++ headers: g++ 11** (`apt install g++-11`; the shipped build used
  11.4.0). These are the same headers CodinGame's g++ 11.2 compiles the paste
  with. **Do not use libc++**: its `std::uniform_int_distribution` draws other
  Zobrist keys from the same seed, so the search differs. g++ 12 headers would
  also work at runtime, but they produce a different binary.

```bash
sha256sum LLVM-23.1.2-Linux-X64.tar.zst      # 6382de1c...52534
gpg --verify LLVM-23.1.2-Linux-X64.tar.zst.sig LLVM-23.1.2-Linux-X64.tar.zst
mkdir -p ~/llvm-23.1.2 && tar --zstd -xf LLVM-23.1.2-Linux-X64.tar.zst -C ~/llvm-23.1.2 --strip-components=1
```

Never commit the toolchain or a binary. The binary exists in the repository
only inside the launcher.

**On a newer distribution, build in a container.** An `ubuntu:22.04` Docker
container supplies glibc 2.35 and g++ 11.4.0, which is the shipped build's
host exactly. Mount the repository and the unpacked LLVM tarball into it,
`apt install g++-11 python3 xz-utils binutils make git libicu70 libxml2`
(`ld.lld` from the release tarball needs ICU 70), then run `build.sh` with
`CF_CLANG` pointing at the mounted clang++. The tarball decompresses only
with zstd's long window (`tar -I "zstd -d --long=31" -xf ...`). The packer
writes inside the repository, so leave `CF_NATIVE_OUT` at its default. The
ProbCut build (improvement log section 67) came from such a container on a
glibc 2.39 cloud VM: it needs only GLIBC_2.34, and two builds a day apart gave
the same binary.

### Commands

```bash
CF_CLANG=~/llvm-23.1.2/bin/clang++ make cg-native     # writes cpp_impl/cg_input_native.py + tools/cg_native/manifest.json
make cg-native-check                                  # fingerprint, book, 40 protocol games
```

Optional settings: `CF_GCC_INSTALL_DIR` (default
`/usr/lib/gcc/x86_64-linux-gnu/11`), `CF_OBJDUMP`, `CF_RUN` (a command prefix
such as `taskset -c 0-3 nice -n 10` on a shared machine), `CF_PROTOCOL_GAMES`
(default 40) and `CF_NATIVE_OUT`. The build takes a few seconds.

The exact command line (`build.sh` records it in the manifest):

```text
clang++ --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11
  -std=gnu++17 -O3 -march=haswell -mtune=haswell -ffp-contract=off -pthread -fno-pie
  -fno-plt -fno-semantic-interposition -ffunction-sections -fdata-sections -flto=thin
  src/native_main.cpp
  -fuse-ld=lld -no-pie -Wl,--gc-sections -Wl,-O2 -Wl,--as-needed -Wl,--hash-style=gnu -Wl,--icf=all -s
```

### Reproducibility

Given the same toolchain (LLVM 23.1.2, g++ 11.4.0 headers, glibc 2.35 host),
the build is **bit-identical**. The PR that added this pipeline rebuilt the
binary twice from a clean export of the repository, and got the binary that is
live on CodinGame both times:

| | sha256 |
| --- | --- |
| binary (283,408 B) | `68381d0c21501eb6074b6d908ee0c003e00d587d25b10b28809adc78cba61e58` |
| xz stream (133,120 B) | `03aa1697980f67e3ba4b5fda86becaacd37e1abd54e83029a6d0e836f32e911f` |
| `cg_input_native.py` (72,056 units, 214,052 UTF-8 bytes) | `aaf96195513a17ad9f2b5607e04368167dfd629d274c2c96637e1389c0848260` |

A different clang, different libstdc++ headers or a different packer give
another binary. That is acceptable if `make cg-native-check` passes, but the
change should be deliberate. The fingerprint is the real gate, not the hash.
The launcher is fixed by the binary's bytes plus liblzma's encoder. Python
3.10 on Linux and Python 3.12 on Windows both pack the binary above into the
same file.

## 4. Checks

**`make cg-native-check`** (Linux; it builds `bin/cg_selfcheck` and
`bin/play_book_text_dump` first) runs three checks:

1. **Static.** It runs `tools/test_cg_native.py` (below).
2. **Identity.** `python3 cg_input_native.py selfcheck 120 d` must equal the
   readable C++ build's `bin/cg_selfcheck 120 d` at d = 5, 7 and 9: node
   counts, search checksum, `book=ok` and the book table checksum. The current
   values are 568,480 / 14701287764179133873, 897,652 / 253444004976430199
   and 1,900,326 / 5993005870751148654, with book=ok, 34,066 entries and table
   checksum 17441813851168678777. These are the numbers `port-check`,
   `cg-min-check` and the CG-flags build give.
3. **Protocol.** It plays 40 CodinGame-protocol games through the launcher
   (`tools/play_book_protocol_check.py`). The book check is exact: the bot must
   play from the book precisely where the text book says. The result at this
   PR (two runs on a shared laptop): 40/40 both times, later moves 90.1-90.2 ms
   median and 90.3-90.4 ms max, first turn at most 480 and 412 ms.

**CI** (`make test` runs `tools/test_cg_native.py`, with no clang):

- the committed file is byte-identical to the manifest's record, parses as
  Python 3, is under 100,000 UTF-16 units and has no surrogates;
- the payload decodes to the recorded xz stream and binary (sha256, size, an
  x86-64 ELF), and the packer's template reproduces the file from it;
- **the sources are unchanged since the build.** The manifest stores
  CRLF-normalized hashes of `codingame_nnue.cpp`, the six headers it bundles,
  `cg_selfcheck.cpp` and `native_main.cpp`. A pull request that changes the
  bot without rebuilding the native submission fails here, because the live
  file would no longer be the bot in the repository.

### The FMA hazard

With FMA enabled by `-march=haswell`, clang contracts `a*b+c` into fused
multiply-adds by default, and so does g++ in its default GNU mode. The NNUE's start-up bake
is float arithmetic, and CodinGame's g++ build of the paste does not fuse it.
A contracted bake produces slightly different integer tables, and a different
search. **`-ffp-contract=off` is required.** The fingerprint catches the
problem: a `-ffp-contract=fast` clang probe gave d5 568,400 /
16181874137760093601 and d9 1,900,242 / 7394445440528116680, failing at every
depth. A g++ probe failed at d9 (1,900,323 / 7272222239604433505).

## 5. Why it is faster, and what did not help

Speed was measured on a ThinkPad P-core. The metric is user-mode cycles
(`perf stat -e cycles:u`) for `selfcheck 120 12`, with the fixed start-up cost
subtracted. Every build searches the same nodes, so the cycle ratio is the
nps ratio. Each figure is the geometric mean of per-round paired ratios
± 2 SE, over 40 interleaved rounds:

| Build | vs CodinGame's build of `cg_input.cpp` | xz bytes |
| --- | --- | --- |
| g++ 11 native (`-O3 -march=haswell`, LTO) | +0.1% ± 1.7 | 141,816 |
| clang -O3, full LTO | +8.0% ± 1.9 | 137,320 |
| **clang -O3, thin LTO (shipped)** | **+7.5% ± 2.0** | **133,120** |
| clang FE PGO | +5.9% ± 2.0 | 141,240 |
| clang IR PGO / CS-IR PGO (30 rounds) | +7.4% ± 2.1 / +6.6% ± 2.0 | ~139,000 |

- **g++ native buys about 1%.** `codingame_nnue.cpp` already gets -O3 code
  from its `#pragma GCC optimize("O3")`, its target pragma and its
  `always_inline` helpers. So the gain comes from clang's code generation, not
  from escaping CodinGame's `-O0`.
- **PGO adds nothing** with either compiler. The clang PGO builds were trained
  on match-mode self-play, not on the benchmark positions. FE PGO against
  plain clang is -1.9% ± 2.0. A g++ PGO build was 2-4% slower.
- **Thin LTO** is as fast as full LTO and compresses 4 KB smaller.
- **UPX** (5.2.1, `--ultra-brute` or `--best --lzma`) is about 4.7 KB larger
  than xz, around 2,500 characters, and adds 9 ms to start-up. xz stays.

TomAlard reports about 20% from the same trick. Our gain is smaller because
the paste is already pragma-optimized. The numbers come from a Raptor Lake
P-core, and CodinGame's Haswell may differ by a few points either way.

### Strength

The clang binary played CodinGame's build of the paste (g++ 11, CodinGame's
flags) at **62 ms** per move. That is the Dell's CG-compute budget, so it only
approximates CG compute on these cores. Openings were paired and random, with
both engines of a game pinned to one CPU:

| | Games | W / D / L | Elo ± 95% |
| --- | --- | --- | --- |
| fixed block | 2,000 | 630 / 809 / 561 | +12.0 ± 11.8 |
| GSPRT continuation | 3,400 | 1,055 / 1,351 / 994 | +6.2 ± 6.3 |
| **all** | **5,400** | **1,685 / 2,160 / 1,555** | **+8.4 ± 5.9** |

**GSPRT [0, 5]** (α = β = 0.05) accepted **H1 at 5,400 games** (LLR +3.10).
The fixed block's W/D/L counts enter as a trinomial block, and the
continuation's pentanomial counts (39, 308, 953, 353, 47) as a pentanomial
block, combined as in `tools/sprt_merge.py`. A stopped SPRT's Elo is biased
upward. The continuation alone gives +6.2, so read the gain as **about +6 to
+8 Elo**. Neither side timed out or forfeited.

### On the ladder

In its first 258 ladder games the native file timed out once. That is the same
rate as the C++ build's.

## 6. Rules

Sending a compiled binary through a Python launcher is a known technique on
CodinGame.

- TomAlard's write-up on reaching first place in Ultimate Tic-Tac-Toe uses it
  (https://tomalard.github.io/posts/fighting-for-1-in-the-ultimate-tic-tac-toe-arena/).
  He says it is banned for active contests, but that he thinks it is fine for
  bot-programming games, which is where this ladder is. He shared the posts
  on CodinGame's main Discord.
- Agade09's CG-Send-Binary (https://github.com/Agade09/CG-Send-Binary) notes
  that it is "flagged as a cheat during contests".
- Several top UTTT bots on the leaderboard use the method.

**Never use the native file in a contest.** For one, submit `cg_input.cpp`.

## 7. Workflow for a bot change

1. Change `codingame_nnue.cpp` (or a net, or the book) and pass the usual
   gates (README, **Shipping to CodinGame**).
2. `make cg-input`, then commit `cg_input.cpp`. The CodinGame performance gate
   checks it in CI.
3. On Linux, run `make cg-native` and `make cg-native-check`. Commit
   `cpp_impl/cg_input_native.py` **and** `tools/cg_native/manifest.json`.
   Without them, `tools/test_cg_native.py` fails in CI.
4. Paste `cg_input_native.py` into the IDE with the language set to
   **Python 3**. The user submits twice and pools both agents. The README's
   CodinGame section lists the steps.

## 8. Switching back to the C++ paste

Set the IDE's language to C++ and paste `cpp_impl/cg_input.cpp`, which is
always current because CI requires it. Nothing in the engine depends on the
native file. To retire the native path for good, delete
`cpp_impl/cg_input_native.py`, `tools/cg_native/manifest.json` and
`tools/test_cg_native.py`, and say so here and in the README.
